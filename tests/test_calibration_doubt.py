# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import copy

import pytest
import torch

from gptqmodel.models.base import BaseQModel
from gptqmodel.utils.calibration import prepare_calibration_dataset
from gptqmodel.utils.calibration_doubt import parse_doubt_mode


class _StubTokenizer:
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        token_ids = [max(1, ord(ch)) for ch in str(text)]
        return {
            "input_ids": torch.tensor([token_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
        }


_TOKEN_DOUBT = {1: 0.0, 2: 1.0, 3: 6.0}


class _TokenDoubtModel:
    """Full-precision stand-in whose per-position doubt is set by the token id.

    Position `t` emits near-uniform logits except that the top entry is pulled
    down by `_TOKEN_DOUBT[t]`, so predictive entropy rises monotonically with
    the doubt weight of the tokens in a row for both `entropy` and
    `confidence`. Doubt ranking then depends only on which row was scored.

    Plain class (not `nn.Module`) so it can be attached to an un-initialized
    `BaseQModel.__new__` stub the same way `test_prepare_dataset.py` attaches
    its `_DummyModel`.
    """

    def __init__(self, vocab_size: int = 8):
        self.vocab_size = vocab_size

    def __call__(self, input_ids=None, attention_mask=None):
        rows = []
        for token in input_ids[0].tolist():
            logits = torch.ones(self.vocab_size)
            logits[0] = 8.0 - _TOKEN_DOUBT.get(int(token), 0.0)
            rows.append(logits)
        return {"logits": torch.stack(rows).unsqueeze(0)}


def _make_qmodel(model) -> BaseQModel:
    qmodel = BaseQModel.__new__(BaseQModel)
    qmodel.tokenizer = _StubTokenizer()
    qmodel.support_batch_quantize = True
    qmodel.model = model
    return qmodel


def _row_dataset(rows):
    return [
        {"input_ids": [list(row)], "attention_mask": [[1] * len(row)]}
        for row in rows
    ]


def test_prepare_dataset_doubt_ignores_unattended_position_tokens():
    qmodel = _make_qmodel(_TokenDoubtModel())

    dataset = [
        {"input_ids": [[3, 3]], "attention_mask": [[1, 0]]},
        {"input_ids": [[3, 3]], "attention_mask": [[1, 1]]},
    ]

    batches = prepare_calibration_dataset(
        qmodel,
        calibration_dataset=dataset,
        calibration_dataset_sort="doubt:entropy:0.5",
        batch_size=1,
        calibration_data_min_length=0,
    )

    assert [batch["input_ids"][0].tolist() for batch in batches] == [
        [3, 3],
        [3, 3],
    ]


def test_prepare_dataset_doubt_orders_high_doubt_rows_first():
    qmodel = _make_qmodel(_TokenDoubtModel())

    batches = prepare_calibration_dataset(
        qmodel,
        calibration_dataset=_row_dataset([(1, 1), (3, 3), (1, 1)]),
        calibration_dataset_sort="doubt",
        batch_size=1,
        calibration_data_min_length=0,
    )

    # ratio defaults to the paper's DPQ-r75 -> 2 doubt rows of 3, then the anchor.
    assert [batch["input_ids"].tolist() for batch in batches] == [
        [[3, 3]],
        [[1, 1]],
        [[1, 1]],
    ]


def test_prepare_dataset_doubt_confidence_signal_and_r50_ratio():
    qmodel = _make_qmodel(_TokenDoubtModel())

    batches = prepare_calibration_dataset(
        qmodel,
        calibration_dataset=_row_dataset([(1, 1), (2, 2), (3, 3), (1, 1)]),
        calibration_dataset_sort="doubt:confidence:r50",
        batch_size=1,
        calibration_data_min_length=0,
    )

    ids = [batch["input_ids"][0].tolist() for batch in batches]
    assert len(batches) == 4
    # 50% doubt (2 of 4 rows) led by the highest-doubt rows, then the generic
    # anchors in native order.
    assert ids[:2] == [[3, 3], [2, 2]]
    assert sorted(map(tuple, ids[2:])) == [(1, 1), (1, 1)]


def test_prepare_dataset_doubt_preserves_every_row_and_native_sort_untouched():
    qmodel = _make_qmodel(_TokenDoubtModel())
    dataset = _row_dataset([(1, 1), (3, 3), (1, 1)])

    doubt_batches = prepare_calibration_dataset(
        qmodel,
        calibration_dataset=copy.deepcopy(dataset),
        calibration_dataset_sort="doubt",
        batch_size=1,
        calibration_data_min_length=0,
    )
    assert len(doubt_batches) == len(dataset)

    # The existing modes keep their behavior with the doubt hook installed.
    # Lengths 4 > 3 > 2 make `desc` order unambiguous and disjoint from doubt order.
    length_dataset = _row_dataset([(1, 1, 1, 1), (3, 3), (1, 1, 1)])
    desc_batches = prepare_calibration_dataset(
        qmodel,
        calibration_dataset=length_dataset,
        calibration_dataset_sort="desc",
        batch_size=1,
        calibration_data_min_length=0,
    )
    assert [batch["input_ids"][0].tolist() for batch in desc_batches] == [
        [1, 1, 1, 1],
        [1, 1, 1],
        [3, 3],
    ]


def test_prepare_dataset_doubt_without_model_raises():
    qmodel = BaseQModel.__new__(BaseQModel)
    qmodel.tokenizer = _StubTokenizer()
    qmodel.support_batch_quantize = True
    qmodel.model = None

    with pytest.raises(ValueError, match="full-precision model"):
        prepare_calibration_dataset(
            qmodel,
            calibration_dataset=_row_dataset([(1, 1), (3, 3)]),
            calibration_dataset_sort="doubt",
            batch_size=1,
            calibration_data_min_length=0,
        )


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("doubt", ("entropy", 0.75)),
        ("doubt:confidence", ("confidence", 0.75)),
        ("doubt:entropy:0.5", ("entropy", 0.5)),
        ("doubt:confidence:r75", ("confidence", 0.75)),
        (None, None),
        ("desc", None),
        ("shuffle", None),
    ],
)
def test_parse_doubt_mode(spec, expected):
    parsed = parse_doubt_mode(spec)
    if expected is None:
        assert parsed is None
    else:
        assert (parsed.signal, parsed.ratio) == expected


@pytest.mark.parametrize(
    "spec",
    [
        "doubt:margin",
        "doubt:entropy:0",
        "doubt:entropy:1.5",
        "doubt:entropy:a:b",
        "doubt:entropy:r200",
    ],
)
def test_parse_doubt_mode_rejects_malformed_specs(spec):
    with pytest.raises(ValueError):
        parse_doubt_mode(spec)
