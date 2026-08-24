# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Target-aware calibration selection that preserves full-precision uncertainty.

Adapted from "Target-Aware Calibration Data Selection for Preserving Uncertainty
in Quantized Language Models" (Doubt-Preserving Quantization, DPQ;
arXiv:2608.21019).  The paper's core recipe is kept intact: score every
calibration row with the *full-precision* model's own predictive uncertainty,
then emit a target-aligned mixture of the highest-doubt rows plus generic
anchors retained in native order, so the calibration distribution keeps corpus
coverage instead of drifting onto only the hard cases (the paper's
mixture-mismatch argument for why no single recipe fits every target).

Substituted for the target repo (Mode 2): the paper scores doubt on labeled
benchmark targets such as SQuAD2 answerability or multiple-choice margins.
Calibration corpora passed to ``quantize()`` carry no labels, so the two
single-signal variants the paper names are computed directly on the model's
next-token predictive distribution instead:

- ``entropy``     mean predictive entropy over attended positions
- ``confidence``  mean ``1 - max_prob`` over attended positions

Both are the paper's own signal axes; only the quantity they are measured on
was swapped for a label-free proxy.  The paper's separate benchmark/eval suite
is intentionally out of scope here -- post-quant scoring belongs downstream.
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional

import torch

from .logger import setup_logger

# Mode prefix recognized on `calibration_dataset_sort`.
DOUBT_MODE = "doubt"

VALID_SIGNALS = ("entropy", "confidence")

# Default ratio is the paper's leading fixed recipe, DPQ-r75: 75% high-doubt
# rows balanced by 25% generic anchors.
DEFAULT_RATIO = 0.75

# Positions are reduced in blocks so a long row never materializes a second
# full [T, V] float32 copy of the logits on top of the one the forward returns.
_POSITION_CHUNK = 512


class DoubtMode(NamedTuple):
    """Parsed ``calibration_dataset_sort`` doubt specification."""

    signal: str
    ratio: float


def parse_doubt_mode(spec: Optional[str]) -> Optional[DoubtMode]:
    """Parse a ``doubt[:signal[:ratio]]`` sort spec.

    Returns ``None`` when ``spec`` is not a doubt mode so the existing
    ``asc`` / ``desc`` / ``shuffle`` values are unaffected.  Malformed doubt
    specs raise so typos do not silently fall back to native order.

    ``ratio`` accepts either ``0.75`` or the paper's ``r75`` label.
    """

    if not spec:
        return None

    parts = [part.strip() for part in spec.split(":")]
    if parts[0] != DOUBT_MODE:
        return None

    signal = "entropy"
    ratio = DEFAULT_RATIO

    if len(parts) > 3:
        raise ValueError(
            f"Calibration: invalid doubt sort `{spec}`. "
            "Expected `doubt[:signal[:ratio]]`."
        )

    if len(parts) > 1 and parts[1]:
        signal = parts[1].lower()
        if signal not in VALID_SIGNALS:
            raise ValueError(
                f"Calibration: unknown doubt signal `{parts[1]}`. "
                f"Expected one of {VALID_SIGNALS}."
            )

    if len(parts) > 2 and parts[2]:
        raw_ratio = parts[2].lower()
        is_label = raw_ratio.startswith("r")
        if is_label:
            raw_ratio = raw_ratio[1:]
        try:
            ratio = float(raw_ratio)
        except ValueError as exc:
            raise ValueError(
                f"Calibration: invalid doubt ratio `{parts[2]}`. "
                "Expected a float in (0, 1] or an `rNN` label."
            ) from exc
        if is_label:
            # `r75` is the paper's DPQ-r75 naming: a percent, not a raw fraction.
            ratio /= 100.0
        if not 0.0 < ratio <= 1.0:
            raise ValueError(
                f"Calibration: doubt ratio `{parts[2]}` must be in (0, 1]. "
                "Use a lower ratio to keep more generic anchors."
            )

    return DoubtMode(signal=signal, ratio=ratio)


def _resolve_device(model: torch.nn.Module):
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return None


def _score_row(
    model: torch.nn.Module,
    row: Dict[str, List[List[int]]],
    signal: str,
) -> float:
    """Return the mean doubt of one calibration row under ``signal``.

    Row layout matches ``prepare_calibration_dataset`` output:
    ``{"input_ids": [[...]], "attention_mask": [[...]]}``.
    """

    row_ids = row["input_ids"][0]
    row_mask = row["attention_mask"][0]

    device = _resolve_device(model)
    input_ids = torch.as_tensor([row_ids], dtype=torch.long)
    attention_mask = torch.as_tensor([row_mask], dtype=torch.long)
    if device is not None:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

    try:
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
    except Exception as exc:
        raise ValueError(
            "Calibration: doubt selection ran the full-precision model on the "
            "calibration rows and the forward failed. Doubt selection expects a "
            "text-only causal LM forward; it is not supported for models needing "
            "extra modal inputs."
        ) from exc

    # HF returns either a ModelOutput (mapping) or a bare logits tensor.
    if hasattr(output, "__getitem__"):
        logits = output["logits"]
    else:
        logits = getattr(output, "logits", None)
    if logits is None:
        raise ValueError(
            "Calibration: doubt selection requires the model forward to "
            "expose `logits`."
        )

    # [T, V]; drop the batch dim scored above.
    logits_row = logits[0]
    attended = torch.nonzero(
        attention_mask[0].to(torch.bool), as_tuple=False
    ).flatten().tolist()

    total = 0.0
    for start in range(0, len(attended), _POSITION_CHUNK):
        positions = attended[start : start + _POSITION_CHUNK]
        log_probs = torch.log_softmax(logits_row[positions].float(), dim=-1)
        probs = log_probs.exp()
        if signal == "entropy":
            total += float(-(probs * log_probs).sum(dim=-1).sum())
        else:  # confidence
            total += float((1.0 - probs.max(dim=-1).values).sum())

    if not attended:
        # Nothing to score (fully masked row); treat as a generic anchor.
        return 0.0

    return total / len(attended)


def select_doubt_calibration(
    model: Any,
    examples: List[Dict[str, List[List[int]]]],
    mode: DoubtMode,
    logger=None,
) -> List[Dict[str, List[List[int]]]]:
    """Score ``examples`` with ``model`` and return the DPQ calibration mixture.

    The high-doubt block is emitted first (descending doubt, stable on ties) so
    the selection is auditable in logs, followed by the unselected generic
    anchors in native order.  Order does not change GPTQ's Hessian -- it is a
    sum over batches -- only batch padding efficiency.
    """

    log = logger or setup_logger()

    if model is None:
        raise ValueError(
            "Calibration: doubt selection requires the full-precision model, "
            "but `model` is not available."
        )

    scores = [_score_row(model, example, mode.signal) for example in examples]

    count = len(examples)
    doubt_count = max(0, min(count, int(round(mode.ratio * count))))
    ranking = sorted(range(count), key=lambda idx: scores[idx], reverse=True)
    doubt_indices = set(ranking[:doubt_count])

    selected = [examples[idx] for idx in ranking[:doubt_count]]
    selected += [examples[idx] for idx in range(count) if idx not in doubt_indices]

    doubt_scores = [scores[idx] for idx in ranking[:doubt_count]]
    anchor_scores = [scores[idx] for idx in range(count) if idx not in doubt_indices]

    log.info(
        "Calibration: doubt selection signal=%s ratio=%s rows=%s doubt=%s anchors=%s "
        "mean_doubt[high=%.4f anchor=%.4f] top=%.4f bottom=%.4f",
        mode.signal,
        mode.ratio,
        count,
        doubt_count,
        count - doubt_count,
        (sum(doubt_scores) / len(doubt_scores)) if doubt_scores else 0.0,
        (sum(anchor_scores) / len(anchor_scores)) if anchor_scores else 0.0,
        scores[ranking[0]] if ranking else 0.0,
        scores[ranking[-1]] if ranking else 0.0,
    )

    if doubt_count == count:
        log.warn(
            "Calibration: doubt ratio %s selected every row, so no generic "
            "anchors remain. The paper's mixture-mismatch result suggests "
            "keeping some anchors to avoid drifting the calibration "
            "distribution off-target.",
            mode.ratio,
        )

    return selected


__all__ = [
    "DOUBT_MODE",
    "DEFAULT_RATIO",
    "DoubtMode",
    "VALID_SIGNALS",
    "parse_doubt_mode",
    "select_doubt_calibration",
]
