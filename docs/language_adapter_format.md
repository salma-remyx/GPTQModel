# Per-Language Adapter Save Format

## Overview

When a model is quantized with per-language EoRA corrections
(Language-Conditional Dequantization, arXiv:2608.11786), `save_language_adapters`
writes one standard HF LoRA adapter directory per language instead of a single
adapter directory. This document is the format contract for that on-disk layout.

## Layout

```
{save_dir}/
  {language}/
    adapter_model.safetensors
    adapter_config.json
  {language}/
    adapter_model.safetensors
    adapter_config.json
  ...
```

- `{language}` is a language tag (e.g. `en`, `ko`). One directory is emitted per
  language that has at least one module correction; languages with no
  corrections are skipped with a warning.
- `adapter_model.safetensors` holds the LoRA weights for that language in the
  standard HF adapter layout: keys are
  `base_model.model.{module}.lora_A.weight` / `.lora_B.weight`, with A stored
  as `rank x in_features` and B as `out_features x rank` (transposed from the
  runtime layout on write).
- `adapter_config.json` is a standard PEFT `LoraConfig` (`r`, `lora_alpha`,
  `target_modules`, optional `rank_pattern`), so each per-language directory
  stays loadable through the existing plain `Lora` path and through PEFT
  tooling without any language-aware code.

## Loading

`LanguageAwareLora.post_init` discovers sibling per-language directories under
the adapter path (any sub-directory containing an `adapter_config.json` is
treated as a language directory) and loads each through the plain `Lora` path.
A path without language sub-directories degrades to plain `Lora` loading, so
single-adapter checkpoints produced before this format existed remain valid.

## Runtime Routing

At inference time `LanguageAwareLora.set_language(language)` routes all
subsequent forward passes to that language's correction. When no language is
set, or the requested language has no saved correction, the default language's
correction is applied, matching plain `Lora` behavior.
