# micro-engram

An [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) plugin that adds a small **hashed n-gram memory** (Engram-style) to the residual stream of a decoder model. It was built for and tested on Gemma 4 (31B).

At each position `t`, the module looks up the 2-grams and 3-grams that **end** at `t` (so it stays causal) in a set of hashed embedding tables. A learned gate decides how much of that memory goes into the hidden state after one decoder layer. Some details:

- **No data pipeline changes.** Hashes are computed directly from `input_ids`.
- **Sample packing works.** `position_ids` stop n-grams from crossing document boundaries.
- **It starts as an exact no-op.** The output projection is zero-initialised, so step 0 behaves exactly like the base model.
- **It works with FSDP2.** The engram lives inside one decoder layer, so it is sharded, offloaded, checkpointed and auto-resumed together with that layer.
- **Token merging is optional.** A "canon" map can merge token variants (e.g. `▁The` / `▁the`) so they share table rows.

---

## Installation

Install into the **same Python environment as Axolotl**:

```bash
git clone https://github.com/som1tokmynam/micro_engram.git
cd micro_engram
pip install -e .
```

To check that it worked:

```bash
python -c "from micro_engram import MicroEngramPlugin; print('ok')"
```

Requirements: `torch`, `transformers`, `axolotl` and `accelerate`, all of which an Axolotl install already provides. `safetensors` and `datasets` are only used by the helper scripts.

---

## Usage

### 1. Register the plugin

```yaml
plugins:
  - micro_engram.MicroEngramPlugin
```

> Without `pip install`: if you clone the repo into the folder you launch Axolotl from, use `micro_engram.micro_engram.MicroEngramPlugin` instead (repo folder → package folder). Don't mix the two: once the package is pip-installed, only `micro_engram.MicroEngramPlugin` works.

### 2. Configure the engram

| Key | Default | Meaning |
|---|---|---|
| `micro_engram_layer` | `0` | Inject the memory into the output of this decoder layer (0-indexed: `0` = after the first block, the Engram paper's best single placement) |
| `micro_engram_table_size` | `65536` | Rows per hash table |
| `micro_engram_dim` | `128` | Width of each table row |
| `micro_engram_heads` | `4` | Hash tables per n-gram order (orders 2 and 3 give `2 × heads` tables in total) |
| `micro_engram_save_steps` | `250` | Non-FSDP only: also save `micro_engram-step{N}.pt` every N steps (`0` = save only at the end) |
| `micro_engram_canon_path` | `null` | Token-merge map from `build_canon.py`. `null` turns merging off |
| `micro_engram_save_trainable_only` | `false` | FSDP2 only: checkpoints hold only the trainable weights (the engram, ~0.5–3 GB) instead of the whole model. Resuming reads them back and takes the frozen weights from `base_model`. See *Outputs* |
| `micro_engram_gate` | `scalar` | How the memory is mixed in: `scalar` (the paper's sigmoid gate), `scalar_sqrt` (DeepSeek's released code: signed-sqrt gate with learnable norm scales, recommended for new runs) or `ple` (a Gemma 3n PLE-style per-dimension gate). See below |
| `micro_engram_stats_steps` | `10` | Every N steps, add engram stats to the training log line (output-weight size, table RMS and share of rows trained per n-gram order, gate mean and saturation). `0` = off. See *Watching training* |

The engram has `2 × heads × table_size × dim` table parameters plus two `(2·heads·dim) × hidden_size` projections. With the settings below on Gemma 4 31B, that is about 268M table parameters and 11M projection parameters.

#### Gate modes

All modes use the same tables and start as an exact no-op. `scalar` and `ple` have the same parameter count; `scalar_sqrt` adds two vectors of `hidden_size`.

- **`scalar`** (default, as in the Engram paper): one gate value per token, `α = sigmoid(rms(h)·rms(W_K·e)/√d)`, then `h += α · W_V·e`. The memory is let in or held back as a whole.
- **`scalar_sqrt`** (as in DeepSeek's released Engram code): like `scalar`, but the score goes through a signed square root, `α = sigmoid(sign(s)·√|s|)`, and both RMSNorms get learnable scales. The score is about `cos·√d`; with Gemma's `d = 5376` (`√d ≈ 73`) the plain sigmoid is already at 0.999 for a cosine of 0.1, where it barely learns. The square root keeps the gate responsive (cosine 0.1 gives 0.94), and the scales let it learn its own sharpness. Recommended for new runs; `scalar` checkpoints keep loading as `scalar`.
- **`ple`** (Gemma 3n per-layer-embedding style): a gate per dimension, computed from the hidden state: `g = gelu(W_down·rms(h))`, then `h += W_up·(g ⊙ e)`. The context chooses *which features* of the memory to use, and each hash head's slice gets its own slice of the gate.

The gate mode is saved in `micro_engram_config.json`, and `infer_engram.py` picks it up automatically. Checkpoints from before this option are `scalar`.

### 3. Make it trainable

**Full-model freeze (recommended).** List the engram in `unfrozen_parameters`. The optional wrapper part of the regex is needed because gradient checkpointing renames the layer:

```yaml
unfrozen_parameters:
  - 'model.language_model.layers.0.(_checkpoint_wrapped_module.)?micro_engram..*'
```

Change the `0` if you change `micro_engram_layer`. You can add other patterns too, for example selected embedding rows.

If you set **neither** `unfrozen_parameters` **nor** `adapter`, the plugin freezes the whole model and trains only the engram on its own.

### 4. Settings it needs

```yaml
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false   # required: with reentrant checkpointing and frozen inputs the engram gets no gradients

sample_packing: true     # supported, as long as position_ids restart at 0 per document (Axolotl's default)
```

Hashed tables usually want a **higher learning rate** than normal fine-tuning (we use `1e-3`) and `weight_decay: 0.0`.

### Several engrams, and starting from a trained one

Each engram sits in its own decoder layer. The settings above describe the main engram; `micro_engram_extra` adds more. Any engram can start from trained weights (`init_from`) and can be kept frozen (`frozen`):

| Setting (main engram / in `micro_engram_extra`) | Meaning |
|---|---|
| `micro_engram_init_from` / `init_from` | Start from a trained engram: an output folder (its newest complete checkpoint is used), a `checkpoint-N` folder, a `micro_engram_config.json`, or the weights themselves (DCP folder, `.pt`, `.safetensors`). With a config, the engram is rebuilt exactly as trained (sizes, gate, **seed**); the YAML can't change those |
| `micro_engram_frozen` / `frozen` | Load it but don't train it (needs `init_from`). A frozen engram costs only its weights, not gradients or optimizer state |
| `micro_engram_seed` / `seed` | Hash seed. Default `10007 × layer` (DeepSeek's scheme), so each layer hashes n-grams to different rows; layer 0 keeps seed 0, as before |
| `layer`, `table_size`, `dim`, `heads`, `gate` (extra only) | `layer` is required (one engram per layer); the others default to the main engram's |

Second engram at mid depth, after the layer-0 one has finished training:

```yaml
micro_engram_layer: 0
micro_engram_init_from: /path/to/first_run_output   # its newest checkpoint
micro_engram_frozen: true
micro_engram_extra:
  - layer: 29          # pick one right after a global-attention layer (see layer_types in config.json)
    gate: scalar_sqrt
unfrozen_parameters:
  - '.+micro_engram'   # matches both; the plugin re-freezes the frozen one before training
```

Both at once: leave out `init_from`/`frozen`. Fine-tuning a 4-bit-trained engram on the bf16 base for a short final stretch: `micro_engram_init_from` without `frozen`.

Every checkpoint's `micro_engram_config.json` lists all engrams under `"engrams"`, each with its own `weights`: a trained one points to that checkpoint, a frozen one to the files it was loaded from (it is not in trainable-only checkpoints). The first engram is also written at the top level, so older loaders still see it. `infer_engram.py`, `serve_engram.py`, the vLLM plugin, `assemble_engram_model.py` and `engram_stats.py` load all of them. The training log uses `engram_L<layer>/...` keys when there is more than one.

### Training on a 4-bit base (NF4, experimental)

The frozen base can be loaded in 4-bit (bitsandbytes NF4, as in QLoRA) while the engram trains in bf16. Gemma 4 31B drops from ~62 GB to ~17 GB, so FSDP no longer needs `offload_params` and training gets much faster, with room for bigger tables. The engram is still served on the normal bf16 base.

Axolotl only loads a 4-bit base for `adapter: qlora`, so the config uses a **placeholder LoRA**: rank 1 on one module. LoRA starts as an exact no-op (its B matrix is zero), and `unfrozen_parameters` keeps it frozen, so it stays a no-op and only the engram trains:

```yaml
base_model: /path/to/bf16/model        # quantized on the fly; serve the engram on this same folder
load_in_4bit: true
adapter: qlora
lora_r: 1
lora_alpha: 1
lora_dropout: 0.0
lora_target_modules:
  - model.language_model.layers.59.self_attn.o_proj   # any single module; it stays frozen at zero

unfrozen_parameters:
  - '.+micro_engram'     # PEFT renames everything to base_model.model...; '.+' is a wildcard here

fsdp_config:
  offload_params: false              # the point of 4-bit
  cpu_ram_efficient_loading: false   # Axolotl rejects true with FSDP2 + 4-bit
micro_engram_save_trainable_only: true
```

Notes:
- **Change the `unfrozen_parameters` pattern.** The old `model.language_model.layers.0...` pattern matches nothing under PEFT (Axolotl matches from the start of the name), so the engram would stay frozen. The plugin logs an error at train start if that happens, and a warning if anything besides the engram is training (e.g. the LoRA).
- Don't set a PEFT init option (`peft_init_lora_weights` / PiSSA / OLoRA etc.): only the default init starts the LoRA at exactly zero.
- Don't merge the LoRA; it is zero. The engram is in the checkpoint as usual (`assemble_engram_model.py`, vLLM `--additional-config`).
- The engram learns against the 4-bit model's hidden states and is served on bf16 ones. Like QLoRA adapters this is usually fine, but measure it: `eval_logprob.py` on the bf16 base, engram on vs off.
- Tested on CPU only (`test_quant.py`: real bitsandbytes NF4 layers plus a PEFT LoRA). Not yet run on GPUs with FSDP2.

### Full example (Gemma 4 31B, FSDP2)

```yaml
base_model: /path/to/gemma-4-31B
sequence_len: 8192
sample_packing: true
pad_to_sequence_len: true

# --- micro-engram ---
micro_engram_layer: 0
micro_engram_table_size: 262144
micro_engram_dim: 128
micro_engram_heads: 4
micro_engram_canon_path: /path/to/canon.pt   # optional, from build_canon.py --paper

unfrozen_parameters:
  - 'model.language_model.layers.0.(_checkpoint_wrapped_module.)?micro_engram..*'

learning_rate: 1e-3
lr_scheduler: constant_with_warmup
warmup_ratio: 0.05
weight_decay: 0.0
optimizer: adamw_torch_fused
max_grad_norm: 2.0

gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false

plugins:
  - axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin
  - micro_engram.MicroEngramPlugin

fsdp_version: 2
fsdp_config:
  offload_params: true
  auto_wrap_policy: TRANSFORMER_BASED_WRAP
  transformer_layer_cls_to_wrap: Gemma4TextDecoderLayer
  state_dict_type: SHARDED_STATE_DICT
  reshard_after_forward: true
```

---

## Outputs

During training, the plugin writes `micro_engram_config.json` (layer, table size, dim, heads, vocab and canon info) to `output_dir` and to every `checkpoint-N/` folder. **Keep this file with the weights**, because inference reads it.

**Without FSDP**, the engram is saved separately as `output_dir/micro_engram.pt`, plus `micro_engram-step{N}.pt` every `micro_engram_save_steps` steps.

**With FSDP**, the engram is part of the normal checkpoint. Merge the sharded checkpoint as usual (e.g. `axolotl merge-sharded-fsdp-weights`). You can then either use the merged model directly, or pull out only the engram:

```bash
python extract_engram.py /path/to/merged_model micro_engram.pt
```

**Smaller checkpoints:** with `micro_engram_save_trainable_only: true` (FSDP2), each checkpoint holds only the trainable weights instead of the whole model, e.g. ~0.5 GB of engram instead of ~62 GB for Gemma 4 31B. Resuming works the same way, and older full checkpoints still resume. The checkpoint is then **not a servable model**: build one with `assemble_engram_model.py`, which links the base weights and adds the engram (no merge step, no copy of the base):

```bash
python assemble_engram_model.py --base /path/to/base_model \
    --engram /path/to/output/checkpoint-400/pytorch_model_fsdp_0 --out /path/to/base_model-engram
```

`--engram` also takes a merged safetensors folder or a `.pt` file. The output folder works with vLLM (plugin) and with `serve_engram.py` / `infer_engram.py`.

**Resuming** (`auto_resume_from_checkpoints: true`) is handled. The plugin patches Accelerate's FSDP state-dict loading so the engram weights are restored even with the checkpoint wrapper. If the canon file in your YAML differs from the one the checkpoint was trained with, resume **fails on purpose**, because otherwise the run would silently train on different hashes.

---

## Helper scripts

| Script | What it does |
|---|---|
| `build_canon.py` | Builds the token-merge map. `python build_canon.py --tokenizer /path/to/model --out canon.pt --paper`. By default it merges only case/NFKC variants (`▁The` = `▁the`). Each option merges more: `--merge-space` (`▁the` = `the`), `--strip-accents` (`é` = `e`, Latin/Greek/Cyrillic diacritics only) and `--merge-whitespace` (all whitespace-only pieces share one id). `--paper` turns on all three, like the paper's tokenizer compression. `--protect names.txt` (one name per line, `#` comments, repeatable) keeps names that are also ordinary words apart from the word, e.g. `Peril` the dragon vs `peril`: a single-token name keeps its own id while the word forms still merge with each other. Names split into several tokens are listed and left alone. Special, added and byte tokens are never merged. It prints the class count for each step so you can compare. |
| `engram_data_stats.py` | Estimates how many n-grams your data trains and how crowded each table size would be. `python engram_data_stats.py /path/to/last_run_prepared/<hash> [--max-tokens N]`. Use it to pick `micro_engram_table_size`. |
| `assemble_engram_model.py` | Builds a servable model folder from the base weights (symlinked, not copied) plus an engram from a DCP checkpoint, a safetensors folder or a `.pt`. See *Outputs*. |
| `extract_engram.py` | Pulls the engram tensors out of merged safetensors into a standalone `.pt` file. |
| `infer_engram.py` | Test generation with the engram attached. See below. |
| `engram_stats.py` | Engram weight stats from checkpoints on disk, **while training keeps running** (CPU only, reads only the engram tensors). `python engram_stats.py /path/to/output_dir` (newest checkpoint), `--all` (every checkpoint on disk), `--watch 120` (print each new checkpoint as it appears). Skips checkpoints that are still being written. |
| `eval_logprob.py` | Measures how well a served model predicts held-out text, through vLLM's prompt log-probabilities, and compares two configs (e.g. with and without the engram): loss, perplexity, per-chunk win rate and a 95% interval. `run --url ... --data heldout/ --out a.json` once per config, then `compare a.json b.json`. |

### Inference

```bash
# merged model (copy micro_engram_config.json into the model folder first)
python infer_engram.py --model /path/to/merged_model --prompt "Hello there"

# compare against the same model with the engram turned off
python infer_engram.py --model /path/to/merged_model --prompt "Hello there" --no-engram

# base model + separate engram file
python infer_engram.py --model /path/to/base_model --engram-pt micro_engram.pt \
    --engram-config micro_engram_config.json --prompt "Hello there"
```

Engram settings are resolved in this order: explicit flag (`--layer`, `--table-size`, `--dim`, `--heads`), then `micro_engram_config.json`, then the shapes of the saved tensors, then the defaults.

> The engram is not part of the Hugging Face model class. Plain `transformers` and llama.cpp loading will **ignore** it. Use `infer_engram.py` / `serve_engram.py`, or vLLM with the plugin below.

### Serving with vLLM

`micro_engram_vllm` is a vLLM plugin (no vLLM fork). Installing this package registers it automatically; it replaces vLLM's text-only Gemma 4 class with one that runs the engram at its layer. Models without a `micro_engram_config.json` keep running as plain Gemma 4.

Install the package **in the vLLM environment** (Axolotl is not needed there):

```bash
pip install -e /path/to/micro_engram
```

Serve the **merged** model folder, with `micro_engram_config.json` copied into it:

```bash
vllm serve /path/to/merged_model --tensor-parallel-size 4 \
    --hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}' \
    --reasoning-parser gemma4        # optional: thinking in reasoning_content
```

- **`--hf-overrides` is required.** The plugin runs on the text-only architecture; without it, vLLM starts the multimodal one and the plugin stops with an error.
- **Model Runner V2 is required.** It is vLLM's default; the plugin stops with an error if something forces the old runner.
- **A/B test:** start with `MICRO_ENGRAM_OFF=1` to load the same weights with the engram switched off.
- **vLLM version:** the plugin uses internal vLLM APIs and was written against commit `64175d3827`. Other versions may need changes.
- The startup log shows `[micro_engram] attached at layer ...` when the engram is active.

#### Choosing the engram at launch

The engram doesn't have to live in the model folder. Serve the **plain base model** and pick an engram at launch, like a LoRA adapter:

```bash
vllm serve /path/to/base_model --tensor-parallel-size 4 \
    --hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}' \
    --additional-config '{"micro_engram": "/path/to/run/checkpoint-400/micro_engram_config.json"}'
```

- `micro_engram` is a `micro_engram_config.json`, or the folder that holds it. Its `"weights"` field says where the engram is, relative to the config file: a DCP checkpoint folder (only the engram is read from it, even from a full checkpoint), a `.safetensors` file or folder, or a `.pt`.
- Training writes `"weights"` for you: FSDP checkpoints point to their `pytorch_model_fsdp_0`, a non-FSDP run's final config points to `micro_engram.pt`. For other files, add it by hand, e.g. `"weights": "my_engram.pt"`.
- Without `--additional-config` (or the `MICRO_ENGRAM_CONFIG` environment variable) the plugin uses a `micro_engram_config.json` next to the model weights (an assembled folder), and otherwise serves plain Gemma 4. A chosen engram replaces any engram already in the model folder.
- The engram must have been trained on this exact base model. Training records a fingerprint of the base weights in the config (`base_fingerprint`: a hash of the embeddings and of every projection in the first, middle and last layer, independent of the folder name). vLLM, `assemble_engram_model.py` and the HF scripts **warn** when the model they load differs, but load it anyway.
- `infer_engram.py` / `serve_engram.py` do the same with `--engram-config path/to/micro_engram_config.json`. Their `--engram-pt` accepts every format above.

---

## Watching training

With `micro_engram_stats_steps` (default every 10 steps) the training log line gets `engram/...` entries, which also reach wandb/tensorboard:

| Entry | Meaning | What to look for |
|---|---|---|
| `engram/out_abs_mean` | mean \|weight\| of the output layer (`value`, or `up` for `ple`) | starts at 0 and should grow; 0 means it is not training |
| `engram/table_rms_2g`, `_3g` | RMS of the 2-gram / 3-gram tables (start at 0.02) | slowly rising |
| `engram/rows_learned_2g`, `_3g` | share of rows whose RMS grew past 1.5× the start | how much of each table is in use; 3-grams fill more slowly |
| `engram/gate_mean` | mean gate value over the tokens since the last log (scalar gates) | not stuck at 0 or 1 |
| `engram/gate_saturated` | share of tokens with gate > 0.99 | high values mean an on/off gate that learns slowly |
| `engram/q_scale_mean`, `k_scale_mean` | learnable norm scales (`scalar_sqrt`) | start at 1; drifting shows the gate tuning its sharpness |

Under FSDP the numbers come from rank 0's shard of each weight, a random sample of rows, so no extra communication is needed.

For a run that started without this option (or to look at past checkpoints), `engram_stats.py` reads the same weight stats from the checkpoints on disk without stopping training.

---

## Tests (CPU only)

Run them from the repo root:

```bash
python test_gate.py                           # all gate modes: no-op start, gradients, bf16, checkpoint names; weight stats
python test_canon.py                          # token-merge hashing and state-dict compatibility
python test_resume_keys.py                    # checkpoint-wrapper key renaming on resume
python test_vllm_ids.py                       # vLLM path: previous tokens per batch type, same output as training
python test_quant.py                          # 4-bit (NF4) base + placeholder QLoRA: dtype, freezing pattern, gradients (needs bitsandbytes, peft)
python test_multi.py                          # several engrams, init_from + frozen, config, inference == training, assemble, stats (needs transformers)
```

---

## Notes / limitations

- It is tested on Gemma 4 only. Other decoders should work as long as `find_text_layers` finds the text decoder `ModuleList` (it prefers a `*.language_model.*layers` path and skips vision/audio towers).
- The n-gram orders are fixed at (2, 3).
- Hash multipliers are rebuilt from a fixed seed, so they are identical in every run and are not loaded from checkpoints.
