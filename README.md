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
| `micro_engram_dtype` | unset | Engram weight dtype. Unset: same as its layer (bf16). `fp32`: fp32 master weights, so Adam's small steps don't round away in bf16 (matters most late in training, at low LR); costs ~2x the engram's share of memory (weights, gradients, Adam state). Under FSDP2 the compute stays bf16. FSDP2 only needs one dtype among the *trainable* params of a unit, so this works inside a frozen bf16 or 4-bit layer (checked on PyTorch 2.15; older versions may differ, the train-start log states the dtype). Without a 4-bit base, recent Accelerate versions may already upcast trainable params to fp32 on their own |
| `micro_engram_optimizer` | unset | Unset: Axolotl's `optimizer`. `lazy_adam`: AdamW, except that a table row whose gradient is zero this step is left exactly as it is (no drift from leftover momentum), like `SparseAdam` but on ordinary gradients, so it works with FSDP2 (rows are sharded, each GPU decides for its own rows). Other trainable params (key/value/scales) use plain AdamW. Uses `learning_rate`, `adam_beta1/2`, `adam_epsilon`, `weight_decay`; the YAML `optimizer` is ignored. Same memory and speed as AdamW: it changes how rarely-seen rows learn, not the cost. State names match AdamW, so you can switch on resume. `rowwise_adam` (needs `micro_engram_sharding: rows`): see [Row-wise Adam](#row-wise-adam-rowwise_adam) |
| `micro_engram_sharding` | unset | `rows`: the tables are split by rows across the GPUs, outside FSDP, and a batch fetches only the rows it uses. See [Row-sharded tables](#row-sharded-tables). Needs FSDP |
| `micro_engram_lr_epoch_decay` | `null` | LR schedule: linear warmup (`warmup_ratio`/`warmup_steps` as usual), then the LR is multiplied by this factor at the start of every epoch (`0.5` = halve each epoch). Replaces `lr_scheduler` / `lr_scheduler_kwargs`. Epoch boundaries come from the step count, and resume continues the curve |
| `micro_engram_stats_steps` | `10` | Every N steps, add engram stats to the training log line (output-weight size, table RMS and share of rows trained per n-gram order, gate mean and saturation). `0` = off. See *Watching training* |

The engram has `2 × heads × table_size × dim` table parameters plus two `(2·heads·dim) × hidden_size` projections. With the settings below on Gemma 4 31B, that is about 268M table parameters and 11M projection parameters.

#### Gate modes

All modes use the same tables and start as an exact no-op. `scalar` and `ple` have the same parameter count; `scalar_sqrt` adds two vectors of `hidden_size`.

- **`scalar`** (default, as in the Engram paper): one gate value per token, `α = sigmoid(rms(h)·rms(W_K·e)/√d)`, then `h += α · W_V·e`. The memory is let in or held back as a whole.
- **`scalar_sqrt`** (as in DeepSeek's released Engram code): like `scalar`, but the score goes through a signed square root, `α = sigmoid(sign(s)·√|s|)`, and both RMSNorms get learnable scales. The score is about `cos·√d`; with Gemma's `d = 5376` (`√d ≈ 73`) the plain sigmoid is already at 0.999 for a cosine of 0.1, where it barely learns. The square root keeps the gate responsive (cosine 0.1 gives 0.94), and the scales let it learn its own sharpness. Recommended for new runs; `scalar` checkpoints keep loading as `scalar`. Inside the module the two scales are stored as their difference from 1 (bf16 cannot move a value sitting at 1.0 by an Adam step: its spacing there is 1/128); saved files still hold the scales themselves, so old checkpoints load unchanged.
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

### Row-sharded tables

```yaml
micro_engram_sharding: rows
```

Without it, FSDP2 also splits each table across the GPUs, but before every use it gathers the **whole** table onto every GPU (in the forward, and again in the backward with activation checkpointing). It also builds a full-size gradient and reduce-scatters it, even though a batch only uses a few thousand rows. For a 2×1M-row engram that is ~1 GiB per gather, and the gradient buffer stays in VRAM until the next layer's reduce-scatter.

With `rows`, each GPU keeps only its own share of the rows (1/number of GPUs), on the GPU, outside FSDP:

- **Forward:** each GPU sends the row numbers its batch needs to the GPUs that own them, and gets just those rows back (`all_to_all`, a few MB per micro-batch). Rows used several times are sent once.
- **Backward:** the gradients of those rows go back to their owners, which add up every GPU's share and divide by the number of GPUs, exactly as FSDP averages gradients.
- **Memory:** the table costs 1/GPUs of its size plus its Adam state (also 1/GPUs), with no full-table gather or gradient buffer. So a bigger table is cheap.
- **Optimizer:** the tables get their own AdamW (lazy AdamW with `micro_engram_optimizer: lazy_adam`, partial row-wise Adam with `rowwise_adam`), stepped right after the Trainer's. It uses `learning_rate`, `adam_beta1/2`, `adam_epsilon` and `weight_decay`, and follows the Trainer's LR schedule (warmup, decay, `micro_engram_lr_epoch_decay`). Key/value/scales stay in FSDP with Axolotl's `optimizer`.
- **Gradient clipping:** the tables are clipped on their own to `max_grad_norm`. The Trainer's `grad_norm` no longer includes them, so their norm is logged as `engram/table_grad_norm`.
- **Checkpoints:** each checkpoint gets `micro_engram_rows/` (the **complete** engram: tables, key/value, scales, buffers, under the usual names) and `micro_engram_rows_optim/` (the tables' Adam state). Both are written before `trainer_state.json`, so a checkpoint that counts as complete has them. The checkpoint's `micro_engram_config.json` points to `micro_engram_rows`, so vLLM, llama.cpp, `assemble_engram_model.py`, `engram_stats.py` and `init_from` read it like any checkpoint. At the end of training it is also written to `output_dir/micro_engram_rows`.
- **Resume:** the rows come back from `micro_engram_rows/`, re-split if the number of GPUs changed. A checkpoint from a run *without* row sharding also works: the tables are read from `pytorch_model_fsdp_0`, but the tables' Adam state starts over (logged).
- **Frozen engrams** are row-sharded too (no full-table gathers at all), and aren't saved (their config points to their source, as before).

The results are the same as without sharding. `test_rows.py` checks this with 3 processes: same output, same gradients (also with activation checkpointing and FSDP2), same tables after training steps, the same clipping norm, and save/load/resume including a different GPU count.

Notes:
- `offload_params` doesn't move the tables to the CPU (they aren't FSDP's), so they stay on the GPU.
- Stats (`rows_learned`, ...) come from rank 0's rows, as before with FSDP shards.

#### Row-wise Adam (`rowwise_adam`)

```yaml
micro_engram_sharding: rows
micro_engram_optimizer: rowwise_adam
```

For the tables only: lazy row updates (as `lazy_adam`), and **one second moment per row** instead of one per value: the running mean of the row's g². This is "partial row-wise Adam" as FBGEMM/TorchRec use for big embedding tables. The momentum stays per value. The tables' optimizer state drops from 2× to ~1× their size. For example, a 2M-row fp32 engram pair on 5 GPUs goes from ~6.9 GB to ~5.2 GB per GPU (weights + gradients + state).

- The second moment is always fp32, also with bf16 weights. In bf16, `v = 0.999·v + 0.001·g²` changes by less than bf16's precision, so `v` can never decay and only moves up on big gradient spikes. With a constant gradient it gets stuck low (steps ~1.8× too large, shown in `test_optim.py`). With noisy, shrinking or sparse gradients, which is what engram rows get, it gets stuck high: steps come out ~0.6–0.9× of the correct size, like a lower learning rate. Plain AdamW on bf16 weights keeps its state in bf16 too, so this applies to bf16 engram runs with any per-value Adam.
- It equals AdamW whenever a row's gradient values have the same size, and approximates it otherwise.
- Key/value/scales keep the YAML `optimizer`.
- Resume works across `lazy_adam` ↔ `rowwise_adam`: per value → per row is exact (the row mean), per row → per value copies the row's value.

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
- Checkpoints: with an adapter, the Trainer saves FSDP checkpoints "adapter only", which in Accelerate means only the LoRA, so the engram would be missing (a ~100 KB `pytorch_model_fsdp_0`). The plugin patches this: checkpoints hold every trainable weight (the engram), and resume reads it back. Check the size of the first checkpoint: about 0.5 GB per 274M-parameter engram in bf16.
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
| `engram_token_diff.py` | Shows **where** the engram pushes within one reply: scores every token with the engram on and off (vLLM prompt log-probabilities, same tokens both times), lists the biggest pushes both ways, and writes an HTML heat map (green: the engram made the token likelier, red: less likely; hover for both probabilities). `run --url ... --messages chat.json --reply reply.txt --out on.json` once per config, then `render on.json off.json --out diff.html`. |

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

`micro_engram_vllm` is a vLLM plugin (no vLLM fork). Installing this package registers it automatically; it replaces vLLM's Gemma 4 classes (text-only and multimodal) with ones that run the engram at its layer(s). Models without a `micro_engram_config.json` keep running as plain Gemma 4.

Install the package **in the vLLM environment** (Axolotl is not needed there):

```bash
pip install -e /path/to/micro_engram
```

Serve the **merged** model folder, with `micro_engram_config.json` copied into it:

```bash
vllm serve /path/to/merged_model --tensor-parallel-size 4 \
    --reasoning-parser gemma4        # optional: thinking in reasoning_content
```

- **Images, video and audio:** the multimodal model is served as usual. Media placeholder tokens get no engram (training never saw them), and the text right after an image sees no earlier tokens through the engram, like the start of a sequence. Text-only serving with `--hf-overrides '{"architectures": ["Gemma4ForCausalLM"]}'` also works. (Multimodal serving is tested on CPU only, not yet on GPUs.)
- **Model Runner V2 is required.** It is vLLM's default; the plugin stops with an error if something forces the old runner.
- **A/B test:** start with `MICRO_ENGRAM_OFF=1` to load the same weights with the engram switched off.
- **vLLM version:** the plugin uses internal vLLM APIs and was written against commit `64175d3827`. Other versions may need changes.
- The startup log shows `[micro_engram] attached at layer ...` when the engram is active.

#### Choosing the engram at launch

The engram doesn't have to live in the model folder. Serve the **plain base model** and pick an engram at launch, like a LoRA adapter:

```bash
vllm serve /path/to/base_model --tensor-parallel-size 4 \
    --additional-config '{"micro_engram": "/path/to/run/checkpoint-400/micro_engram_config.json"}'
```

- `micro_engram` is a `micro_engram_config.json`, or the folder that holds it. Its `"weights"` field says where the engram is, relative to the config file: a DCP checkpoint folder (only the engram is read from it, even from a full checkpoint), a `.safetensors` file or folder, or a `.pt`.
- Training writes `"weights"` for you: FSDP checkpoints point to their `pytorch_model_fsdp_0`, a non-FSDP run's final config points to `micro_engram.pt`. For other files, add it by hand, e.g. `"weights": "my_engram.pt"`.
- Without `--additional-config` (or the `MICRO_ENGRAM_CONFIG` environment variable) the plugin uses a `micro_engram_config.json` next to the model weights (an assembled folder), and otherwise serves plain Gemma 4. A chosen engram replaces any engram already in the model folder.
- The engram must have been trained on this exact base model. Training records a fingerprint of the base weights in the config (`base_fingerprint`: a hash of the embeddings and of every projection in the first, middle and last layer, independent of the folder name). vLLM, `assemble_engram_model.py` and the HF scripts **warn** when the model they load differs, but load it anyway.
- `infer_engram.py` / `serve_engram.py` do the same with `--engram-config path/to/micro_engram_config.json`. Their `--engram-pt` accepts every format above.

---

### Serving with llama.cpp (Gemma 4)

llama.cpp has no plugin mechanism for models, so support comes as a small patch set in `llama_cpp/`: one new file (`src/llama-micro-engram.{h,cpp}`) plus 5 small edits to existing files (`micro-engram.patch`, ~25 changed lines). The engram goes **inside the GGUF**, so `llama-server` and the other tools need no new flags.

1. **Patch and build llama.cpp** (tested on llama.cpp `13ec8f6`):
   ```bash
   ./llama_cpp/apply.sh /path/to/llama.cpp
   cd /path/to/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j --target llama-server
   ```
   Only `libllama` changes (no CUDA code). To ship a prebuilt `libllama.so`, it must come from the same llama.cpp commit as the `llama-server` and ggml libraries it is used with.
2. **Add the engram to a GGUF** (any quantization; the engram is stored as BF16):
   ```bash
   pip install gguf    # or PYTHONPATH=/path/to/llama.cpp/gguf-py
   python llama_cpp/gguf_add_engram.py --gguf gemma-4-31B-it-Q4_K_M.gguf \
       --engram /path/to/output/checkpoint-400 --out gemma-4-31B-it-Q4_K_M-engram.gguf
   ```
   `--engram` takes a checkpoint folder, an assembled folder or a `micro_engram_config.json`; every engram it lists is added (all must share one token-merge map). Split GGUFs must be merged first.
3. **Run it as usual:** `llama-server -m gemma-4-31B-it-Q4_K_M-engram.gguf`. The load log shows `micro_engram: N engram(s)`.

How it works: on every batch, each token's previous two tokens come from llama.cpp's KV cache, which already stores token ids per sequence (added upstream for DeepSeek-V4.1's Engram). That covers chunked prompts, token-by-token generation, parallel slots and prompt-cache reuse. The patch also keeps that history in saved/restored slot states. The hash runs on the CPU (same formula as training), and the tables are looked up on the GPU.

Notes:
- A patched llama.cpp still loads every other GGUF normally. An **unpatched** llama.cpp refuses a GGUF that contains an engram (unknown tensors).
- Use the same tokenizer as training: the token-merge map is per token id.
- Image tokens (multimodal input) get no engram, and text right after an image sees no earlier tokens through the engram (same as vLLM).

Tested on CPU against the PyTorch engram (`llama_cpp/test/compare.py`, a tiny random Gemma 4 with two engrams per model, all three gates, f32 and bf16 storage):
- the logits match within the drift that llama.cpp shows even without an engram (~2e-2 on a tiny model)
- the same check also passes for a chunked prompt, token-by-token decoding, two sequences in one batch, and a state save/restore
- a negative control (hashing without token merging) misses by ~100x more, so a hashing bug cannot pass
- without the patch's cache fix, decoding after a state restore misses by 1.5, so the history check catches that too

Not yet tested on a GPU or with the real 31B model.

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
python test_optim.py                          # lazy_adam / rowwise_adam: vs AdamW, unused rows untouched, bf16 second-moment freeze, FSDP2 + state save/load
python test_rows.py                           # micro_engram_sharding: rows, 3 processes: == full tables (forward, gradients, training, clipping), save/resume/reshard
python test_token_diff.py                     # engram_token_diff.py against a fake vLLM server
python test_resume_keys.py                    # checkpoint-wrapper key renaming on resume
python test_vllm_ids.py                       # vLLM path: previous tokens per batch type, same output as training
python test_quant.py                          # 4-bit (NF4) base + placeholder QLoRA: dtype, freezing pattern, gradients (needs bitsandbytes, peft)
python test_multi.py                          # several engrams, init_from + frozen, config, inference == training, assemble, stats (needs transformers)
python llama_cpp/test/compare.py --llama-cpp <patched llama.cpp> --driver <engram_logits>   # llama.cpp patch vs PyTorch (needs a build; see the file)
```

---

## Notes / limitations

- It is tested on Gemma 4 only. Other decoders should work as long as `find_text_layers` finds the text decoder `ModuleList` (it prefers a `*.language_model.*layers` path and skips vision/audio towers).
- The n-gram orders are fixed at (2, 3).
- Hash multipliers are rebuilt from a fixed seed, so they are identical in every run and are not loaded from checkpoints.
