# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Protenix is an open-source AlphaFold3-based biomolecular structure prediction model by ByteDance. It predicts 3D structures of proteins, nucleic acids, and small molecules.

## Common Commands

```bash
# Install (development)
pip install -e .
pip install pre-commit && pre-commit install

# Inference
protenix pred -i examples/input.json -o ./output -n protenix_base_default_v1.0.0
protenix pred --input examples/input.json --use_msa false --enable_cache true

# Data conversion
protenix json --input ./examples/7pzb.pdb --out_dir ./output

# MSA/Template preprocessing
protenix prep --input examples/input.json --out_dir ./output

# Training (single GPU)
python runner/train.py --model_name "protenix_base_default_v1.0.0" [ARGS]

# Training (multi-GPU)
torchrun --nproc_per_node=8 runner/train.py --model_name "protenix_base_default_v1.0.0" [ARGS]

# Demo scripts
bash inference_demo.sh <model_name> <input_json> <output_dir> <dtype> <use_msa>
bash train_demo.sh
bash finetune_demo.sh
```

## Architecture Overview

### Main Forward Pass (Algorithm 1 in AF3)

`protenix/model/protenix.py` - `Protenix` class is the top-level model:
1. `get_pairformer_output()`: Input embedding → Template/MSA → Pairformer → `(s_inputs, s, z)`
2. `_main_inference_loop()`: Pairformer output → diffusion sampling → confidence head
3. `forward()`: Routes to `main_train_loop()` or `main_inference_loop()`

### Key Model Components

| Module | File | Description |
|--------|------|-------------|
| `InputFeatureEmbedder` | `modules/embedders.py` | Atom-level features → token embeddings (Algorithm 2) |
| `ConstraintEmbedder` | `modules/embedders.py` | Contact/pocket/substructure constraints → pair bias `z_constraint` |
| `PairformerStack` | `modules/pairformer.py` | Core triangular attention on pair repr `(s, z)` |
| `MSAModule` | `modules/pairformer.py` | MSA processing |
| `TemplateEmbedder` | `modules/pairformer.py` | Template structural features |
| `DiffusionModule` | `modules/diffusion.py` | Atom-level denoising network |
| `ConfidenceHead` | `modules/confidence.py` | pLDDT, PAE, PDE, resolved score prediction |

### Diffusion Sampling

`protenix/model/generator.py` - `sample_diffusion()`:
- Implements Algorithm 18 (AF3): iterative denoising from high to low noise
- Inner loop at `_chunk_sample_diffusion()`: for each step, adds noise → denoises via `denoise_net` → Euler step
- `x_denoised` computed at each step is the key point for restraint injection

### Data Pipeline

`protenix/data/inference/` - Inference path:
- `json_parser.py`: Parses input JSON → sequences/constraints
- `json_to_feature.py`: Converts to feature tensors
- `infer_dataloader.py`: Batches for inference

`protenix/data/constraint/constraint_featurizer.py` - `ConstraintFeatureGenerator`:
- `generate_from_json()`: Contact/pocket/substructure specs → feature tensors
- Constraint features added to `z_init` (pair repr) before Pairformer

### Constraint System (Existing)

Protenix already supports **soft constraints** via pairformer bias:
- **Contact**: distance range (min/max) between token/atom pairs across chains
- **Pocket**: binder chain ↔ pocket residues proximity
- **Substructure**: distance map from known structure fragment

Constraints are embedded into `z_init` at `protenix.py:get_pairformer_output()` (line ~204-228), before the recycling loop.

### Configuration System

`configs/` + `protenix/config/config.py`:
- Hierarchical config via `ml_collections`
- Model variants defined as named configs (e.g., `protenix_base_default_v1.0.0`)
- Override via CLI: `--sample_diffusion.N_sample 5`

### Key Config Parameters for Inference

```python
configs.sample_diffusion.N_sample      # number of output structures
configs.sample_diffusion.N_step        # diffusion steps (default 200)
configs.model.N_cycle                  # recycling iterations
configs.infer_setting.chunk_size       # memory/speed tradeoff
configs.enable_diffusion_shared_vars_cache  # caches pair_z between samples
```

## Input JSON Format

The `constraint` field in input JSON (see `docs/infer_json_format.md`) enables soft constraints:
```json
{
  "constraint": {
    "contact": [{"entity1":1,"copy1":1,"position1":5,"entity2":2,"copy2":1,"position2":10,"max_distance":6}],
    "pocket": {"binder_chain":{"entity":2,"copy":1},"contact_residues":[...],"max_distance":6}
  }
}
```

## Implementation Plan: Restraint-Guided Inference

See memory file `restraint_plan.md` for the full implementation plan.
