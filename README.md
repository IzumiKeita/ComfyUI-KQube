# ComfyUI_KQube

Custom nodes for loading and applying KQUBE (.kqube) projection layers onto diffusion models (SDXL, Flux, SD3).

## Installation

```bash
# Option A: Git clone (recommended)
cd ComfyUI/custom_nodes
git clone https://github.com/IzumiKeita/ComfyUI_KQube.git KQube

# Option B: Download ZIP and extract as KQube/
#   Download the ZIP, extract it, and rename the folder to "KQube"
#   Copy KQube/ into ComfyUI/custom_nodes/
```

> **Important**: The folder inside `custom_nodes/` must be named exactly `KQube`.

## Requirements

```bash
pip install safetensors numpy
```

> **Note**: ComfyUI already includes `torch`, `torchvision` and `pillow`. You only need `safetensors` and `numpy` additionally.

## Nodes

### KQubeLoader
Loads a .kqube file and applies it to the model.
- **model**: SDXL
- **kqube_name**: pick a .kqube from `models/kqube/`
- **strength**: 0.0-2.0 (1.0 = full)
- **strict_dna**: if True, requires exact DNA match

### KQubeStacker
Stacks up to 4 .kqube layers (character, clothing, background, quality).
- **model**: base model
- **kqube_1..4**: .kqube files from `models/kqube/`
- **weight_1..4**: per-layer weight
- **mode**: chain (sequential) or parallel (sum)

### KQubeInspector
Previews compatibility without injecting.
- **kqube_name**: .kqube file from `models/kqube/`

### KQubeHook
Applies KQUBE weights as post-LoRA forward hooks.
- **model**: SDXL
- **kqube_name**: .kqube file from `models/kqube/`
- **strength**: scaling factor
- **hook_position**: 'post' (after LoRA) or 'pre' (before)

### KQubeConstraint
Applies KQUBE as a parallel constraint layer.

### KQubeHookRemover
Removes previously applied KQUBE hooks.

## Compatibility
A .kqube trained on SDXL base works on Pony, Illustrious, etc.
See [architecture_detector.py](architecture_detector.py) for supported families.

## Project Structure

```
KQube/
├── __init__.py              # ComfyUI node registration
├── nodes_loader.py          # KQubeLoader, KQubeStacker
├── nodes_hook.py            # KQubeHook, KQubeConstraint, KQubeHookRemover
├── nodes_inspector.py       # KQubeInspector
├── kqube_core.py            # Math modules (CubicProjection, SubspaceManager)
├── kqube_format.py          # .kqube format v3.0 (load/save)
├── kqube_hook.py            # CubicHook (forward hooks)
├── architecture_detector.py # Architecture family detector
├── utils.py                 # Shared utilities
├── requirements.txt         # Minimal dependencies
└── pyproject.toml           # Package metadata
```

## Development

```bash
# Clone for development
git clone https://github.com/IzumiKeita/ComfyUI_KQube.git
cd ComfyUI_KQube

# Install dependencies
pip install -r requirements.txt

# Run tests (if applicable)
pytest
```

## License

MIT
