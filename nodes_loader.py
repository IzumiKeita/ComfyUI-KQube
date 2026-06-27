"""
src/comfyui/nodes_loader.py — Nodos ComfyUI para cargar y apilar KQUBEs.

KQubeLoader: carga un archivo .kqube y aplica las proyecciones cubicas
             al modelo de difusion (SDXL) en tiempo de inferencia.
             Espera claves NATIVAS de ComfyUI (input_blocks/output_blocks).

KQubeStacker: apila N archivos .kqube con pesos independientes.

v3.0 — Simplificado: sin conversion Diffusers->ComfyUI.
       Los .kqube se entrenan directamente sobre el UNet de ComfyUI.
"""

import os
import sys
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

# Lazy import: comfy will be imported when actually needed
ModelPatcher = None

def _lazy_import_comfy():
    global ModelPatcher
    if ModelPatcher is None:
        try:
            from comfy.model_patcher import ModelPatcher as _MP
            ModelPatcher = _MP
        except ImportError as e:
            print(f"[KQubeLoader] Warning: Could not import comfy.model_patcher: {e}")
            raise

# ── Resolución de carpeta models/kqube ──
def _get_kqube_dir() -> str:
    """Devuelve la ruta a models/kqube (crea si no existe)."""
    try:
        import folder_paths
        kqube_dir = os.path.join(folder_paths.models_dir, "kqube")
    except Exception:
        # Fallback si no se ejecuta dentro de ComfyUI
        kqube_dir = os.path.join(os.path.dirname(__file__), "..", "models", "kqube")
    os.makedirs(kqube_dir, exist_ok=True)
    return kqube_dir


def _list_kqube_files() -> List[str]:
    """Lista archivos .kqube disponibles en models/kqube/."""
    try:
        kqube_dir = _get_kqube_dir()
        files = [f for f in os.listdir(kqube_dir) if f.endswith(".kqube")]
        files.sort()
        if not files:
            files = ["(sin archivos .kqube)"]
        return files
    except Exception:
        return ["(sin archivos .kqube)"]


def _resolve_kqube_path(filename: str) -> str:
    """Convierte nombre de archivo a ruta completa en models/kqube/."""
    if not filename or filename.startswith("("):
        return ""
    if os.path.isabs(filename):
        return filename
    return os.path.join(_get_kqube_dir(), filename)


# Imports relativos (todos los módulos están en el mismo directorio)
try:
    from .kqube_core import CubicProjection, SubspaceManager
    from .kqube_format import load_kqube_meta, load_kqube_weights
except ImportError as e:
    print(f"[KQubeLoader] Warning: could not import KQube modules: {e}")
    raise


# ============================================================
# Utilidad: extraer UNet de modelo ComfyUI
# ============================================================

def _get_unet_from_comfy_model(model) -> Tuple[Any, str]:
    _lazy_import_comfy()
    if isinstance(model, dict):
        if "model" in model:
            return _get_unet_from_comfy_model(model["model"])
        raise ValueError("Modelo ComfyUI no reconocido como dict")
    if isinstance(model, ModelPatcher):
        inner = model.model
    elif hasattr(model, "diffusion_model"):
        inner = model.diffusion_model
    else:
        inner = model
    if hasattr(inner, "diffusion_model"):
        return inner.diffusion_model, "SDXL-UNet"
    elif hasattr(inner, "model") and hasattr(inner.model, "diffusion_model"):
        return inner.model.diffusion_model, "SDXL-UNet"
    elif hasattr(inner, "transformer_blocks"):
        return inner, "Flux-Transformer"
    elif hasattr(inner, "double_blocks"):
        return inner, "SD3-MMDiT"
    return inner, "Generic"


# ============================================================
# Wrapping: reemplazar nn.Linear con CubicProjection
# ============================================================

def _find_and_wrap_linears(module: nn.Module, layer_idx: int, total: int,
                           r_cubic: int) -> int:
    wrapped = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and child.in_features > 8:
            r = _dynamic_r(layer_idx, total, name, r_cubic)
            setattr(module, name, CubicProjection(child, r))
            wrapped += 1
        elif isinstance(child, nn.Module):
            wrapped += _find_and_wrap_linears(child, layer_idx, total, r_cubic)
    return wrapped


def _dynamic_r(layer_idx: int, total: int, proj_name: str, base_r: int) -> int:
    r = base_r
    if total > 1:
        pos = layer_idx / max(1, total - 1)
        mid = 0.75 + 0.75 * (1.0 - abs(pos * 2.0 - 1.0))
        r = int(round(base_r * mid))
    if proj_name in ("to_out", "o_proj", "ff_net_2"):
        r = int(r * 1.12)
    return max(4, min(base_r * 2, r))


# ============================================================
# Carga de pesos KQUBE en el UNet
# ============================================================

def _load_and_map_weights(unet, weights: Dict[str, torch.Tensor]) -> Tuple[int, str]:
    """
    Carga pesos KQUBE directamente en el UNet.
    
    Las claves del .kqube (entrenado sobre UNet ComfyUI) deben coincidir
    exactamente con las claves del state_dict del UNet tras el wrapping.
    
    Returns:
        Tuple de (numero_de_coincidencias, mensaje_de_estado)
    """
    current_state = unet.state_dict()
    current_keys = set(current_state.keys())
    weight_keys = list(weights.keys())
    
    print(f"[KQubeLoader] {len(weight_keys)} tensores KQUBE vs {len(current_keys)} en modelo")

    # ── Coincidencia directa ──
    valid_weights = {}
    for wk in weight_keys:
        if wk in current_keys and weights[wk].shape == current_state[wk].shape:
            valid_weights[wk] = weights[wk]

    if not valid_weights:
        # ── Fallback: buscar prefijos extra ──
        for prefix in ["diffusion_model.", "model.", "model.diffusion_model."]:
            temp = {}
            for wk in weight_keys:
                fk = prefix + wk
                if fk in current_keys and weights[wk].shape == current_state[fk].shape:
                    temp[fk] = weights[wk]
            if len(temp) > len(valid_weights):
                valid_weights = temp

    loaded = len(valid_weights)
    if loaded == 0:
        print("[KQubeLoader] DIAGNOSTIC: Sin coincidencias de claves")
        print(f"  Ejemplos KQUBE: {weight_keys[:3]}")
        print(f"  Ejemplos modelo: {list(current_keys)[:3]}")
        return 0, "INCOMPATIBLE (sin coincidencias de claves)"

    try:
        missing, unexpected = unet.load_state_dict(valid_weights, strict=False)
        ratio = loaded / max(1, len(weight_keys))
        if ratio >= 0.99:
            return loaded, f"{loaded}/{len(weight_keys)} cargados (perfecto)"
        elif ratio > 0.7:
            return loaded, f"{loaded}/{len(weight_keys)} cargados (parcial)"
        else:
            return loaded, f"{loaded}/{len(weight_keys)} cargados (bajo)"
    except RuntimeError as e:
        return 0, f"Error: {str(e)[:100]}"


# ============================================================
# apply_kqube_to_model — funcion principal
# ============================================================

def apply_kqube_to_model(
    comfy_model,
    kqube_path: str,
    strength: float = 1.0,
    strict_dna: bool = False,
) -> Tuple[Any, str]:
    if not os.path.exists(kqube_path):
        raise FileNotFoundError(f"Archivo .kqube no encontrado: {kqube_path}")

    meta = load_kqube_meta(kqube_path)
    family = meta.get("architecture_family", "desconocida")
    stage = meta.get("stage_signature", "?")
    r_cubic = meta.get("hyperparameters", {}).get("r_body", 16)
    version = meta.get("kqube_version", "?")

    unet, unet_type = _get_unet_from_comfy_model(comfy_model)

    # Compatibilidad (informativo)
    compat_msg = f"Modo: family={family}"
    if version < "2.1":
        compat_msg = "Modo: sin verificacion"

    # Contar bloques para r dinamico
    blocks = []
    if hasattr(unet, "input_blocks"):
        for blk in unet.input_blocks:
            if hasattr(blk, "attentions") and blk.attentions:
                blocks.extend(blk.attentions)
        if hasattr(unet, "middle_block") and hasattr(unet.middle_block, "attentions"):
            blocks.extend(unet.middle_block.attentions)
        if hasattr(unet, "output_blocks"):
            for blk in unet.output_blocks:
                if hasattr(blk, "attentions") and blk.attentions:
                    blocks.extend(blk.attentions)
    elif hasattr(unet, "transformer_blocks"):
        blocks = list(unet.transformer_blocks)
    elif hasattr(unet, "double_blocks"):
        blocks = list(unet.double_blocks)
        if hasattr(unet, "single_blocks"):
            blocks.extend(list(unet.single_blocks))
    else:
        for child in unet.children():
            if any(isinstance(m, nn.Linear) for m in child.modules()):
                blocks.append(child)

    total = max(1, len(blocks))

    # ── 1. Wrapping ──
    total_wrapped = 0
    for idx, block in enumerate(blocks):
        total_wrapped += _find_and_wrap_linears(block, idx, total, r_cubic)

    # ── 2. Cargar pesos (claves nativas ComfyUI) ──
    weights = load_kqube_weights(kqube_path)
    loaded_count, mapping_msg = _load_and_map_weights(unet, weights)

    # ── 3. Strength ──
    if strength != 1.0:
        for m in unet.modules():
            if isinstance(m, CubicProjection):
                m.alpha = strength

    # ── 4. Subspace ──
    applied_subspace = 0
    try:
        from .kqube_format import _apply_subspace_to_model
        applied_subspace = _apply_subspace_to_model(unet, meta)
    except Exception:
        pass

    log_msg = (
        f"[KQubeLoader] {os.path.basename(kqube_path)}\n"
        f"  Etapa: {stage} | Familia: {family} | r={r_cubic}\n"
        f"  {total_wrapped} capas cubicizadas en {total} bloques ({unet_type})\n"
        f"  Cargados: {len(weights)} tensores ({mapping_msg})\n"
        f"  Strength: {strength:.2f} | Subspace: {applied_subspace}\n"
        f"  {compat_msg}"
    )

    return comfy_model, log_msg


# ============================================================
# Nodo 1: KQubeLoader
# ============================================================

class KQubeLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "kqube_name": (_list_kqube_files(), {
                    "default": "(sin archivos .kqube)",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
            },
            "optional": {
                "strict_dna": ("BOOLEAN", {
                    "default": False, "label_on": "exacto", "label_off": "familia",
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "info")
    FUNCTION = "apply_kqube"
    CATEGORY = "KQube"
    OUTPUT_NODE = False

    def apply_kqube(self, model, kqube_name: str = "", strength: float = 1.0,
                    strict_dna: bool = False):
        kqube_path = _resolve_kqube_path(kqube_name)
        if not kqube_path or kqube_name.startswith("("):
            return (model, "[KQubeLoader] Sin archivo .kqube seleccionado")
        try:
            modified_model, log_msg = apply_kqube_to_model(
                model, kqube_path, strength, strict_dna)
            print(f"\033[92m{log_msg}\033[0m")
            return (modified_model, log_msg)
        except Exception as e:
            import traceback
            error_msg = f"[KQubeLoader ERROR] {e}\n{traceback.format_exc()}"
            print(f"\033[91m{error_msg}\033[0m")
            return (model, error_msg)


# ============================================================
# Nodo 2: KQubeStacker
# ============================================================

class KQubeStacker:
    @classmethod
    def INPUT_TYPES(cls):
        kqube_list = _list_kqube_files()
        inputs = {
            "required": {
                "model": ("MODEL",),
                "mode": (["chain", "parallel"], {"default": "chain"}),
            },
            "optional": {},
        }
        for i in range(1, 5):
            inputs["optional"][f"kqube_{i}"] = (
                kqube_list,
                {"default": "(sin archivos .kqube)"},
            )
            inputs["optional"][f"weight_{i}"] = ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
            })
        return inputs

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "info")
    FUNCTION = "stack_kqubes"
    CATEGORY = "KQube"
    OUTPUT_NODE = False

    def stack_kqubes(self, model, mode="chain", **kwargs):
        logs = []
        current_model = model
        loaded_subspaces: List[Tuple[str, SubspaceManager]] = []

        for i in range(1, 5):
            kqube_name = kwargs.get(f"kqube_{i}", "")
            weight = kwargs.get(f"weight_{i}", 1.0)
            if not kqube_name or kqube_name.startswith("("):
                continue
            path = _resolve_kqube_path(kqube_name)
            if not os.path.exists(path):
                logs.append(f"[{i}] No encontrado: {path}")
                continue

            try:
                meta = load_kqube_meta(path)
                family = meta.get("architecture_family", "?")
                stage = meta.get("stage_signature", "?")

                new_subspace = SubspaceManager.from_meta(meta)
                if new_subspace is not None:
                    for prev_path, prev_subspace in loaded_subspaces:
                        collides, cos_sim, _ = SubspaceManager.check_collision(
                            prev_subspace, new_subspace)
                        if collides:
                            logs.append(
                                f"[{i}] WARN: colision subspace con "
                                f"{os.path.basename(prev_path)} (|cos|={cos_sim:.4f})")

                current_model, log_msg = apply_kqube_to_model(
                    current_model, path, strength=weight)

                if new_subspace is not None:
                    loaded_subspaces.append((path, new_subspace))

                status = (
                    f"[{i}] {stage} ({family}) w={weight:.2f}"
                    f"{' subsp=' + str(new_subspace.seed) if new_subspace else ''} OK")
                logs.append(status)
            except Exception as e:
                import traceback
                logs.append(f"[{i}] ERROR: {e}\n{traceback.format_exc()}")

        summary = f"[KQubeStacker] {mode}\n" + "\n".join(logs)
        print(f"\033[92m{summary}\033[0m")
        return (current_model, summary)
