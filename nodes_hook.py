"""
src/comfyui/nodes_hook.py — Nodos ComfyUI: KQUBE como hooks post-LoRA.

KQubeHook: Aplica pesos KQUBE como forward hooks registrados en
           modulos lineales, permitiendo que LoRAs actuen primero.

KQubeConstraint: Aplica KQUBE como capa de restriccion en paralelo.

KQubeHookRemover: Remueve hooks previamente aplicados.

v3.0 — Simplificado: carga directa de claves nativas ComfyUI.
       Los hooks no reemplazan modulos, registran forward hooks.
"""

import os
import sys
from typing import Any, List, Tuple, Optional

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
            print(f"[KQubeHook] Warning: {e}")
            raise

# Imports relativos (todos los módulos están en el mismo directorio)
try:
    from .nodes_loader import _get_unet_from_comfy_model, _list_kqube_files, _resolve_kqube_path
    from .kqube_format import load_kqube_meta, load_kqube_weights
    from .kqube_hook import CubicHook, apply_hooks_to_linears, remove_all_hooks
except ImportError as e:
    print(f"[KQubeHook] Warning: {e}")
    raise


# ============================================================
# apply_kqube_hooks
# ============================================================

def apply_kqube_hooks(
    comfy_model,
    kqube_path: str,
    strength: float = 1.0,
    hook_position: str = 'post',
    strict_dna: bool = False,
) -> Tuple[Any, str, List[CubicHook]]:
    if not os.path.exists(kqube_path):
        raise FileNotFoundError(f"Archivo .kqube no encontrado: {kqube_path}")

    meta = load_kqube_meta(kqube_path)
    family = meta.get("architecture_family", "desconocida")
    stage = meta.get("stage_signature", "?")
    r_cubic = meta.get("hyperparameters", {}).get("r_body", 16)
    version = meta.get("kqube_version", "?")

    unet, unet_type = _get_unet_from_comfy_model(comfy_model)

    compat_msg = f"Modo: family={family}"
    if version < "2.1":
        compat_msg = "Modo: sin verificacion"

    # ── 1. Crear hooks sobre los modulos lineales del UNet ──
    hooks = apply_hooks_to_linears(
        unet,
        r_cubic=r_cubic,
        hook_position=hook_position,
        filter_fn=lambda linear: linear.in_features > 8)

    # ── 2. Cargar pesos KQUBE y emparejar con hooks ──
    weights = load_kqube_weights(kqube_path)

    # Agrupar pesos por prefijo
    kqube_groups = {}
    param_suffixes = ('in_proj.weight', 'out_proj.weight', 'w_r', 'w_i', 'w_j', 'w_k')
    for key in weights:
        matched_suffix = None
        for suffix in param_suffixes:
            if key.endswith('.' + suffix):
                matched_suffix = suffix
                break
        if matched_suffix is None:
            continue
        prefix = key[:-(len(matched_suffix) + 1)]
        kqube_groups.setdefault(prefix, {})[matched_suffix] = weights[key]

    # Indexar por (in_features, 4*r)
    shape_index = {}
    for prefix, group in kqube_groups.items():
        ip = group.get('in_proj.weight')
        if ip is None:
            continue
        shape_index.setdefault((ip.shape[1], ip.shape[0]), []).append(prefix)

    # Emparejar hooks con grupos KQUBE por dimensiones del target Linear
    loaded = 0
    matched = set()
    for hook in hooks:
        in_f = hook.target.in_features
        out_f = hook.target.out_features
        cp_r4 = hook.in_proj.weight.shape[0]
        candidates = shape_index.get((in_f, cp_r4), [])
        best_prefix = None
        for prefix in candidates:
            if prefix in matched:
                continue
            op = kqube_groups[prefix].get('out_proj.weight')
            if op is not None and op.shape[0] == out_f:
                best_prefix = prefix
                break
        if best_prefix is None:
            continue
        matched.add(best_prefix)
        group = kqube_groups[best_prefix]
        try:
            with torch.no_grad():
                for pname in ('in_proj.weight', 'out_proj.weight', 'w_r', 'w_i', 'w_j', 'w_k'):
                    if pname not in group:
                        continue
                    tensor = group[pname].to(
                        device=hook.in_proj.weight.device,
                        dtype=hook.in_proj.weight.dtype)
                    target_obj = hook
                    for attr in pname.split('.'):
                        target_obj = getattr(target_obj, attr)
                    if hasattr(target_obj, 'copy_'):
                        target_obj.copy_(tensor)
            loaded += 1
        except Exception:
            pass

    if loaded > 0:
        mapping_msg = f"hooks ({loaded} cargados por shape)"
    else:
        mapping_msg = "hooks (sin coincidencias)"

    total_hooks = len(hooks)

    # ── 3. Strength ──
    if strength != 1.0:
        for hook in hooks:
            hook.alpha = strength

    log_msg = (
        f"[KQubeHook] {os.path.basename(kqube_path)}\n"
        f"  Etapa: {stage} | Familia: {family} | r={r_cubic}\n"
        f"  {total_hooks} hooks registrados ({hook_position}-forward)\n"
        f"  Cargados: {len(weights)} tensores, {loaded} hooks ({mapping_msg})\n"
        f"  Strength: {strength:.2f} | Modo: hook\n"
        f"  {compat_msg}"
    )

    return comfy_model, log_msg, hooks


# ============================================================
# apply_kqube_constraint
# ============================================================

def apply_kqube_constraint(
    comfy_model, kqube_path: str, strength: float = 1.0,
    constraint_mode: str = 'additive', strict_dna: bool = False,
) -> Tuple[Any, str]:
    if constraint_mode != 'additive':
        print(f"[KQubeConstraint] Modo '{constraint_mode}' no implementado, usando 'additive'")
    model, log_msg, hooks = apply_kqube_hooks(
        comfy_model, kqube_path, strength, hook_position='post', strict_dna=strict_dna)
    log_msg = log_msg.replace("[KQubeHook]", "[KQubeConstraint]")
    log_msg = log_msg.replace("Modo: hook", f"Modo: constraint ({constraint_mode})")
    return model, log_msg


# ============================================================
# Nodo 1: KQubeHook
# ============================================================

class KQubeHook:
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
                "hook_position": (["post", "pre"], {
                    "default": "post", "label": "Posicion del hook",
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
    FUNCTION = "apply_hook"
    CATEGORY = "KQube/Hook"
    OUTPUT_NODE = False

    def apply_hook(self, model, kqube_name: str = "", strength: float = 1.0,
                   hook_position: str = 'post', strict_dna: bool = False):
        kqube_path = _resolve_kqube_path(kqube_name)
        if not kqube_path or kqube_name.startswith("("):
            return (model, "[KQubeHook] Sin archivo .kqube seleccionado")
        try:
            modified_model, log_msg, hooks = apply_kqube_hooks(
                model, kqube_path, strength, hook_position, strict_dna)
            print(f"\033[93m{log_msg}\033[0m")
            if not hasattr(modified_model, '_kqube_hooks'):
                modified_model._kqube_hooks = []
            modified_model._kqube_hooks.extend(hooks)
            return (modified_model, log_msg)
        except Exception as e:
            error_msg = f"[KQubeHook] Error: {e}"
            print(f"\033[91m{error_msg}\033[0m")
            import traceback; traceback.print_exc()
            return (model, error_msg)


# ============================================================
# Nodo 2: KQubeConstraint
# ============================================================

class KQubeConstraint:
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
                "constraint_mode": (["additive", "gated", "residual"], {
                    "default": "additive", "label": "Modo de restriccion",
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
    FUNCTION = "apply_constraint"
    CATEGORY = "KQube/Constraint"
    OUTPUT_NODE = False

    def apply_constraint(self, model, kqube_name: str = "", strength: float = 1.0,
                         constraint_mode: str = 'additive', strict_dna: bool = False):
        kqube_path = _resolve_kqube_path(kqube_name)
        if not kqube_path or kqube_name.startswith("("):
            return (model, "[KQubeConstraint] Sin archivo .kqube")
        try:
            modified_model, log_msg = apply_kqube_constraint(
                model, kqube_path, strength, constraint_mode, strict_dna)
            print(f"\033[94m{log_msg}\033[0m")
            return (modified_model, log_msg)
        except Exception as e:
            error_msg = f"[KQubeConstraint] Error: {e}"
            print(f"\033[91m{error_msg}\033[0m")
            import traceback
            traceback.print_exc()
            return (model, error_msg)


# ============================================================
# Nodo 3: KQubeHookRemover
# ============================================================

class KQubeHookRemover:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "info")
    FUNCTION = "remove_hooks"
    CATEGORY = "KQube/Hook"
    OUTPUT_NODE = False

    def remove_hooks(self, model):
        try:
            if hasattr(model, '_kqube_hooks'):
                hooks = model._kqube_hooks
                for hook in hooks:
                    hook.remove_hook()
                removed_count = len(hooks)
                del model._kqube_hooks
                log_msg = f"[KQubeHookRemover] {removed_count} hooks removidos"
                print(f"\033[92m{log_msg}\033[0m")
                return (model, log_msg)
            else:
                return (model, "[KQubeHookRemover] No se encontraron hooks")
        except Exception as e:
            error_msg = f"[KQubeHookRemover] Error: {e}"
            print(f"\033[91m{error_msg}\033[0m")
            return (model, error_msg)
