# KQube — Custom Nodes para ComfyUI
# Este archivo hace que KQube/ sea un paquete Python válido,
# permitiendo imports relativos entre módulos.

import sys, os

# HACK: PyArmor genera `from pyarmor_runtime_000000 import __pyarmor__`
# como import absoluto. Para que funcione, pyarmor_runtime_000000 debe
# ser un módulo top-level accesible desde sys.path. Agregamos el
# directorio de este paquete para que el import absoluto funcione.
# Esto además resuelve el problema de carpetas con guiones (kqube-comfyui).
_self_dir = os.path.dirname(os.path.abspath(__file__))
if _self_dir not in sys.path:
    sys.path.insert(0, _self_dir)

from .nodes_loader import KQubeLoader, KQubeStacker
from .nodes_hook import KQubeHook, KQubeConstraint, KQubeHookRemover
from .nodes_inspector import KQubeInspector

NODE_CLASS_MAPPINGS = {
    "KQubeLoader": KQubeLoader,
    "KQubeStacker": KQubeStacker,
    "KQubeHook": KQubeHook,
    "KQubeConstraint": KQubeConstraint,
    "KQubeHookRemover": KQubeHookRemover,
    "KQubeInspector": KQubeInspector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KQubeLoader": "KQube Loader (v1)",
    "KQubeStacker": "KQube Stacker (v1)",
    "KQubeHook": "KQube Hook Node (v1)",
    "KQubeConstraint": "KQube Constraint (v1)",
    "KQubeHookRemover": "KQube Remover (v1)",
    "KQubeInspector": "KQube Inspector (v1)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
