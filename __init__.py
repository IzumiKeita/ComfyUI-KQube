# KQube — Custom Nodes para ComfyUI
# Este archivo hace que KQube/ sea un paquete Python válido,
# permitiendo imports relativos entre módulos.

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
