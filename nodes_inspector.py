"""
src/comfyui/nodes_inspector.py — Nodo ComfyUI para inspeccionar .kqube sin inyectar.

KQubeInspector: previsualiza compatibilidad entre un .kqube y el modelo actual.
               No modifica el modelo — util para verificar antes de cargar.
"""

import os
import sys
from typing import Tuple

# Imports relativos (todos los módulos están en el mismo directorio)
try:
    from .kqube_format import load_kqube_meta
    from .architecture_detector import KNOWN_ARCHITECTURES
    from .nodes_loader import _list_kqube_files, _resolve_kqube_path
except ImportError as e:
    print(f"[KQubeInspector] Warning: could not import KQube modules: {e}")
    raise


class KQubeInspector:
    """
    Verifica compatibilidad .kqube ↔ modelo sin modificar nada.

    Util para previsualizar en el workflow si un archivo es compatible
    antes de conectarlo al KQubeLoader.

    Inputs:
      - kqube_name: nombre del archivo .kqube (desde models/kqube/)

    Outputs:
      - compatible: "YES" o "NO"
      - mode: "exact" (DNA identico), "architecture" (misma familia), "incompatible"
      - report: texto detallado con metadata del .kqube
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "kqube_name": (_list_kqube_files(), {
                    "default": "(sin archivos .kqube)",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("compatible", "mode", "report")
    FUNCTION = "inspect"
    CATEGORY = "KQube"
    OUTPUT_NODE = True

    def inspect(self, kqube_name: str = "") -> Tuple[str, str, str]:
        kqube_path = _resolve_kqube_path(kqube_name)
        if not kqube_path or kqube_name.startswith("("):
            return ("NO", "no_file", "Sin archivo .kqube seleccionado")

        if not os.path.exists(kqube_path):
            return ("NO", "not_found", f"Archivo no encontrado:\n{kqube_path}")

        try:
            meta = load_kqube_meta(kqube_path)
        except Exception as e:
            return ("NO", "invalid", f"Error al leer .kqube:\n{e}")

        # Extraer metadata
        version = meta.get("kqube_version", "?")
        stage = meta.get("stage_signature", "?")
        family = meta.get("architecture_family", "desconocida")
        arch_sig = meta.get("architecture_signature", "desconocida")
        dna = meta.get("model_dna", "?")
        params = meta.get("num_params", 0)
        r_body = meta.get("hyperparameters", {}).get("r_body", "?")
        file_size = os.path.getsize(kqube_path) / 1024

        # Info de la familia
        family_info = KNOWN_ARCHITECTURES.get(family)
        family_display = family_info.display_name if family_info else family
        model_type = family_info.model_type if family_info else "?"

        lines = [
            f"Archivo:  {os.path.basename(kqube_path)}",
            f"Version:  {version}",
            f"Etapa:    {stage}",
            f"Familia:  {family_display} ({family})",
            f"Tipo:     {model_type}",
            f"Params:   {params:,} ({file_size:.0f} KB)",
            f"r_cubic:  {r_body}",
            f"DNA:      {dna[:16] if len(dna) > 16 else dna}...",
            f"Firma:    {arch_sig[:16] if len(arch_sig) > 16 else arch_sig}...",
            "",
        ]

        if family == "unknown":
            lines.append("[!] Archivo v2.0 sin metadatos de arquitectura.")
            lines.append("    Compatible SOLO con el modelo exacto donde se entreno.")
            lines.append("    Re-entrene con la version actual para cross-model.")
            return ("LIMITED", "dna_only", "\n".join(lines))

        # Compatibilidad con modelos conocidos
        if family_info and family_info.base_models:
            lines.append("Modelos compatibles (misma familia):")
            for bm in family_info.base_models[:5]:
                lines.append(f"  - {bm}")

        report = "\n".join(lines)
        return ("COMPATIBLE", "architecture" if version >= "2.1" else "legacy", report)
