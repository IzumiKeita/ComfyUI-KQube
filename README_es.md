# ComfyUI_KQube

Custom nodes para cargar y aplicar proyecciones KQUBE (.kqube) en modelos de difusion (SDXL, Flux, SD3).

## Instalacion

```bash
# Opcion A: Git clone (recomendado)
cd ComfyUI/custom_nodes
git clone https://github.com/IzumiKeita/ComfyUI_KQube.git KQube

# Opcion B: Descarga ZIP y extrae como KQube/
#   Descarga el ZIP, extraelo y renombra la carpeta a "KQube"
#   Copia KQube/ en ComfyUI/custom_nodes/
```

> **Importante**: La carpeta en `custom_nodes/` debe llamarse exactamente `KQube`.

## Requisitos

```bash
pip install safetensors numpy
```

> **Nota**: ComfyUI ya incluye `torch`, `torchvision` y `pillow`. Solo necesitas `safetensors` y `numpy` adicionalmente.

## Nodos

### KQubeLoader
Carga un archivo .kqube y lo aplica al modelo.
- **model**: modelo SDXL
- **kqube_path**: ruta al .kqube
- **strength**: 0.0-2.0 (1.0 = completo)
- **strict_dna**: si True, exige DNA exacto

### KQubeStacker
Apila hasta 4 .kqube (personaje, ropa, fondo, calidad).
- **model**: modelo base
- **kqube_1..4**: rutas a .kqube
- **weight_1..4**: pesos por capa
- **mode**: chain (secuencial) o parallel (suma)

### KQubeInspector
Previsualiza compatibilidad sin inyectar.
- **kqube_path**: ruta al .kqube

### KQubeHook
Aplica pesos KQUBE como forward hooks post-LoRA.
- **model**: modelo SDXL
- **kqube_path**: ruta al .kqube
- **strength**: factor de escala
- **hook_position**: 'post' (despues de LoRA) o 'pre' (antes)

### KQubeConstraint
Aplica KQUBE como capa de restriccion en paralelo.

### KQubeHookRemover
Remueve hooks KQUBE previamente aplicados.

## Compatibilidad
Un .kqube entrenado en SDXL base funciona en Pony, Illustrious, etc.
Ver [architecture_detector.py](architecture_detector.py) para familias soportadas.

## Estructura del Proyecto

```
KQube/
├── __init__.py             # Registro de nodos ComfyUI
├── nodes_loader.py         # KQubeLoader, KQubeStacker
├── nodes_hook.py           # KQubeHook, KQubeConstraint, KQubeHookRemover
├── nodes_inspector.py      # KQubeInspector
├── kqube_core.py           # Modulos matematicos (CubicProjection, SubspaceManager)
├── kqube_format.py         # Formato .kqube v2.1 (carga/guardado)
├── kqube_hook.py           # CubicHook (forward hooks)
├── architecture_detector.py # Detector de familias de arquitectura
├── utils.py                # Utilidades compartidas
├── requirements.txt        # Dependencias minimas
└── pyproject.toml          # Metadatos del paquete
```

## Desarrollo

```bash
# Clonar para desarrollo
git clone https://github.com/IzumiKeita/KQube.git
cd KQube

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar tests (si aplica)
pytest
```

## Licencia

MIT
