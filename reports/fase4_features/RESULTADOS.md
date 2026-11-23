# Ingeniería de características: resultados

Modelo: `PoissonRegressor` con el mismo split temporal de Fase 2 (entrenamiento
hasta 2025-12-31, validación 2026-01 a 2026-06, prueba 2026-07 a 2026-09) y el
mismo dataset (`historico_filtrado`, MD5 `171e28ac…`). Corridas en Linux con
Python 3.14.5, experimento MLflow local `mibici-feature-engineering`.

Los conjuntos son acumulativos y se definen en `MiBici/feature_engineering/sets.py`.
Todas las características se ajustan solo con entrenamiento dentro del pipeline;
el modelo recibe únicamente `fecha` y `estacion_id`.

## Validación (selección)

MAE en salidas por estación y día. Δ contra la referencia v1 con `alpha=0.01` (12.0654).

| Set | Columnas | Mejor `alpha` | MAE | Sesgo | MAE estación conocida | MAE estación nueva | Δ MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 (referencia, `alpha=0.01`) | 406 | 0.01 | 12.0654 | +8.2521 | 11.9230 | 21.8087 | — |
| v1 | 406 | 0.001 | 11.6443 | +7.9930 | 11.5312 | 19.3813 | −3.5 % |
| v2_calendario | 410 | 0.001 | 11.5084 | +7.9999 | 11.3929 | 19.4141 | −4.6 % |
| v3_tendencia | 423 | 0.001 | 8.6492 | +2.5010 | 8.6476 | 8.7612 | −28.3 % |
| v4_perfil_estacion | 426 | 0.3 | 7.8443 | +2.1958 | 7.8315 | 8.7158 | −35.0 % |
| v5_interacciones | 1198 | 0.1 | 7.9976 | +2.2194 | 7.9827 | 9.0175 | −33.7 % |
| v6_catalogo_espacial | 1240 | 0.1 | 8.1538 | +2.3157 | 8.1492 | 8.4639 | −32.4 % |
| **v7_rezago_anual** | 1242 | **0.1** | **7.7185** | **+2.2259** | 7.6966 | 9.2159 | **−36.0 %** |

Búsqueda de `alpha`: 0.001, 0.01 y 0.1 para todos los sets, más 0.3 y 1.0 para v4 y v7.
En v7 el óptimo (0.1) queda dentro del rango: 0.3 da 7.7217 y 1.0 da 7.7992.

## Decisión: se adopta `v4_perfil_estacion` con `alpha=0.3`

Se elige v4 en lugar de v7 por simplicidad. En validación está a 0.13 de MAE del
mejor set (7.8443 frente a 7.7185), pero usa 426 columnas en lugar de 1,242 y no
depende del catálogo de estaciones. Sus pasos son `CalendarStep`, `HolidayStep`,
`StationAgeStep` y `StationProfileStep`.

## Prueba (una sola corrida por modelo)

| Modelo | MAE | Sesgo | MAE estación conocida | MAE estación nueva |
|---|---:|---:|---:|---:|
| v1, Poisson `alpha=0.01` | 13.6711 | +7.9761 | 11.6840 | 22.1809 |
| **v4_perfil_estacion, Poisson `alpha=0.3` (adoptado)** | **8.4730** | **−0.9775** | 8.9280 | 6.5246 |
| v7_rezago_anual, Poisson `alpha=0.1` | 8.3031 | −1.0102 | 8.5920 | 7.0660 |

Con v4, el MAE en prueba baja 38.0 % frente a v1 y el sesgo pasa de +7.98 a −0.98.
Modelos registrados localmente en `mibici-poisson-features`:
- **Versión 3 = v4 servido** (run `3efad342cebd454cb4cc9cd03a51f982`), exportado a
  `models/mibici-poisson-v4/` y versionado con `models/mibici-poisson-v4.dvc`.
- Versión 2 = v4, misma configuración y métricas, pero registrada antes de que
  `FeaturePipeline` publicara `feature_names_in_` (no usar).
- Versión 1 = v7 (run `1b9d9fdca71e42e4af8aef0bf80c7922`).

La prueba de v7 se corrió antes de elegir. La decisión se basó en validación y
en simplicidad, no en el resultado de prueba, que favorece ligeramente a v7.

## Lectura

- **El salto principal viene de `v3_tendencia`.** Al quitar el año lineal (que
  con enlace log extrapolaba una tendencia exponencial) y usar la antigüedad de
  la estación con el año como categoría topada, el sesgo baja de +8.0 a +2.5 y
  el error en estaciones nuevas pasa de 19.4 a 8.8.
- `v4_perfil_estacion` da la segunda mejora y casi iguala al ganador con 426
  columnas: es la alternativa más simple si se prefiere un modelo pequeño.
- `v5` y `v6` no mejoran el MAE global por sí solos. `v6` sí mejora a las
  estaciones nuevas (8.46, el mejor valor), porque usa la demanda de las
  estaciones vecinas del catálogo.
- `v2_calendario` aporta poco (−1.2 % frente a v1 con el mismo `alpha`).
- En validación todavía queda un sesgo positivo de alrededor de +2.2 salidas.

## Servicio

La API (`MiBici/api.py`) y el `Dockerfile` usan `models/mibici-poisson-v4` por
defecto. `select_model_input` entrega a cada modelo las columnas con las que se
entrenó, así que `models:/mibici-poisson/2` (v1) sigue funcionando con
`MIBICI_MODEL_URI`.

Pendiente: ejecutar `dvc push` desde un equipo con el remoto de OneDrive
configurado, y construir y publicar una imagen nueva en Docker Hub.
