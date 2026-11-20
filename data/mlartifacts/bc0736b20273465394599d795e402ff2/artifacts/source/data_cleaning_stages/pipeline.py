"""Ejecuta etapas de limpieza encadenadas y junta sus conteos."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class StageOutcome:
    """Conteos, reglas y detalles que aporta una etapa."""

    rows_removed: dict[str, int]
    rules: tuple[str, ...] = ()
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class CleaningResult:
    """Filas que quedan tras el pipeline y los datos para el reporte de calidad."""

    frame: pd.DataFrame
    rows_read: int
    rows_kept: int
    rows_removed: dict[str, int]
    rows_removed_total: int
    timestamp_formats: dict[str, int]
    details: dict[str, object]
    cleaning_rules: list[str]


Stage = Callable[[pd.DataFrame], tuple[pd.DataFrame, StageOutcome]]


def run_pipeline(frame: pd.DataFrame, stages: Sequence[Stage]) -> CleaningResult:
    """Aplica cada etapa en orden y fusiona exclusiones, reglas y detalles.

    El orden es el de ``stages``. Dos etapas no pueden informar la misma clave
    en ``rows_removed`` o en ``details``.
    """
    rows_read = len(frame)
    rows_removed: dict[str, int] = {}
    details: dict[str, object] = {}
    cleaning_rules: list[str] = []

    for stage in stages:
        frame, outcome = stage(frame)
        removed_overlap = set(rows_removed) & set(outcome.rows_removed)
        if removed_overlap:
            raise ValueError(
                f"Claves de exclusión repetidas: {', '.join(sorted(removed_overlap))}"
            )
        detail_overlap = set(details) & set(outcome.details)
        if detail_overlap:
            raise ValueError(
                f"Claves de detalle repetidas: {', '.join(sorted(detail_overlap))}"
            )
        rows_removed.update(outcome.rows_removed)
        details.update(outcome.details)
        cleaning_rules.extend(outcome.rules)

    raw_formats = details.get("timestamp_formats", {})
    if not isinstance(raw_formats, dict):
        raise TypeError("timestamp_formats debe ser un diccionario")
    timestamp_formats = {str(key): int(value) for key, value in raw_formats.items()}
    if "timestamp_formats" in details:
        details["timestamp_formats"] = timestamp_formats

    rows_removed_total = sum(rows_removed.values())
    rows_kept = rows_read - rows_removed_total
    if len(frame) != rows_kept:
        raise ValueError(
            "Las filas conservadas no coinciden con los conteos de las etapas: "
            f"quedan {len(frame)} y el conteo indica {rows_kept}"
        )

    return CleaningResult(
        frame=frame,
        rows_read=rows_read,
        rows_kept=rows_kept,
        rows_removed=rows_removed,
        rows_removed_total=rows_removed_total,
        timestamp_formats=timestamp_formats,
        details=details,
        cleaning_rules=cleaning_rules,
    )
