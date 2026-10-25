import json
import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd
from MiBici import eda


class EdaTests(unittest.TestCase):
    def test_constant_series_has_undefined_correlation(self):
        data = pd.DataFrame(
            {
                "fecha": pd.date_range("2025-01-01", periods=40),
                "estacion_id": 1,
                "salidas": 5,
            }
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            table = eda.build_tables(data)["relaciones_temporales"]
        self.assertTrue(table[["pearson", "spearman"]].isna().all().all())
        self.assertFalse(
            any("constant" in str(item.message).lower() for item in captured)
        )

    def test_validation_rejects_bad_keys_counts_and_dates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            base = pd.DataFrame(
                {"fecha": ["2026-01-01"], "estacion_id": [1], "salidas": [2]}
            )
            for column, value in [
                ("fecha", "bad"),
                ("estacion_id", 0),
                ("salidas", -1),
                ("salidas", 1.5),
            ]:
                changed = base.copy()
                changed[column] = value
                changed.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    eda.load_and_validate(path)
            pd.concat([base, base]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicadas"):
                eda.load_and_validate(path)

    def test_report_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, quality = (
                Path(directory) / "input.csv",
                Path(directory) / "quality.json",
            )
            pd.DataFrame(
                {"fecha": ["2026-01-01"], "estacion_id": [1], "salidas": [2]}
            ).to_csv(path, index=False)
            quality.write_text(
                json.dumps({"rows_kept": 3, "output_rows": 1}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "no coincide"):
                eda.load_and_validate(path, quality)

    def test_missing_days_are_not_zero_filled(self):
        data = pd.DataFrame(
            {
                "fecha": pd.to_datetime(["2026-01-01", "2026-01-03"]),
                "estacion_id": [1, 1],
                "salidas": [2, 4],
            }
        )
        tables = eda.build_tables(data)
        self.assertTrue(pd.isna(tables["diario"].loc["2026-01-02", "salidas"]))
        self.assertEqual(tables["mensual"].iloc[0].promedio_diario, 3)
        self.assertFalse(tables["anual"].iloc[0].anio_completo)

    def test_totals_and_iqr_do_not_change_input(self):
        data = pd.DataFrame(
            {
                "fecha": pd.date_range("2025-01-01", periods=365),
                "estacion_id": 1,
                "salidas": [2] * 364 + [200],
            }
        )
        before = data.copy(deep=True)
        tables = eda.build_tables(data)
        pd.testing.assert_frame_equal(data, before)
        self.assertTrue(tables["anual"].iloc[0].anio_completo)
        self.assertEqual(tables["diario"].salidas.sum(), data.salidas.sum())
        self.assertEqual(tables["estaciones"].iloc[0].alertas_iqr, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
