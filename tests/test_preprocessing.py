from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from MiBici import features


def fixture():
    return pd.DataFrame(
        {
            "fecha": pd.to_datetime(
                [
                    "2024-01-01",
                    "2025-01-01",
                    "2025-06-01",
                    "2026-01-01",
                    "2026-07-01",
                    "2026-09-01",
                ]
            ),
            "estacion_id": [1, 2, 1, 1, 99, 99],
            "salidas": [3, 7, 2, 5, 4, 1],
        }
    )


class PreprocessingTests(unittest.TestCase):
    def test_calendar_and_splits_preserve_rows_and_target(self):
        data = fixture()
        before = data.copy(deep=True)
        frame = features.build_calendar_features(data)
        pd.testing.assert_frame_equal(data, before)
        np.testing.assert_array_equal(frame.salidas, data.salidas)
        self.assertEqual(frame.iloc[0].dia_semana, "0")
        parts = features.temporal_split(frame, "2025-12-31", "2026-06-30")
        self.assertEqual(
            [len(parts[name]) for name in ["train", "validation", "test"]], [3, 1, 2]
        )
        self.assertEqual(
            sum(p.salidas.sum() for p in parts.values()), data.salidas.sum()
        )

    def test_fit_is_train_only_unknown_station_and_no_target_leakage(self):
        frame = features.build_calendar_features(fixture())
        parts = features.temporal_split(frame, "2025-12-31", "2026-06-30")
        model = features.fit_preprocessor(parts["train"])
        self.assertEqual(
            set(model.named_transformers_["categoria"].categories_[0]), {"1", "2"}
        )
        self.assertAlmostEqual(
            model.named_transformers_["numero"].mean_[0], (2024 + 2025 + 2025) / 3
        )
        original = model.transform(parts["test"][features.PREDICTORS])
        self.assertEqual(original.shape[1], len(model.get_feature_names_out()))
        changed = parts["train"].copy()
        changed["salidas"] += 10000
        another = features.fit_preprocessor(changed)
        np.testing.assert_allclose(
            original.toarray(),
            another.transform(parts["test"][features.PREDICTORS]).toarray(),
        )
        station_columns = [
            i
            for i, n in enumerate(model.get_feature_names_out())
            if n.startswith("categoria__estacion_id_")
        ]
        self.assertEqual(original[:, station_columns].nnz, 0)

    def test_saved_matrices_and_labels_align_with_csv(self):
        parts = features.temporal_split(
            features.build_calendar_features(fixture()), "2025-12-31", "2026-06-30"
        )
        model = features.fit_preprocessor(parts["train"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            report = features.save_transformed(parts, model, output)
            restored = joblib.load(output / "preprocessor.joblib")
            for name in parts:
                matrix = sparse.load_npz(output / f"X_{name}.npz")
                target = np.load(output / f"y_{name}.npy")
                csv = pd.read_csv(output / f"{name}.csv")
                np.testing.assert_array_equal(target, csv.salidas)
                self.assertEqual(matrix.shape[0], len(csv))
                self.assertTrue(np.isfinite(matrix.data).all())
                np.testing.assert_allclose(
                    matrix.toarray(),
                    restored.transform(parts[name][features.PREDICTORS]).toarray(),
                    rtol=1e-6,
                )
            self.assertEqual(report["test"]["stations_unseen_in_training"], 1)
            self.assertEqual(report["test"]["rows_unseen_station"], 2)

    def test_invalid_inputs_and_empty_splits_are_rejected(self):
        frame = features.build_calendar_features(fixture())
        with self.assertRaises(ValueError):
            features.temporal_split(frame, "2027-01-01", "2026-06-30")
        with self.assertRaisesRegex(ValueError, "contener registros"):
            features.temporal_split(frame, "2020-01-01", "2021-01-01")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            bad = fixture()
            bad.loc[0, "salidas"] = -1
            bad.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "salidas"):
                features.load_daily_counts(path)
            pd.concat([fixture(), fixture().iloc[[0]]]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicadas"):
                features.load_daily_counts(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
