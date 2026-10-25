"""Comprueba equivalencia entre limpieza completa y por bloques."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(os.environ.get('MIBICI_REPO', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
from MiBici.prepare_data import prepare_trip_data

MODULE = Path(os.environ.get('MIBICI_STREAMING_FILE', ROOT / 'MiBici/prepare_data_streaming.py'))
spec = importlib.util.spec_from_file_location('streaming_under_test', MODULE)
streaming = importlib.util.module_from_spec(spec)
spec.loader.exec_module(streaming)


class StreamingFilterTests(unittest.TestCase):
    def test_boundaries_and_duplicates_across_files_and_chunks(self):
        start = '2026-01-01 10:00:00'
        def row(tid, end, destination='1', origin='1', beginning=start):
            return [str(tid), beginning, end, origin, destination]
        rows = [
            row(1, '2026-01-01 10:01:59'),  # Excluido; reserva clave.
            row(2, '2026-01-01 10:02:00'),  # Límite exacto: conservar.
            row(3, '2026-01-01 10:00:00'),  # Duración cero: excluir.
            row(4, '2026-01-01 10:01:00', '2'),
            row(5, 'invalido'),
            row(6, '2026-01-01 09:59:59'),
            row(7, '2026-01-01 10:00:30', 'invalido'),
            row(8, '2026-01-01 10:00:30', '1.5'),
            row(9, '2026-01-01 10:03:00'),
            row(9, '2026-01-01 10:00:30'),  # Duplicado dentro del bloque.
            row(1, '2026-01-01 10:30:00'),  # No debe resucitar al excluido.
            row(2, '2026-01-01 10:00:30'),
            row('bad', '2026-01-01 10:03:00'),
            row(10, '2026-01-01 10:03:00', beginning='bad'),
            row(11, '2026-01-01 10:03:00', origin='0'),
            row(1, '2026-02-01 10:03:00', beginning='2026-02-01 10:00:00'),
        ]
        columns = ['Viaje_Id', 'Inicio_del_viaje', 'Fin_del_viaje', 'Origen_Id', 'Destino_Id']
        frame = pd.DataFrame(rows, columns=columns, dtype='string')
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            files = [tmp / 'datos_abiertos_2026_01.csv', tmp / 'datos_abiertos_2026_02.csv']
            frame.iloc[:10].to_csv(files[0], index=False)
            frame.iloc[10:].to_csv(files[1], index=False)
            expected, reference = prepare_trip_data(frame, files)
            self.assertEqual(reference['rows_kept'], 8)
            self.assertEqual(reference['rows_removed']['short_same_station_trips'], 2)
            self.assertEqual(reference['rows_removed']['duplicate_trip_ids'], 3)
            for chunk_size in (1, 3, 50):
                with self.subTest(chunk_size=chunk_size):
                    output, report = streaming.process_files(files, tmp / f'out{chunk_size}', chunk_size)
                    actual = pd.read_csv(output, dtype={'fecha': str})
                    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
                    self.assertEqual(report['rows_removed'], reference['rows_removed'])
                    self.assertEqual(report['rows_read'], len(frame))
                    self.assertEqual(report['rows_kept'], 8)
                    self.assertEqual(sum(f['rows_kept'] for f in report['files']), 8)
                    self.assertEqual(report['rows_read'], report['rows_kept'] + report['rows_removed_total'])

    def test_all_filtered_rejects_without_overwriting_previous_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / 'datos_abiertos_2026_01.csv'
            source.write_text('Viaje_Id,Inicio_del_viaje,Fin_del_viaje,Origen_Id,Destino_Id\n1,2026-01-01 10:00:00,2026-01-01 10:01:00,1,1\n', encoding='utf-8')
            output_dir = tmp / 'out'
            output_dir.mkdir()
            old = output_dir / streaming.OUTPUT_NAME
            old.write_text('previous', encoding='utf-8')
            with self.assertRaises(ValueError):
                streaming.process_files([source], output_dir, 1)
            self.assertEqual(old.read_text(encoding='utf-8'), 'previous')


if __name__ == '__main__':
    unittest.main()
