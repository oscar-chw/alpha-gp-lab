import copy
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from helpers import inputs, small_config
from alpha_gp_lab import store
from alpha_gp_lab.config import canonical, digest, read_json


def snapshot(out):
    return {p.name: p.read_bytes() for p in Path(out).iterdir()}


def rehash(out, name):
    manifest = read_json((out / 'completion.json').read_bytes())
    manifest[name] = digest((out / name).read_bytes())
    (out / 'completion.json').write_bytes(canonical(manifest))


class Bundles(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.config = small_config()
        self.raw, self.panel, self.llm = inputs(self.config, self.tmp)

    def test_run_verify_and_identical_rerun(self):
        out = self.tmp / 'run'
        report = store.run(self.raw, self.panel, self.llm, out)
        before = snapshot(out)
        self.assertEqual(store.verify(out), report)
        self.assertEqual(store.run(self.raw, self.panel, self.llm, out), report)
        self.assertEqual(snapshot(out), before)
        with sqlite3.connect(out / 'state.sqlite') as conn:
            n = conn.execute('SELECT count(*) FROM nodes').fetchone()[0]
            self.assertEqual(n, report['folds'][0]['counts']['occurrences'])
            self.assertEqual(conn.execute("SELECT count(*) FROM results WHERE split='test'").fetchone()[0], 1)
            self.assertEqual(conn.execute("SELECT count(*) FROM llm_proposals WHERE status='rejected'").fetchone()[0], 1)

    def test_tables_are_append_only(self):
        out = self.tmp / 'run'
        store.run(self.raw, self.panel, self.llm, out)
        with sqlite3.connect(out / 'state.sqlite') as conn:
            for sql in ("UPDATE nodes SET record = '{}'", 'DELETE FROM edges', "UPDATE selection SET record = '{}'"):
                with self.subTest(sql=sql), self.assertRaisesRegex(sqlite3.IntegrityError, 'append-only'):
                    conn.execute(sql)

    def test_conflicting_rerun_is_refused_without_changes(self):
        out = self.tmp / 'run'
        store.run(self.raw, self.panel, self.llm, out)
        before = snapshot(out)
        other = copy.deepcopy(self.config)
        other['evaluation']['fee_bps'] = 9
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            store.run(canonical(other), self.panel, self.llm, out)
        self.assertEqual(snapshot(out), before)

    def test_tampering_is_detected_even_with_a_rehashed_manifest(self):
        original = self.tmp / 'run'
        store.run(self.raw, self.panel, self.llm, original)

        plain = self.tmp / 'plain'
        shutil.copytree(original, plain)
        (plain / 'report.md').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            store.verify(plain)

        forged = self.tmp / 'forged'
        shutil.copytree(original, forged)
        report = read_json((forged / 'report.json').read_bytes())
        report['folds'][0]['test']['mean_ic'] = 0.5
        (forged / 'report.json').write_bytes(canonical(report))
        rehash(forged, 'report.json')
        with self.assertRaisesRegex(ValueError, 'numerical report replay'):
            store.verify(forged)

        db = self.tmp / 'db'
        shutil.copytree(original, db)
        with sqlite3.connect(db / 'state.sqlite') as conn:
            conn.execute('DROP TRIGGER edges_no_delete')
            conn.execute('DELETE FROM edges')
        rehash(db, 'state.sqlite')
        with self.assertRaisesRegex(ValueError, 'schema or missing append-only triggers'):
            store.verify(db)

    def test_interrupted_run_is_kept_and_refused(self):
        out = self.tmp / 'interrupted'
        with self.assertRaises(InterruptedError):
            store.run(self.raw, self.panel, self.llm, out, _interrupt_after_generation=(0, 0))
        self.assertFalse((out / 'completion.json').exists())
        with sqlite3.connect(out / 'state.sqlite') as conn:
            self.assertEqual(conn.execute('SELECT max(generation) FROM nodes').fetchone()[0], 0)
            self.assertGreater(conn.execute('SELECT count(*) FROM nodes').fetchone()[0], 0)
        before = snapshot(out)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            store.run(self.raw, self.panel, self.llm, out)
        self.assertEqual(snapshot(out), before)
        self.assertEqual(store.run(self.raw, self.panel, self.llm, self.tmp / 'fresh')['folds'][0]['status'], 'SELECTED')

    def test_code_identity_mismatch_is_refused(self):
        out = self.tmp / 'run'
        store.run(self.raw, self.panel, self.llm, out)
        with patch.object(store, 'code_identity', return_value={'changed': True}):
            with self.assertRaisesRegex(ValueError, 'code/runtime'):
                store.verify(out)


if __name__ == '__main__':
    unittest.main()
