import json
import tempfile
import unittest
from pathlib import Path

from helpers import ROOT, small_config, write_replay
from alpha_gp_lab import llm_seed
from alpha_gp_lab.llm_seed import FIXTURE_SOURCE, LIVE_SOURCE, build_prompt, check_record, parse_response, prompt_sha256, propose

REPLAY = ROOT / 'fixtures' / 'llm_replay.json'
LIVE_REPLAY = ROOT / 'fixtures' / 'llm_replay_live.json'


def refuse(prompt):
    raise AssertionError('the LLM must not be called outside live mode')


class Replay(unittest.TestCase):
    def test_fixture_is_labelled_and_covers_both_configs(self):
        cache = json.loads(REPLAY.read_text())
        for entry in cache['entries'].values():
            self.assertEqual(entry['source'], FIXTURE_SOURCE)
        if any(e['source'] == FIXTURE_SOURCE for e in cache['entries'].values()):
            self.assertIn('HAND-WRITTEN FIXTURE', cache['note'])
        for name in ('demo_config.json', 'demo_no_seeds_config.json', 'walkforward_config.json'):
            llm = json.loads((ROOT / 'fixtures' / name).read_text())['llm']
            self.assertIn(prompt_sha256(build_prompt(llm['brief'], llm['n'])), cache['entries'], name)

    def test_demo_fixture_verdicts(self):
        llm = json.loads((ROOT / 'fixtures' / 'demo_config.json').read_text())['llm']
        record = propose(llm['brief'], llm['n'], REPLAY, runner=refuse)
        self.assertEqual((len(record['accepted']), len(record['rejected']), record['duplicates']), (6, 3, 1))
        self.assertEqual(record['source'], FIXTURE_SOURCE)
        check_record(record, llm['brief'], llm['n'])

    def test_parse_response_validates_every_line(self):
        text = '```\n1. -returns\n* rank(close)\n2) ts_mean(volume, 5)\n# a comment\n\nrank(vwap)\n- returns\n3. -returns\n```'
        with self.assertLogs('alpha_gp_lab.llm_seed', 'WARNING') as logs:
            accepted, rejected, duplicates = parse_response(text)
        self.assertEqual(accepted, ['-returns', 'rank(close)', 'ts_mean(volume, 5)'])
        # '- returns' is a bullet or a negation; it is rejected rather than guessed.
        self.assertEqual([r['line'] for r in rejected], ['rank(vwap)', '- returns'])
        self.assertEqual(duplicates, 1)
        self.assertIn('rank(vwap)', logs.output[0])

    def test_missing_entry_fails_without_calling_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(KeyError):
                propose('a brief nobody cached', 3, Path(tmp) / 'replay.json', runner=refuse)

    def test_live_refresh_writes_the_cache_through_the_runner(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_replay(tmp, config)
            calls = []

            def fake(prompt):
                calls.append(prompt)
                return 'ts_rank(volume, 10)\nrank(open, 3)\n', 'claude -p (test), model fake-model'
            record = propose(config['llm']['brief'], config['llm']['n'], path, live=True, runner=fake)
            self.assertEqual(len(calls), 1)
            self.assertTrue(record['source'].startswith(LIVE_SOURCE + ': claude -p (test), model fake-model, '))
            self.assertEqual((record['accepted'], len(record['rejected'])), (['ts_rank(volume, 10)'], 1))
            cache = json.loads(path.read_text())
            self.assertNotIn('HAND-WRITTEN', cache['note'])
            self.assertIn(LIVE_SOURCE, cache['note'])
            replayed = propose(config['llm']['brief'], config['llm']['n'], path, runner=refuse)
            self.assertEqual(replayed, record)

    def test_tampered_entries_and_records_are_refused(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_replay(tmp, config)
            cache = json.loads(path.read_text())
            next(iter(cache['entries'].values()))['prompt'] = 'something else'
            path.write_text(json.dumps(cache))
            with self.assertRaises(ValueError):
                propose(config['llm']['brief'], config['llm']['n'], path)
            write_replay(tmp, config)
            record = propose(config['llm']['brief'], config['llm']['n'], path)
            record['accepted'] = record['accepted'] + ['close']
            with self.assertRaises(ValueError):
                check_record(record, config['llm']['brief'], config['llm']['n'])

    def test_live_runner_is_the_claude_cli(self):
        self.assertIs(propose.__defaults__[-1], llm_seed.claude_cli)


if __name__ == '__main__':
    unittest.main()
