import contextlib
import inspect
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from helpers import ROOT, small_config, write_replay
from alpha_gp_lab import cli, llm_seed
from alpha_gp_lab.llm_seed import (FIXTURE_SOURCE, LIVE_SOURCE, MAX_BODY_BYTES, build_prompt, check_record, openrouter,
                                   parse_response, prompt_sha256, propose)

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
                return 'ts_rank(volume, 10)\nrank(open, 3)\n', 'OpenRouter (test), response model fake-model'
            record = propose(config['llm']['brief'], config['llm']['n'], path, live=True, runner=fake)
            self.assertEqual(len(calls), 1)
            self.assertTrue(record['source'].startswith(LIVE_SOURCE + ': OpenRouter (test), response model fake-model, '))
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

    def test_live_runner_is_openrouter_over_urllib(self):
        self.assertIs(inspect.signature(propose).parameters['runner'].default, llm_seed.openrouter)
        self.assertIs(inspect.signature(openrouter).parameters['post'].default, llm_seed.http_post)


KEY = 'fake-openrouter-key-0123'   # not a real key; it must never reach a saved file or a message
PINNED = 'qwen/qwen3.8-27b:free'


def answer(content='rank(close)\nts_mean(volume, 5)\n', model=PINNED, finish_reason='stop', **top):
    out = dict(id='gen-fake-1', provider='FakeProvider', model=model,
               choices=[dict(finish_reason=finish_reason, message=dict(role='assistant', content=content))])
    out.update(top)
    return out


class FakePost:
    """Stands in for the HTTP call: records every request and returns one canned (status, body)."""

    def __init__(self, status=200, body=None, raw=None):
        self.status, self.raw, self.calls = status, raw if raw is not None else json.dumps(body or answer()).encode(), []

    def __call__(self, url, headers, body, timeout):
        self.calls.append(dict(url=url, headers=headers, body=json.loads(body), timeout=timeout))
        return self.status, self.raw


@mock.patch.dict(os.environ, {'OPENROUTER_API_KEY': KEY})
class OpenRouter(unittest.TestCase):
    def refused(self, post, pattern):
        with self.assertRaisesRegex(RuntimeError, pattern) as ctx:
            openrouter('PROMPT', post=post)
        self.assertEqual(len(post.calls), 1, 'a refusal must not be retried: one call per --live')
        self.assertNotIn(KEY, str(ctx.exception))

    def test_one_request_with_the_pinned_body_and_bearer_key(self):
        post = FakePost()
        text, via = openrouter('PROMPT', post=post, timeout=7)
        self.assertEqual(text, 'rank(close)\nts_mean(volume, 5)\n')
        self.assertEqual(len(post.calls), 1)
        call = post.calls[0]
        self.assertEqual(call['url'], 'https://openrouter.ai/api/v1/chat/completions')
        self.assertEqual(call['headers']['Authorization'], 'Bearer ' + KEY)
        self.assertEqual(call['body'], {'model': PINNED, 'messages': [{'role': 'user', 'content': 'PROMPT'}],
                                        'temperature': 0, 'max_tokens': 1024, 'reasoning': {'effort': 'none'}})
        self.assertEqual(call['timeout'], 7)
        self.assertEqual(via, f'OpenRouter {PINNED}, weights Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 '
                              f'(Apache-2.0), response model {PINNED}, provider FakeProvider, id gen-fake-1')

    def test_response_model_without_free_suffix_is_the_same_model(self):
        text, via = openrouter('PROMPT', post=FakePost(body=answer(model='qwen/qwen3.8-27b')))
        self.assertIn('response model qwen/qwen3.8-27b, provider', via)

    def test_any_other_response_model_is_refused(self):
        for model in ('qwen/qwen3.8-27b-instruct', 'openai/gpt-4o', 'anthropic/claude-x', 'qwen/qwen3.8-27b:free:x',
                      'qwen/qwen3.8-27b:free:nitro', 'qwen/qwen3.8-27b:nitro', '', None):
            with self.subTest(model=model):
                self.refused(FakePost(body=answer(model=model)), 'response came from model')

    def test_http_errors_raise_with_status_and_message_tail(self):
        self.refused(FakePost(429, {'error': {'code': 429, 'message': 'Rate limit exceeded: free-models-per-min'}}),
                     r'HTTP 429 \(rate limit\): Rate limit exceeded: free-models-per-min$')
        self.refused(FakePost(502, raw=b'upstream gateway said no'), 'HTTP 502: upstream gateway said no$')
        self.refused(FakePost(401, {'error': {'message': 'bad key ' + KEY}}), 'HTTP 401: bad key \\[key\\]$')

    def test_errors_inside_a_200_are_refused(self):
        self.refused(FakePost(body=answer(error={'message': 'provider failed'})), 'error in response.*provider failed')
        bad = answer()
        bad['choices'][0]['error'] = {'message': 'stream cut'}
        self.refused(FakePost(body=bad), 'error in choice.*stream cut')
        self.refused(FakePost(body=answer(choices=[])), 'no choices')
        self.refused(FakePost(raw=b'<html>not json</html>'), 'not a JSON object')

    def test_unfinished_answers_are_refused(self):
        for reason in ('length', 'content_filter', 'error', None):
            with self.subTest(reason=reason):
                self.refused(FakePost(body=answer(finish_reason=reason)), 'finish_reason')

    def test_empty_or_non_text_content_is_refused(self):
        for content in ('', '  \n', None, 42, ['rank(close)']):
            with self.subTest(content=content):
                self.refused(FakePost(body=answer(content=content)), 'content is empty or not text')
        self.refused(FakePost(body=answer(choices=[{'finish_reason': 'stop', 'message': 'rank(close)'}])),
                     'content is empty or not text')

    def test_oversized_body_is_refused_before_parsing(self):
        self.refused(FakePost(raw=b' ' * (MAX_BODY_BYTES + 1)), 'over 1000000 bytes')
        self.assertEqual(openrouter('PROMPT', post=FakePost(raw=json.dumps(answer()).encode().ljust(MAX_BODY_BYTES)))[0],
                         'rank(close)\nts_mean(volume, 5)\n')

    def test_missing_key_fails_loudly_before_any_request(self):
        for value in (None, '', '  \n'):
            with self.subTest(value=value), mock.patch.dict(os.environ):
                os.environ.pop('OPENROUTER_API_KEY')
                if value is not None:
                    os.environ['OPENROUTER_API_KEY'] = value
                post = FakePost()
                with self.assertRaisesRegex(RuntimeError, 'OPENROUTER_API_KEY'):
                    openrouter('PROMPT', post=post)
                self.assertEqual(post.calls, [])

    def test_live_refresh_saves_provenance_and_never_the_key(self):
        config = small_config()
        brief, n = config['llm']['brief'], config['llm']['n']
        with tempfile.TemporaryDirectory() as tmp:
            path = write_replay(tmp, config)
            post = FakePost(body=answer(content='ts_rank(volume, 10)\n'))
            record = propose(brief, n, path, live=True, runner=lambda prompt: openrouter(prompt, post=post))
            self.assertEqual(len(post.calls), 1)
            self.assertEqual(post.calls[0]['body']['messages'][0]['content'], build_prompt(brief, n))
            self.assertRegex(record['source'], rf'^{LIVE_SOURCE}: OpenRouter {PINNED}, .*response model {PINNED}, '
                                               r'provider FakeProvider, id gen-fake-1, \d{4}-\d{2}-\d{2}$')
            self.assertEqual(record['accepted'], ['ts_rank(volume, 10)'])
            saved = path.read_bytes()
            self.assertNotIn(KEY.encode(), saved)
            self.assertIn(b'open-weight', saved)
            self.assertEqual(propose(brief, n, path, runner=refuse), record)

    def test_a_refused_response_saves_nothing(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_replay(tmp, config)
            before = path.read_bytes()
            post = FakePost(body=answer(finish_reason='length'))
            with self.assertRaisesRegex(RuntimeError, 'finish_reason'):
                propose(config['llm']['brief'], config['llm']['n'], path, live=True,
                        runner=lambda prompt: openrouter(prompt, post=post))
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), [path.name])


class KeyFile(unittest.TestCase):
    def test_cli_reads_the_key_file_only_when_the_variable_is_unset(self):
        with tempfile.TemporaryDirectory() as tmp:
            key_file = Path(tmp) / 'api_key'
            key_file.write_text(KEY + '\n')
            with mock.patch.dict(os.environ):
                os.environ.pop('OPENROUTER_API_KEY', None)
                cli.load_key_file(Path(tmp) / 'absent')
                self.assertNotIn('OPENROUTER_API_KEY', os.environ)
                cli.load_key_file(key_file)
                self.assertEqual(os.environ['OPENROUTER_API_KEY'], KEY)
                os.environ['OPENROUTER_API_KEY'] = 'from-the-environment'
                cli.load_key_file(key_file)
                self.assertEqual(os.environ['OPENROUTER_API_KEY'], 'from-the-environment')

    def test_seeds_live_loads_the_key_file_and_never_prints_the_key(self):
        seen = []

        def fake_propose(brief, n, replay_path, live=False):
            seen.append((live, os.environ.get('OPENROUTER_API_KEY')))
            return dict(source='s', prompt_sha256='h', requested=n, accepted=[], rejected=[], duplicates=0)
        with tempfile.TemporaryDirectory() as home, mock.patch.dict(os.environ, {'HOME': home}), \
                mock.patch.object(cli, 'propose', fake_propose):
            os.environ.pop('OPENROUTER_API_KEY', None)
            (Path(home) / '.config' / 'openrouter').mkdir(parents=True)
            (Path(home) / '.config' / 'openrouter' / 'api_key').write_text(KEY)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                cli.main(['seeds'])
                cli.main(['seeds', '--live'])
        self.assertEqual(seen, [(False, None), (True, KEY)])
        self.assertNotIn(KEY, out.getvalue())

    def test_default_key_file_is_under_the_home_config(self):
        with tempfile.TemporaryDirectory() as home, mock.patch.dict(os.environ, {'HOME': home}):
            os.environ.pop('OPENROUTER_API_KEY', None)
            (Path(home) / '.config' / 'openrouter').mkdir(parents=True)
            (Path(home) / '.config' / 'openrouter' / 'api_key').write_text(KEY)
            cli.load_key_file()
            self.assertEqual(os.environ['OPENROUTER_API_KEY'], KEY)


if __name__ == '__main__':
    unittest.main()
