"""LLM seed proposer: ask a model for N grammar expressions, given a short research brief.

The model is an untrusted proposer. It sees the brief and the grammar, never data, and it
cannot promote anything: each line it returns must parse under ``grammar.parse`` or it is
logged and rejected, and accepted seeds still have to earn their place in the GP on train.

Responses are cached in a JSON replay file keyed by the SHA-256 of the exact prompt. Normal
runs and every test read the replay file only. ``live=True`` (the CLI's ``--live``) calls the
local ``claude -p`` CLI to refresh the cache entry; that spends the operator's quota and is
never invoked by tests or the demo.
"""
from datetime import date
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .grammar import canonical, grammar_doc, parse

log = logging.getLogger(__name__)
FIXTURE_SOURCE = 'HAND-WRITTEN FIXTURE (not real LLM output)'
LIVE_SOURCE = 'REAL LLM OUTPUT'
_BULLET = re.compile(r'^(?:\*\s+|\d+[.)]\s*)')   # '* x', '1. x', '2) x'


def build_prompt(brief, n):
    return (
        'You propose seed expressions for a genetic-programming search over cross-sectional factors.\n'
        f'Research brief: {brief.strip()}\n\n'
        f'Grammar (anything else is rejected):\n{grammar_doc()}\n\n'
        f'Return exactly {n} expressions, one per line, with no numbering, commentary or code fences.\n')


def prompt_sha256(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


def parse_response(text):
    """Split a response into accepted canonical expressions and rejected lines with reasons."""
    accepted, rejected, seen, duplicates = [], [], set(), 0
    for raw in text.splitlines():
        line = _BULLET.sub('', raw.strip().strip('`').strip()).strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('- '):
            # A '- ' bullet and a unary minus look the same; guessing could flip the sign.
            log.warning('rejected LLM proposal %r: ambiguous leading "- "', line)
            rejected.append(dict(line=line, reason='ambiguous leading "- " (bullet or minus?)'))
            continue
        try:
            tree = parse(line)
        except ValueError as exc:
            log.warning('rejected LLM proposal %r: %s', line, exc)
            rejected.append(dict(line=line, reason=str(exc)))
            continue
        key = canonical(tree)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        accepted.append(str(tree))
    return accepted, rejected, duplicates


def claude_cli(prompt, timeout=600):
    """Ask the local Claude Code CLI in print mode; return (response text, label naming CLI and model).

    Only reached through ``live=True``. The prompt goes in on stdin, every tool is disabled and
    the working directory is empty, so the model sees the brief and the grammar and cannot read
    any data file.
    """
    exe = shutil.which('claude')
    if exe is None:
        raise RuntimeError('live mode needs the `claude` CLI on PATH')
    version = subprocess.run([exe, '--version'], capture_output=True, text=True, timeout=60).stdout.strip()
    with tempfile.TemporaryDirectory() as empty:
        done = subprocess.run([exe, '-p', '--output-format', 'json', '--no-session-persistence', '--tools', ''],
                              input=prompt, capture_output=True, text=True, timeout=timeout, cwd=empty)
    if done.returncode != 0:
        raise RuntimeError(f'claude -p exited {done.returncode}: {(done.stderr or done.stdout).strip()[:500]}')
    out = json.loads(done.stdout)
    if out.get('is_error') or not isinstance(out.get('result'), str):
        raise RuntimeError(f'claude -p returned an error: {str(out)[:500]}')
    models = ', '.join(sorted(out.get('modelUsage') or {})) or 'model not reported'
    return out['result'], f'claude -p ({version}), model {models}'


def _note(entries):
    if any(e['source'] == FIXTURE_SOURCE for e in entries.values()):
        return 'Contains HAND-WRITTEN FIXTURE entries (not real LLM output); each entry names its source.'
    return 'REAL LLM OUTPUT: every entry is a live claude -p response; each entry names its model and date.'


def _load(path):
    path = Path(path)
    if not path.exists():
        return dict(note=_note({}), entries={})
    cache = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(cache, dict) or not isinstance(cache.get('entries'), dict):
        raise ValueError(f'{path}: replay file needs an "entries" object')
    return cache


def propose(brief, n, replay_path, live=False, runner=claude_cli):
    """Return the seed record used by a run: prompt, source label, raw response and the verdicts."""
    prompt = build_prompt(brief, n)
    key = prompt_sha256(prompt)
    cache = _load(replay_path)
    if live:
        response, via = runner(prompt)
        cache['entries'][key] = dict(source=f'{LIVE_SOURCE}: {via}, {date.today().isoformat()}',
                                     prompt=prompt, response=response)
        cache['note'] = _note(cache['entries'])
        tmp = Path(str(replay_path) + '.tmp')
        tmp.write_text(json.dumps(cache, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        os.replace(tmp, replay_path)
    entry = cache['entries'].get(key)
    if entry is None:
        raise KeyError(f'no replay entry for prompt {key[:12]}; refresh it with --live (spends LLM quota)')
    if entry.get('prompt') != prompt:
        raise ValueError(f'replay entry {key[:12]} stores a different prompt')
    accepted, rejected, duplicates = parse_response(entry['response'])
    return dict(source=entry['source'], prompt=prompt, prompt_sha256=key, response=entry['response'],
                requested=n, accepted=accepted, rejected=rejected, duplicates=duplicates)


def check_record(record, brief, n):
    """Re-derive a stored seed record from its own response; used by ``verify``."""
    prompt = build_prompt(brief, n)
    accepted, rejected, duplicates = parse_response(record['response'])
    expected = dict(source=record['source'], prompt=prompt, prompt_sha256=prompt_sha256(prompt),
                    response=record['response'], requested=n, accepted=accepted, rejected=rejected,
                    duplicates=duplicates)
    if record != expected:
        raise ValueError('LLM seed record does not re-derive from its prompt and response')
