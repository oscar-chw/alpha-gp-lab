"""LLM seed proposer: ask a model for N grammar expressions, given a short research brief.

The model is an untrusted proposer. It sees the brief and the grammar, never data, and it
cannot promote anything: each line it returns must parse under ``grammar.parse`` or it is
logged and rejected, and accepted seeds still have to earn their place in the GP on train.

Responses are cached in a JSON replay file keyed by the SHA-256 of the exact prompt. Normal
runs and every test read the replay file only. ``live=True`` (the CLI's ``--live``) makes one
request to a pinned open-weight model on OpenRouter to refresh the cache entry; that spends the
operator's free-tier allowance and is never invoked by tests or the demo.
"""
from datetime import date
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import urllib.error
import urllib.request

from .grammar import canonical, grammar_doc, parse

log = logging.getLogger(__name__)
FIXTURE_SOURCE = 'HAND-WRITTEN FIXTURE (not real LLM output)'
LIVE_SOURCE = 'REAL LLM OUTPUT'
NO_SEEDS_SOURCE = 'none: use_seeds is false, no LLM output read'
_BULLET = re.compile(r'^(?:\*\s+|\d+[.)]\s*)')   # '* x', '1. x', '2) x'

# The live proposer. Oscar's decision (2026-10-05): no Anthropic or OpenAI model, in any form.
# Free hosted open weights: Hugging Face Qwen/Qwen3.8-27B at revision
# 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0, Apache-2.0 (HF createdAt 2026-08-05; OpenRouter
# listed it 2026-08-14); docs/llm-seeds.md names the sources. Free tier: 20 requests/minute, 50/day
# without purchased credits, so a refusal is never retried.
OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'
MODEL = 'qwen/qwen3.8-27b:free'
MODEL_WEIGHTS = 'Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 (Apache-2.0)'
KEY_ENV = 'OPENROUTER_API_KEY'
# Hidden reasoning counts against this too, so it is sized for reasoning plus a short list of
# expressions; a longer answer stops with finish_reason 'length' and is refused rather than
# saved truncated.
MAX_TOKENS = 8192
# The listing offers efforts xhigh, medium and low (default xhigh) and not 'none', which could be
# refused or silently run at xhigh; 'low' is the smallest offered. exclude keeps the reasoning
# text out of the response.
REASONING = dict(effort='low', exclude=True)
MAX_BODY_BYTES = 1_000_000   # a body this large is not an answer to the prompt; refuse it unparsed


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


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect would resend the Authorization header to wherever it points; refuse it instead."""

    def redirect_request(self, *args, **kwargs):
        return None


def http_post(url, headers, body, timeout):
    """POST ``body``; return (status, at most MAX_BODY_BYTES + 1 bytes of the response body)."""
    request = urllib.request.Request(url, data=body, headers=headers, method='POST')
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=timeout) as response:
            return response.status, response.read(MAX_BODY_BYTES + 1)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(MAX_BODY_BYTES + 1)


def openrouter(prompt, post=http_post, timeout=120):
    """Make one request to the pinned model; return (response text, label naming model, provider, id).

    Only reached through ``live=True``. The model sees the prompt and nothing else. The key is read
    from the ``OPENROUTER_API_KEY`` environment variable only, and every message that could be
    shown has it removed. Any doubtful response raises, so ``propose`` saves nothing.
    """
    key = os.environ.get(KEY_ENV, '').strip()
    if not key:
        raise RuntimeError(f'live mode needs {KEY_ENV} set (an OpenRouter API key); nothing was sent')
    if any(c.isspace() or not c.isprintable() for c in key):
        # A two-line key file made http.client raise ValueError('Invalid header value b"Bearer <key>"'), unredacted.
        raise RuntimeError(f'{KEY_ENV} contains whitespace or control characters (a multi-line key file?); nothing was sent')

    def refuse(why):
        raise RuntimeError(f'OpenRouter {MODEL}: {why}'.replace(key, '[key]'))
    body = json.dumps(dict(model=MODEL, messages=[dict(role='user', content=prompt)], temperature=0,
                           max_tokens=MAX_TOKENS, reasoning=REASONING)).encode()
    try:
        status, raw = post(OPENROUTER_URL, {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
                           body, timeout)
    except Exception as exc:
        # Transport errors can quote the request headers; show a redacted message and drop the chained original.
        raise RuntimeError(f'OpenRouter {MODEL}: request failed: {type(exc).__name__}: {exc}'.replace(key, '[key]')) from None
    if len(raw) > MAX_BODY_BYTES:
        refuse(f'response body over {MAX_BODY_BYTES} bytes (HTTP {status})')
    try:
        out = json.loads(raw)
    except ValueError:
        out = None
    if status != 200:
        detail = out.get('error') if isinstance(out, dict) else None
        detail = detail.get('message', detail) if isinstance(detail, dict) else detail or raw.decode(errors='replace')
        refuse(f'HTTP {status}{" (rate limit)" if status == 429 else ""}: {str(detail).strip()[-500:]}')
    if not isinstance(out, dict):
        refuse('response is not a JSON object')
    if out.get('error'):
        refuse(f'error in response: {str(out["error"])[-500:]}')
    choices = out.get('choices')
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    if choice is None:
        refuse('response has no choices')
    if choice.get('error'):
        refuse(f'error in choice: {str(choice["error"])[-500:]}')
    if choice.get('finish_reason') != 'stop':
        refuse(f'finish_reason {choice.get("finish_reason")!r}, not "stop"')
    message = choice.get('message')
    content = message.get('content') if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        refuse('response content is empty or not text')
    if out.get('model') not in (MODEL, MODEL.removesuffix(':free')):
        refuse(f'response came from model {out.get("model")!r}, not the pinned {MODEL!r}')
    return content, (f'OpenRouter {MODEL}, weights {MODEL_WEIGHTS}, response model {out["model"]}, '
                     f'provider {out.get("provider") or "not reported"}, id {out.get("id") or "not reported"}')


def _note(entries):
    if any(e['source'] == FIXTURE_SOURCE for e in entries.values()):
        return 'Contains HAND-WRITTEN FIXTURE entries (not real LLM output); each entry names its source.'
    return ('REAL LLM OUTPUT: every entry is a live OpenRouter response from the pinned open-weight model; '
            'each entry names its model, provider, response id and date.')


def _load(path):
    path = Path(path)
    if not path.exists():
        return dict(note=_note({}), entries={})
    cache = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(cache, dict) or not isinstance(cache.get('entries'), dict):
        raise ValueError(f'{path}: replay file needs an "entries" object')
    return cache


def propose(brief, n, replay_path, live=False, runner=openrouter):
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
        raise KeyError(f'no replay entry for prompt {key[:12]}; refresh it with --live (one OpenRouter request)')
    if entry.get('prompt') != prompt:
        raise ValueError(f'replay entry {key[:12]} stores a different prompt')
    accepted, rejected, duplicates = parse_response(entry['response'])
    return dict(source=entry['source'], prompt=prompt, prompt_sha256=key, response=entry['response'],
                requested=n, accepted=accepted, rejected=rejected, duplicates=duplicates)


def no_seeds(brief, n):
    """The record of a run that uses no LLM seeds: same shape, empty response, nothing read."""
    prompt = build_prompt(brief, n)
    return dict(source=NO_SEEDS_SOURCE, prompt=prompt, prompt_sha256=prompt_sha256(prompt), response='',
                requested=n, accepted=[], rejected=[], duplicates=0)


def check_record(record, brief, n):
    """Re-derive a stored seed record from its own response; used by ``verify``."""
    prompt = build_prompt(brief, n)
    accepted, rejected, duplicates = parse_response(record['response'])
    expected = dict(source=record['source'], prompt=prompt, prompt_sha256=prompt_sha256(prompt),
                    response=record['response'], requested=n, accepted=accepted, rejected=rejected,
                    duplicates=duplicates)
    if record != expected:
        raise ValueError('LLM seed record does not re-derive from its prompt and response')
