"""Run bundles: write once, never update, verify by hash, read-back and numerical replay.

A bundle directory holds the exact inputs (config bytes, panel, LLM seed record), the code
identity, an append-only SQLite lineage and the report. ``completion.json`` hashes every
member and is written last, so a directory without it is an interrupted run: it is kept for
inspection and refused for reuse. Triggers make every table append-only, so a later
UPDATE or DELETE through SQLite fails instead of silently rewriting lineage.

This detects accidental or casual alteration. It is not a signature: someone who controls
the code and every file can forge a consistent bundle.
"""
from pathlib import Path
import platform
import sqlite3
import sys

from .config import canonical, digest, read_json, validate
from .data import Panel, synthetic_from_config
from .gp import compute_report
from .llm_seed import check_record

PACKAGE = Path(__file__).resolve().parent
MEMBERS = ('config.json', 'panel.json', 'llm.json', 'code.json', 'identity.json', 'state.sqlite',
           'report.json', 'report.md')
MAX_MEMBER_BYTES = 64 * 1024 * 1024
TABLES = {   # name -> (columns, primary-key columns)
    'nodes': ('id TEXT PRIMARY KEY, fold INTEGER NOT NULL, generation INTEGER NOT NULL, record TEXT NOT NULL', ('id',)),
    'edges': ('child TEXT NOT NULL REFERENCES nodes(id), ordinal INTEGER NOT NULL, '
              'parent TEXT NOT NULL REFERENCES nodes(id), PRIMARY KEY (child, ordinal)', ('child', 'ordinal')),
    'results': ('node TEXT NOT NULL REFERENCES nodes(id), split TEXT NOT NULL, record TEXT NOT NULL, '
                'PRIMARY KEY (node, split)', ('node', 'split')),
    'selection': ('fold INTEGER PRIMARY KEY, record TEXT NOT NULL', ('fold',)),
    'llm_proposals': ('ordinal INTEGER PRIMARY KEY, status TEXT NOT NULL, expression TEXT NOT NULL, reason TEXT',
                      ('ordinal',)),
}


def code_identity():
    # Platform is part of the identity: libm differences can change low-order float bits.
    files = sorted(PACKAGE.glob('*.py'))
    return dict(python=list(sys.version_info[:3]), platform=[sys.platform, platform.machine()],
                files={'src/alpha_gp_lab/' + p.name: digest(p.read_bytes()) for p in files})


def _ddl():
    """(name, sql) for every table and trigger. UPDATE and DELETE abort; so does an INSERT that
    would collide with an existing key, which is how INSERT OR REPLACE would otherwise rewrite a
    row without firing the DELETE trigger."""
    out = []
    for name, (cols, key) in TABLES.items():
        out.append((name, f'CREATE TABLE {name} ({cols})'))
        abort = f"BEGIN SELECT RAISE(ABORT, '{name} is append-only'); END"
        for event in ('UPDATE', 'DELETE'):
            out.append((f'{name}_no_{event.lower()}', f'CREATE TRIGGER {name}_no_{event.lower()} BEFORE {event} ON {name} {abort}'))
        match = ' AND '.join(f'{k} = NEW.{k}' for k in key)
        out.append((f'{name}_no_replace', f'CREATE TRIGGER {name}_no_replace BEFORE INSERT ON {name} '
                                          f'WHEN EXISTS (SELECT 1 FROM {name} WHERE {match}) {abort}'))
    return out


def _schema(conn):
    for _, sql in _ddl():
        conn.execute(sql)


def _insert(conn, nodes, results):
    for n in nodes:
        conn.execute('INSERT INTO nodes VALUES (?,?,?,?)', (n['id'], n['fold'], n['generation'], canonical(n).decode()))
        for i, parent in enumerate(n['parents']):
            conn.execute('INSERT INTO edges VALUES (?,?,?)', (n['id'], i, parent))
    for r in results:
        conn.execute('INSERT INTO results VALUES (?,?,?)', (r['node'], r['split'], canonical(r).decode()))


def _selection(fold):
    keys = ('fold', 'status', 'splits', 'selected_id', 'selected_expression', 'selected_origin', 'hall_of_fame',
            'shortlist', 'correlation_rejected')
    return {k: fold[k] for k in keys}


def _llm_rows(llm):
    rows = [('accepted', e, None) for e in llm['accepted']] + [('rejected', r['line'], r['reason']) for r in llm['rejected']]
    return [(i, *row) for i, row in enumerate(rows)]


def _check_panel(config, panel_raw):
    """A synthetic panel must be exactly what its config generates; a CSV panel is pinned by its hash."""
    if config['data']['kind'] == 'synthetic' and canonical(synthetic_from_config(config['data']).to_json()) != panel_raw:
        raise ValueError('panel does not match the synthetic data config')


def markdown(report):
    lines = [f"# Run `{report['name']}` ({report['mode']})", '',
             f"Data: {report['data']['label']}, {report['data']['symbols']} symbols, {report['data']['dates']} dates "
             f"({report['data']['first_date']} to {report['data']['last_date']}).",
             f"LLM seeds: {len(report['llm']['accepted'])} accepted, {len(report['llm']['rejected'])} rejected; "
             f"source: {report['llm']['source']}.", '',
             'Train picks parents, validation picks the winner, test scores only the final pick.', '',
             '| Fold | Test regime | Selected | Validation IC | Test IC | Test turnover | Test net/interval |',
             '|---:|---|---|---:|---:|---:|---:|']
    for f in report['folds']:
        v, t = f['validation'] or {}, f['test'] or {}
        regime = ', '.join(f['regimes']['test']) if f['regimes']['test'] else 'n/a'
        lines.append(f"| {f['fold']} | {regime} | `{f['selected_expression']}` | {v.get('mean_ic')} | "
                     f"{t.get('mean_ic')} | {t.get('mean_turnover')} | {t.get('mean_net')} |")
    lines += ['', 'Synthetic data and hypothetical interval returns; not evidence of market performance.']
    return ('\n'.join(lines) + '\n').encode()


def _write(out, name, raw):
    with (out / name).open('xb') as fh:   # 'x': never overwrite
        fh.write(raw)


def run(config_raw, panel, llm, out, *, _interrupt_after_generation=None):
    """Compute and persist one experiment into the fresh directory ``out``; return the report."""
    config = validate(read_json(config_raw))
    panel_raw = canonical(panel.to_json())
    _check_panel(config, panel_raw)
    llm_raw = canonical(llm)
    code = code_identity()
    identity = dict(config_sha256=digest(config_raw), panel_sha256=digest(panel_raw), llm_sha256=digest(llm_raw),
                    code_sha256=digest(canonical(code)))
    out = Path(out)
    if out.exists():
        report = verify(out)
        if read_json((out / 'identity.json').read_bytes()) != identity:
            raise ValueError('conflicting run identity: use a fresh output directory')
        return report
    out.mkdir(parents=True)
    for name, raw in (('config.json', config_raw), ('panel.json', panel_raw), ('llm.json', llm_raw),
                      ('code.json', canonical(code)), ('identity.json', canonical(identity))):
        _write(out, name, raw)
    conn = sqlite3.connect(out / 'state.sqlite')
    try:
        conn.execute('PRAGMA foreign_keys=ON')
        _schema(conn)
        conn.executemany('INSERT INTO llm_proposals VALUES (?,?,?,?)', _llm_rows(llm))
        conn.commit()

        def on_generation(fold, generation, nodes, results):
            _insert(conn, nodes, results)
            conn.commit()   # each generation is durable before the next one starts
            if _interrupt_after_generation == (fold, generation):
                raise InterruptedError(f'injected stop after fold {fold} generation {generation}')

        report = compute_report(config, panel, llm, on_generation)
        for fold in report['folds']:
            _insert(conn, [], [r for r in fold['results'] if r['split'] != 'train'])
            conn.execute('INSERT INTO selection VALUES (?,?)', (fold['fold'], canonical(_selection(fold)).decode()))
        conn.commit()
    finally:
        conn.close()
    _write(out, 'report.json', canonical(report))
    _write(out, 'report.md', markdown(report))
    _write(out, 'completion.json', canonical({n: digest((out / n).read_bytes()) for n in MEMBERS}))
    verify(out)
    return report


def verify(out):
    """Replay a completed bundle: hashes, code identity, inputs, numbers, then SQLite read-back."""
    out = Path(out)
    if out.is_symlink() or not out.is_dir():
        raise ValueError('a regular run directory is required')
    if {p.name for p in out.iterdir()} != set(MEMBERS) | {'completion.json'}:
        raise ValueError('incomplete or unexpected run members (an interrupted run is never reused)')
    if any(p.is_symlink() or not p.is_file() or p.stat().st_size > MAX_MEMBER_BYTES for p in out.iterdir()):
        raise ValueError('run members must be regular, bounded files')
    manifest = read_json((out / 'completion.json').read_bytes())
    if set(manifest) != set(MEMBERS):
        raise ValueError('manifest does not list exactly the run members')
    for name in MEMBERS:
        if digest((out / name).read_bytes()) != manifest[name]:
            raise ValueError('run member hash mismatch: ' + name)
    code = read_json((out / 'code.json').read_bytes())
    if code != code_identity():
        raise ValueError('code/runtime identity mismatch: replay needs the exact code and Python version')
    raw = {n: (out / n).read_bytes() for n in ('config.json', 'panel.json', 'llm.json')}
    identity = dict(config_sha256=digest(raw['config.json']), panel_sha256=digest(raw['panel.json']),
                    llm_sha256=digest(raw['llm.json']), code_sha256=digest(canonical(code)))
    if read_json((out / 'identity.json').read_bytes()) != identity:
        raise ValueError('input identity mismatch')
    config = validate(read_json(raw['config.json']))
    panel = Panel.from_json(read_json(raw['panel.json']))
    if canonical(panel.to_json()) != raw['panel.json']:
        raise ValueError('panel does not round-trip')
    _check_panel(config, raw['panel.json'])
    llm = read_json(raw['llm.json'])
    check_record(llm, config['llm']['brief'], config['llm']['n'])
    report = read_json((out / 'report.json').read_bytes())
    rebuilt = compute_report(config, panel, llm)
    if canonical(report) != canonical(rebuilt) or (out / 'report.md').read_bytes() != markdown(rebuilt):
        raise ValueError('numerical report replay mismatch')
    with sqlite3.connect((out / 'state.sqlite').resolve().as_uri() + '?mode=ro', uri=True) as conn:
        if conn.execute('PRAGMA integrity_check').fetchone() != ('ok',) or conn.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('database integrity failure')
        schema = conn.execute("SELECT name, sql FROM sqlite_master WHERE type IN ('table', 'trigger') ORDER BY name")
        if schema.fetchall() != sorted(_ddl()):
            raise ValueError('unexpected database schema or altered append-only triggers')
        nodes = [n for f in report['folds'] for n in f['nodes']]
        results = [r for f in report['folds'] for r in f['results']]
        checks = [
            ('SELECT id, fold, generation, record FROM nodes ORDER BY id',
             sorted((n['id'], n['fold'], n['generation'], canonical(n).decode()) for n in nodes), 'node'),
            ('SELECT child, ordinal, parent FROM edges ORDER BY child, ordinal',
             sorted((n['id'], i, p) for n in nodes for i, p in enumerate(n['parents'])), 'lineage'),
            ('SELECT node, split, record FROM results ORDER BY node, split',
             sorted((r['node'], r['split'], canonical(r).decode()) for r in results), 'result'),
            ('SELECT fold, record FROM selection ORDER BY fold',
             [(f['fold'], canonical(_selection(f)).decode()) for f in report['folds']], 'selection'),
            ('SELECT ordinal, status, expression, reason FROM llm_proposals ORDER BY ordinal', _llm_rows(llm), 'llm proposal'),
        ]
        for sql, expected, what in checks:
            if conn.execute(sql).fetchall() != expected:
                raise ValueError(what + ' read-back mismatch')
    return report
