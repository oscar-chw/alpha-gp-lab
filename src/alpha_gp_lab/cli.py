"""Command line: demo, walkforward, run, verify, seeds, verify-data."""
import argparse
import json
from pathlib import Path
import sys
import time

from . import store
from .config import read_json, validate
from .data import check_universe, load_csv_dir, synthetic_from_config
from .llm_seed import no_seeds, propose

ROOT = Path(__file__).resolve().parents[2]
DEMO_CONFIG = ROOT / 'fixtures' / 'demo_config.json'
WALK_FORWARD_CONFIG = ROOT / 'fixtures' / 'walkforward_config.json'
UNIVERSE = ROOT / 'fixtures' / 'binance_universe.json'


def seed_record(config, config_path, live=False):
    llm = config['llm']
    return propose(llm['brief'], llm['n'], Path(config_path).parent / llm['replay'], live=live)


def load_inputs(config_path):
    """Read a config and build its panel and LLM seed record. Relative paths resolve from the config's folder."""
    config_path = Path(config_path)
    raw = config_path.read_bytes()
    config = validate(read_json(raw))
    data = config['data']
    if data['kind'] == 'synthetic':
        panel = synthetic_from_config(data)
    else:
        panel, dropped = load_csv_dir(config_path.parent / data['path'])
        if any(dropped.values()):
            print(json.dumps(dict(rows_dropped_to_align_dates=dropped)), file=sys.stderr)
    # The ablation reads no LLM output, so it needs no replay entry (and runs before any live call).
    llm = seed_record(config, config_path) if config['llm']['use_seeds'] else no_seeds(config['llm']['brief'], config['llm']['n'])
    return raw, panel, llm


def summary(report):
    """The printed result: what a reader needs, nothing that is not in report.json."""
    keep = ('mean_ic', 'ic_tstat', 'mean_turnover', 'mean_gross', 'mean_net', 'net_tstat', 'sum_net', 'intervals',
            'valid_ic_intervals')
    folds = []
    for f in report['folds']:
        folds.append(dict(fold=f['fold'], status=f['status'], regimes=f['regimes'], split_dates=f['split_dates'],
                          selected=f['selected_expression'],
                          selected_origin=f['selected_origin'],
                          validation={k: f['validation'][k] for k in keep} if f['validation'] else None,
                          test={k: f['test'][k] for k in keep} if f['test'] else None,
                          occurrences=f['counts']['occurrences'], unique_expressions=f['unique_expressions'],
                          # every candidate scored on validation: the multiple-testing count of the pick
                          validation_candidates=sum(r['split'] == 'validation' for r in f['results']),
                          baselines={n: dict(expression=b['expression'], **{k: {m: b[k][m] for m in keep}
                                                                            for k in ('validation', 'test')})
                                     for n, b in f['baselines'].items()},
                          rejected={k: v for k, v in f['counts'].items() if k.startswith('rejected')},
                          shortlist=[s['expression'] for s in f['shortlist']],
                          correlation_rejected=len(f['correlation_rejected'])))
    return dict(name=report['name'], mode=report['mode'], data=report['data']['label'],
                llm=dict(used_as_seeds=report['llm']['used_as_seeds'], source=report['llm']['source'],
                         accepted=len(report['llm']['accepted']),
                         rejected=len(report['llm']['rejected'])),
                folds=folds, summary=report['summary'])


def main(argv=None):
    parser = argparse.ArgumentParser(prog='alpha_gp_lab', description='GP alpha search with strict train/validation/test roles.')
    sub = parser.add_subparsers(dest='command', required=True)
    for name, text in (('demo', 'single split on the SYNTHETIC multi-regime demo config'),
                       ('walkforward', 'rolling train/validation/test folds on the SYNTHETIC walk-forward config')):
        p = sub.add_parser(name, help=text)
        p.add_argument('--out', required=True, help='fresh output directory')
    p = sub.add_parser('run', help='run any config')
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p = sub.add_parser('verify', help='replay a completed run directory')
    p.add_argument('out')
    p = sub.add_parser('verify-data', help='check a local CSV directory against the pinned Binance universe')
    p.add_argument('--data', default='data/binance-daily')
    p.add_argument('--universe', default=str(UNIVERSE))
    p = sub.add_parser('seeds', help='show the LLM seed proposals for a config (replay file by default)')
    p.add_argument('--config', default=str(DEMO_CONFIG))
    p.add_argument('--live', action='store_true',
                   help='call the local `claude -p` CLI and refresh the replay cache (spends LLM quota)')
    args = parser.parse_args(argv)

    if args.command == 'verify-data':
        universe = read_json(Path(args.universe).read_bytes())
        problems = check_universe(args.data, universe)
        print(json.dumps(dict(universe=universe['name'], symbols=len(universe['symbols']),
                              first_date=universe['first_date'], last_date=universe['last_date'],
                              ok=not problems, problems=problems), indent=2))
        return 1 if problems else 0
    if args.command == 'seeds':
        llm = seed_record(validate(read_json(Path(args.config).read_bytes())), args.config, live=args.live)
        print(json.dumps({k: llm[k] for k in ('source', 'prompt_sha256', 'requested', 'accepted', 'rejected', 'duplicates')}, indent=2))
        return 0
    started = time.monotonic()
    if args.command == 'verify':
        report = store.verify(args.out)
    else:
        config_path = {'demo': DEMO_CONFIG, 'walkforward': WALK_FORWARD_CONFIG}.get(args.command) or args.config
        report = store.run(*load_inputs(config_path), args.out)
    print(json.dumps(summary(report), indent=2, sort_keys=True))
    print(f'{args.command} finished in {time.monotonic() - started:.1f}s', file=sys.stderr)
    return 0
