"""Small, offline GP-versus-random comparison using the existing search and analysis paths."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from alpha_gp_lab import store
from alpha_gp_lab.budget import audit_budget
from alpha_gp_lab.cli import load_inputs, summary
from alpha_gp_lab.config import digest, read_json, validate

ROOT = Path(__file__).resolve().parents[1]


def markdown(report, config, audit):
    lines = ['# Small GP-versus-random comparison', '',
             'SYNTHETIC mechanics example; not evidence of a market edge.', '',
             f"Delay: {config['evaluation']['delay']} bars. Fee: {config['evaluation']['fee_bps']} bps per side, "
             'charged on drifted-book turnover. No spread, slippage, borrow or funding.', '',
             'Train breeds/ranks; validation qualifies/selects; test evaluates each selected pick.',
             'No qualifying candidate means no pick or test score, never a zero return.', '',
             ('Admitted-score budgets: COMPLETE.' if audit['eligible'] else
              'Winner verdict: WITHHELD because declared admitted-score budgets were incomplete.'),
             'Accounting includes GP elites; not equal compute. See budget_audit.json for every seed.', '',
             '| Search | Seed | Status | Expression | Test rank IC | Net / interval | Train scores |',
             '|---|---:|---|---|---:|---:|---:|']
    for row in report['per_seed']:
        test = row['test'] or {}
        lines.append(f"| {row['kind']} | {row['seed']} | {row['status']} | `{row['selected']}` | "
                     f"{test.get('mean_ic')} | {test.get('mean_net')} | {row['trials']['train']} |")
    lines += ['', 'GP minus random across selected seeds (original 2-SE rule; two seeds are a mechanics sample):',
              '', '```json', json.dumps(audit['comparison'], indent=2, sort_keys=True), '```', '',
              'Read audited_comparison.json for the canonical controls, daily series and effective inference.',
              'comparison.json is the raw legacy analysis; its winner arithmetic is unadjusted.',
              'Read primary/report.md and primary/state.sqlite for the primary search and lineage.',
              'Inputs, selection thresholds and split spans are in config.json and analysis_plan.json.',
              'Historical real-data conclusions remain in docs/results.md and results/binance_analysis.json.']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'fixtures' / 'comparison_small_config.json'))
    parser.add_argument('--out', required=True, help='fresh output directory; incomplete runs are retained')
    args = parser.parse_args(argv)
    config_path, out = Path(args.config).resolve(), Path(args.out).resolve()
    config = validate(read_json(config_path.read_bytes()))
    if config['data']['kind'] != 'synthetic' or config['llm']['use_seeds']:
        parser.error('offline example requires synthetic data and llm.use_seeds=false')
    data, gp = config['data'], config['gp']
    if (data['assets'] > 16 or sum(s['days'] for s in data['segments']) > 200
            or gp['population'] * gp['generations'] > 128 or gp['max_attempts'] > 10
            or not isinstance(config.get('splits'), dict)):
        parser.error('small example requires <=16 assets, <=200 dates, <=128 train scores, '
                     'max_attempts<=10 and one split; '
                     'use the general run/analysis commands for other studies')
    if out.exists():
        parser.error('use a fresh output directory; existing evidence is never overwritten')
    raw, panel, llm = load_inputs(config_path)
    out.mkdir(parents=True)
    (out / 'config.json').write_bytes(raw)
    primary = store.run(raw, panel, llm, out / 'primary')
    (out / 'main_result.json').write_text(json.dumps(summary(primary), indent=2, sort_keys=True) + '\n')
    seeds = [config['seed'], (config['seed'] + 1) % (2**63)]
    budget = gp['population'] * gp['generations']
    plan = dict(name='small-offline-comparison', main_config='config.json', main_result='main_result.json',
                primary_seed=config['seed'], seeds=seeds, random_search_budget=budget,
                control={'range_5d': '-ts_mean(((close - low) / close), 5)'},
                interpretation_references={'range_5d': '-ts_mean(((close - low) / close), 5)'},
                bootstrap=dict(block=5, reps=200, seed=1, level=0.9), newey_west_lags=[3],
                bonferroni_counts=['validation_candidates', 'unique_expressions'])
    (out / 'analysis_plan.json').write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
    with (out / 'analysis.stdout.json').open('x') as stdout, (out / 'analysis.stderr.log').open('x') as stderr:
        run = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'analyze_binance.py'), '--config',
                              str(out / 'analysis_plan.json'), '--out', str(out / 'comparison.json'), '--workers', '1'],
                             cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / 'src')),
                             stdout=stdout, stderr=stderr, check=False)
    if run.returncode:
        print(f'analysis failed (exit {run.returncode}); inspect {out / "analysis.stderr.log"}; '
              'use a fresh directory after correcting the cause', file=sys.stderr)
        return run.returncode
    report = read_json((out / 'comparison.json').read_bytes())
    audit = audit_budget(report, seeds, budget)
    (out / 'budget_audit.json').write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n')
    audited = copy.deepcopy(report)
    audited['schema'] = 'alpha-gp-audited-comparison/v1'
    audited['unadjusted_gp_minus_random'] = copy.deepcopy(report['seeds']['gp_minus_random'])
    audited['seeds']['gp_minus_random'] = copy.deepcopy(audit['comparison'])
    audited['budget_audit'] = {k: v for k, v in audit.items() if k not in ('comparison', 'unadjusted_comparison')}
    (out / 'audited_comparison.json').write_text(json.dumps(audited, indent=2, sort_keys=True) + '\n')
    (out / 'report.md').write_text(markdown(report, config, audit))
    sources = [Path(__file__).resolve(), ROOT / 'scripts' / 'analyze_binance.py', *sorted(store.PACKAGE.glob('*.py'))]
    manifest = dict(schema='alpha-gp-small-comparison/v1', canonical_result='audited_comparison.json',
                    data_label=panel.label,
                    input_sha256=digest(raw), panel_sha256=digest((out / 'primary' / 'panel.json').read_bytes()),
                    delay=config['evaluation']['delay'], fee_bps_per_side=config['evaluation']['fee_bps'],
                    candidate_budget_per_search=budget, seeds=seeds,
                    source_sha256={str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in sources},
                    files={str(p.relative_to(out)): digest(p.read_bytes()) for p in sorted(out.rglob('*')) if p.is_file()})
    (out / 'example_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(data=panel.label, seeds=seeds, candidate_budget_per_search=budget,
                         report=str(out / 'report.md'), comparison=audit['comparison'],
                         budget_eligible=audit['eligible'], canonical_result='audited_comparison.json'), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
