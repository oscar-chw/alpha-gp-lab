"""Shared test inputs: a small SYNTHETIC config that runs the whole search in well under a second."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from alpha_gp_lab.config import canonical  # noqa: E402
from alpha_gp_lab.data import Panel, synthetic_from_config  # noqa: E402
from alpha_gp_lab.llm_seed import FIXTURE_SOURCE, build_prompt, prompt_sha256, propose  # noqa: E402

SEEDS = '-returns\ngroup_neutralize(-returns, industry)\nts_mean(returns, 5)\nrank(vwap)\n'


def small_config():
    c = json.loads((ROOT / 'fixtures' / 'demo_config.json').read_text())
    c['name'] = 'test-small'
    c['data'].update(assets=12, industries=3, strength=0.008,   # strong effect: 12 assets are noisy
                     segments=[{'regime': 'reversal', 'days': 90}, {'regime': 'adverse', 'days': 30}])
    c['splits'] = {'train': [20, 70], 'validation': [71, 95], 'test': [96, 119]}
    c['gp'].update(population=12, generations=3, tournament=3, elitism=2, hall_of_fame=6, max_attempts=10)
    c['llm'].update(n=4, replay='replay.json')
    return c


def write_replay(folder, config, response=SEEDS):
    prompt = build_prompt(config['llm']['brief'], config['llm']['n'])
    entry = dict(source=FIXTURE_SOURCE, prompt=prompt, response=response)
    path = Path(folder) / config['llm']['replay']
    path.write_text(json.dumps(dict(note='test replay', entries={prompt_sha256(prompt): entry})))
    return path


def panel_for(config):
    return synthetic_from_config(config['data'])


def inputs(config, folder, response=SEEDS):
    """(config bytes, panel, LLM record) exactly as the CLI would build them."""
    llm = propose(config['llm']['brief'], config['llm']['n'], write_replay(folder, config, response))
    return canonical(config), panel_for(config), llm


def perturbed(panel, from_index):
    """The same panel with every bar from ``from_index`` on scaled by an uneven factor per asset."""
    bars = {k: [[v * (1.5 + 0.37 * ((a * 7 + t) % 5)) if t >= from_index else v for a, v in enumerate(row)]
                for t, row in enumerate(rows)] for k, rows in panel.bars.items()}
    return Panel(panel.symbols, panel.groups, panel.dates, bars, panel.synthetic, panel.label, panel.regimes)
