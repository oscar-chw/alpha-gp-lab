"""Strict experiment-config validation: unknown or missing keys refuse, bounds are explicit."""
import hashlib
import json
import math

from .data import REGIMES
from .grammar import MAX_DEPTH, MAX_WINDOW, parse


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(raw):
    """Parse JSON refusing duplicate keys and NaN/Infinity, which ``json`` accepts by default."""
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError(f'duplicate JSON key {k!r}')
            out[k] = v
        return out

    def reject(_):
        raise ValueError('non-finite JSON constant')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def _keys(obj, required, optional=(), where='config'):
    if not isinstance(obj, dict):
        raise ValueError(f'{where} must be an object')
    missing, unknown = set(required) - set(obj), set(obj) - set(required) - set(optional)
    if missing or unknown:
        raise ValueError(f'{where}: missing {sorted(missing)}, unknown {sorted(unknown)}')


def _int(v, low, high, name):
    if type(v) is not int or not low <= v <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')


def _num(v, low, high, name):
    if type(v) not in (int, float) or not math.isfinite(v) or not low <= v <= high:
        raise ValueError(f'{name} must be a finite number in {low}..{high}')


def _span(pair, name):
    if not isinstance(pair, list) or len(pair) != 2:
        raise ValueError(f'{name} must be [start, end]')
    _int(pair[0], 0, 10**6, name + ' start')
    _int(pair[1], pair[0] + 1, 10**6, name + ' end')


def validate(config):
    _keys(config, ['schema_version', 'name', 'seed', 'data', 'evaluation', 'gp', 'fitness', 'selection', 'llm'],
          ['splits', 'walk_forward'])
    if config['schema_version'] != 1:
        raise ValueError('unsupported config schema_version')
    if not isinstance(config['name'], str) or not 1 <= len(config['name']) <= 64:
        raise ValueError('name must be a short string')
    _int(config['seed'], 0, 2**63 - 1, 'seed')
    if ('splits' in config) == ('walk_forward' in config):
        raise ValueError('give exactly one of splits or walk_forward')

    data = config['data']
    if not isinstance(data, dict) or data.get('kind') not in ('synthetic', 'csv'):
        raise ValueError('data.kind must be synthetic or csv')
    if data['kind'] == 'synthetic':
        _keys(data, ['kind', 'seed', 'assets', 'industries', 'start_date', 'segments'], ['strength'], 'data')
        _int(data['seed'], 0, 2**63 - 1, 'data.seed')
        _int(data['assets'], 4, 500, 'data.assets')
        _int(data['industries'], 2, data['assets'], 'data.industries')
        if 'strength' in data:
            _num(data['strength'], 0, 0.05, 'data.strength')
        if not isinstance(data['segments'], list) or not data['segments']:
            raise ValueError('data.segments must be a non-empty list')
        for seg in data['segments']:
            _keys(seg, ['regime', 'days'], where='segment')
            if seg['regime'] not in REGIMES:
                raise ValueError(f'unknown regime {seg["regime"]!r}; known: {sorted(REGIMES)}')
            _int(seg['days'], 1, 100000, 'segment days')
    else:
        _keys(data, ['kind', 'path'], where='data')
        if not isinstance(data['path'], str) or not data['path']:
            raise ValueError('data.path must be a directory path')

    ev = config['evaluation']
    _keys(ev, ['delay', 'fee_bps'], where='evaluation')
    _int(ev['delay'], 1, 5, 'evaluation.delay')
    _num(ev['fee_bps'], 0, 100, 'evaluation.fee_bps')

    if 'splits' in config:
        splits = config['splits']
        _keys(splits, ['train', 'validation', 'test'], where='splits')
        previous = -1
        for name in ('train', 'validation', 'test'):
            _span(splits[name], 'splits.' + name)
            if splits[name][0] <= previous:
                raise ValueError('splits must be ordered train < validation < test and disjoint')
            previous = splits[name][1]
    else:
        wf = config['walk_forward']
        _keys(wf, ['warmup', 'train', 'validation', 'test', 'step'], where='walk_forward')
        for k in ('warmup', 'train', 'validation', 'test', 'step'):
            _int(wf[k], 0 if k == 'warmup' else 1, 100000, 'walk_forward.' + k)
        if wf['step'] < wf['test']:
            raise ValueError('walk_forward.step must be >= test so test windows never overlap')

    gp = config['gp']
    _keys(gp, ['population', 'generations', 'tournament', 'elitism', 'crossover_prob', 'max_depth', 'max_nodes',
               'init_depth', 'windows', 'hall_of_fame', 'max_attempts'], where='gp')
    _int(gp['population'], 4, 1024, 'gp.population')
    _int(gp['generations'], 1, 200, 'gp.generations')
    _int(gp['tournament'], 2, gp['population'], 'gp.tournament')
    _int(gp['elitism'], 0, gp['population'] - 1, 'gp.elitism')
    _num(gp['crossover_prob'], 0, 1, 'gp.crossover_prob')
    _int(gp['max_depth'], 2, MAX_DEPTH, 'gp.max_depth')
    _int(gp['max_nodes'], 3, 128, 'gp.max_nodes')
    _span(gp['init_depth'], 'gp.init_depth')
    if gp['init_depth'][1] > gp['max_depth']:
        raise ValueError('gp.init_depth must not exceed gp.max_depth')
    if not isinstance(gp['windows'], list) or not gp['windows'] or len(set(gp['windows'])) != len(gp['windows']):
        raise ValueError('gp.windows must be a non-empty list of distinct integers')
    for w in gp['windows']:
        _int(w, 1, MAX_WINDOW, 'gp.windows')
    if max(gp['windows']) < 2:
        raise ValueError('gp.windows needs a window >= 2 for ts_std/ts_rank/ts_corr')
    _int(gp['hall_of_fame'], 1, gp['population'], 'gp.hall_of_fame')
    _int(gp['max_attempts'], 1, 1000, 'gp.max_attempts')

    fit = config['fitness']
    _keys(fit, ['turnover_penalty', 'complexity_penalty'], where='fitness')
    _num(fit['turnover_penalty'], 0, 1, 'fitness.turnover_penalty')
    _num(fit['complexity_penalty'], 0, 1, 'fitness.complexity_penalty')

    sel = config['selection']
    _keys(sel, ['min_ic', 'max_turnover', 'max_corr', 'select_k', 'existing_alphas'], where='selection')
    _num(sel['min_ic'], -1, 1, 'selection.min_ic')
    _num(sel['max_turnover'], 0, 2, 'selection.max_turnover')
    _num(sel['max_corr'], 0, 1, 'selection.max_corr')
    _int(sel['select_k'], 1, 50, 'selection.select_k')
    if not isinstance(sel['existing_alphas'], list) or len(sel['existing_alphas']) > 50:
        raise ValueError('selection.existing_alphas must be a list of at most 50 expressions')
    for expr in sel['existing_alphas']:
        parse(expr)

    llm = config['llm']
    _keys(llm, ['brief', 'n', 'replay', 'use_seeds'], where='llm')
    if type(llm['use_seeds']) is not bool:
        raise ValueError('llm.use_seeds must be true or false (false = ablation without LLM seeds)')
    if not isinstance(llm['brief'], str) or not 1 <= len(llm['brief']) <= 2000:
        raise ValueError('llm.brief must be 1..2000 characters')
    _int(llm['n'], 1, 50, 'llm.n')
    if not isinstance(llm['replay'], str) or not llm['replay']:
        raise ValueError('llm.replay must be a file path')
    return config


def folds(config, n_dates):
    """Train/validation/test index spans: one fold for ``splits``, rolling folds for ``walk_forward``."""
    if 'splits' in config:
        s = config['splits']
        if s['test'][1] > n_dates - 1:
            raise ValueError('splits extend past the last date')
        return [s]
    wf, out, k = config['walk_forward'], [], 0
    while True:
        ts = wf['warmup'] + k * wf['step']
        te = ts + wf['train']
        vs, ve = te + 1, te + 1 + wf['validation']
        xs, xe = ve + 1, ve + 1 + wf['test']
        if xe > n_dates - 1:
            break
        out.append(dict(train=[ts, te], validation=[vs, ve], test=[xs, xe]))
        k += 1
    if not out:
        raise ValueError('panel too short for one walk-forward fold')
    return out
