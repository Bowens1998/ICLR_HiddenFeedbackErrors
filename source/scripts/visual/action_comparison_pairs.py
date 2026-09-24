"""Matched comparisons for the complete 72-route action baseline matrix."""
from action_comparison_manifest import specifications


def comparison_pairs(rows):
    keys = ('replica', 'arm', 'mode', 'checkpoint', 'algorithm')
    models, routes = specifications()
    expected = {tuple({**models[r['model_index']], **r}[k] for k in keys) for r in routes}
    actual = {tuple(row[k] for k in keys): row['route_index'] for row in rows}
    if len(rows) != 72 or len(actual) != 72 or set(actual) != expected:
        raise ValueError('Action analysis requires all 72 unique frozen combinations')
    pairs = []
    for key, index in actual.items():
        replica, arm, mode, checkpoint, algorithm = key
        if algorithm == 'cem':
            pairs.append((index, actual[(replica, arm, mode, checkpoint, 'random')], 'CEM minus random'))
        controls = {'none': [], 'inverse': ['none'], 'inverse_goal': ['none', 'inverse']}[mode]
        for control in controls:
            pairs.append((index, actual[(replica, arm, control, checkpoint, algorithm)], f'{mode} minus {control}'))
    return pairs
