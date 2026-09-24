"""Complete factorial effects; training-pool replication stays explicit."""
import itertools
import numpy as np

ARMS=['transformer_jepa','gru_jepa','transformer_state','gru_state']
KEYS=['replica','arm','checkpoint','episodes','updates','algorithm']
LEVELS=[range(3),ARMS,['best','last'],[256,1024],[5250,21000],['random','cem']]


def analyze(tables):
    expected=set(itertools.product(*LEVELS));lookup={tuple(r[k] for k in KEYS):r for r in tables}
    if len(tables)!=192 or len(lookup)!=192 or set(lookup)!=expected:
        raise ValueError('Require the complete unique 192-route scaling matrix')
    if not all(np.isfinite(r['mean_cost']) and 0<=r['success']<=1 for r in tables):raise ValueError('Invalid metrics')
    def get(settings):return lookup[tuple(settings[k] for k in KEYS)]
    aggregates=[]
    for values in itertools.product(*LEVELS[1:]):
        settings=dict(zip(KEYS[1:],values));rows=[get(dict(replica=r,**settings)) for r in range(3)]
        cost=[r['mean_cost'] for r in rows];success=[r['success'] for r in rows]
        aggregates.append(dict(**settings,pool_costs=cost,pool_successes=success,mean_cost=float(np.mean(cost)),between_pool_cost_sd=float(np.std(cost,ddof=1)),mean_success=float(np.mean(success)),between_pool_success_sd=float(np.std(success,ddof=1))))
    effects=[]
    def add(label,settings,terms):
        costs=[];successes=[];routes=[]
        for replica in range(3):
            rows=[(weight,get(dict(settings,replica=replica,**changes))) for weight,changes in terms]
            costs.append(float(sum(w*r['mean_cost'] for w,r in rows)));successes.append(float(sum(w*r['success'] for w,r in rows)))
            routes.append([dict(weight=w,route=r['route_index']) for w,r in rows])
        effects.append(dict(contrast=label,**settings,pool_cost_effects=costs,pool_success_effects=successes,mean_cost_effect=float(np.mean(costs)),between_pool_cost_effect_sd=float(np.std(costs,ddof=1)),mean_success_effect=float(np.mean(successes)),routes_by_pool=routes))
    for arm,checkpoint,algorithm in itertools.product(ARMS,['best','last'],['random','cem']):
        base=dict(arm=arm,checkpoint=checkpoint,algorithm=algorithm)
        for updates in [5250,21000]:add('1024 minus 256 episodes',dict(base,updates=updates),[(1,dict(episodes=1024)),(-1,dict(episodes=256))])
        for episodes in [256,1024]:add('21000 minus 5250 updates',dict(base,episodes=episodes),[(1,dict(updates=21000)),(-1,dict(updates=5250))])
        add('data-by-update interaction',base,[(1,dict(episodes=1024,updates=21000)),(-1,dict(episodes=256,updates=21000)),(-1,dict(episodes=1024,updates=5250)),(1,dict(episodes=256,updates=5250))])
    for arm,checkpoint,episodes,updates in itertools.product(ARMS,['best','last'],[256,1024],[5250,21000]):
        add('CEM minus random',dict(arm=arm,checkpoint=checkpoint,episodes=episodes,updates=updates),[(1,dict(algorithm='cem')),(-1,dict(algorithm='random'))])
    for checkpoint,episodes,updates,algorithm in itertools.product(['best','last'],[256,1024],[5250,21000],['random','cem']):
        base=dict(checkpoint=checkpoint,episodes=episodes,updates=updates,algorithm=algorithm)
        for architecture in ['transformer','gru']:
            add(f'JEPA minus state implementation within {architecture}',base,[(1,dict(arm=architecture+'_jepa')),(-1,dict(arm=architecture+'_state'))])
        for target in ['jepa','state']:
            add(f'Transformer minus GRU within {target}',base,[(1,dict(arm='transformer_'+target)),(-1,dict(arm='gru_'+target))])
    return dict(aggregates=aggregates,effects=effects,scope='Complete development matrix; primary fixed-last and separate best sensitivity. All three independent pool effects retained. SD is descriptive training-pool variation, not a confidence interval. Objective/architecture comparisons also change supervision or capacity; not pure causal effects.')
