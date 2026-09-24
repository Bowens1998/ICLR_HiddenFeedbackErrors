"""All action-supervision modes, independent pools, checkpoints and planners."""
def specifications():
    models=[];routes=[]
    for replica in range(3):
        for arm in ['transformer_jepa','gru_jepa']:
            for mode in ['none','inverse','inverse_goal']:
                for checkpoint in ['best','last']:
                    idx=len(models)
                    models.append(dict(replica=replica,arm=arm,mode=mode,checkpoint=checkpoint,training_path=f'replica_{replica}/{arm}/{mode}',episodes=256,updates=21000,seed=3072))
                    for algorithm in ['random','cem']:routes.append(dict(model_index=idx,algorithm=algorithm,parameterization='full'))
    return models,routes
