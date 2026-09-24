"""Complete prespecified checkpoint/solver matrices; never rank by task outcomes."""
ARMS=['transformer_jepa','gru_jepa','transformer_state','gru_state']

def specifications(layout):
    if layout=='scaling':
        cells=[(f'replica_{r}/n{n}/u{u}',{'replica':r,'episodes':n,'updates':u}) for r in range(3) for n in [256,1024] for u in [5250,21000]]
        checkpoints=['best','last'];policies=[('random','full'),('cem','full')]
    elif layout=='seed':
        cells=[('',{})];checkpoints=['best'];policies=[('random','held'),('random','full'),('cem','held'),('cem','full')]
    elif layout=='engineering':
        cells=[('',{})];checkpoints=['best','last'];policies=[('random','held'),('random','full'),('cem','held'),('cem','full')]
    else:raise ValueError(layout)
    models=[];routes=[]
    for cell,metadata in cells:
        for arm in ARMS:
            for checkpoint in checkpoints:
                index=len(models);models.append({'training_path':f'{cell}/{arm}'.lstrip('/'),'cell_path':cell,'arm':arm,'checkpoint':checkpoint,**metadata})
                for algorithm,parameterization in policies:routes.append({'model_index':index,'algorithm':algorithm,'parameterization':parameterization})
    return models,routes
