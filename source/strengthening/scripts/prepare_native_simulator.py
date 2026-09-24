"""Separate native physics environment; deterministic replay smoke on development seeds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'strengthening/adapters'))
from contracts import sha, atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--output',required=True);p.add_argument('--inside',action='store_true');a=p.parse_args()
    base=Path(a.base);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if not a.inside:
        env=base/'envs/native-simulator-v1'
        if not (env/'bin/python').exists():subprocess.run([sys.executable,'-m','venv','--system-site-packages',str(env)],check=True)
        python=str(env/'bin/python')
        subprocess.run([python,'-m','pip','install','--disable-pip-version-check','gym==0.23.1','pymunk==6.8.0','pygame==2.5.2',
            'shapely==2.0.3','opencv-python-headless==4.8.1.78','scikit-image==0.22.0','matplotlib==3.8.4','einops==0.8.1','omegaconf==2.3.0'],check=True)
        freeze=subprocess.check_output([python,'-m','pip','freeze','--all'],text=True);(out/'environment.freeze.txt').write_text(freeze)
        subprocess.run([python,str(Path(__file__).resolve()),'--base',a.base,'--output',a.output,'--inside'],check=True);return
    os.environ['SDL_VIDEODRIVER']='dummy';os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
    sys.path.insert(0,str(ROOT/'strengthening/external/dino_wm'))
    import numpy as np
    import pymunk
    from native_simulator import load_native_pusht
    PushTEnv=load_native_pusht(ROOT/'strengthening/external/dino_wm')
    from contracts import namespace_seed
    def replay(seed,actions):
        env=PushTEnv(with_velocity=True,with_target=True);env.seed(seed);obs,state=env.reset()
        frames=[obs['visual'].copy()];states=[state.copy()];proprio=[obs['proprio'].copy()]
        for u in actions:
            obs,_,_,info=env.step(u);frames.append(obs['visual'].copy());states.append(info['state'].copy());proprio.append(obs['proprio'].copy())
        env.close();return np.stack(frames),np.stack(states),np.stack(proprio)
    rows=[]
    for i in range(2):
        seed=namespace_seed(20260919,'B_native_simulator_smoke')+i
        actions=np.repeat(np.random.default_rng(seed).uniform(-.35,.35,(7,2)),5,axis=0).astype(np.float32)
        left=replay(seed,actions);right=replay(seed,actions)
        for x,y in zip(left,right):np.testing.assert_array_equal(x,y)
        assert left[0].shape==(36,224,224,3) and left[1].shape==(36,7)
        np.testing.assert_array_equal(left[2],left[1][:,[0,1,5,6]])
        assert np.isfinite(left[1]).all()
        np.savez_compressed(out/f'smoke_{i}.npz',pixels=left[0][::5],states=left[1],proprio=left[2],actions=actions,seed=seed)
        rows.append(dict(seed=seed,exact_physics_and_pixels_replay=True,proprio_matches_physical_channels=True,file_sha256=sha(out/f'smoke_{i}.npz')))
    atomic_json(out/'report.json',dict(status='PASS_NATIVE_SIMULATOR_DETERMINISTIC_REPLAY',python=sys.executable,pymunk_version=pymunk.version,
        environment_lock_sha256=sha(out/'environment.freeze.txt'),simulator_source_sha256=sha(ROOT/'strengthening/external/dino_wm/env/pusht/pusht_env.py'),
        rows=rows,source_sha256=sha(__file__),scope='Two fresh development seeds; native relative controls and proprio. Not a replay of compressed dataset videos or a confirmation.'))
    (out/'DONE').write_text('accepted\n')


if __name__=='__main__':main()
