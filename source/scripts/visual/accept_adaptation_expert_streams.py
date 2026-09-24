"""Read every prescribed expert window and independently check boundary identities."""
import argparse,json
from pathlib import Path
import numpy as np
from adaptation_streams import expert_windows,sha

def main():
    p=argparse.ArgumentParser()
    for key in ('data','freeze','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();rows=[];fr=json.loads((Path(a.freeze)/'report.json').read_text())
    for rep in range(3):
        for split in ('train','validation'):
            d=Path(a.data)/f'replica_{rep}'/'n256'/split;ep=np.load(d/'episodes.npz')
            expected=[]
            for identity,length in zip(ep['source_episode_ids'],ep['lengths']):
                expected.extend([[int(identity),s] for s in range(0,int(length)-15,5)])
            if split=='train':expected=[expected[i] for i in fr['pools'][rep]['selected_indices']];assert len(expected)==1280
            actual=[]
            for identity,(pixels,actions,states) in expert_windows(a.data,a.freeze,rep,split):
                assert pixels.shape==(4,224,224,3) and pixels.dtype==np.uint8
                assert actions.shape==(3,10) and states.shape==(4,7)
                assert np.isfinite(actions).all() and np.isfinite(states).all();actual.append(identity)
            assert actual==expected and len({tuple(x) for x in actual})==len(actual)
            rows.append(dict(replica=rep,split=split,windows=len(actual),identities=actual))
    report=dict(status='PASS_ALL_FIXED_EXPERT_TRAIN_AND_VALIDATION_WINDOWS',rows=rows,freeze_report_sha256=sha(Path(a.freeze)/'report.json'),source_sha256=sha(__file__),stream_source_sha256=sha(Path(__file__).with_name('adaptation_streams.py')),scope='Every selected training and every complete validation window read; shape, finite labels/actions, unique identity and independently enumerated episode boundaries checked. No encoding or fitting yet.')
    with Path(a.output).open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print('PASS',[(r['replica'],r['split'],r['windows']) for r in rows])

if __name__=='__main__':main()
