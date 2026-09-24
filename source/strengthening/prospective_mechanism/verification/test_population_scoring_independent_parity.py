"""Independent generated-data oracles for the population/scoring interface."""
from pathlib import Path
import sys
import unittest
import json
import numpy as np
PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PHASE/'scripts'))
from score_reserved_readouts import assemble_group_shard, decode_all, frozen_bootstrap
from independent_measurement_reference import independent_gelu_forward


class IndependentInterfaceParity(unittest.TestCase):
    def test_every_objective_constraint_goal_stream_branch_horizon_axis(self):
        streams={}; n=5
        for stream in range(4):
            tokens=np.zeros((2,2,n,4,5,192),np.float32)
            truth=np.zeros((n,5,6),np.float64)
            for oi in range(2):
                for ci in range(2):
                    for goal in range(n):
                        for branch in range(4):
                            for horizon in range(5):
                                tokens[oi,ci,goal,branch,horizon,:]=100000*oi+10000*ci+1000*goal+100*stream+10*branch+horizon
            for goal in range(n):
                for horizon in range(5): truth[goal,horizon,:]=1000*goal+100*stream+horizon
            streams[stream]=dict(tokens=tokens,truth=truth)
        assembled,truth=assemble_group_shard(streams,1,4)
        self.assertEqual(assembled.shape,(2,2,3,4,4,5,192))
        for oi in range(2):
            for ci in range(2):
                for local,goal in enumerate(range(1,4)):
                    for stream in range(4):
                        for branch in range(4):
                            for horizon in range(5):
                                expected=100000*oi+10000*ci+1000*goal+100*stream+10*branch+horizon
                                np.testing.assert_array_equal(assembled[oi,ci,local,stream,branch,horizon],np.full(192,expected))
        for local,goal in enumerate(range(1,4)):
            for stream in range(4):
                for horizon in range(5):np.testing.assert_array_equal(truth[local,stream,horizon],np.full(6,1000*goal+100*stream+horizon))
        with self.assertRaises(ValueError):assemble_group_shard({k:v for k,v in streams.items() if k<3},1,4)

    def test_actual_reserved_architecture_physical_forward_matches_independent_erf(self):
        rng=np.random.default_rng(7091)
        h=dict(mean=rng.normal(size=192),scale=rng.uniform(.5,2,size=192),target_mean=rng.normal(size=6),target_scale=rng.uniform(.5,3,size=6))
        shapes={'input':(256,192),'output':(6,256),'skip':(6,192)}
        for block in range(2):
            for fc in ('fc1','fc2'):shapes[f'blocks.{block}.{fc}']=(256,256)
        for layer,shape in shapes.items():
            h[layer+'.weight']=(rng.normal(size=shape)*.03).astype(np.float32)
            if layer!='skip':h[layer+'.bias']=(rng.normal(size=shape[0])*.1).astype(np.float32)
        tokens=rng.normal(size=(2,3,192)).astype(np.float32)
        actual=decode_all(tokens,h,'D');reference=independent_gelu_forward(tokens,h)
        self.assertEqual(actual.shape,(2,3,6));np.testing.assert_allclose(actual,reference,rtol=1e-12,atol=1e-9)
        self.__class__.maximum_forward_difference=float(np.max(np.abs(actual-reference)))

    def test_frozen_bootstrap_complete_draw_matrix(self):
        cfg=json.loads((PHASE/'protocol/DESIGN.lock.json').read_text())
        actual=frozen_bootstrap(cfg)
        expected=np.random.Generator(np.random.PCG64(3350567987)).integers(0,256,(20000,256),dtype=np.int64)
        np.testing.assert_array_equal(actual,expected)


if __name__=='__main__':unittest.main(verbosity=2)
