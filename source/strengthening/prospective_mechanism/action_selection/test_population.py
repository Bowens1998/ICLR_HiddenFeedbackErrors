"""Focused new assembly checks using synthetic inputs only."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import accept_inputs as acceptance
import population as producer
from test_common_prefix import fixture, StatefulSimulator
from test_projection_eight import fixture as head_fixture


class AssemblyTests(unittest.TestCase):
    def test_recipient_full_saved_alignment(self):
        c, a, p, row = fixture()
        inputs, outcomes, checks = acceptance.common.build_compact_case(StatefulSimulator, c, a, p, row, 32)
        case = dict(row, checks=checks, outcomes={'synthetic': True})
        acceptance.validate_saved(case, inputs, c, a, p, row, 'recipient', 1, outcomes)
        corrupt = copy.deepcopy(inputs)
        # Keep a structurally valid set of source actions but change frozen RNG order.
        corrupt['source_candidate_indices'][[2, 3]] = corrupt['source_candidate_indices'][[3, 2]]
        corrupt['suffix_actions'][[2, 3]] = corrupt['suffix_actions'][[3, 2]]
        with self.assertRaises(AssertionError):
            acceptance.validate_saved(case, corrupt, c, a, p, row, 'recipient', 1, outcomes)

    def test_physical_misalignment_blocks(self):
        c, a, p, row = fixture()
        inputs, outcomes, checks = acceptance.common.build_compact_case(StatefulSimulator, c, a, p, row, 0)
        case = dict(row, checks=checks, outcomes={'synthetic': True})
        outcomes['terminal_states'][2, 2] += 1
        with self.assertRaises(AssertionError):
            acceptance.validate_saved(case, inputs, c, a, p, row, 'recipient', 0, outcomes)

    def test_donor_acceptance_omits_outcomes(self):
        c, a, p, row = fixture()
        inputs, checks = acceptance.common.build_donor_case(c, a, p, row, 64)
        case = dict(row, checks=checks)
        acceptance.validate_saved(case, inputs, c, a, p, row, 'donor', 2)
        with self.assertRaises(ValueError):
            acceptance.validate_saved(dict(case, outcomes={}), inputs, c, a, p, row, 'donor', 2)

    def test_goal_broadcast_and_cost_units(self):
        head, _, _, _ = head_fixture()
        head['4.weight'][2, 0] = 2
        head['4.bias'][5] = 1
        tokens = np.zeros((3, 2, 2, 4, 32, 192), np.float32)
        tokens[..., 0] = np.arange(32)
        goal = np.zeros((3, 192), np.float32)
        goal[:, 0] = [0, 1, 2]
        result = producer.predicted_costs(tokens, goal, head)
        self.assertEqual(result.shape, (3, 2, 2, 4, 32))
        for i in range(3):
            np.testing.assert_allclose(result[i], np.broadcast_to(36*(np.arange(32)-i)**2, result[i].shape))

    def test_named_model_axis_mapping(self):
        roots = np.arange(2*2*5*192, dtype=np.float32).reshape(2, 2, 5, 192)
        mapped = producer.predicted_map(roots, 3)
        for ci, condition in enumerate(producer.stats.CONDITIONS):
            for oi, objective in enumerate(producer.stats.OBJECTIVES):
                np.testing.assert_array_equal(mapped[condition][objective], roots[ci, oi, 3])


if __name__ == '__main__': unittest.main()
