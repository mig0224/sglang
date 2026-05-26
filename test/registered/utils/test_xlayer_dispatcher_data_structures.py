import unittest

import sglang.srt.layers.moe.utils as moe_utils
import torch

from sglang.srt.layers.moe.token_dispatcher.deepep import (
    ExpertSlotInfo,
    PartialAggregator,
    PartialAggregatorState,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")


class TestXLayerDispatcherDataStructures(unittest.TestCase):
    def test_xlayer_dispatcher_flag_defaults_false(self):
        moe_utils.ENABLE_XLAYER_DISPATCHER = None
        self.assertFalse(moe_utils.is_xlayer_dispatcher_enabled())

    def test_partial_aggregator_tracks_partner_bits(self):
        accum = torch.zeros((2, 3), dtype=torch.bfloat16)
        aggr = PartialAggregator(
            y_accum=accum,
            ep_handle=object(),
            partner_expected_bits=(1 << 1) | (1 << 3),
        )
        aggr.mark_dispatch_submitted()
        self.assertEqual(aggr.state, PartialAggregatorState.S_DISPATCH_SUBMITTED)
        self.assertEqual(aggr.expected_partner_count, 2)

        part0 = torch.ones_like(accum)
        self.assertFalse(aggr.add_partial(1, part0))
        self.assertEqual(aggr.state, PartialAggregatorState.S_AWAITING_PARTIALS)
        self.assertEqual(aggr.received_partner_count, 1)

        part1 = torch.full_like(accum, 2)
        self.assertTrue(aggr.add_partial(3, part1))
        self.assertEqual(aggr.state, PartialAggregatorState.S_AGGREGATE_COMPLETE)
        self.assertEqual(aggr.received_partner_count, 2)
        self.assertTrue(torch.equal(aggr.y_accum, part0 + part1))

    def test_expert_slot_info_order_uses_arrival_tick(self):
        slots = [
            ExpertSlotInfo(layer_id=2, arrival_tick=5, request_id="b", rank_id=1),
            ExpertSlotInfo(layer_id=2, arrival_tick=3, request_id="z", rank_id=7),
            ExpertSlotInfo(layer_id=1, arrival_tick=9, request_id="a", rank_id=0),
        ]
        sorted_slots = sorted(slots)
        self.assertEqual(
            [(s.layer_id, s.arrival_tick) for s in sorted_slots],
            [(1, 9), (2, 3), (2, 5)],
        )


if __name__ == "__main__":
    unittest.main()
