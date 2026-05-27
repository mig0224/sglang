import unittest
from unittest import mock

import sglang.srt.layers.moe.utils as moe_utils
import torch

import sglang.srt.layers.moe.token_dispatcher.deepep as deepep_mod
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    ExpertSlotInfo,
    PartialAggregator,
    PartialAggregatorState,
    _DeepEPDispatcherImplXLayer,
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

    def test_xlayer_impl_uses_slot_poll_take_sequence(self):
        impl = _DeepEPDispatcherImplXLayer.__new__(_DeepEPDispatcherImplXLayer)
        impl.layer_id = 4
        impl._arrival_tick = 0
        impl._last_request_id = None
        impl._aggregators = {}
        impl._expert_slot_infos = {}
        impl._rank = 0
        impl.async_finish = True
        impl.handle = object()

        def _raise_unexpected_fallback(exc):
            raise AssertionError(f"Unexpected fallback: {exc}")

        impl._warn_and_fallback = _raise_unexpected_fallback

        call_log = []
        poll_slots = [[17], [21], [22]]
        partial1 = torch.ones((2, 3), dtype=torch.float32)
        partial2 = torch.full((2, 3), 2.0, dtype=torch.float32)
        dispatch_ret = (
            partial1,
            torch.zeros((2, 2), dtype=torch.int64),
            torch.ones((2, 2), dtype=torch.float32),
            [2, 0],
            "handle-after-dispatch",
            "dispatch-event",
        )
        combine_returns = {
            21: (partial1, 1 << 1, "combine-event-1"),
            22: (partial2, 1 << 3, "combine-event-2"),
        }
        expected_bits = (1 << 1) | (1 << 3)

        def fake_call_xlayer(method_name: str, **kwargs):
            call_log.append((method_name, kwargs))
            if method_name == "xlayer_dispatch":
                return None
            if method_name == "xlayer_poll":
                return poll_slots.pop(0)
            if method_name == "xlayer_take_dispatch":
                self.assertEqual(kwargs["slot_idx"], 17)
                return dispatch_ret
            if method_name == "involved_rank_bitmask_for":
                return expected_bits
            if method_name == "xlayer_combine":
                self.assertEqual(kwargs["active_partner_bits"], expected_bits)
                return None
            if method_name == "xlayer_take_combine":
                return combine_returns[kwargs["slot_idx"]]
            raise AssertionError(f"Unexpected method: {method_name}")

        impl._call_xlayer = fake_call_xlayer
        x = torch.zeros((2, 3), dtype=torch.float32)
        topk_ids = torch.zeros((2, 2), dtype=torch.int64)
        topk_weights = torch.ones((2, 2), dtype=torch.float32)
        dummy_cfg = type(
            "DummyDeepEPConfig",
            (),
            {"normal_dispatch_config": None, "normal_combine_config": None},
        )()

        with mock.patch.object(
            deepep_mod.DeepEPConfig, "get_instance", return_value=dummy_cfg
        ):
            impl._dispatch_core(x, topk_ids, topk_weights, previous_event=None)
            key = (impl._last_request_id, impl.layer_id)
            combined_1, _ = impl._combine_core(x, previous_event=None)
            self.assertTrue(torch.equal(combined_1, partial1))
            self.assertIn(key, impl._aggregators)
            self.assertEqual(
                impl._aggregators[key].state, PartialAggregatorState.S_AWAITING_PARTIALS
            )
            combined_2, _ = impl._combine_core(x, previous_event=None)
            self.assertTrue(torch.equal(combined_2, partial1 + partial2))
            self.assertNotIn(key, impl._aggregators)

        method_order = [name for name, _ in call_log]
        self.assertEqual(
            method_order[:3], ["xlayer_dispatch", "xlayer_poll", "xlayer_take_dispatch"]
        )
        self.assertIn(("xlayer_take_combine", {"slot_idx": 21}), call_log)
        self.assertIn(("xlayer_take_combine", {"slot_idx": 22}), call_log)


if __name__ == "__main__":
    unittest.main()
