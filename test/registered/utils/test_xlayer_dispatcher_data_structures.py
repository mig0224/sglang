import unittest
from unittest import mock

import torch

import sglang.srt.layers.moe.token_dispatcher.deepep as deepep_mod
import sglang.srt.layers.moe.utils as moe_utils
from sglang.srt.layers.moe.token_dispatcher.deepep import (
    ExpertSlotInfo,
    PartialAggregator,
    PartialAggregatorState,
    PhaseScheduler,
    _DeepEPDispatcherImplXLayer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")


class TestXLayerDispatcherDataStructures(CustomTestCase):
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

    def test_phase_scheduler_deterministic_plans(self):
        scheduler = PhaseScheduler(k_d=2, k_c=2)
        scheduler.enqueue_dispatch(
            ("r2", 2),
            ExpertSlotInfo(layer_id=2, arrival_tick=9, request_id="r2", rank_id=1),
            payload={"name": "d2"},
        )
        scheduler.enqueue_dispatch(
            ("r1", 1),
            ExpertSlotInfo(layer_id=1, arrival_tick=10, request_id="r1", rank_id=0),
            payload={"name": "d1"},
        )
        scheduler.enqueue_dispatch(
            ("r3", 2),
            ExpertSlotInfo(layer_id=2, arrival_tick=3, request_id="r3", rank_id=3),
            payload={"name": "d3"},
        )
        scheduler.enqueue_combine(
            ("c2", 2),
            ExpertSlotInfo(layer_id=2, arrival_tick=1, request_id="c2", rank_id=3),
            payload={"name": "c2"},
        )
        scheduler.enqueue_combine(
            ("c1", 1),
            ExpertSlotInfo(layer_id=1, arrival_tick=9, request_id="c1", rank_id=2),
            payload={"name": "c1"},
        )
        scheduler.enqueue_combine(
            ("c0", 1),
            ExpertSlotInfo(layer_id=1, arrival_tick=1, request_id="c0", rank_id=7),
            payload={"name": "c0"},
        )

        self.assertEqual(scheduler.plan_dispatch(), [("r1", 1), ("r3", 2)])
        self.assertEqual(scheduler.plan_combine(), [("c0", 1), ("c1", 1)])

    def test_xlayer_impl_uses_slot_poll_take_sequence(self):
        impl = _DeepEPDispatcherImplXLayer.__new__(_DeepEPDispatcherImplXLayer)
        impl.layer_id = 4
        impl._arrival_tick = 0
        impl._allow_inline_phase_driving = True
        impl._last_request_id = None
        impl._aggregators = {}
        impl._expert_slot_infos = {}
        impl._rank = 0
        impl.async_finish = True
        impl.handle = object()
        impl._phase_state = PhaseScheduler(k_d=8, k_c=8)

        def _raise_unexpected_fallback(exc):
            raise AssertionError(f"Unexpected fallback: {exc}")

        impl._warn_and_fallback = _raise_unexpected_fallback

        call_log = []
        dispatch_slot = 17
        poll_schedule = {
            "dispatch": [[dispatch_slot], [], []],
            "combine": [[], [21], [22]],
        }
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
                self.assertEqual(
                    set(kwargs),
                    {
                        "x",
                        "topk_idx",
                        "topk_weights",
                        "request_id",
                        "layer_id",
                        "num_sms",
                        "do_cpu_sync",
                    },
                )
                self.assertTrue(torch.equal(kwargs["x"], x))
                self.assertTrue(torch.equal(kwargs["topk_idx"], topk_ids))
                self.assertTrue(torch.equal(kwargs["topk_weights"], topk_weights))
                return None
            if method_name == "enter_phase":
                return None
            if method_name == "xlayer_poll":
                kind = kwargs["kind"]
                return poll_schedule[kind].pop(0)
            if method_name == "xlayer_take_dispatch":
                self.assertEqual(kwargs["slot_idx"], dispatch_slot)
                return dispatch_ret
            if method_name == "involved_rank_bitmask_for":
                self.assertEqual(kwargs["request_id"], impl._last_request_id)
                self.assertEqual(kwargs["layer_id"], impl.layer_id)
                return expected_bits
            if method_name == "xlayer_combine":
                self.assertEqual(set(kwargs), {"x", "handle", "num_sms"})
                self.assertEqual(kwargs["handle"], impl.handle)
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
            {
                "normal_dispatch_config": None,
                "normal_combine_config": None,
                "num_sms": 0,
            },
        )()

        with mock.patch.object(
            deepep_mod.DeepEPConfig, "get_instance", return_value=dummy_cfg
        ):
            impl._dispatch_core(x, topk_ids, topk_weights, previous_event=None)
            key = (impl._last_request_id, impl.layer_id)
            combined, _ = impl._combine_core(x, previous_event=None)
            self.assertTrue(torch.equal(combined, partial1 + partial2))
            self.assertNotIn(key, impl._aggregators)

        method_order = [name for name, _ in call_log]
        self.assertIn("enter_phase", method_order)
        self.assertIn("xlayer_dispatch", method_order)
        self.assertIn(("xlayer_take_combine", {"slot_idx": 21}), call_log)
        self.assertIn(("xlayer_take_combine", {"slot_idx": 22}), call_log)

    def test_configure_xlayer_scheduler_uses_real_signature(self):
        impl = _DeepEPDispatcherImplXLayer.__new__(_DeepEPDispatcherImplXLayer)
        impl.group = object()
        impl.router_topk = 2
        impl.num_experts = 64
        impl.num_max_dispatch_tokens_per_rank = 256
        impl._phase_state = PhaseScheduler(k_d=8, k_c=8)
        impl._phase_state.register_layer(0)

        scheduler = mock.Mock()

        with (
            mock.patch.object(deepep_mod.dist, "get_world_size", return_value=8),
            mock.patch.object(
                deepep_mod.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", False
            ),
        ):
            impl._configure_xlayer_scheduler(scheduler)

        scheduler.xlayer_set_config.assert_called_once_with(
            num_max_inflight_pairs=16,
            num_max_tokens_per_rank=256,
            num_experts=64,
            num_topk=2,
            expert_alignment=1,
        )

    def test_call_xlayer_positional_fallback_preserves_real_api_args(self):
        impl = _DeepEPDispatcherImplXLayer.__new__(_DeepEPDispatcherImplXLayer)

        class PositionalOnlyScheduler:
            def xlayer_dispatch(
                self,
                x,
                topk_idx,
                topk_weights,
                request_id,
                layer_id,
                num_sms=0,
                do_cpu_sync=False,
                /,
            ):
                return (
                    x,
                    topk_idx,
                    topk_weights,
                    request_id,
                    layer_id,
                    num_sms,
                    do_cpu_sync,
                )

            def xlayer_combine(self, x, handle, num_sms=0, /):
                return x, handle, num_sms

        impl._scheduler = PositionalOnlyScheduler()

        dispatch_ret = impl._call_xlayer(
            "xlayer_dispatch",
            x="x",
            topk_idx="ids",
            topk_weights="weights",
            request_id="req-1",
            layer_id=3,
            num_sms=7,
            do_cpu_sync=True,
        )
        combine_ret = impl._call_xlayer(
            "xlayer_combine",
            x="y",
            handle="handle-1",
            num_sms=5,
        )

        self.assertEqual(dispatch_ret, ("x", "ids", "weights", "req-1", 3, 7, True))
        self.assertEqual(combine_ret, ("y", "handle-1", 5))


if __name__ == "__main__":
    unittest.main()
