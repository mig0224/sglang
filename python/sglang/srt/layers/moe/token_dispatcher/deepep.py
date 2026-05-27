from __future__ import annotations

import logging
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Deque,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    BaseDispatcherConfig,
    CombineInput,
    CombineInputFormat,
    DispatcherBaseHooks,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    get_deepep_config,
    get_moe_runner_backend,
    is_tbo_enabled,
)
from sglang.srt.utils import (
    get_bool_env_var,
    is_blackwell,
    is_hip,
    is_npu,
    load_json_config,
)

_is_npu = is_npu()

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import CombineOverlapArgs

try:
    from deep_ep import Buffer, Config
    from deep_ep.buffers.xlayers import XLayerScheduler

    if not _is_npu:
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

    use_deepep = True
except ImportError:
    use_deepep = False
    XLayerScheduler = None

import torch
import torch.distributed as dist

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()

logger = logging.getLogger(__name__)
_PARTNER_BITS_MASK = (1 << 64) - 1


def _deepep_precompile_tp_barrier() -> None:
    # DeepEP's all-to-all operation has a much shorter timeout compared to torch.distributed,
    # so if different ranks compile at different speeds, it may quickly trigger a timeout.
    # To avoid this, we use torch.distributed's barrier during the compile stage.
    # We apply this barrier only in the compile stage to prevent extra all-reduce overhead at runtime.
    if envs.SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE.get():
        get_tp_group().barrier()


class DeepEPPDispatchHooks(DispatcherBaseHooks):
    def __call__(self, dispatcher: BaseDispatcher):
        for hook_fun in self.hook_dict.values():
            hook_fun(dispatcher)


class DeepEPNormalDispatchOutput(NamedTuple):
    """DeepEP normal dispatch output."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: List[int]

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_NORMAL


class DeepEPLLDispatchOutput(NamedTuple):
    """DeepEP low latency dispatch output."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    masked_m: torch.Tensor
    expected_m: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_LL


assert isinstance(DeepEPNormalDispatchOutput, DispatchOutput)
assert isinstance(DeepEPLLDispatchOutput, DispatchOutput)


class DeepEPNormalCombineInput(NamedTuple):
    """DeepEP normal combine input."""

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_NORMAL


class DeepEPLLCombineInput(NamedTuple):
    """DeepEP low latency combine input."""

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_LL


assert isinstance(DeepEPNormalCombineInput, CombineInput)
assert isinstance(DeepEPLLCombineInput, CombineInput)


class DeepEPDispatchMode(IntEnum):
    NORMAL = auto()
    LOW_LATENCY = auto()


class DeepEPBuffer:
    _buffer = None
    _dispatch_mode: Optional[DeepEPDispatchMode] = None
    _hidden_size: Optional[int] = None
    _num_max_dispatch_tokens_per_rank: Optional[int] = None
    _num_experts: Optional[int] = None

    @classmethod
    def get_deepep_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        param_bytes: int,
        deepep_mode: DeepEPMode,
        num_max_dispatch_tokens_per_rank: int = -1,
        num_experts: int = -1,
    ):
        if cls._buffer is not None:
            return cls._buffer

        cls._hidden_size = hidden_size
        cls._num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        cls._num_experts = num_experts

        num_nvl_bytes, num_rdma_bytes = 0, 0
        if deepep_mode.enable_normal():
            hidden_bytes = hidden_size * param_bytes
            for config in (
                DeepEPConfig.get_instance().normal_dispatch_config
                or Buffer.get_dispatch_config(group.size()),
                DeepEPConfig.get_instance().normal_combine_config
                or Buffer.get_combine_config(group.size()),
            ):
                num_nvl_bytes = max(
                    config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
                    num_nvl_bytes,
                )
                num_rdma_bytes = max(
                    config.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
                    num_rdma_bytes,
                )
        if deepep_mode.enable_low_latency():
            assert num_max_dispatch_tokens_per_rank != -1
            assert num_experts != -1 and num_experts % group.size() == 0
            num_rdma_bytes = max(
                Buffer.get_low_latency_rdma_size_hint(
                    num_max_dispatch_tokens_per_rank,
                    hidden_size,
                    group.size(),
                    num_experts,
                ),
                num_rdma_bytes,
            )

        # We should calculate num_qps_per_rank consistently with DeepEP's test script logic:
        if deepep_mode == DeepEPMode.NORMAL:
            # refer: https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_internode.py#L235
            num_qps_per_rank = DeepEPConfig.get_instance().num_sms
        elif deepep_mode == DeepEPMode.LOW_LATENCY:
            # refer: https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_low_latency.py#L176
            num_qps_per_rank = num_experts // group.size()
        elif deepep_mode == DeepEPMode.AUTO:
            # low-latency and normal mode all need run
            # refer: https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_internode.py#L235
            num_qps_per_rank = max(
                DeepEPConfig.get_instance().num_sms, num_experts // group.size()
            )
        else:
            raise NotImplementedError

        if not _is_npu:
            total_num_sms = torch.cuda.get_device_properties(
                device="cuda"
            ).multi_processor_count
            if (
                (deepep_mode != DeepEPMode.LOW_LATENCY)
                and not is_tbo_enabled()
                and (DeepEPConfig.get_instance().num_sms < total_num_sms // 2)
            ):
                logger.warning(
                    f"Only use {DeepEPConfig.get_instance().num_sms} SMs for DeepEP communication. "
                    f"This may result in highly suboptimal performance. "
                    f"Consider using --deepep-config to change the behavior."
                )

        cls._buffer = Buffer(
            group,
            num_nvl_bytes,
            num_rdma_bytes,
            low_latency_mode=deepep_mode.enable_low_latency(),
            num_qps_per_rank=num_qps_per_rank,
            # TODO can be false when unneeded
            allow_mnnvl=True,
        )
        return cls._buffer

    @classmethod
    def clean_buffer(cls):
        if not cls._buffer.low_latency_mode:
            return
        cls._buffer.clean_low_latency_buffer(
            cls._num_max_dispatch_tokens_per_rank,
            cls._hidden_size,
            cls._num_experts,
        )

    @classmethod
    def set_dispatch_mode_as_normal(cls):
        cls._dispatch_mode = DeepEPDispatchMode.NORMAL

    @classmethod
    def set_dispatch_mode_as_low_latency(cls):
        if cls._dispatch_mode == DeepEPDispatchMode.NORMAL:
            cls.clean_buffer()
        cls._dispatch_mode = DeepEPDispatchMode.LOW_LATENCY

    @classmethod
    def set_dispatch_mode(cls, mode: DeepEPMode):
        if mode.is_low_latency():
            cls.set_dispatch_mode_as_low_latency()
        elif mode.is_normal():
            cls.set_dispatch_mode_as_normal()
        else:
            raise Exception("unsupported mode")


class DeepEPConfig(BaseDispatcherConfig):
    _instance = None

    def __init__(self):
        config_str = get_deepep_config()
        if config_str:
            config_parsed = load_json_config(config_str)
            if torch.distributed.get_rank() == 0:
                logger.info(f"Use DeepEP Config: {config_parsed}")
            config_dispatch = config_parsed["normal_dispatch"]
            config_combine = config_parsed["normal_combine"]

            self.normal_dispatch_config = Config(**config_dispatch)
            self.normal_combine_config = Config(**config_combine)

            assert config_dispatch["num_sms"] == config_combine["num_sms"]
            self.num_sms = config_dispatch["num_sms"]
        else:
            self.normal_dispatch_config = None
            self.normal_combine_config = None
            self.num_sms = Buffer.num_sms

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = DeepEPConfig()
        return cls._instance


class _DeepEPDispatcherImplBase:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        deepep_mode: DeepEPMode,
    ):
        if not use_deepep:
            raise ImportError(
                "DeepEP is not installed. Please install DeepEP package from "
                "https://github.com/deepseek-ai/deepep."
            )

        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode

        self.params_bytes = 2
        # A large value will lead to large memory occupation, thus users should change it accordingly
        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )
        # DeepEP internode_ll dispatch uses FINISHED_SUM_TAG=1024
        # and the logic requires num-tokens-sent-from-one-rank-to-another-rank less than it
        assert self.num_max_dispatch_tokens_per_rank <= 1024

        self.handle = None

        self.quant_config: Optional[dict] = None

        self.overlap_args: Optional[CombineOverlapArgs] = None
        self.meta_overlap_args: Optional[dict] = None

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        raise NotImplementedError

    def dispatch_b(self, *args, **kwargs):
        raise NotImplementedError

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        raise NotImplementedError

    def combine_b(self, *args, **kwargs):
        raise NotImplementedError

    def _get_buffer(self):
        raise NotImplementedError

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config

    def set_overlap_args(
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict
    ) -> None:
        self.overlap_args = combine_overlap_args
        self.meta_overlap_args = meta_overlap_args

    def clear_overlap_args(self) -> None:
        self.overlap_args = None
        self.meta_overlap_args = None


class _DeepEPDispatcherImplNormal(_DeepEPDispatcherImplBase):
    def __init__(self, async_finish: bool, **kwargs):
        super().__init__(**kwargs)

        self.async_finish = async_finish
        self.src2dst = None
        self.quant_config = {}

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        topk_ids = topk_ids.to(torch.int64)
        if (
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and not get_moe_runner_backend().is_cutlass()
            and not envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
        ):
            # TODO hard code 128 block quant,use fp8 communication
            hidden_states = sglang_per_token_group_quant_fp8(
                hidden_states,
                128,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
        previous_event = Buffer.capture() if self.async_finish else None
        return hidden_states, topk_ids, topk_weights, previous_event

    def dispatch_b(self, hidden_states, topk_ids, topk_weights, previous_event):
        (
            hidden_states,
            topk_ids,
            topk_weights,
            num_recv_tokens_per_expert,
            event,
        ) = self._dispatch_core(hidden_states, topk_ids, topk_weights, previous_event)
        event.current_stream_wait() if self.async_finish else ()

        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states
        else:
            hidden_states_scale = None

        return DeepEPNormalDispatchOutput(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            num_recv_tokens_per_expert,
        )

    def _dispatch_core(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        previous_event,
    ):
        buffer = self._get_buffer()
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(
            topk_ids,
            self.num_experts,
            previous_event=previous_event,
            async_finish=self.async_finish,
            allocate_on_comm_stream=previous_event is not None,
        )
        # FIXME: `handle` should be transmitted with tokens from dispatch to combine.
        # However, doing this would incur an unknown synchronization error, but keeping
        # `handle` as a member variable works.

        _deepep_precompile_tp_barrier()
        (
            recv_x,
            recv_topk_ids,
            recv_topk_weights,
            num_recv_tokens_per_expert,
            self.handle,
            event,
        ) = buffer.dispatch(
            x,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=self.async_finish,
            allocate_on_comm_stream=(previous_event is not None) and self.async_finish,
            expert_alignment=128 if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM else 1,
            config=DeepEPConfig.get_instance().normal_dispatch_config,
        )
        get_global_expert_distribution_recorder().on_deepep_dispatch_normal(
            num_recv_tokens_per_expert,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            num_tokens_per_expert=num_tokens_per_expert,
        )

        return (
            recv_x,
            recv_topk_ids,
            recv_topk_weights,
            num_recv_tokens_per_expert,
            event,
        )

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):

        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM or _use_aiter or _is_npu:
            output = hidden_states
        else:
            raise NotImplementedError()  # triton runner was supported but it's temporarily disabled

        previous_event = Buffer.capture() if self.async_finish else None
        return output, previous_event

    def combine_b(self, output, previous_event):
        hidden_states, event = self._combine_core(output, previous_event)
        event.current_stream_wait() if self.async_finish else ()
        self.handle = None
        self.src2dst = None
        return hidden_states

    def _combine_core(self, x: torch.Tensor, previous_event):
        buffer = self._get_buffer()
        _deepep_precompile_tp_barrier()
        combined_x, _, event = buffer.combine(
            x,
            self.handle,
            async_finish=self.async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=previous_event is not None,
            config=DeepEPConfig.get_instance().normal_combine_config,
        )
        return combined_x, event

    def _get_buffer(self):
        DeepEPBuffer.set_dispatch_mode_as_normal()

        return DeepEPBuffer.get_deepep_buffer(
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
        )


class _DeepEPDispatcherImplLowLatency(_DeepEPDispatcherImplBase):
    def __init__(self, return_recv_hook: bool, **kwargs):
        super().__init__(**kwargs)

        """
        num_max_dispatch_tokens_per_rank: the actual batch size in the decoding engine should be less than 256
        https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-inference-decoding
        """
        self.return_recv_hook = return_recv_hook
        self.device_module = torch.get_device_module()
        self.quant_config = {}

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        buffer = self._get_buffer()
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        topk_ids = topk_ids.to(torch.int64)
        expected_m = (
            hidden_states.shape[0] * buffer.group_size * topk_ids.shape[1]
            + self.num_experts
        ) // self.num_experts
        hidden_states, masked_m, event, hook = self._dispatch_core(
            hidden_states,
            topk_ids,
        )
        return (
            hidden_states,
            topk_ids,
            topk_weights,
            masked_m,
            expected_m,
            event,
            hook,
        )

    def dispatch_b(
        self,
        hidden_states,
        topk_ids,
        topk_weights,
        masked_m,
        expected_m,
        event,
        hook,
    ):
        hook() if self.return_recv_hook else event.current_stream_wait()

        get_global_expert_distribution_recorder().on_deepep_dispatch_low_latency(
            masked_m
        )

        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states
        else:
            hidden_states_scale = None

        deepep_output = DeepEPLLDispatchOutput(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            masked_m,
            expected_m,
        )
        return deepep_output

    def _dispatch_core(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        use_nvfp4 = use_fp8 = False
        input_global_scale = self.quant_config.get("input_global_scale", None)
        if input_global_scale is not None:
            use_nvfp4 = True
        elif not get_moe_runner_backend().is_flashinfer_cutedsl():
            # flashinfer_cutedsl expects BF16 dispatch when NVFP4 dispatch is
            # off; its kernel quantizes to NVFP4 internally.
            use_fp8 = True

        # round_scale / use_ue8m0 are FP8-DeepGEMM specific; they cause DeepEP
        # to return int32-packed UE8M0 scales that don't feed the flashinfer
        # cutedsl kernel.
        fp8_deepgemm_scale_opts = (
            dict(
                round_scale=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,
                use_ue8m0=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,
            )
            if use_fp8
            else dict()
        )

        buffer = self._get_buffer()
        _deepep_precompile_tp_barrier()
        packed_recv_hidden, self.packed_recv_count, self.handle, event, hook = (
            buffer.low_latency_dispatch(
                hidden_states,
                topk_ids,
                self.num_max_dispatch_tokens_per_rank,
                self.num_experts,
                use_fp8=use_fp8,
                **(dict(use_nvfp4=True) if use_nvfp4 else dict()),
                **(
                    dict(x_global_scale=input_global_scale)
                    if input_global_scale is not None
                    else dict()
                ),
                async_finish=not self.return_recv_hook,
                return_recv_hook=self.return_recv_hook,
                **fp8_deepgemm_scale_opts,
            )
        )
        return packed_recv_hidden, self.packed_recv_count, event, hook

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        hidden_states, event, hook = self._combine_core(
            hidden_states,
            topk_ids,
            topk_weights,
        )
        return hidden_states, event, hook

    def combine_b(self, hidden_states, event, hook):
        overlap_args = self.overlap_args
        if overlap_args is not None:
            overlap_args.stream.wait_stream(self.device_module.current_stream())

        hook() if self.return_recv_hook else event.current_stream_wait()

        if overlap_args is not None:
            self.device_module.current_stream().wait_stream(overlap_args.stream)

        return hidden_states

    def _combine_core(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        buffer = self._get_buffer()
        overlap_args = self.overlap_args
        meta_overlap_args = self.meta_overlap_args

        ctx = nullcontext()
        if overlap_args is not None:
            overlap_args.stream.wait_event(overlap_args.wait_event)
            ctx = torch.cuda.stream(overlap_args.stream)

            if is_blackwell():
                overlap_args_dict = dict(
                    overlap=overlap_args.overlap,
                    src_signals=overlap_args.signal,
                    src_signal_expect_value=overlap_args.threshold,
                )
            else:
                overlap_args_dict = dict(
                    overlap=overlap_args.overlap,
                    packed_recv_count=self.packed_recv_count,
                    comp_signal=overlap_args.signal,
                    block_m=meta_overlap_args["block_m"],
                    threshold=meta_overlap_args["threshold"],
                    num_sms=overlap_args.num_sms,
                )
        else:
            overlap_args_dict = {}

        with ctx:
            _deepep_precompile_tp_barrier()
            combined_hidden_states, event, hook = buffer.low_latency_combine(
                x=hidden_states,
                topk_idx=topk_ids,
                topk_weights=topk_weights,
                handle=self.handle,
                async_finish=not self.return_recv_hook,
                return_recv_hook=self.return_recv_hook,
                **overlap_args_dict,
            )

        self.packed_recv_count = self.handle = None
        return combined_hidden_states, event, hook

    def _get_buffer(self):
        DeepEPBuffer.set_dispatch_mode_as_low_latency()
        return DeepEPBuffer.get_deepep_buffer(
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
        )


class PartialAggregatorState(Enum):
    S_INIT = auto()
    S_DISPATCH_SUBMITTED = auto()
    S_AWAITING_PARTIALS = auto()
    S_AGGREGATE_COMPLETE = auto()
    S_RELEASED = auto()


@dataclass(order=True)
class ExpertSlotInfo:
    # Ordering key used by phase planning: layer first, then per-layer arrival/request.
    layer_id: int
    arrival_tick: int
    request_id: str
    # Exclude rank_id from ordering to keep cross-rank plan ordering lockstep-safe.
    rank_id: int = field(compare=False)


class PhaseScheduler:
    def __init__(self, k_d: int, k_c: int):
        self.k_d = max(1, int(k_d))
        self.k_c = max(1, int(k_c))
        self.phase_id = 0
        self.max_seen_layer_id = -1
        self._pending_dispatch: Dict[Tuple[str, int], Tuple[ExpertSlotInfo, dict]] = {}
        self._ffn_ready: Dict[Tuple[str, int], Tuple[ExpertSlotInfo, dict]] = {}
        self._dispatch_submitted: Deque[Tuple[str, int]] = deque()
        self._combine_submitted: Deque[Tuple[str, int]] = deque()
        self._dispatch_ready: Dict[Tuple[str, int], Any] = {}
        self._combine_ready: Dict[Tuple[str, int], List[Any]] = {}

    def register_layer(self, layer_id: int) -> int:
        self.max_seen_layer_id = max(self.max_seen_layer_id, int(layer_id))
        return self.num_max_inflight_pairs()

    def num_max_inflight_pairs(self) -> int:
        return max(1, (self.max_seen_layer_id + 1) * 2)

    def enqueue_dispatch(
        self, key: Tuple[str, int], slot_info: ExpertSlotInfo, payload: dict
    ) -> None:
        self._pending_dispatch[key] = (slot_info, payload)

    def enqueue_combine(
        self, key: Tuple[str, int], slot_info: ExpertSlotInfo, payload: dict
    ) -> None:
        self._ffn_ready[key] = (slot_info, payload)

    def plan_combine(self) -> List[Tuple[str, int]]:
        # C-micro priority follows design: (layer_id, request_id/rid, arrival_tick).
        plan = sorted(
            self._ffn_ready.keys(),
            key=lambda key: (
                self._ffn_ready[key][0].layer_id,
                self._ffn_ready[key][0].request_id,
                self._ffn_ready[key][0].arrival_tick,
            ),
        )
        return plan[: self.k_c]

    def plan_dispatch(self) -> List[Tuple[str, int]]:
        # D-micro priority follows design: (layer_id, arrival_tick, request_id/rid).
        plan = sorted(
            self._pending_dispatch.keys(),
            key=lambda key: (
                self._pending_dispatch[key][0].layer_id,
                self._pending_dispatch[key][0].arrival_tick,
                self._pending_dispatch[key][0].request_id,
            ),
        )
        return plan[: self.k_d]

    def pop_dispatch_payload(self, key: Tuple[str, int]) -> dict:
        _, payload = self._pending_dispatch.pop(key)
        self._dispatch_submitted.append(key)
        return payload

    def pop_combine_payload(self, key: Tuple[str, int]) -> dict:
        _, payload = self._ffn_ready.pop(key)
        self._combine_submitted.append(key)
        return payload

    def mark_dispatch_ready(self, ret: Any) -> Optional[Tuple[str, int]]:
        if not self._dispatch_submitted:
            return None
        key = self._dispatch_submitted.popleft()
        self._dispatch_ready[key] = ret
        return key

    def mark_combine_ready(self, ret: Any) -> Optional[Tuple[str, int]]:
        if not self._combine_submitted:
            return None
        key = self._combine_submitted.popleft()
        self._combine_ready.setdefault(key, []).append(ret)
        return key

    def take_dispatch_ready(self, key: Tuple[str, int]) -> Optional[Any]:
        return self._dispatch_ready.pop(key, None)

    def take_combine_ready(self, key: Tuple[str, int]) -> Optional[Any]:
        ready = self._combine_ready.get(key)
        if not ready:
            return None
        ret = ready.pop(0)
        if not ready:
            del self._combine_ready[key]
        return ret

    def advance_phase(self) -> None:
        self.phase_id += 1


@dataclass
class PartialAggregator:
    y_accum: torch.Tensor
    ep_handle: Any
    partner_expected_bits: int
    partner_received_bits: int = 0
    expected_partner_count: int = 0
    received_partner_count: int = 0
    state: PartialAggregatorState = PartialAggregatorState.S_INIT

    def __post_init__(self):
        self.partner_expected_bits &= _PARTNER_BITS_MASK
        self.expected_partner_count = int(self.partner_expected_bits.bit_count())

    def mark_dispatch_submitted(self) -> None:
        self.state = PartialAggregatorState.S_DISPATCH_SUBMITTED

    def add_partial(self, partner_rank: int, partial_output: torch.Tensor) -> bool:
        self.y_accum.add_(partial_output)
        self.state = PartialAggregatorState.S_AWAITING_PARTIALS
        if 0 <= partner_rank < 64:
            partner_bit = 1 << partner_rank
            if (self.partner_expected_bits & partner_bit) != 0 and (
                self.partner_received_bits & partner_bit
            ) == 0:
                self.partner_received_bits |= partner_bit
                self.received_partner_count += 1
        if self.received_partner_count >= self.expected_partner_count:
            self.state = PartialAggregatorState.S_AGGREGATE_COMPLETE
            return True
        return False

    def add_partial_bits(
        self, active_partner_bits: int, partial_output: torch.Tensor
    ) -> bool:
        self.y_accum.add_(partial_output)
        self.state = PartialAggregatorState.S_AWAITING_PARTIALS
        masked_bits = int(active_partner_bits) & _PARTNER_BITS_MASK
        newly_received_bits = (
            masked_bits & self.partner_expected_bits & (~self.partner_received_bits)
        )
        if newly_received_bits:
            self.partner_received_bits |= newly_received_bits
            self.received_partner_count += newly_received_bits.bit_count()
        if self.received_partner_count >= self.expected_partner_count:
            self.state = PartialAggregatorState.S_AGGREGATE_COMPLETE
            return True
        return False

    def release(self) -> None:
        self.state = PartialAggregatorState.S_RELEASED


class _DeepEPDispatcherImplXLayer(_DeepEPDispatcherImplNormal):
    _warned_fallback = False
    _shared_phase_scheduler: Optional[PhaseScheduler] = None

    def __init__(self, layer_id: int, **kwargs):
        super().__init__(**kwargs)
        self.layer_id = layer_id
        self._arrival_tick = 0
        self._allow_inline_phase_driving = False
        self._last_request_id: Optional[str] = None
        self._aggregators: Dict[Tuple[str, int], PartialAggregator] = {}
        self._expert_slot_infos: Dict[Tuple[str, int], ExpertSlotInfo] = {}
        self._rank = dist.get_rank(group=self.group)
        self._buffer = self._get_buffer()
        self._phase_state = self._get_phase_scheduler()
        self._scheduler = self._init_scheduler()

    def _get_phase_scheduler(self) -> PhaseScheduler:
        if self.__class__._shared_phase_scheduler is None:
            world_size = dist.get_world_size(group=self.group)
            self.__class__._shared_phase_scheduler = PhaseScheduler(
                k_d=world_size, k_c=world_size
            )
        phase_state = self.__class__._shared_phase_scheduler
        phase_state.register_layer(self.layer_id)
        return phase_state

    def _init_scheduler(self):
        if XLayerScheduler is None:
            return None
        try:
            scheduler = XLayerScheduler(self._buffer)
            self._configure_xlayer_scheduler(scheduler)
            return scheduler
        except Exception as e:
            logger.warning(
                "Failed to initialize XLayerScheduler (XLayer path will fall back to DeepEP): %s",
                e,
            )
            return None

    def _configure_xlayer_scheduler(self, scheduler: Any) -> None:
        xlayer_set_config = getattr(scheduler, "xlayer_set_config", None)
        if xlayer_set_config is None:
            return
        config_kwargs = {
            # Keep enough inflight room for at least one dispatch/combine pair per rank
            # even before all MoE layers have registered with the shared phase state.
            "num_max_inflight_pairs": max(
                self._phase_state.num_max_inflight_pairs(),
                dist.get_world_size(group=self.group) * 2,
            ),
            "num_max_tokens_per_rank": self.num_max_dispatch_tokens_per_rank,
            "num_experts": self.num_experts,
            "num_topk": self.router_topk,
            "expert_alignment": (
                128 if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM else 1
            ),
        }
        try:
            xlayer_set_config(**config_kwargs)
        except TypeError:
            xlayer_set_config(
                config_kwargs["num_max_inflight_pairs"],
                config_kwargs["num_max_tokens_per_rank"],
                config_kwargs["num_experts"],
                config_kwargs["num_topk"],
                config_kwargs["expert_alignment"],
            )

    def _build_xlayer_dispatch_payload(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        request_id: str,
    ) -> dict:
        return {
            "x": x,
            "topk_idx": topk_ids,
            "topk_weights": topk_weights,
            "request_id": request_id,
            "layer_id": self.layer_id,
            "num_sms": DeepEPConfig.get_instance().num_sms,
            "do_cpu_sync": False,
        }

    def _build_xlayer_combine_payload(self, x: torch.Tensor) -> dict:
        return {
            "x": x,
            "handle": self.handle,
            "num_sms": DeepEPConfig.get_instance().num_sms,
        }

    def _inline_phase_driving_enabled(self) -> bool:
        return self._allow_inline_phase_driving

    def _warn_and_fallback(self, exc: Exception):
        if not self._warned_fallback:
            logger.warning(
                "Falling back to DeepEP normal dispatcher because the XLayer path is unavailable: %s",
                exc,
            )
            self.__class__._warned_fallback = True

    def _call_xlayer(self, method_name: str, **kwargs):
        if self._scheduler is None:
            raise RuntimeError("XLayer scheduler is not initialized")
        method = getattr(self._scheduler, method_name)
        try:
            return method(**kwargs)
        except TypeError:
            if method_name == "xlayer_poll":
                return method()
            if method_name in ("xlayer_take_dispatch", "xlayer_take_combine"):
                return method(kwargs["slot_idx"])
            if method_name == "xlayer_dispatch":
                args = [
                    kwargs["x"],
                    kwargs["topk_idx"],
                    kwargs["topk_weights"],
                    kwargs["request_id"],
                    kwargs["layer_id"],
                ]
                if "num_sms" in kwargs:
                    args.append(kwargs["num_sms"])
                if "do_cpu_sync" in kwargs:
                    args.append(kwargs["do_cpu_sync"])
                return method(*args)
            if method_name == "xlayer_combine":
                args = [kwargs["x"], kwargs["handle"]]
                if "num_sms" in kwargs:
                    args.append(kwargs["num_sms"])
                return method(*args)
            raise

    def _unpack_dispatch_ret(self, ret):
        if isinstance(ret, tuple):
            if len(ret) == 6:
                return ret
            if len(ret) == 5:
                (
                    recv_x,
                    recv_topk_ids,
                    recv_topk_weights,
                    num_recv_tokens_per_expert,
                    event,
                ) = ret
                return (
                    recv_x,
                    recv_topk_ids,
                    recv_topk_weights,
                    num_recv_tokens_per_expert,
                    self.handle,
                    event,
                )
        return (
            ret.recv_x,
            ret.recv_topk_ids,
            ret.recv_topk_weights,
            ret.num_recv_tokens_per_expert,
            ret.ep_handle,
            ret.event,
        )

    def _unpack_combine_ret(self, ret):
        if isinstance(ret, tuple):
            if len(ret) >= 3:
                a, b, c = ret[0], ret[1], ret[2]
                if isinstance(b, int):
                    return a, c, int(b)
                if isinstance(c, int):
                    return a, b, int(c)
                return a, b, None
            if len(ret) >= 2:
                return ret[0], ret[1], None
            return ret[0], None, None
        return ret.combined_x, ret.event, getattr(ret, "active_partner_bits", None)

    def _normalize_ready_slots(self, ready_slots: Any) -> List[int]:
        if ready_slots is None:
            return []
        if isinstance(ready_slots, int):
            return [int(ready_slots)]
        return [int(slot_idx) for slot_idx in ready_slots]

    def _run_micro_phase_driver(self) -> None:
        self._call_xlayer("enter_phase")

        for key in self._phase_state.plan_combine():
            self._call_xlayer(
                "xlayer_combine", **self._phase_state.pop_combine_payload(key)
            )

        for key in self._phase_state.plan_dispatch():
            self._call_xlayer(
                "xlayer_dispatch", **self._phase_state.pop_dispatch_payload(key)
            )

        dispatch_slots = self._normalize_ready_slots(
            self._call_xlayer("xlayer_poll", kind="dispatch")
        )
        for slot_idx in dispatch_slots:
            ret = self._call_xlayer("xlayer_take_dispatch", slot_idx=slot_idx)
            self._phase_state.mark_dispatch_ready(ret)

        combine_slots = self._normalize_ready_slots(
            self._call_xlayer("xlayer_poll", kind="combine")
        )
        for slot_idx in combine_slots:
            ret = self._call_xlayer("xlayer_take_combine", slot_idx=slot_idx)
            self._phase_state.mark_combine_ready(ret)

        self._phase_state.advance_phase()

    def _dispatch_core(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        previous_event,
    ):
        arrival_tick = self._arrival_tick
        # Keep request_id rank-independent to preserve cross-rank lockstep semantics.
        request_id = f"L{self.layer_id}:T{arrival_tick}"
        self._arrival_tick += 1
        self._last_request_id = request_id
        key = (request_id, self.layer_id)
        slot_info = ExpertSlotInfo(
            layer_id=self.layer_id,
            arrival_tick=arrival_tick,
            request_id=request_id,
            rank_id=self._rank,
        )
        self._expert_slot_infos[key] = slot_info
        try:
            if not self._inline_phase_driving_enabled():
                raise RuntimeError(
                    "XLayer inline micro-phase driving is disabled by default until model-runner integration enables it"
                )
            self._phase_state.enqueue_dispatch(
                key=key,
                slot_info=slot_info,
                payload=self._build_xlayer_dispatch_payload(
                    x=x,
                    topk_ids=topk_ids,
                    topk_weights=topk_weights,
                    request_id=request_id,
                ),
            )
            self._run_micro_phase_driver()
            ret = self._phase_state.take_dispatch_ready(key)
            if ret is None:
                raise RuntimeError(
                    f"XLayer dispatch did not produce ready output for key={key} in the first inline micro-phase"
                )
            (
                recv_x,
                recv_topk_ids,
                recv_topk_weights,
                num_recv_tokens_per_expert,
                handle,
                event,
            ) = self._unpack_dispatch_ret(ret)
            self.handle = handle
            return (
                recv_x,
                recv_topk_ids,
                recv_topk_weights,
                num_recv_tokens_per_expert,
                event,
            )
        except Exception as e:
            if get_bool_env_var("SGLANG_XLAYER_STRICT"):
                raise
            self._warn_and_fallback(e)
            return super()._dispatch_core(x, topk_ids, topk_weights, previous_event)

    def _combine_core(self, x: torch.Tensor, previous_event):
        if self._last_request_id is None:
            return super()._combine_core(x, previous_event)
        key = (self._last_request_id, self.layer_id)
        try:
            if not self._inline_phase_driving_enabled():
                raise RuntimeError(
                    "XLayer inline micro-phase driving is disabled by default until model-runner integration enables it"
                )
            partner_expected_bits = self._call_xlayer(
                "involved_rank_bitmask_for",
                request_id=self._last_request_id,
                layer_id=self.layer_id,
            )
            aggregator = self._aggregators.get(key)
            if aggregator is None:
                aggregator = PartialAggregator(
                    y_accum=torch.zeros_like(x),
                    ep_handle=self.handle,
                    partner_expected_bits=int(partner_expected_bits),
                )
                aggregator.mark_dispatch_submitted()
                self._aggregators[key] = aggregator

            event = None
            expected_combine_submissions = max(1, aggregator.expected_partner_count)
            is_complete = False
            for _ in range(expected_combine_submissions):
                self._phase_state.enqueue_combine(
                    key=key,
                    slot_info=self._expert_slot_infos[key],
                    payload=self._build_xlayer_combine_payload(x),
                )
                self._run_micro_phase_driver()
                ret = self._phase_state.take_combine_ready(key)
                if ret is None:
                    raise RuntimeError(
                        f"XLayer combine did not produce partial output for key={key} in the first inline micro-phase"
                    )
                partial, event, active_partner_bits = self._unpack_combine_ret(ret)
                is_complete = aggregator.add_partial_bits(
                    (
                        int(partner_expected_bits)
                        if active_partner_bits is None
                        else int(active_partner_bits)
                    ),
                    partial,
                )
                if is_complete:
                    break
            if not is_complete:
                raise RuntimeError(
                    f"XLayer combine did not reach completion for key={key} within {expected_combine_submissions} combine submissions"
                )

            combined = aggregator.y_accum
            aggregator.release()
            del self._aggregators[key]
            return combined, event
        except Exception as e:
            if get_bool_env_var("SGLANG_XLAYER_STRICT"):
                raise
            self._warn_and_fallback(e)
            if key in self._aggregators:
                del self._aggregators[key]
            return super()._combine_core(x, previous_event)


@dataclass
class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


class DeepEPDispatcher(BaseDispatcher):
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        super().__init__()

        self.deepep_mode = deepep_mode

        common_kwargs = dict(
            group=group,
            router_topk=router_topk,
            permute_fusion=permute_fusion,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=deepep_mode,
        )

        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher = _DeepEPDispatcherImplLowLatency(
                return_recv_hook=return_recv_hook,
                **common_kwargs,
            )
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher = _DeepEPDispatcherImplNormal(
                async_finish=async_finish,
                **common_kwargs,
            )

        self._stage = _Stage.INITIAL
        self._deepep_dispatch_hooks = DeepEPPDispatchHooks()

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> DispatchOutput:
        self.dispatch_a(hidden_states, topk_output)
        if self._deepep_dispatch_hooks is not None:
            self._deepep_dispatch_hooks(self)
        ret = self.dispatch_b()
        return ret

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        inner_state = self._get_impl().dispatch_a(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        self._dispatch_intermediate_state = inner_state

    def dispatch_b(self):
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
        inner_state = self._dispatch_intermediate_state
        del self._dispatch_intermediate_state
        return self._get_impl().dispatch_b(*inner_state)

    def combine(
        self,
        combine_input: CombineInput,
    ) -> torch.Tensor:
        self.combine_a(combine_input)
        ret = self.combine_b()
        return ret

    def combine_a(
        self,
        combine_input: CombineInput,
    ):
        hidden_states, topk_ids, topk_weights = combine_input
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        inner_state = self._get_impl().combine_a(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        self._combine_intermediate_state = inner_state

    def combine_b(self):
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        inner_state = self._combine_intermediate_state
        del self._combine_intermediate_state
        return self._get_impl().combine_b(*inner_state)

    def _get_impl(self) -> _DeepEPDispatcherImplBase:
        is_extend_in_batch = get_is_extend_in_batch()
        resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)
        if resolved_deepep_mode == DeepEPMode.NORMAL:
            return self._normal_dispatcher
        elif resolved_deepep_mode == DeepEPMode.LOW_LATENCY:
            return self._low_latency_dispatcher
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")

    def _update_stage(self, old_stage, new_stage):
        assert self._stage == old_stage
        self._stage = new_stage

    def set_quant_config(self, quant_config: dict):
        super().set_quant_config(quant_config)
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.set_quant_config(quant_config)
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.set_quant_config(quant_config)

    def set_overlap_args(
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict
    ):
        super().set_overlap_args(combine_overlap_args, meta_overlap_args)
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )

    def clear_overlap_args(self):
        super().clear_overlap_args()
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.clear_overlap_args()
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.clear_overlap_args()

    def register_deepep_dispatch_hook(self, hook):
        return self._deepep_dispatch_hooks.register_hook(hook)


class XLayerDeepEPDispatcher(DeepEPDispatcher):
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.NORMAL,
        async_finish: bool = False,
        return_recv_hook: bool = False,
        layer_id: int = 0,
    ):
        BaseDispatcher.__init__(self)
        del return_recv_hook
        if deepep_mode.enable_low_latency():
            raise ValueError(
                "XLayerDeepEPDispatcher currently supports DeepEP normal mode only."
            )
        self.deepep_mode = deepep_mode
        self._normal_dispatcher = _DeepEPDispatcherImplXLayer(
            layer_id=layer_id,
            async_finish=async_finish,
            group=group,
            router_topk=router_topk,
            permute_fusion=permute_fusion,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=deepep_mode,
        )
        self._stage = _Stage.INITIAL
        self._deepep_dispatch_hooks = DeepEPPDispatchHooks()
