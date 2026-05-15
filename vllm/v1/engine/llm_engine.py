# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections.abc import Callable, Mapping
from copy import copy
from typing import TYPE_CHECKING, Any

from vllm.v1.dag.session import DAGSession

if TYPE_CHECKING:
    from vllm.v1.dag.topology import DAGTopology

import torch.nn as nn
from typing_extensions import TypeVar

import vllm.envs as envs
from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.distributed.parallel_state import get_dp_group
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import EngineInput, PromptType
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.renderers import renderer_from_config
from vllm.renderers.inputs.preprocess import extract_prompt_components
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.tokenizers import TokenizerLike
from vllm.tracing import init_tracer
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest, PauseMode
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.executor import Executor
from vllm.v1.metrics.loggers import StatLoggerFactory, StatLoggerManager
from vllm.v1.metrics.reader import Metric, get_metrics_snapshot
from vllm.v1.metrics.stats import IterationStats
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)

_R = TypeVar("_R", default=Any)


class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        aggregate_engine_logging: bool = False,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        multiprocess_mode: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.observability_config = vllm_config.observability_config

        tracing_endpoint = self.observability_config.otlp_traces_endpoint
        if tracing_endpoint is not None:
            init_tracer("vllm.llm_engine", tracing_endpoint)

        self.log_stats = log_stats

        parallel_config = vllm_config.parallel_config
        executor_backend = parallel_config.distributed_executor_backend

        self.external_launcher_dp = (
            parallel_config.data_parallel_size > 1
            and executor_backend == "external_launcher"
        )
        # important: init dp group before init the engine_core
        # In the decoupled engine case this is handled in EngineCoreProc.
        if (
            not multiprocess_mode
            and parallel_config.data_parallel_size > 1
            and not self.external_launcher_dp
        ):
            self.dp_group = parallel_config.stateless_init_dp_group()
        else:
            self.dp_group = None
        self.should_execute_dummy_batch = False

        self.renderer = renderer = renderer_from_config(self.vllm_config)

        # Convert EngineInput --> EngineCoreRequest.
        self.input_processor = InputProcessor(self.vllm_config, renderer)

        # Converts EngineCoreOutputs --> RequestOutput.
        self.output_processor = OutputProcessor(
            renderer.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
            tracing_enabled=tracing_endpoint is not None,
        )

        # EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
        )

        self.logger_manager: StatLoggerManager | None = None
        if self.log_stats:
            self.logger_manager = StatLoggerManager(
                vllm_config=vllm_config,
                custom_stat_loggers=stat_loggers,
                enable_default_loggers=log_stats,
                aggregate_engine_logging=aggregate_engine_logging,
            )
            self.logger_manager.log_engine_initialized()

        if not multiprocess_mode:
            # for v0 compatibility
            self.model_executor = self.engine_core.engine_core.model_executor  # type: ignore

        if self.external_launcher_dp:
            # If we use DP in external launcher mode, we reuse the
            # existing DP group used for data communication.
            self.dp_group = get_dp_group().cpu_group

        # Don't keep the dummy data in memory
        self.reset_mm_cache()

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        disable_log_stats: bool = False,
    ) -> "LLMEngine":
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            log_stats=(not disable_log_stats),
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=envs.VLLM_ENABLE_V1_MULTIPROCESSING,
        )

    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_multiprocessing: bool = False,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.debug("Enabling multiprocessing for LLMEngine.")
            enable_multiprocessing = True

        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=enable_multiprocessing,
        )

    def generate_dag(
        self,
        dag: "DAGTopology",
        sampling_params: SamplingParams,
        return_timing: bool = False,
        include_parent_context: bool = True,
    ) -> "dict[str, RequestOutput] | tuple[dict[str, RequestOutput], dict[str, float]]":
        """Generate outputs for all nodes in a DAG-structured agent workflow.

        Nodes are processed in topological order.  Parallel branches receive the
        same RoPE start offset, enabling zero-prefill KV-cache reuse at merge
        points via DAG-RoPE position assignment.

        Args:
            dag: The DAG topology.
            sampling_params: Applied to every node.
            return_timing: When True, also return per-node TTFT (ms) as a second
                value.  TTFT = wall-clock from add_request() to first token in step().
            include_parent_context: When True (default), prepend each parent's
                output text to the child's prompt so the model has conversational
                context even without full KV-inheritance.  Set to False for pure
                TTFT measurement: each node only prefills its own short prompt,
                matching the zero-prefill semantics DAG-RoPE provides via the
                block table.

        Returns:
            results dict mapping node_id → RequestOutput.
            If return_timing=True, also returns {node_id: ttft_ms}.
        """
        import copy as _copy

        from vllm.v1.dag import DAGSession

        session = DAGSession(dag)
        results: dict[str, RequestOutput] = {}
        per_node_ttft_ms: dict[str, float] = {}
        node_text_outputs: dict[str, str] = {}

        _scheduler = None
        try:
            _scheduler = self.engine_core.engine_core.scheduler  # type: ignore[attr-defined]
        except AttributeError:
            pass

        # Pre-tokenize each node's own prompt to compute tail-alignment deltas.
        tokenizer = self.renderer.tokenizer
        node_own_token_counts: dict[str, int] = {}
        for nid in dag.topo_sort():
            raw_prompt = dag.get_node(nid).prompt
            try:
                node_own_token_counts[nid] = len(
                    tokenizer.encode(raw_prompt, add_special_tokens=False)
                )
            except Exception:
                node_own_token_counts[nid] = len(raw_prompt.split())

        # For each branch node that feeds a merge node, compute the tail delta
        # (positive shift so shorter branches align their last token with the
        # longest branch).  Only used when include_parent_context=False.
        tail_delta: dict[str, int] = {}
        if not include_parent_context:
            for nid in dag.topo_sort():
                node = dag.get_node(nid)
                if len(node.parents) > 1:
                    # nid is a merge node; compute per-parent expected lengths.
                    expected: dict[str, int] = {
                        pid: node_own_token_counts[pid] + sampling_params.max_tokens
                        for pid in node.parents
                    }
                    L_max = max(expected.values())
                    for pid, L in expected.items():
                        delta = L_max - L
                        # Keep the larger delta if a node feeds multiple merges.
                        if delta > tail_delta.get(pid, 0):
                            tail_delta[pid] = delta

        # Per-node pinned block IDs (pure DAG-RoPE mode only).
        # Keeps ancestor KV blocks alive until a downstream merge node finishes.
        pinned_by_node: dict[str, list[int]] = {}
        # Track each node's own (unformatted) prompt for conversation history.
        node_own_prompts: dict[str, str] = {}

        # Helper: format a message list with the model's chat template.
        # Falls back to newline-joined content if the tokenizer lacks one.
        def _apply_template(messages: list[dict]) -> str:
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                return "\n".join(m["content"] for m in messages if m.get("content"))

        for node_id in dag.topo_sort():
            node = dag.get_node(node_id)
            node_own_prompts[node_id] = node.prompt
            position_offset = session.compute_offset(node_id) + tail_delta.get(
                node_id, 0
            )

            # Build effective prompt using proper chat-template roles.
            #
            # include_parent_context=True  → quality mode: build a multi-turn
            #   conversation that lists all ancestors in topological order,
            #   alternating user (node prompt) / assistant (node output) turns.
            #   This correctly marks LLM outputs as assistant messages rather
            #   than smuggling them into user turns.
            #
            # include_parent_context=False → pure DAG-RoPE mode: each node
            #   sees only its own prompt; ancestor context is provided through
            #   the inherited KV-cache block table, not repeated as text.
            if include_parent_context and node.parents:
                ancestors_ordered = [
                    nid
                    for nid in dag.topo_sort()
                    if nid in session._ancestors_cache[node_id]
                ]
                messages: list[dict] = []
                for anc_id in ancestors_ordered:
                    if anc_id in node_own_prompts:
                        messages.append(
                            {"role": "user", "content": node_own_prompts[anc_id]}
                        )
                    if anc_id in node_text_outputs:
                        messages.append(
                            {"role": "assistant", "content": node_text_outputs[anc_id]}
                        )
                messages.append({"role": "user", "content": node.prompt})
                effective_prompt = _apply_template(messages)
            else:
                effective_prompt = _apply_template(
                    [{"role": "user", "content": node.prompt}]
                )

            node_params = _copy.copy(sampling_params)
            extra = dict(node_params.extra_args) if node_params.extra_args else {}
            extra["dag_position_offset"] = position_offset

            # Prevent partial prefix-cache hits for nodes with a non-zero DAG
            # position offset.
            #
            # Problem: with include_parent_context=True the effective prompt for
            # branch B starts with the root A's formatted content, so vLLM finds
            # a partial prefix-cache hit (N_A tokens, cached from A's run at
            # positions [0, N_A-1]).  B's remaining tokens are then prefilled at
            # positions N_A + q + dag_offset.  During that prefill B attends to
            # the cached A-KV, whose keys encode positions [0, N_A-1] — but
            # B's queries expect those keys at positions [dag_offset, dag_offset
            # + N_A-1].  The relative-distance mismatch makes attention
            # incoherent → the first-ever run of B/C/D produces "!!!".
            #
            # Fix: force a full re-prefill for every node that has a non-zero
            # offset.  The correct KV (computed at the right DAG positions) is
            # then cached and can be safely reused on future identical calls.
            if position_offset != 0:
                node_params.skip_reading_prefix_cache = True

            # For merge nodes in pure DAG-RoPE mode, pass inherited block IDs.
            # NOTE: we pass ONLY the block IDs, NOT a "num_inherited_tokens"
            # count.  The scheduler will prepend these blocks to D's block
            # table after normal slot allocation.  D's num_computed_tokens
            # stays 0, so D re-prefills its own prompt in full — the inherited
            # blocks merely extend the block table so D can attend to ancestor
            # KV during that prefill.
            if (
                not include_parent_context
                and len(node.parents) > 1
                and _scheduler is not None
            ):
                inherited = session.get_inherited_block_ids(node_id)
                if inherited and inherited[0]:
                    extra["dag_inherited_block_ids"] = inherited[0]

            node_params.extra_args = extra

            req_id = f"dag__{node_id}"
            t_submit = time.time()
            self.add_request(
                request_id=req_id,
                prompt=effective_prompt,
                params=node_params,
            )

            first_token_time: float | None = None
            node_output: RequestOutput | None = None
            while node_output is None:
                step_outputs = self.step()
                for out in step_outputs:
                    if out.request_id == req_id:
                        if (
                            first_token_time is None
                            and out.outputs
                            and out.outputs[0].token_ids
                        ):
                            first_token_time = time.time()
                        if out.finished:
                            node_output = out
                            break

            ttft_ms = (
                (first_token_time - t_submit) * 1000.0
                if first_token_time is not None
                else 0.0
            )
            per_node_ttft_ms[node_id] = ttft_ms
            results[node_id] = node_output
            node_text_outputs[node_id] = node_output.outputs[0].text

            block_ids_for_session: list[list[int]] = [[]]
            if _scheduler is not None:
                try:
                    raw = _scheduler.kv_cache_manager.get_block_ids(req_id)
                    block_ids_for_session = [list(g) for g in raw]
                except Exception:
                    pass

            prompt_token_count = (
                len(node_output.prompt_token_ids)
                if node_output.prompt_token_ids is not None
                else len(effective_prompt.split())
            )
            session.register_completion(
                node_id=node_id,
                num_tokens=prompt_token_count + len(node_output.outputs[0].token_ids),
                block_ids=block_ids_for_session,
            )

            if not include_parent_context and _scheduler is not None:
                bp = _scheduler.kv_cache_manager.block_pool
                if len(node.parents) > 1:
                    # Merge node completed: unpin all ancestor blocks now that
                    # D has finished and its KV blocks are in the block table.
                    for anc_id in session._ancestors_cache[node_id]:
                        for bid in pinned_by_node.pop(anc_id, []):
                            bp.blocks[bid].ref_cnt -= 1
                else:
                    # Non-merge node: pin own blocks for downstream merge nodes.
                    own_pins = []
                    for bid in block_ids_for_session[0]:
                        bp.blocks[bid].ref_cnt += 1
                        own_pins.append(bid)
                    pinned_by_node[node_id] = own_pins

        # Release any remaining pins (leaf nodes with no downstream merge).
        if not include_parent_context and _scheduler is not None:
            bp = _scheduler.kv_cache_manager.block_pool
            for bid_list in pinned_by_node.values():
                for bid in bid_list:
                    bp.blocks[bid].ref_cnt -= 1
            pinned_by_node.clear()

        if return_timing:
            return results, per_node_ttft_ms
        return results

    def get_num_unfinished_requests(self) -> int:
        return self.output_processor.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        has_unfinished = self.output_processor.has_unfinished_requests()
        if self.dp_group is None:
            return has_unfinished or self.engine_core.dp_engines_running()
        return self.has_unfinished_requests_dp(has_unfinished)

    def has_unfinished_requests_dp(self, has_unfinished: bool) -> bool:
        aggregated_has_unfinished = ParallelConfig.has_unfinished_dp(
            self.dp_group, has_unfinished
        )
        if not has_unfinished and aggregated_has_unfinished:
            self.should_execute_dummy_batch = True
        return aggregated_has_unfinished

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        if not hasattr(self, "_supported_tasks"):
            # Cache the result
            self._supported_tasks = self.engine_core.get_supported_tasks()

        return self._supported_tasks

    def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
        """Remove request_ids from EngineCore and Detokenizer."""

        request_ids = self.output_processor.abort_requests(request_ids, internal)
        self.engine_core.abort_requests(request_ids)

    def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType | EngineInput,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        prompt_text: str | None = None,
    ) -> str:
        # Validate the request_id type.
        if not isinstance(request_id, str):
            raise TypeError(f"request_id must be a string, got {type(request_id)}")

        # Process raw inputs into the request.
        if isinstance(prompt, EngineCoreRequest):
            logger.warning_once(
                "Passing EngineCoreRequest to LLMEngine.generate() and .add_requests() "
                "is deprecated and will be removed in v0.18. You should instead pass "
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
            )

            request = prompt
            if request_id != request.request_id:
                logger.warning_once(
                    "LLMEngine.add_request() was passed a request_id parameter that "
                    "does not match the EngineCoreRequest.request_id attribute. The "
                    "latter will be used, and the former will be ignored."
                )
        else:
            request = self.input_processor.process_inputs(
                request_id,
                prompt,
                params,
                supported_tasks=self.get_supported_tasks(),
                arrival_time=arrival_time,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
            )
            prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)

        self.input_processor.assign_request_id(request)

        req_id = request.request_id

        # Use cloned params that may have been updated in process_inputs()
        params = request.params

        n = params.n if isinstance(params, SamplingParams) else 1

        if n == 1:
            # Make a new RequestState and queue.
            self.output_processor.add_request(request, prompt_text, None, 0)
            # Add the request to EngineCore.
            self.engine_core.add_request(request)
            return req_id

        # Fan out child requests (for n>1).
        parent_req = ParentRequest(request)
        for idx in range(n):
            request_id, child_params = parent_req.get_child_info(idx)
            child_request = request if idx == n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = child_params

            # Make a new RequestState and queue.
            self.output_processor.add_request(
                child_request, prompt_text, parent_req, idx
            )
            # Add the request to EngineCore.
            self.engine_core.add_request(child_request)

        return req_id

    def add_dag_request(
        self,
        prompt: list[int],
        nodeid: str,
        session: DAGSession,
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None = None,
        priority: int = 0,
    ) -> str:
        request_id = f"dag{session.session_id}_{nodeid}"

        request = self.input_processor.process_inputs(
            request_id,
            prompt,
            sampling_params,
            dag_context=session.get_context(nodeid),
            supported_tasks=self.get_supported_tasks(),
            arrival_time=None,
            lora_request=lora_request,
            tokenization_kwargs=None,
            trace_headers=None,
            priority=priority,
        )

        prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)
        self.input_processor.assign_request_id(request)

        self.output_processor.add_request(request, prompt_text, None, 0)
        self.engine_core.add_request(request)
        return request.request_id

    def run_dag_request(
        self,
        prompt: EngineInput,
        anc_prompt: list[int],
        nodeid: str,
        session: DAGSession,
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None = None,
        priority: int = 0,
    ) -> tuple[RequestOutput, float]:
        t_submit = time.time()
        req_id = f"dag{session.session_id}_{nodeid}"
        prompt_token_ids = self.input_processor.process_inputs(
            req_id, prompt, sampling_params, supported_tasks=self.get_supported_tasks()
        ).prompt_token_ids

        assert prompt_token_ids
        # Internal request id
        internal_req_id = self.add_dag_request(
            anc_prompt + prompt_token_ids,
            nodeid,
            session,
            sampling_params,
            lora_request,
            priority,
        )
        first_token_time: float | None = None
        node_output: RequestOutput | None = None

        while node_output is None:
            step_outputs = self.step()
            for out in step_outputs:
                if out.request_id == req_id:
                    if (
                        first_token_time is None
                        and out.outputs
                        and out.outputs[0].token_ids
                    ):
                        first_token_time = time.time()
                    if out.finished:
                        node_output = out
                        break
        session.register_completion(
            nodeid,
            len(prompt_token_ids) + len(node_output.outputs[0].token_ids),
            internal_req_id,
        )
        ttft_ms = (
            (first_token_time - t_submit) * 1000.0
            if first_token_time is not None
            else 0.0
        )
        return (node_output, ttft_ms)

    def step(self) -> list[RequestOutput | PoolingRequestOutput]:
        if self.should_execute_dummy_batch:
            self.should_execute_dummy_batch = False
            self.engine_core.execute_dummy_batch()
            return []

        # 1) Get EngineCoreOutput from the EngineCore.
        with record_function_or_nullcontext("llm_engine step: get_output"):
            outputs = self.engine_core.get_output()

        # 2) Process EngineCoreOutputs.
        with record_function_or_nullcontext("llm_engine step: process_outputs"):
            iteration_stats = IterationStats() if self.log_stats else None
            processed_outputs = self.output_processor.process_outputs(
                outputs.outputs,
                engine_core_timestamp=outputs.timestamp,
                iteration_stats=iteration_stats,
            )
            self.output_processor.update_scheduler_stats(outputs.scheduler_stats)

        # 3) Abort any reqs that finished due to stop strings.
        with record_function_or_nullcontext("llm_engine step: abort_requests"):
            self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

        # 4) Record stats
        with record_function_or_nullcontext("llm_engine step: record_stats"):
            if (
                self.logger_manager is not None
                and outputs.scheduler_stats is not None
                and len(outputs.outputs) > 0
            ):
                self.logger_manager.record(
                    scheduler_stats=outputs.scheduler_stats,
                    iteration_stats=iteration_stats,
                    mm_cache_stats=self.renderer.stat_mm_cache(),
                )
                self.do_log_stats_with_interval()

        return processed_outputs.request_outputs

    def start_profile(self, profile_prefix: str | None = None):
        self.engine_core.profile(True, profile_prefix)

    def stop_profile(self):
        self.engine_core.profile(False)

    def reset_mm_cache(self):
        self.renderer.clear_mm_cache()
        self.engine_core.reset_mm_cache()

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return self.engine_core.reset_prefix_cache(
            reset_running_requests, reset_connector
        )

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings computed with old weights are not reused.
        """
        self.engine_core.reset_encoder_cache()

    def sleep(self, level: int = 1, mode: PauseMode = "abort"):
        self.engine_core.sleep(level, mode)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(1, level)

    def wake_up(self, tags: list[str] | None = None):
        self.engine_core.wake_up(tags)

        if self.logger_manager is not None:
            self.logger_manager.record_sleep_state(0, 0)

    def is_sleeping(self) -> bool:
        return self.engine_core.is_sleeping()

    def get_metrics(self) -> list[Metric]:
        assert self.log_stats, "Stat logging disabled"
        return get_metrics_snapshot()

    @property
    def tokenizer(self) -> TokenizerLike | None:
        return self.renderer.tokenizer

    def get_tokenizer(self) -> TokenizerLike:
        return self.renderer.get_tokenizer()

    def do_log_stats(self) -> None:
        """Log stats if logging is enabled."""
        if self.logger_manager:
            self.logger_manager.log()

    def do_log_stats_with_interval(self) -> None:
        """Log stats when the time interval has passed."""
        now = time.time()
        if not hasattr(self, "_last_log_time"):
            self._last_log_time = now
        if now - self._last_log_time >= envs.VLLM_LOG_STATS_INTERVAL:
            self.do_log_stats()
            self._last_log_time = now

    def add_lora(self, lora_request: LoRARequest) -> bool:
        """Load a new LoRA adapter into the engine for future requests."""
        return self.engine_core.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        """Remove an already loaded LoRA adapter."""
        return self.engine_core.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        """List all registered adapters."""
        return self.engine_core.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        """Prevent an adapter from being evicted."""
        return self.engine_core.pin_lora(lora_id)

    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.engine_core.collective_rpc(method, timeout, args, kwargs)

    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]:
        return self.collective_rpc("apply_model", args=(func,))

    def __del__(self):
        dp_group = getattr(self, "dp_group", None)
        if dp_group is not None and not self.external_launcher_dp:
            stateless_destroy_torch_distributed_process_group(dp_group)
