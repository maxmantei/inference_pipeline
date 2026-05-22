from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import pytest

from inference_pipeline.adapters.sinks import ManagedSinkAdapter
from inference_pipeline.adapters.sources import ManagedSourceAdapter
from inference_pipeline.configs import (
    BaseBroadcastStageConfig,
    BaseConsumerConfig,
    BaseProcessorConfig,
    BaseProducerConfig,
    BaseSplitStageConfig,
)
from inference_pipeline.fanout import BroadcastStage, SplitStage
from inference_pipeline.pipeline import (
    DuplicateRunId,
    Pipeline,
    PipelineBuilder,
    PipelineManager,
    PipelineRunActive,
    PipelineRunCompleted,
    PipelineRunNotFound,
    PipelineSpec,
    PipelineState,
    ResourceLimitExceeded,
    RunInfo,
)
from inference_pipeline.stages import ConsumerStage, ProcessorStage, ProducerStage


class _CollectStr(ConsumerStage[str, BaseConsumerConfig]):
    def __init__(self, *, config: BaseConsumerConfig) -> None:
        super().__init__(config=config)
        self.items: list[str] = []
        self.fail_on = config.name

    def consume(self, item: str) -> None:
        if item == self.fail_on:
            raise RuntimeError(f"failed on {item}")
        self.items.append(item)


class _RangeSource(ProducerStage[int, BaseProducerConfig]):
    def produce(self, stop: threading.Event | None = None) -> Iterator[int]:
        for value in [1, 2, 3]:
            if stop is not None and stop.is_set():
                return
            yield value


class _RenderingProcessor(ProcessorStage[int, str, BaseProcessorConfig]):
    def process(self, item: int) -> str | None:
        return f"value:{item}"


class _ParitySplit(SplitStage[int, str, int, BaseSplitStageConfig]):
    def split(self, item: int) -> tuple[str | None, int | None]:
        if item % 2 == 0:
            return f"even:{item}", None
        return None, item


class _CollectInt(ConsumerStage[int, BaseConsumerConfig]):
    def __init__(self, *, config: BaseConsumerConfig) -> None:
        super().__init__(config=config)
        self.items: list[int] = []
        self.fail_on = config.name

    def consume(self, item: int) -> None:
        if self.fail_on is not None and str(item) == self.fail_on:
            raise RuntimeError(f"failed on {item}")
        self.items.append(item)


class _RecordingSourceAdapter(ManagedSourceAdapter[int, BaseProducerConfig]):
    def __init__(self, *, config: BaseProducerConfig) -> None:
        super().__init__(config=config)
        self.start_calls = 0
        self.stop_calls = 0

    def _start_impl(self) -> None:
        self.start_calls += 1
        for value in [1, 2, 3]:
            self.output.put(value)

    def _stop_impl(self) -> None:
        self.stop_calls += 1


class _RecordingSinkAdapter(ManagedSinkAdapter[str, BaseConsumerConfig]):
    def __init__(self, *, config: BaseConsumerConfig) -> None:
        super().__init__(config=config)
        self.start_calls = 0
        self.stop_calls = 0
        self.items: list[str] = []

    def _start_impl(self) -> None:
        self.start_calls += 1

    def _stop_impl(self) -> None:
        self.stop_calls += 1

    def consume(self, item: str) -> None:
        self.items.append(item)


def _source_config() -> BaseProducerConfig:
    return BaseProducerConfig(out_maxsize=8, out_timeout=0.01)


def _processor_config() -> BaseProcessorConfig:
    return BaseProcessorConfig(out_maxsize=8, out_timeout=0.01)


def _consumer_config() -> BaseConsumerConfig:
    return BaseConsumerConfig()


def _split_config() -> BaseSplitStageConfig:
    return BaseSplitStageConfig(
        out_a_maxsize=8,
        out_b_maxsize=8,
        out_a_timeout=0.01,
        out_b_timeout=0.01,
    )


def _broadcast_config(n_outputs: int) -> BaseBroadcastStageConfig[int]:
    return BaseBroadcastStageConfig(
        n_outputs=n_outputs,
        out_maxsize=8,
        out_timeout=0.01,
    )


def _linear_spec() -> PipelineSpec:
    builder = PipelineBuilder("inference")
    (
        builder.source("frames", _RangeSource, config=_source_config())
        .then("render", _RenderingProcessor, config=_processor_config())
        .sink("writer", _CollectStr, config=_consumer_config())
    )
    return builder.freeze()


def _runtime_spec(*, fail_on: str | None = None) -> PipelineSpec:
    builder = PipelineBuilder("runtime")
    (
        builder.source("frames", _RangeSource, config=_source_config())
        .then("render", _RenderingProcessor, config=_processor_config())
        .sink("writer", _CollectStr, config=BaseConsumerConfig(name=fail_on))
    )
    return builder.freeze()


def _split_runtime_spec(*, fail_on_reject: str | None = None) -> PipelineSpec:
    builder = PipelineBuilder("split-runtime")
    accepted, rejected = builder.source(
        "frames", _RangeSource, config=_source_config()
    ).split("route", _ParitySplit, config=_split_config())
    accepted.sink("writer", _CollectStr, config=BaseConsumerConfig())
    rejected.sink(
        "rejects", _CollectInt, config=BaseConsumerConfig(name=fail_on_reject)
    )
    return builder.freeze()


def _broadcast_runtime_spec(*, fail_on_preview: str | None = None) -> PipelineSpec:
    builder = PipelineBuilder("broadcast-runtime")
    writer_branch, preview_branch = builder.source(
        "frames",
        _RangeSource,
        config=_source_config(),
    ).broadcast(
        "copies",
        BroadcastStage,
        n_outputs=2,
        config=_broadcast_config(2),
    )
    writer_branch.sink("writer", _CollectInt, config=BaseConsumerConfig())
    preview_branch.sink(
        "preview",
        _CollectInt,
        config=BaseConsumerConfig(name=fail_on_preview),
    )
    return builder.freeze()


def _adapter_runtime_spec() -> PipelineSpec:
    builder = PipelineBuilder("adapter-runtime")
    (
        builder.source("frames", _RecordingSourceAdapter, config=_source_config())
        .then("render", _RenderingProcessor, config=_processor_config())
        .sink("writer", _RecordingSinkAdapter, config=BaseConsumerConfig())
    )
    return builder.freeze()


def test_pipeline_manager_starts_with_no_active_or_completed_runs() -> None:
    manager = PipelineManager()

    assert manager.active == {}
    assert manager.completed == {}


def test_manager_create_registers_created_pipeline_as_active() -> None:
    manager = PipelineManager()

    pipeline = manager.create(_linear_spec())

    assert isinstance(pipeline, Pipeline)
    assert pipeline.state is PipelineState.CREATED
    assert manager.active == {"inference-run-1": pipeline}
    assert manager.completed == {}


def test_manager_create_generates_readable_incrementing_run_ids() -> None:
    manager = PipelineManager()
    spec = _linear_spec()

    first = manager.create(spec)
    second = manager.create(spec)

    assert manager.active["inference-run-1"] is first
    assert manager.active["inference-run-2"] is second


def test_manager_create_rejects_duplicate_explicit_run_id() -> None:
    manager = PipelineManager()
    spec = _linear_spec()

    manager.create(spec, run_id="custom-run")

    with pytest.raises(DuplicateRunId, match="custom-run"):
        manager.create(spec, run_id="custom-run")


def test_manager_pipeline_returns_active_pipeline_by_run_id() -> None:
    manager = PipelineManager()
    pipeline = manager.create(_linear_spec())

    assert manager.pipeline("inference-run-1") is pipeline


def test_manager_pipeline_returns_completed_pipeline_by_run_id() -> None:
    manager = PipelineManager()
    spec = _runtime_spec()

    pipeline = manager.run(spec)

    assert manager.pipeline("runtime-run-1") is pipeline


def test_manager_pipeline_rejects_unknown_run_id() -> None:
    manager = PipelineManager()

    with pytest.raises(PipelineRunNotFound, match="missing"):
        manager.pipeline("missing")


def test_manager_start_creates_and_starts_pipeline() -> None:
    manager = PipelineManager()
    spec = _runtime_spec()

    pipeline = manager.start(spec)

    assert pipeline.state is PipelineState.RUNNING
    assert manager.active == {"runtime-run-1": pipeline}
    assert manager.completed == {}

    manager.stop("runtime-run-1")
    manager.join("runtime-run-1")


def test_manager_run_creates_runs_and_completes_pipeline() -> None:
    manager = PipelineManager()
    spec = _runtime_spec()

    pipeline = manager.run(spec)

    assert pipeline.state is PipelineState.SUCCEEDED
    assert manager.active == {}
    assert manager.completed == {"runtime-run-1": pipeline}


def test_manager_run_preserves_failed_pipeline_in_completed() -> None:
    manager = PipelineManager()
    spec = _runtime_spec(fail_on="value:2")

    with pytest.raises(RuntimeError, match="failed on value:2"):
        manager.run(spec)

    failed = manager.completed["runtime-run-1"]
    assert failed.state is PipelineState.FAILED
    assert isinstance(failed.error, RuntimeError)


def test_manager_join_moves_active_pipeline_to_completed() -> None:
    manager = PipelineManager()
    spec = _runtime_spec()

    pipeline = manager.start(spec)

    joined = manager.join("runtime-run-1")

    assert joined is pipeline
    assert joined.state is PipelineState.SUCCEEDED
    assert manager.active == {}
    assert manager.completed == {"runtime-run-1": pipeline}


def test_manager_join_returns_completed_pipeline_when_run_already_finished() -> None:
    manager = PipelineManager()
    spec = _runtime_spec()

    pipeline = manager.run(spec)
    joined = manager.join("runtime-run-1")

    assert joined is pipeline
    assert joined.state is PipelineState.SUCCEEDED


def test_manager_join_rejects_unknown_run_id() -> None:
    manager = PipelineManager()

    with pytest.raises(PipelineRunNotFound, match="missing"):
        manager.join("missing")


def test_manager_stop_signals_active_pipeline() -> None:
    manager = PipelineManager()
    spec = _runtime_spec()

    pipeline = manager.start(spec)

    manager.stop("runtime-run-1")
    joined = manager.join("runtime-run-1")

    assert joined is pipeline
    assert joined.state in {PipelineState.STOPPED, PipelineState.SUCCEEDED}


def test_manager_stop_rejects_unknown_run_id() -> None:
    manager = PipelineManager()

    with pytest.raises(PipelineRunNotFound, match="missing"):
        manager.stop("missing")


def test_manager_stop_rejects_completed_run() -> None:
    manager = PipelineManager()
    spec = _runtime_spec()

    manager.run(spec)

    with pytest.raises(PipelineRunCompleted, match="already completed"):
        manager.stop("runtime-run-1")


def test_manager_discard_removes_completed_pipeline() -> None:
    manager = PipelineManager()
    spec = _runtime_spec()

    pipeline = manager.run(spec)
    manager.discard("runtime-run-1")

    assert pipeline.state is PipelineState.SUCCEEDED
    assert manager.completed == {}


def test_manager_discard_rejects_active_pipeline() -> None:
    manager = PipelineManager()

    manager.create(_linear_spec())

    with pytest.raises(PipelineRunActive, match="still active"):
        manager.discard("inference-run-1")


def test_manager_discard_rejects_unknown_run_id() -> None:
    manager = PipelineManager()

    with pytest.raises(PipelineRunNotFound, match="missing"):
        manager.discard("missing")


def test_manager_can_run_same_spec_multiple_times() -> None:
    manager = PipelineManager()
    spec = _runtime_spec()

    first = manager.run(spec)
    second = manager.run(spec)

    assert first is not second
    assert manager.completed["runtime-run-1"] is first
    assert manager.completed["runtime-run-2"] is second


def test_manager_run_supports_adapter_based_pipeline() -> None:
    manager = PipelineManager()
    spec = _adapter_runtime_spec()

    pipeline = manager.run(spec)

    assert pipeline.state is PipelineState.SUCCEEDED
    source = pipeline.step("frames")
    writer = pipeline.step("writer")
    assert isinstance(source, _RecordingSourceAdapter)
    assert isinstance(writer, _RecordingSinkAdapter)
    assert source.start_calls == 1
    assert source.stop_calls == 1
    assert writer.start_calls == 1
    assert writer.stop_calls == 1
    assert writer.items == ["value:1", "value:2", "value:3"]


def test_manager_run_supports_broadcast_pipeline() -> None:
    manager = PipelineManager()
    spec = _broadcast_runtime_spec()

    pipeline = manager.run(spec)

    assert pipeline.state is PipelineState.SUCCEEDED


def test_manager_run_supports_split_pipeline() -> None:
    manager = PipelineManager()
    spec = _split_runtime_spec()

    pipeline = manager.run(spec)

    assert pipeline.state is PipelineState.SUCCEEDED


# ------------------------------------------------------------------
# Resource limits
# ------------------------------------------------------------------


def test_manager_create_rejects_exceeded_max_active() -> None:
    manager = PipelineManager(max_active_runs=1)
    spec = _linear_spec()

    manager.create(spec)

    with pytest.raises(ResourceLimitExceeded, match="Max active"):
        manager.create(spec)


def test_manager_start_rejects_exceeded_max_active() -> None:
    manager = PipelineManager(max_active_runs=1)
    spec = _runtime_spec()

    pipeline = manager.start(spec)
    manager.stop(pipeline.spec.name + "-run-1")

    with pytest.raises(ResourceLimitExceeded, match="Max active"):
        manager.start(spec)


def test_manager_run_rejects_exceeded_max_active() -> None:
    manager = PipelineManager(max_active_runs=1)
    spec = _linear_spec()

    manager.create(spec)

    with pytest.raises(ResourceLimitExceeded, match="Max active"):
        manager.create(spec)


def test_manager_max_active_zero_allows_unlimited() -> None:
    manager = PipelineManager(max_active_runs=0)
    spec = _linear_spec()

    for _ in range(10):
        manager.create(spec)

    assert manager.active_count == 10


# ------------------------------------------------------------------
# Observability: RunInfo
# ------------------------------------------------------------------


def test_manager_run_info_returns_lightweight_status() -> None:
    manager = PipelineManager()
    spec = _runtime_spec()
    created_before = time.time()

    manager.run(spec)
    info = manager.run_info("runtime-run-1")

    assert isinstance(info, RunInfo)
    assert info.run_id == "runtime-run-1"
    assert info.spec_name == "runtime"
    assert info.state is PipelineState.SUCCEEDED
    assert info.created_at >= created_before
    assert info.started_at is not None and info.started_at >= info.created_at
    assert info.finished_at is not None and info.finished_at >= info.started_at
    assert info.error is None


def test_manager_run_info_with_failure() -> None:
    manager = PipelineManager()
    spec = _runtime_spec(fail_on="value:2")

    with pytest.raises(RuntimeError):
        manager.run(spec)

    info = manager.run_info("runtime-run-1")
    assert info.state is PipelineState.FAILED
    assert info.error is not None
    assert "failed on" in info.error
    assert info.finished_at is not None


def test_manager_run_info_for_active_run() -> None:
    manager = PipelineManager()
    spec = _runtime_spec()

    manager.start(spec)
    info = manager.run_info("runtime-run-1")

    assert info.state is PipelineState.RUNNING
    assert info.started_at is not None
    assert info.finished_at is None

    manager.stop("runtime-run-1")
    manager.join("runtime-run-1")


def test_manager_run_info_rejects_unknown() -> None:
    manager = PipelineManager()

    with pytest.raises(PipelineRunNotFound):
        manager.run_info("nobody")


# ------------------------------------------------------------------
# Observability: list_runs
# ------------------------------------------------------------------


def test_manager_list_runs() -> None:
    manager = PipelineManager()
    spec = _runtime_spec()

    manager.run(spec)
    manager.run(spec)

    runs = manager.list_runs()
    assert len(runs) == 2
    assert all(isinstance(r, RunInfo) for r in runs)


def test_manager_list_runs_mixed_active_and_completed() -> None:
    manager = PipelineManager()
    spec = _runtime_spec()

    manager.start(spec)
    manager.stop("runtime-run-1")
    manager.run(spec)

    runs = manager.list_runs()
    assert len(runs) == 2


# ------------------------------------------------------------------
# Observability: counts
# ------------------------------------------------------------------


def test_manager_active_count() -> None:
    manager = PipelineManager()

    manager.create(_linear_spec())
    assert manager.active_count == 1

    manager.create(_linear_spec())
    assert manager.active_count == 2

    manager.discard_completed()
    assert manager.active_count == 2


def test_manager_completed_count() -> None:
    manager = PipelineManager()
    spec = _runtime_spec()

    assert manager.completed_count == 0

    manager.run(spec)
    assert manager.completed_count == 1


def test_manager_empty_counts() -> None:
    manager = PipelineManager()

    assert manager.active_count == 0
    assert manager.completed_count == 0


# ------------------------------------------------------------------
# Observability: oldest_active_run
# ------------------------------------------------------------------


def test_manager_oldest_active_run() -> None:
    manager = PipelineManager()

    manager.create(_linear_spec())
    oldest = manager.oldest_active_run
    assert oldest is not None
    assert oldest.run_id == "inference-run-1"


def test_manager_oldest_active_run_none_when_empty() -> None:
    manager = PipelineManager()

    assert manager.oldest_active_run is None


def test_manager_oldest_active_run_returns_oldest() -> None:
    manager = PipelineManager()
    spec = _linear_spec()

    manager.create(spec)
    manager.create(spec)

    oldest = manager.oldest_active_run
    assert oldest is not None
    assert oldest.run_id == "inference-run-1"


# ------------------------------------------------------------------
# Observability: runs_by_state
# ------------------------------------------------------------------


def test_manager_runs_by_state() -> None:
    manager = PipelineManager()

    manager.create(_linear_spec())
    manager.run(_runtime_spec())

    counts = manager.runs_by_state()
    assert counts.get(PipelineState.CREATED, 0) == 1
    assert counts.get(PipelineState.SUCCEEDED, 0) == 1


# ------------------------------------------------------------------
# Observability: health
# ------------------------------------------------------------------


def test_manager_health_returns_expected_keys() -> None:
    manager = PipelineManager()
    spec = _runtime_spec()

    manager.run(spec)

    h = manager.health()
    assert h["active_count"] == 0
    assert h["completed_count"] == 1
    assert h["oldest_active_seconds"] is None
    assert h["oldest_active_run_id"] is None
    assert isinstance(h["runs_by_state"], dict)


def test_manager_health_with_active_run() -> None:
    manager = PipelineManager()

    manager.create(_linear_spec())

    h = manager.health()
    assert h["active_count"] == 1
    assert h["completed_count"] == 0
    assert h["oldest_active_seconds"] is not None
    assert h["oldest_active_run_id"] == "inference-run-1"


# ------------------------------------------------------------------
# cancel
# ------------------------------------------------------------------


def test_manager_cancel_active_removes_run() -> None:
    manager = PipelineManager()
    spec = _runtime_spec()

    manager.start(spec)
    manager.cancel("runtime-run-1")

    assert "runtime-run-1" not in manager._active
    assert "runtime-run-1" not in manager._completed


def test_manager_cancel_completed_removes_run() -> None:
    manager = PipelineManager()
    spec = _runtime_spec()

    manager.run(spec)
    manager.cancel("runtime-run-1")

    assert "runtime-run-1" not in manager._completed


def test_manager_cancel_rejects_unknown() -> None:
    manager = PipelineManager()

    with pytest.raises(PipelineRunNotFound):
        manager.cancel("missing")


# ------------------------------------------------------------------
# discard_completed
# ------------------------------------------------------------------


def test_manager_discard_completed_all() -> None:
    manager = PipelineManager()

    manager.run(_runtime_spec())
    manager.run(_runtime_spec())

    count = manager.discard_completed()
    assert count == 2
    assert manager.completed_count == 0


def test_manager_discard_completed_older_than() -> None:
    manager = PipelineManager()

    manager.run(_runtime_spec())
    time.sleep(0.01)

    count = manager.discard_completed(older_than=0.005)
    assert count == 1
    assert manager.completed_count == 0


def test_manager_discard_completed_older_than_none_expired() -> None:
    manager = PipelineManager()

    manager.run(_runtime_spec())

    count = manager.discard_completed(older_than=3600)
    assert count == 0
    assert manager.completed_count == 1


# ------------------------------------------------------------------
# Context manager
# ------------------------------------------------------------------


def test_manager_context_manager_enter_exit() -> None:
    with PipelineManager() as manager:
        assert isinstance(manager, PipelineManager)
        manager.run(_runtime_spec())
        assert manager.completed_count == 1


def test_manager_context_manager_stops_active() -> None:
    spec = _runtime_spec()
    manager = PipelineManager()

    pipeline = manager.start(spec)
    manager.close()

    assert pipeline.state in {PipelineState.STOPPED, PipelineState.SUCCEEDED}


# ------------------------------------------------------------------
# TTL auto-cleanup
# ------------------------------------------------------------------


def test_manager_ttl_auto_removes_completed() -> None:
    manager = PipelineManager(completed_run_ttl=0.1)

    manager.run(_runtime_spec())
    assert manager.completed_count == 1

    time.sleep(0.3)

    assert manager.completed_count == 0


def test_manager_ttl_daemon_thread_does_not_block() -> None:
    manager = PipelineManager(completed_run_ttl=0.1)

    # Cleanup thread is a daemon and won't block process exit
    assert manager._cleanup_thread is not None
    assert manager._cleanup_thread.daemon is True
    assert manager._cleanup_thread.is_alive()

    manager.run(_runtime_spec())

    time.sleep(0.3)
    assert manager.completed_count == 0

    manager.close()


# ------------------------------------------------------------------
# Thread safety
# ------------------------------------------------------------------


def test_manager_thread_safety_concurrent_create() -> None:
    import concurrent.futures

    manager = PipelineManager()
    spec = _linear_spec()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(manager.create, spec) for _ in range(20)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 20
    assert manager.active_count == 20
    # All run IDs should be unique - verify no duplicates
    run_ids_in_manager = set(manager._active)
    assert len(run_ids_in_manager) == 20
    # Every result should be a Pipeline
    assert all(isinstance(r, Pipeline) for r in results)


def test_manager_thread_safety_stop_and_join() -> None:
    import concurrent.futures

    manager = PipelineManager()
    spec = _runtime_spec()

    pipeline = manager.start(spec)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        stop_future = ex.submit(manager.stop, "runtime-run-1")
        join_future = ex.submit(manager.join, "runtime-run-1")
        stop_future.result()
        join_future.result()

    assert pipeline.state in {PipelineState.STOPPED, PipelineState.SUCCEEDED}


# ------------------------------------------------------------------
# Discard of already-discarded run
# ------------------------------------------------------------------


def test_manager_discard_twice_raises() -> None:
    manager = PipelineManager()

    manager.run(_runtime_spec())
    manager.discard("runtime-run-1")

    with pytest.raises(PipelineRunNotFound):
        manager.discard("runtime-run-1")
