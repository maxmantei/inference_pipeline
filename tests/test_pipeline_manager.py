from __future__ import annotations

import threading
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
    Pipeline,
    PipelineBuilder,
    PipelineManager,
    PipelineSpec,
    PipelineState,
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

    with pytest.raises(ValueError, match="custom-run"):
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

    with pytest.raises(KeyError, match="missing"):
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

    with pytest.raises(KeyError, match="missing"):
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

    with pytest.raises(KeyError, match="missing"):
        manager.stop("missing")


def test_manager_stop_rejects_completed_run() -> None:
    manager = PipelineManager()
    spec = _runtime_spec()

    manager.run(spec)

    with pytest.raises(RuntimeError, match="completed|active"):
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

    with pytest.raises(RuntimeError, match="active|completed"):
        manager.discard("inference-run-1")


def test_manager_discard_rejects_unknown_run_id() -> None:
    manager = PipelineManager()

    with pytest.raises(KeyError, match="missing"):
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
