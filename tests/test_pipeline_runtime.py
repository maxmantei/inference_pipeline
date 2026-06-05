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
    PipelineOutcome,
    PipelinePhase,
    PipelineStopMode,
)
from inference_pipeline.stages import ConsumerStage, ProcessorStage, ProducerStage


class _NumberSource(ProducerStage[int, BaseProducerConfig]):
    def produce(self, stop=None) -> Iterator[int]:  # type: ignore[override]
        raise NotImplementedError


class _IntToStr(ProcessorStage[int, str, BaseProcessorConfig]):
    def process(self, item: int) -> str | None:
        raise NotImplementedError


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
    def __init__(self, *, config: BaseProducerConfig) -> None:
        super().__init__(config=config)
        self._values: list[int] = []

    def produce(self, stop: threading.Event | None = None) -> Iterator[int]:
        for value in self._values:
            if stop is not None and stop.is_set():
                return
            yield value


class _RenderingProcessor(ProcessorStage[int, str, BaseProcessorConfig]):
    def __init__(self, *, config: BaseProcessorConfig) -> None:
        super().__init__(config=config)

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


def _linear_spec():
    builder = PipelineBuilder("inference")
    (
        builder.source("frames", _NumberSource, config=_source_config())
        .then("decode", _IntToStr, config=_processor_config())
        .sink("writer", _CollectStr, config=_consumer_config())
    )
    return builder.freeze()


def _runtime_pipeline(*, fail_on: str | None = None) -> Pipeline:
    builder = PipelineBuilder("runtime")
    (
        builder.source(
            "frames",
            _RangeSource,
            config=BaseProducerConfig(out_maxsize=8, out_timeout=0.01),
        )
        .then(
            "render",
            _RenderingProcessor,
            config=BaseProcessorConfig(out_maxsize=8, out_timeout=0.01),
        )
        .sink(
            "writer",
            _CollectStr,
            config=BaseConsumerConfig(name=fail_on),
        )
    )
    spec = builder.freeze()
    pipeline = Pipeline.create(spec)

    writer = pipeline.step("writer")
    assert isinstance(writer, _CollectStr)
    writer.fail_on = fail_on
    source = pipeline.step("frames")
    assert isinstance(source, _RangeSource)
    source._values = [1, 2, 3]  # pyright: ignore[reportPrivateUsage]

    return pipeline


def _split_runtime_pipeline(*, fail_on_reject: str | None = None) -> Pipeline:
    builder = PipelineBuilder("split-runtime")
    accepted, rejected = builder.source(
        "frames",
        _RangeSource,
        config=BaseProducerConfig(out_maxsize=8, out_timeout=0.01),
    ).split(
        "route",
        _ParitySplit,
        config=_split_config(),
    )
    accepted.sink("writer", _CollectStr, config=BaseConsumerConfig())
    rejected.sink("rejects", _CollectInt, config=BaseConsumerConfig(name=fail_on_reject))
    pipeline = Pipeline.create(builder.freeze())

    source = pipeline.step("frames")
    assert isinstance(source, _RangeSource)
    source._values = [1, 2, 3, 4]  # pyright: ignore[reportPrivateUsage]
    return pipeline


def _broadcast_runtime_pipeline(*, fail_on_preview: str | None = None) -> Pipeline:
    builder = PipelineBuilder("broadcast-runtime")
    writer_branch, preview_branch = builder.source(
        "frames",
        _RangeSource,
        config=BaseProducerConfig(out_maxsize=8, out_timeout=0.01),
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
    pipeline = Pipeline.create(builder.freeze())

    source = pipeline.step("frames")
    assert isinstance(source, _RangeSource)
    source._values = [1, 2, 3]  # pyright: ignore[reportPrivateUsage]
    return pipeline


def _adapter_runtime_pipeline() -> Pipeline:
    builder = PipelineBuilder("adapter-runtime")
    (
        builder.source(
            "frames",
            _RecordingSourceAdapter,
            config=BaseProducerConfig(out_maxsize=8, out_timeout=0.01),
        )
        .then(
            "render",
            _RenderingProcessor,
            config=BaseProcessorConfig(out_maxsize=8, out_timeout=0.01),
        )
        .sink(
            "writer",
            _RecordingSinkAdapter,
            config=BaseConsumerConfig(),
        )
    )
    return Pipeline.create(builder.freeze())


def test_pipeline_create_is_canonical_construction_path() -> None:
    spec = _linear_spec()

    pipeline = Pipeline.create(spec)

    assert isinstance(pipeline, Pipeline)


def test_pipeline_create_sets_created_phase() -> None:
    pipeline = Pipeline.create(_linear_spec())

    assert pipeline.phase is PipelinePhase.CREATED
    assert pipeline.outcome is None
    assert pipeline.done is False


def test_pipeline_create_preserves_spec() -> None:
    spec = _linear_spec()

    pipeline = Pipeline.create(spec)

    assert pipeline.spec == spec


def test_pipeline_create_sets_error_to_none() -> None:
    pipeline = Pipeline.create(_linear_spec())

    assert pipeline.error is None


def test_pipeline_exposes_steps_by_builder_name() -> None:
    pipeline = Pipeline.create(_linear_spec())

    assert set(pipeline.steps) == {"frames", "decode", "writer"}
    assert pipeline.step("frames") is pipeline.steps["frames"]
    assert pipeline.step("decode") is pipeline.steps["decode"]
    assert pipeline.step("writer") is pipeline.steps["writer"]


def test_pipeline_exposes_tasks_by_builder_name() -> None:
    pipeline = Pipeline.create(_linear_spec())

    assert set(pipeline.tasks) == {"frames", "decode", "writer"}
    assert pipeline.task("frames") is pipeline.tasks["frames"]
    assert pipeline.task("decode") is pipeline.tasks["decode"]
    assert pipeline.task("writer") is pipeline.tasks["writer"]


def test_pipeline_step_lookup_rejects_unknown_name() -> None:
    pipeline = Pipeline.create(_linear_spec())

    with pytest.raises(KeyError, match="missing"):
        pipeline.step("missing")


def test_pipeline_task_lookup_rejects_unknown_name() -> None:
    pipeline = Pipeline.create(_linear_spec())

    with pytest.raises(KeyError, match="missing"):
        pipeline.task("missing")


def test_same_spec_can_create_multiple_distinct_pipeline_instances() -> None:
    spec = _linear_spec()

    first = Pipeline.create(spec)
    second = Pipeline.create(spec)

    assert first is not second
    assert first.spec == second.spec == spec
    assert first.steps is not second.steps
    assert first.tasks is not second.tasks


def test_pipeline_start_transitions_to_running() -> None:
    pipeline = _runtime_pipeline()

    pipeline.start()

    assert pipeline.phase is PipelinePhase.RUNNING
    assert pipeline.outcome is None
    assert pipeline.started_at is not None


def test_pipeline_run_completes_successfully() -> None:
    pipeline = _runtime_pipeline()

    pipeline.run()

    assert pipeline.phase is PipelinePhase.TERMINATED
    assert pipeline.outcome is PipelineOutcome.SUCCEEDED
    assert pipeline.done is True
    writer = pipeline.step("writer")
    assert isinstance(writer, _CollectStr)
    assert writer.items == ["value:1", "value:2", "value:3"]


def test_pipeline_start_is_single_use() -> None:
    pipeline = _runtime_pipeline()

    pipeline.start()

    with pytest.raises(RuntimeError, match="single-use|already"):
        pipeline.start()


def test_pipeline_request_stop_is_idempotent() -> None:
    pipeline = _runtime_pipeline()

    pipeline.start()
    pipeline.request_stop()
    first_requested_at = pipeline.stop_mode_requested_at
    pipeline.request_stop()
    pipeline.wait()

    assert pipeline.stop_mode is PipelineStopMode.IMMEDIATE
    assert first_requested_at is not None
    assert pipeline.stop_mode_requested_at == first_requested_at
    assert pipeline.outcome is PipelineOutcome.STOPPED


def test_pipeline_request_drain_finishes_as_succeeded() -> None:
    pipeline = _runtime_pipeline()

    pipeline.start()
    pipeline.request_drain()
    pipeline.wait()

    assert pipeline.phase is PipelinePhase.TERMINATED
    assert pipeline.stop_mode is PipelineStopMode.DRAIN
    assert pipeline.outcome is PipelineOutcome.SUCCEEDED


def test_pipeline_failure_transitions_to_failed() -> None:
    pipeline = _runtime_pipeline(fail_on="value:2")

    with pytest.raises(RuntimeError, match="failed on value:2"):
        pipeline.run()

    assert pipeline.phase is PipelinePhase.TERMINATED
    assert pipeline.outcome is PipelineOutcome.FAILED
    assert isinstance(pipeline.error, RuntimeError)


def test_pipeline_raise_for_error_reraises_worker_failure() -> None:
    pipeline = _runtime_pipeline(fail_on="value:2")

    pipeline.start()
    with pytest.raises(RuntimeError, match="Worker thread"):
        pipeline.wait()

    with pytest.raises(RuntimeError, match="failed on value:2"):
        pipeline.raise_for_error()


def test_split_pipeline_run_completes_successfully() -> None:
    pipeline = _split_runtime_pipeline()

    pipeline.run()

    assert pipeline.outcome is PipelineOutcome.SUCCEEDED
    writer = pipeline.step("writer")
    rejects = pipeline.step("rejects")
    assert isinstance(writer, _CollectStr)
    assert isinstance(rejects, _CollectInt)
    assert writer.items == ["even:2", "even:4"]
    assert rejects.items == [1, 3]


def test_broadcast_pipeline_run_completes_successfully() -> None:
    pipeline = _broadcast_runtime_pipeline()

    pipeline.run()

    assert pipeline.outcome is PipelineOutcome.SUCCEEDED
    writer = pipeline.step("writer")
    preview = pipeline.step("preview")
    assert isinstance(writer, _CollectInt)
    assert isinstance(preview, _CollectInt)
    assert writer.items == [1, 2, 3]
    assert preview.items == [1, 2, 3]


def test_branch_failure_transitions_pipeline_to_failed() -> None:
    pipeline = _broadcast_runtime_pipeline(fail_on_preview="2")

    with pytest.raises(RuntimeError, match="failed on 2"):
        pipeline.run()

    assert pipeline.outcome is PipelineOutcome.FAILED
    assert isinstance(pipeline.error, RuntimeError)


def test_pipeline_run_supports_managed_source_and_sink_adapters() -> None:
    pipeline = _adapter_runtime_pipeline()

    pipeline.run()

    source = pipeline.step("frames")
    writer = pipeline.step("writer")
    assert isinstance(source, _RecordingSourceAdapter)
    assert isinstance(writer, _RecordingSinkAdapter)
    assert source.start_calls == 1
    assert source.stop_calls == 1
    assert writer.start_calls == 1
    assert writer.stop_calls == 1
    assert writer.items == ["value:1", "value:2", "value:3"]


def test_pipeline_request_stop_before_wait_keeps_stopped_terminal_outcome() -> None:
    pipeline = _runtime_pipeline()

    pipeline.start()
    pipeline.request_stop()
    pipeline.wait()

    assert pipeline.phase is PipelinePhase.TERMINATED
    assert pipeline.outcome is PipelineOutcome.STOPPED


def test_pipeline_wait_is_repeatable_after_success() -> None:
    pipeline = _runtime_pipeline()

    pipeline.start()
    pipeline.wait()
    pipeline.wait()

    assert pipeline.outcome is PipelineOutcome.SUCCEEDED


def test_pipeline_cancel_requests_stop_and_waits() -> None:
    pipeline = _runtime_pipeline()

    pipeline.start()
    pipeline.cancel()

    assert pipeline.phase is PipelinePhase.TERMINATED
    assert pipeline.outcome is PipelineOutcome.STOPPED


def test_pipeline_task_names_use_builder_names() -> None:
    pipeline = _runtime_pipeline()

    assert pipeline.task("frames").name == "frames-worker"
    assert pipeline.task("render").name == "render-worker"
    assert pipeline.task("writer").name == "writer-worker"
