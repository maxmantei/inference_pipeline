from __future__ import annotations

from collections.abc import Iterator

import pytest

from inference_pipeline.configs import (
    BaseBroadcastStageConfig,
    BaseConsumerConfig,
    BaseProcessorConfig,
    BaseProducerConfig,
    BaseSplitStageConfig,
)
from inference_pipeline.fanout import BroadcastStage, SplitStage
from inference_pipeline.pipeline import (
    PipelineBuilder,
    PipelineSpec,
    StepKind,
    StreamBuilder,
)
from inference_pipeline.stages import ConsumerStage, ProcessorStage, ProducerStage


class _NumberSource(ProducerStage[int, BaseProducerConfig]):
    def produce(self, stop=None) -> Iterator[int]:  # type: ignore[override]
        raise NotImplementedError


class _IntToStr(ProcessorStage[int, str, BaseProcessorConfig]):
    def process(self, item: int) -> str | None:
        raise NotImplementedError


class _StrToInt(ProcessorStage[str, int, BaseProcessorConfig]):  # pyright: ignore[reportUnusedClass]
    def process(self, item: str) -> int | None:
        raise NotImplementedError


class _CollectStr(ConsumerStage[str, BaseConsumerConfig]):
    def consume(self, item: str) -> None:
        raise NotImplementedError


class _CollectInt(ConsumerStage[int, BaseConsumerConfig]):
    def consume(self, item: int) -> None:
        raise NotImplementedError


class _ParitySplit(SplitStage[int, str, int, BaseSplitStageConfig]):
    def split(self, item: int) -> tuple[str | None, int | None]:
        raise NotImplementedError


def _source_config(name: str | None = None) -> BaseProducerConfig:
    return BaseProducerConfig(name=name, out_maxsize=8, out_timeout=0.01)


def _processor_config(name: str | None = None) -> BaseProcessorConfig:
    return BaseProcessorConfig(name=name, out_maxsize=8, out_timeout=0.01)


def _consumer_config(name: str | None = None) -> BaseConsumerConfig:
    return BaseConsumerConfig(name=name)


def _split_config(name: str | None = None) -> BaseSplitStageConfig:
    return BaseSplitStageConfig(
        name=name,
        out_a_maxsize=8,
        out_b_maxsize=8,
        out_a_timeout=0.01,
        out_b_timeout=0.01,
    )


def _broadcast_config(
    *,
    n_outputs: int,
    name: str | None = None,
) -> BaseBroadcastStageConfig[int]:
    return BaseBroadcastStageConfig(
        name=name,
        n_outputs=n_outputs,
        out_maxsize=8,
        out_timeout=0.01,
    )


def test_pipeline_builder_exposes_name_and_starts_unfrozen() -> None:
    builder = PipelineBuilder("inference")

    assert builder.name == "inference"
    assert builder.is_frozen is False


def test_source_returns_stream_builder_bound_to_builder() -> None:
    builder = PipelineBuilder("inference")

    stream = builder.source("frames", _NumberSource, config=_source_config())

    assert isinstance(stream, StreamBuilder)
    assert stream.pipeline is builder


def test_linear_pipeline_freeze_builds_expected_spec() -> None:
    builder = PipelineBuilder("inference")

    result = (
        builder.source("frames", _NumberSource, config=_source_config())
        .then("decode", _IntToStr, config=_processor_config())
        .sink("writer", _CollectStr, config=_consumer_config())
    )
    spec = builder.freeze()

    assert result is builder
    assert isinstance(spec, PipelineSpec)
    assert spec.name == "inference"
    assert tuple(step.name for step in spec.steps) == ("frames", "decode", "writer")
    assert tuple(step.kind for step in spec.steps) == (
        StepKind.SOURCE,
        StepKind.PROCESSOR,
        StepKind.SINK,
    )
    assert len(spec.edges) == 2


def test_freeze_is_idempotent() -> None:
    builder = PipelineBuilder("inference")
    (
        builder.source("frames", _NumberSource, config=_source_config())
        .then("decode", _IntToStr, config=_processor_config())
        .sink("writer", _CollectStr, config=_consumer_config())
    )

    first = builder.freeze()
    second = builder.freeze()

    assert first == second
    assert builder.is_frozen is True


def test_builder_step_name_is_authoritative_over_config_name() -> None:
    builder = PipelineBuilder("inference")
    (
        builder.source(
            "frames", _NumberSource, config=_source_config(name="camera-mode")
        )
        .then("infer", _IntToStr, config=_processor_config(name="fast-mode"))
        .sink("writer", _CollectStr, config=_consumer_config(name="jsonl"))
    )

    spec = builder.freeze()

    infer_step = spec.step("infer")
    assert infer_step.name == "infer"
    assert infer_step.config.name == "fast-mode"


def test_duplicate_step_names_are_rejected() -> None:
    builder = PipelineBuilder("inference")

    stream = builder.source("frames", _NumberSource, config=_source_config())

    with pytest.raises(ValueError, match="name"):
        stream.then("frames", _IntToStr, config=_processor_config())


def test_stream_cannot_be_reused_after_then() -> None:
    builder = PipelineBuilder("inference")
    stream = builder.source("frames", _NumberSource, config=_source_config())

    stream.then("decode", _IntToStr, config=_processor_config())

    with pytest.raises(RuntimeError, match="consum"):
        stream.sink("writer", _CollectInt, config=_consumer_config())


def test_stream_cannot_be_reused_after_sink() -> None:
    builder = PipelineBuilder("inference")
    stream = builder.source("frames", _NumberSource, config=_source_config())

    stream.sink("writer", _CollectInt, config=_consumer_config())

    with pytest.raises(RuntimeError, match="consum"):
        stream.then("decode", _IntToStr, config=_processor_config())


def test_freeze_rejects_pipeline_without_source() -> None:
    builder = PipelineBuilder("inference")

    with pytest.raises(ValueError, match="source"):
        builder.freeze()


def test_freeze_rejects_pipeline_without_sink() -> None:
    builder = PipelineBuilder("inference")
    builder.source("frames", _NumberSource, config=_source_config())

    with pytest.raises(ValueError, match="sink"):
        builder.freeze()


def test_split_returns_two_independent_streams() -> None:
    builder = PipelineBuilder("inference")
    stream = builder.source("frames", _NumberSource, config=_source_config())

    left, right = stream.split("route", _ParitySplit, config=_split_config())

    assert isinstance(left, StreamBuilder)
    assert isinstance(right, StreamBuilder)
    assert left.pipeline is builder
    assert right.pipeline is builder
    assert left is not right


def test_split_pipeline_freeze_builds_expected_spec() -> None:
    builder = PipelineBuilder("inference")
    source = builder.source("frames", _NumberSource, config=_source_config())
    accepted, rejected = source.split(
        "route",
        _ParitySplit,
        config=_split_config(),
    )
    accepted.sink("writer", _CollectStr, config=_consumer_config())
    rejected.sink("rejects", _CollectInt, config=_consumer_config())

    spec = builder.freeze()

    assert tuple(step.name for step in spec.steps) == (
        "frames",
        "route",
        "writer",
        "rejects",
    )
    assert tuple(step.kind for step in spec.steps) == (
        StepKind.SOURCE,
        StepKind.SPLIT,
        StepKind.SINK,
        StepKind.SINK,
    )
    assert len(spec.edges) == 3
    assert spec.step("route").output_count == 2


def test_split_consumes_upstream_stream() -> None:
    builder = PipelineBuilder("inference")
    stream = builder.source("frames", _NumberSource, config=_source_config())

    stream.split("route", _ParitySplit, config=_split_config())

    with pytest.raises(RuntimeError, match="consum"):
        stream.then("decode", _IntToStr, config=_processor_config())


def test_freeze_rejects_unconsumed_split_branch() -> None:
    builder = PipelineBuilder("inference")
    stream = builder.source("frames", _NumberSource, config=_source_config())
    accepted, _rejected = stream.split("route", _ParitySplit, config=_split_config())
    accepted.sink("writer", _CollectStr, config=_consumer_config())

    with pytest.raises(ValueError, match="open|dangling|branch"):
        builder.freeze()


def test_broadcast_two_way_returns_two_streams() -> None:
    builder = PipelineBuilder("inference")
    stream = builder.source("frames", _NumberSource, config=_source_config())

    first, second = stream.broadcast(
        "copies",
        BroadcastStage,
        n_outputs=2,
        config=_broadcast_config(n_outputs=2),
    )

    assert isinstance(first, StreamBuilder)
    assert isinstance(second, StreamBuilder)
    assert first.pipeline is builder
    assert second.pipeline is builder
    assert first is not second


def test_broadcast_three_way_returns_three_streams() -> None:
    builder = PipelineBuilder("inference")
    stream = builder.source("frames", _NumberSource, config=_source_config())

    first, second, third = stream.broadcast(
        "copies",
        BroadcastStage,
        n_outputs=3,
        config=_broadcast_config(n_outputs=3),
    )

    assert isinstance(first, StreamBuilder)
    assert isinstance(second, StreamBuilder)
    assert isinstance(third, StreamBuilder)
    assert len({id(first), id(second), id(third)}) == 3


def test_broadcast_pipeline_freeze_builds_expected_spec() -> None:
    builder = PipelineBuilder("inference")
    stream = builder.source("frames", _NumberSource, config=_source_config())
    infer, preview = stream.broadcast(
        "copies",
        BroadcastStage,
        n_outputs=2,
        config=_broadcast_config(n_outputs=2),
    )
    infer.sink("writer", _CollectInt, config=_consumer_config())
    preview.sink("preview", _CollectInt, config=_consumer_config())

    spec = builder.freeze()

    assert tuple(step.name for step in spec.steps) == (
        "frames",
        "copies",
        "writer",
        "preview",
    )
    assert tuple(step.kind for step in spec.steps) == (
        StepKind.SOURCE,
        StepKind.BROADCAST,
        StepKind.SINK,
        StepKind.SINK,
    )
    assert len(spec.edges) == 3
    assert spec.step("copies").output_count == 2


def test_broadcast_consumes_upstream_stream() -> None:
    builder = PipelineBuilder("inference")
    stream = builder.source("frames", _NumberSource, config=_source_config())

    stream.broadcast(
        "copies",
        BroadcastStage,
        n_outputs=2,
        config=_broadcast_config(n_outputs=2),
    )

    with pytest.raises(RuntimeError, match="consum"):
        stream.then("decode", _IntToStr, config=_processor_config())


def test_freeze_rejects_unconsumed_broadcast_branch() -> None:
    builder = PipelineBuilder("inference")
    stream = builder.source("frames", _NumberSource, config=_source_config())
    first, _second = stream.broadcast(
        "copies",
        BroadcastStage,
        n_outputs=2,
        config=_broadcast_config(n_outputs=2),
    )
    first.sink("writer", _CollectInt, config=_consumer_config())

    with pytest.raises(ValueError, match="open|dangling|branch"):
        builder.freeze()


def test_broadcast_rejects_n_outputs_less_than_one() -> None:
    builder = PipelineBuilder("inference")
    stream = builder.source("frames", _NumberSource, config=_source_config())

    with pytest.raises(ValueError, match="n_outputs"):
        stream.broadcast(
            "copies",
            BroadcastStage,
            n_outputs=0,
            config=_broadcast_config(n_outputs=0),
        )


def test_broadcast_requires_n_outputs_to_match_config() -> None:
    builder = PipelineBuilder("inference")
    stream = builder.source("frames", _NumberSource, config=_source_config())

    with pytest.raises(ValueError, match="n_outputs"):
        stream.broadcast(
            "copies",
            BroadcastStage,
            n_outputs=2,
            config=_broadcast_config(n_outputs=3),
        )


def test_apply_supports_reusable_linear_fragment() -> None:
    def add_decode(current: StreamBuilder[int]) -> StreamBuilder[str]:
        return current.then("decode", _IntToStr, config=_processor_config())

    builder = PipelineBuilder("inference")
    decoded = builder.source("frames", _NumberSource, config=_source_config()).apply(
        add_decode
    )
    decoded.sink("writer", _CollectStr, config=_consumer_config())

    spec = builder.freeze()

    assert tuple(step.name for step in spec.steps) == ("frames", "decode", "writer")


def test_apply_supports_fragment_returning_split_branches() -> None:
    def add_route(
        current: StreamBuilder[int],
    ) -> tuple[StreamBuilder[str], StreamBuilder[int]]:
        return current.split("route", _ParitySplit, config=_split_config())

    builder = PipelineBuilder("inference")
    accepted, rejected = builder.source(
        "frames",
        _NumberSource,
        config=_source_config(),
    ).apply(add_route)
    accepted.sink("writer", _CollectStr, config=_consumer_config())
    rejected.sink("rejects", _CollectInt, config=_consumer_config())

    spec = builder.freeze()

    assert tuple(step.name for step in spec.steps) == (
        "frames",
        "route",
        "writer",
        "rejects",
    )
