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
from inference_pipeline.pipeline import PipelineBuilder, StepKind
from inference_pipeline.stages import ConsumerStage, ProcessorStage, ProducerStage


class _NumberSource(ProducerStage[int, BaseProducerConfig]):
    def produce(self, stop=None) -> Iterator[int]:  # type: ignore[override]
        raise NotImplementedError


class _IntToStr(ProcessorStage[int, str, BaseProcessorConfig]):
    def process(self, item: int) -> str | None:
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


def _split_spec():
    builder = PipelineBuilder("inference")
    accepted, rejected = builder.source(
        "frames",
        _NumberSource,
        config=_source_config(),
    ).split("route", _ParitySplit, config=_split_config())
    accepted.sink("writer", _CollectStr, config=_consumer_config())
    rejected.sink("rejects", _CollectInt, config=_consumer_config())
    return builder.freeze()


def _broadcast_spec():
    builder = PipelineBuilder("inference")
    first, second = builder.source(
        "frames",
        _NumberSource,
        config=_source_config(),
    ).broadcast(
        "copies",
        BroadcastStage,
        n_outputs=2,
        config=_broadcast_config(2),
    )
    first.sink("writer", _CollectInt, config=_consumer_config())
    second.sink("preview", _CollectInt, config=_consumer_config())
    return builder.freeze()


def test_pipeline_spec_filters_steps_by_kind() -> None:
    spec = _broadcast_spec()

    assert tuple(step.name for step in spec.sources) == ("frames",)
    assert tuple(step.name for step in spec.processors) == ()
    assert tuple(step.name for step in spec.sinks) == ("writer", "preview")
    assert tuple(step.name for step in spec.splits) == ()
    assert tuple(step.name for step in spec.broadcasts) == ("copies",)


def test_pipeline_spec_step_lookup_by_name() -> None:
    spec = _linear_spec()

    step = spec.step("decode")

    assert step.name == "decode"
    assert step.kind == StepKind.PROCESSOR


def test_pipeline_spec_step_lookup_by_id() -> None:
    spec = _linear_spec()
    decode = spec.step("decode")

    same = spec.step_by_id(decode.id)

    assert same == decode


def test_pipeline_spec_step_lookup_rejects_unknown_name() -> None:
    spec = _linear_spec()

    with pytest.raises(KeyError, match="missing"):
        spec.step("missing")


def test_pipeline_spec_step_lookup_rejects_unknown_id() -> None:
    spec = _linear_spec()

    with pytest.raises(KeyError, match="999"):
        spec.step_by_id(999)


def test_pipeline_spec_upstream_and_downstream_for_linear_graph() -> None:
    spec = _linear_spec()
    frames = spec.step("frames")
    decode = spec.step("decode")
    writer = spec.step("writer")

    assert spec.upstream_of(frames.id) == ()
    assert tuple(edge.to_step for edge in spec.downstream_of(frames.id)) == (decode.id,)
    assert tuple(edge.from_step for edge in spec.upstream_of(decode.id)) == (frames.id,)
    assert tuple(edge.to_step for edge in spec.downstream_of(decode.id)) == (writer.id,)
    assert tuple(edge.from_step for edge in spec.upstream_of(writer.id)) == (decode.id,)
    assert spec.downstream_of(writer.id) == ()


def test_pipeline_spec_upstream_and_downstream_for_split_graph() -> None:
    spec = _split_spec()
    frames = spec.step("frames")
    route = spec.step("route")
    writer = spec.step("writer")
    rejects = spec.step("rejects")

    assert tuple(edge.to_step for edge in spec.downstream_of(frames.id)) == (route.id,)
    assert tuple(edge.from_step for edge in spec.upstream_of(route.id)) == (frames.id,)
    assert tuple(edge.to_step for edge in spec.downstream_of(route.id)) == (
        writer.id,
        rejects.id,
    )


def test_pipeline_spec_upstream_and_downstream_for_broadcast_graph() -> None:
    spec = _broadcast_spec()
    frames = spec.step("frames")
    copies = spec.step("copies")
    writer = spec.step("writer")
    preview = spec.step("preview")

    assert tuple(edge.to_step for edge in spec.downstream_of(frames.id)) == (copies.id,)
    assert tuple(edge.from_step for edge in spec.upstream_of(copies.id)) == (frames.id,)
    assert tuple(edge.to_step for edge in spec.downstream_of(copies.id)) == (
        writer.id,
        preview.id,
    )


def test_step_kinds_and_output_counts_are_correct() -> None:
    linear = _linear_spec()
    split = _split_spec()
    broadcast = _broadcast_spec()

    assert linear.step("frames").kind == StepKind.SOURCE
    assert linear.step("frames").output_count == 1
    assert linear.step("decode").kind == StepKind.PROCESSOR
    assert linear.step("decode").output_count == 1
    assert linear.step("writer").kind == StepKind.SINK
    assert linear.step("writer").output_count == 0
    assert split.step("route").kind == StepKind.SPLIT
    assert split.step("route").output_count == 2
    assert broadcast.step("copies").kind == StepKind.BROADCAST
    assert broadcast.step("copies").output_count == 2
