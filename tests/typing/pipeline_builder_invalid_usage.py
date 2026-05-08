from __future__ import annotations

from inference_pipeline.configs import (
    BaseConsumerConfig,
    BaseProcessorConfig,
    BaseProducerConfig,
)
from inference_pipeline.pipeline import PipelineBuilder
from inference_pipeline.stages import ConsumerStage, ProcessorStage, ProducerStage


class NumberSource(ProducerStage[int, BaseProducerConfig]):
    def produce(self, stop=None):  # type: ignore[override]
        raise NotImplementedError


class IntToStr(ProcessorStage[int, str, BaseProcessorConfig]):
    def process(self, item: int) -> str | None:
        raise NotImplementedError


class CollectStr(ConsumerStage[str, BaseConsumerConfig]):
    def consume(self, item: str) -> None:
        raise NotImplementedError


builder = PipelineBuilder("inference")
stream = builder.source(
    "frames",
    NumberSource,
    config=BaseProducerConfig(out_maxsize=8, out_timeout=0.01),
)

stream.then(
    "decode",
    CollectStr,
    config=BaseConsumerConfig(),
)

stream.sink(
    "writer",
    IntToStr,
    config=BaseProcessorConfig(out_maxsize=8, out_timeout=0.01),
)
