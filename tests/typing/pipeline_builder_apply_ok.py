from __future__ import annotations

from typing_extensions import reveal_type

from inference_pipeline.configs import BaseProcessorConfig, BaseProducerConfig
from inference_pipeline.pipeline import PipelineBuilder, StreamBuilder
from inference_pipeline.stages import ProcessorStage, ProducerStage


class NumberSource(ProducerStage[int, BaseProducerConfig]):
    def produce(self, stop=None):  # type: ignore[override]
        raise NotImplementedError


class IntToStr(ProcessorStage[int, str, BaseProcessorConfig]):
    def process(self, item: int) -> str | None:
        raise NotImplementedError


def add_decode(stream: StreamBuilder[int]) -> StreamBuilder[str]:
    return stream.then(
        "decode",
        IntToStr,
        config=BaseProcessorConfig(out_maxsize=8, out_timeout=0.01),
    )


builder = PipelineBuilder("inference")
source = builder.source(
    "frames",
    NumberSource,
    config=BaseProducerConfig(out_maxsize=8, out_timeout=0.01),
)
decoded = source.apply(add_decode)
reveal_type(decoded)
