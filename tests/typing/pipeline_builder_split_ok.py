from __future__ import annotations

from typing_extensions import reveal_type

from inference_pipeline.configs import BaseProducerConfig, BaseSplitStageConfig
from inference_pipeline.fanout import SplitStage
from inference_pipeline.pipeline import PipelineBuilder
from inference_pipeline.stages import ProducerStage


class NumberSource(ProducerStage[int, BaseProducerConfig]):
    def produce(self, stop=None):  # type: ignore[override]
        raise NotImplementedError


class ParitySplit(SplitStage[int, str, int, BaseSplitStageConfig]):
    def split(self, item: int) -> tuple[str | None, int | None]:
        raise NotImplementedError


builder = PipelineBuilder("inference")
stream = builder.source(
    "frames",
    NumberSource,
    config=BaseProducerConfig(out_maxsize=8, out_timeout=0.01),
)
left, right = stream.split(
    "route",
    ParitySplit,
    config=BaseSplitStageConfig(
        out_a_maxsize=8,
        out_b_maxsize=8,
        out_a_timeout=0.01,
        out_b_timeout=0.01,
    ),
)

reveal_type(left)
reveal_type(right)
