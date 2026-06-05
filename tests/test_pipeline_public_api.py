from __future__ import annotations

import inference_pipeline


def test_pipeline_types_are_exported_from_package_root() -> None:
    assert hasattr(inference_pipeline, "Pipeline")
    assert hasattr(inference_pipeline, "PipelineBuilder")
    assert hasattr(inference_pipeline, "PipelineSpec")
    assert hasattr(inference_pipeline, "PipelinePhase")
    assert hasattr(inference_pipeline, "PipelineOutcome")
    assert hasattr(inference_pipeline, "StreamBuilder")
    assert hasattr(inference_pipeline, "SourceStepSpec")
    assert hasattr(inference_pipeline, "ProcessorStepSpec")
    assert hasattr(inference_pipeline, "SinkStepSpec")
    assert hasattr(inference_pipeline, "SplitStepSpec")
    assert hasattr(inference_pipeline, "BroadcastStepSpec")
    assert hasattr(inference_pipeline, "StepKind")
    assert hasattr(inference_pipeline, "EdgeSpec")


def test_pipeline_registry_is_not_exported_from_package_root() -> None:
    assert not hasattr(inference_pipeline, "PipelineRegistry")
