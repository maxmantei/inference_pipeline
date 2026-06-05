"""Pipeline builder, specification, and runtime API surfaces."""

from __future__ import annotations

import threading
import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal, Protocol, Self, cast, overload

from inference_pipeline.buffer import BufferQ
from inference_pipeline.protocols import (
    BaseBroadcastStageConfigI,
    BaseConsumerConfigI,
    BaseProcessorConfigI,
    BaseProducerConfigI,
    BaseSplitStageConfigI,
    BroadcastLike,
    ConsumerLike,
    ProcessorLike,
    ProducerLike,
    SplitLike,
)
from inference_pipeline.runtime import ThreadTask

type StepId = int
type PortIndex = int


class ProducerCtor[OutT, PConfT: BaseProducerConfigI](Protocol):
    """Callable constructor for producer-like pipeline steps."""

    def __call__(self, *, config: PConfT) -> ProducerLike[OutT]: ...


class ProcessorCtor[InT, OutT, RConfT: BaseProcessorConfigI](Protocol):
    """Callable constructor for processor-like pipeline steps."""

    def __call__(self, *, config: RConfT) -> ProcessorLike[InT, OutT]: ...


class ConsumerCtor[InT, CConfT: BaseConsumerConfigI](Protocol):
    """Callable constructor for consumer-like pipeline steps."""

    def __call__(self, *, config: CConfT) -> ConsumerLike[InT]: ...


class SplitCtor[InT, OutAT, OutBT, SConfT: BaseSplitStageConfigI](Protocol):
    """Callable constructor for split-style pipeline steps."""

    def __call__(self, *, config: SConfT) -> SplitLike[InT, OutAT, OutBT]: ...


class BroadcastCtor[T](Protocol):
    """Callable constructor for broadcast-style pipeline steps."""

    def __call__(self, *, config: BaseBroadcastStageConfigI[T]) -> BroadcastLike[T]: ...


class StepKind(StrEnum):
    """Kinds of pipeline steps supported by the builder."""

    SOURCE = "source"
    PROCESSOR = "processor"
    SINK = "sink"
    SPLIT = "split"
    BROADCAST = "broadcast"


@dataclass(frozen=True, slots=True)
class StreamRef:
    """Opaque reference to one concrete output port in the pipeline graph."""

    step_id: StepId
    port: PortIndex


@dataclass(frozen=True, slots=True)
class StepSpecBase:
    """Common metadata shared by all pipeline step specifications.

    ``name`` is the builder-assigned graph name and is authoritative within the
    pipeline graph, even if the underlying config also carries a ``name``.
    """

    id: StepId
    name: str

    @property
    def kind(self) -> StepKind:
        raise NotImplementedError

    @property
    def output_count(self) -> int:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class SourceStepSpec[OutT, PConfT: BaseProducerConfigI](StepSpecBase):
    """Specification for a source or producer step."""

    ctor: ProducerCtor[OutT, PConfT]
    config: PConfT

    @property
    def kind(self) -> Literal[StepKind.SOURCE]:
        return StepKind.SOURCE

    @property
    def output_count(self) -> Literal[1]:
        return 1


@dataclass(frozen=True, slots=True)
class ProcessorStepSpec[InT, OutT, RConfT: BaseProcessorConfigI](StepSpecBase):
    """Specification for a processor step."""

    ctor: ProcessorCtor[InT, OutT, RConfT]
    config: RConfT

    @property
    def kind(self) -> Literal[StepKind.PROCESSOR]:
        return StepKind.PROCESSOR

    @property
    def output_count(self) -> Literal[1]:
        return 1


@dataclass(frozen=True, slots=True)
class SinkStepSpec[InT, CConfT: BaseConsumerConfigI](StepSpecBase):
    """Specification for a terminal consumer or sink step."""

    ctor: ConsumerCtor[InT, CConfT]
    config: CConfT

    @property
    def kind(self) -> Literal[StepKind.SINK]:
        return StepKind.SINK

    @property
    def output_count(self) -> Literal[0]:
        return 0


@dataclass(frozen=True, slots=True)
class SplitStepSpec[InT, OutAT, OutBT, SConfT: BaseSplitStageConfigI](StepSpecBase):
    """Specification for a two-output split step."""

    ctor: SplitCtor[InT, OutAT, OutBT, SConfT]
    config: SConfT

    @property
    def kind(self) -> Literal[StepKind.SPLIT]:
        return StepKind.SPLIT

    @property
    def output_count(self) -> Literal[2]:
        return 2


@dataclass(frozen=True, slots=True)
class BroadcastStepSpec[T](StepSpecBase):
    """Specification for a broadcast step with ``N`` homogeneous outputs."""

    ctor: BroadcastCtor[T]
    config: BaseBroadcastStageConfigI[T]

    @property
    def kind(self) -> Literal[StepKind.BROADCAST]:
        return StepKind.BROADCAST

    @property
    def output_count(self) -> int:
        return self.config.n_outputs


type StepSpec = (
    SourceStepSpec[Any, Any]
    | ProcessorStepSpec[Any, Any, Any]
    | SinkStepSpec[Any, Any]
    | SplitStepSpec[Any, Any, Any, Any]
    | BroadcastStepSpec[Any]
)


@dataclass(frozen=True, slots=True)
class EdgeSpec:
    """Directed connection from one step output port to one downstream step."""

    from_step: StepId
    from_port: PortIndex
    to_step: StepId


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    """Immutable specification for one reusable pipeline definition."""

    name: str
    steps: tuple[StepSpec, ...]
    edges: tuple[EdgeSpec, ...]

    @property
    def sources(self) -> tuple[SourceStepSpec[Any, Any], ...]:
        return tuple(
            cast(SourceStepSpec[Any, Any], step)  # pyright: ignore[reportUnnecessaryCast]
            for step in self.steps
            if step.kind == StepKind.SOURCE
        )

    @property
    def processors(self) -> tuple[ProcessorStepSpec[Any, Any, Any], ...]:
        return tuple(
            cast(ProcessorStepSpec[Any, Any, Any], step)  # pyright: ignore[reportUnnecessaryCast]
            for step in self.steps
            if step.kind == StepKind.PROCESSOR
        )

    @property
    def sinks(self) -> tuple[SinkStepSpec[Any, Any], ...]:
        return tuple(
            cast(SinkStepSpec[Any, Any], step)  # pyright: ignore[reportUnnecessaryCast]
            for step in self.steps
            if step.kind == StepKind.SINK
        )

    @property
    def splits(self) -> tuple[SplitStepSpec[Any, Any, Any, Any], ...]:
        return tuple(
            cast(SplitStepSpec[Any, Any, Any, Any], step)  # pyright: ignore[reportUnnecessaryCast]
            for step in self.steps
            if step.kind == StepKind.SPLIT
        )

    @property
    def broadcasts(self) -> tuple[BroadcastStepSpec[Any], ...]:
        return tuple(
            cast(BroadcastStepSpec[Any], step)  # pyright: ignore[reportUnnecessaryCast]
            for step in self.steps
            if step.kind == StepKind.BROADCAST
        )

    def step(self, name: str) -> StepSpec:
        for step in self.steps:
            if step.name == name:
                return step
        raise KeyError(name)

    def step_by_id(self, step_id: StepId) -> StepSpec:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(step_id)

    def upstream_of(self, step_id: StepId) -> tuple[EdgeSpec, ...]:
        return tuple(edge for edge in self.edges if edge.to_step == step_id)

    def downstream_of(self, step_id: StepId) -> tuple[EdgeSpec, ...]:
        return tuple(edge for edge in self.edges if edge.from_step == step_id)


class _PipelineRegistry:
    """Mutable graph registry used by the builder while assembling a pipeline."""

    def __init__(self) -> None:
        self._steps: list[StepSpec] = []
        self._steps_by_id: dict[StepId, StepSpec] = {}
        self._step_ids_by_name: dict[str, StepId] = {}
        self._edges: list[EdgeSpec] = []
        self._open_streams: set[StreamRef] = set()
        self._next_step_id: StepId = 1
        self._frozen_spec: PipelineSpec | None = None

    @property
    def steps(self) -> tuple[StepSpec, ...]:
        return tuple(self._steps)

    @property
    def edges(self) -> tuple[EdgeSpec, ...]:
        return tuple(self._edges)

    @property
    def open_streams(self) -> frozenset[StreamRef]:
        return frozenset(self._open_streams)

    def _ensure_mutable(self) -> None:
        if self._frozen_spec is not None:
            raise RuntimeError("Pipeline builder is already frozen.")

    def _next_id(self) -> StepId:
        step_id = self._next_step_id
        self._next_step_id += 1
        return step_id

    def _ensure_unique_name(self, name: str) -> None:
        if not name:
            raise ValueError("Step name must not be empty.")
        if name in self._step_ids_by_name:
            raise ValueError(f"Duplicate step name: {name!r}.")

    def _ensure_open_stream(self, upstream: StreamRef) -> None:
        if upstream not in self._open_streams:
            raise RuntimeError("Stream has already been consumed or does not exist.")

    def _register_step(self, step: StepSpec) -> None:
        self._steps.append(step)
        self._steps_by_id[step.id] = step
        self._step_ids_by_name[step.name] = step.id

    def _connect(self, upstream: StreamRef, step_id: StepId) -> None:
        self._edges.append(EdgeSpec(from_step=upstream.step_id, from_port=upstream.port, to_step=step_id))

    def _consume_stream(self, upstream: StreamRef) -> None:
        self._open_streams.remove(upstream)

    def add_source[OutT, PConfT: BaseProducerConfigI](
        self,
        name: str,
        ctor: ProducerCtor[OutT, PConfT],
        *,
        config: PConfT,
    ) -> StreamRef:
        self._ensure_mutable()
        self._ensure_unique_name(name)

        step_id = self._next_id()
        step = SourceStepSpec(id=step_id, name=name, ctor=ctor, config=config)
        self._register_step(step)

        stream = StreamRef(step_id=step_id, port=0)
        self._open_streams.add(stream)
        return stream

    def add_processor[InT, OutT, RConfT: BaseProcessorConfigI](
        self,
        upstream: StreamRef,
        name: str,
        ctor: ProcessorCtor[InT, OutT, RConfT],
        *,
        config: RConfT,
    ) -> StreamRef:
        self._ensure_mutable()
        self._ensure_unique_name(name)
        self._ensure_open_stream(upstream)

        step_id = self._next_id()
        step = ProcessorStepSpec(id=step_id, name=name, ctor=ctor, config=config)
        self._register_step(step)
        self._connect(upstream, step_id)
        self._consume_stream(upstream)

        stream = StreamRef(step_id=step_id, port=0)
        self._open_streams.add(stream)
        return stream

    def add_sink[InT, CConfT: BaseConsumerConfigI](
        self,
        upstream: StreamRef,
        name: str,
        ctor: ConsumerCtor[InT, CConfT],
        *,
        config: CConfT,
    ) -> StepId:
        self._ensure_mutable()
        self._ensure_unique_name(name)
        self._ensure_open_stream(upstream)

        step_id = self._next_id()
        step = SinkStepSpec(id=step_id, name=name, ctor=ctor, config=config)
        self._register_step(step)
        self._connect(upstream, step_id)
        self._consume_stream(upstream)
        return step_id

    def add_split[InT, OutAT, OutBT, SConfT: BaseSplitStageConfigI](
        self,
        upstream: StreamRef,
        name: str,
        ctor: SplitCtor[InT, OutAT, OutBT, SConfT],
        *,
        config: SConfT,
    ) -> tuple[StreamRef, StreamRef]:
        self._ensure_mutable()
        self._ensure_unique_name(name)
        self._ensure_open_stream(upstream)

        step_id = self._next_id()
        step = SplitStepSpec(id=step_id, name=name, ctor=ctor, config=config)
        self._register_step(step)
        self._connect(upstream, step_id)
        self._consume_stream(upstream)

        outputs = (
            StreamRef(step_id=step_id, port=0),
            StreamRef(step_id=step_id, port=1),
        )
        self._open_streams.update(outputs)
        return outputs

    @overload
    def add_broadcast[T](
        self,
        upstream: StreamRef,
        name: str,
        ctor: BroadcastCtor[T],
        *,
        n_outputs: Literal[2],
        config: BaseBroadcastStageConfigI[T],
    ) -> tuple[StreamRef, StreamRef]: ...

    @overload
    def add_broadcast[T](
        self,
        upstream: StreamRef,
        name: str,
        ctor: BroadcastCtor[T],
        *,
        n_outputs: Literal[3],
        config: BaseBroadcastStageConfigI[T],
    ) -> tuple[StreamRef, StreamRef, StreamRef]: ...

    @overload
    def add_broadcast[T](
        self,
        upstream: StreamRef,
        name: str,
        ctor: BroadcastCtor[T],
        *,
        n_outputs: Literal[4],
        config: BaseBroadcastStageConfigI[T],
    ) -> tuple[StreamRef, StreamRef, StreamRef, StreamRef]: ...

    @overload
    def add_broadcast[T](
        self,
        upstream: StreamRef,
        name: str,
        ctor: BroadcastCtor[T],
        *,
        n_outputs: Literal[5],
        config: BaseBroadcastStageConfigI[T],
    ) -> tuple[StreamRef, StreamRef, StreamRef, StreamRef, StreamRef]: ...

    @overload
    def add_broadcast[T](
        self,
        upstream: StreamRef,
        name: str,
        ctor: BroadcastCtor[T],
        *,
        n_outputs: int,
        config: BaseBroadcastStageConfigI[T],
    ) -> tuple[StreamRef, ...]: ...

    def add_broadcast[T](
        self,
        upstream: StreamRef,
        name: str,
        ctor: BroadcastCtor[T],
        *,
        n_outputs: int,
        config: BaseBroadcastStageConfigI[T],
    ) -> tuple[StreamRef, ...]:
        self._ensure_mutable()
        self._ensure_unique_name(name)
        self._ensure_open_stream(upstream)

        if n_outputs < 1:
            raise ValueError("n_outputs must be >= 1")
        if n_outputs != config.n_outputs:
            raise ValueError("n_outputs must match config.n_outputs")

        step_id = self._next_id()
        step = BroadcastStepSpec(id=step_id, name=name, ctor=ctor, config=config)
        self._register_step(step)
        self._connect(upstream, step_id)
        self._consume_stream(upstream)

        outputs = tuple(StreamRef(step_id=step_id, port=index) for index in range(n_outputs))
        self._open_streams.update(outputs)
        return outputs

    def freeze(self, *, name: str) -> PipelineSpec:
        """Return an immutable spec for the current graph.

        Freezing is idempotent: repeated calls return the same effective graph
        definition.
        """
        if self._frozen_spec is not None:
            return self._frozen_spec
        if not self._steps or not any(step.kind == StepKind.SOURCE for step in self._steps):
            raise ValueError("Pipeline must contain at least one source.")
        if not any(step.kind == StepKind.SINK for step in self._steps):
            raise ValueError("Pipeline must contain at least one sink.")
        if self._open_streams:
            raise ValueError("Pipeline has dangling open branch streams.")

        self._frozen_spec = PipelineSpec(
            name=name,
            steps=tuple(self._steps),
            edges=tuple(self._edges),
        )
        return self._frozen_spec


class PipelineBuilder:
    """Typed fluent entry point for assembling pipeline specifications."""

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("Pipeline name must not be empty.")
        self._name = name
        self._registry = _PipelineRegistry()
        self._frozen_spec: PipelineSpec | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_frozen(self) -> bool:
        return self._frozen_spec is not None

    def _ensure_mutable(self) -> None:
        if self._frozen_spec is not None:
            raise RuntimeError("Pipeline builder is already frozen.")

    def source[OutT, PConfT: BaseProducerConfigI](
        self,
        name: str,
        ctor: ProducerCtor[OutT, PConfT],
        *,
        config: PConfT,
    ) -> StreamBuilder[OutT]:
        self._ensure_mutable()
        stream = self._registry.add_source(name, ctor, config=config)
        return StreamBuilder(self, stream)

    def freeze(self) -> PipelineSpec:
        """Return an immutable spec for the current graph.

        Freezing is idempotent: repeated calls return the same effective graph
        definition.
        """
        if self._frozen_spec is None:
            self._frozen_spec = self._registry.freeze(name=self._name)
        return self._frozen_spec


class StreamBuilder[T]:
    """Typed handle for one open stream endpoint in the builder graph."""

    def __init__(self, pipeline: PipelineBuilder, stream: StreamRef) -> None:
        self._pipeline = pipeline
        self._stream = stream
        self._consumed = False

    @property
    def pipeline(self) -> PipelineBuilder:
        return self._pipeline

    @property
    def stream(self) -> StreamRef:
        return self._stream

    def _consume(self) -> StreamRef:
        self._pipeline._ensure_mutable()  # pyright: ignore[reportPrivateUsage]
        if self._consumed:
            raise RuntimeError("Stream has already been consumed.")
        self._consumed = True
        return self._stream

    def then[OutT, RConfT: BaseProcessorConfigI](
        self,
        name: str,
        ctor: ProcessorCtor[T, OutT, RConfT],
        *,
        config: RConfT,
    ) -> StreamBuilder[OutT]:
        upstream = self._consume()
        stream = self._pipeline._registry.add_processor(  # pyright: ignore[reportPrivateUsage]
            upstream,
            name,
            ctor,
            config=config,
        )
        return StreamBuilder(self._pipeline, stream)

    def sink[CConfT: BaseConsumerConfigI](
        self,
        name: str,
        ctor: ConsumerCtor[T, CConfT],
        *,
        config: CConfT,
    ) -> PipelineBuilder:
        upstream = self._consume()
        self._pipeline._registry.add_sink(upstream, name, ctor, config=config)  # pyright: ignore[reportPrivateUsage]
        return self._pipeline

    def split[OutAT, OutBT, SConfT: BaseSplitStageConfigI](
        self,
        name: str,
        ctor: SplitCtor[T, OutAT, OutBT, SConfT],
        *,
        config: SConfT,
    ) -> tuple[StreamBuilder[OutAT], StreamBuilder[OutBT]]:
        upstream = self._consume()
        left, right = self._pipeline._registry.add_split(  # pyright: ignore[reportPrivateUsage]
            upstream,
            name,
            ctor,
            config=config,
        )
        return StreamBuilder(self._pipeline, left), StreamBuilder(self._pipeline, right)

    @overload
    def broadcast(
        self,
        name: str,
        ctor: BroadcastCtor[T],
        *,
        n_outputs: Literal[2],
        config: BaseBroadcastStageConfigI[T],
    ) -> tuple[StreamBuilder[T], StreamBuilder[T]]: ...

    @overload
    def broadcast(
        self,
        name: str,
        ctor: BroadcastCtor[T],
        *,
        n_outputs: Literal[3],
        config: BaseBroadcastStageConfigI[T],
    ) -> tuple[StreamBuilder[T], StreamBuilder[T], StreamBuilder[T]]: ...

    @overload
    def broadcast(
        self,
        name: str,
        ctor: BroadcastCtor[T],
        *,
        n_outputs: Literal[4],
        config: BaseBroadcastStageConfigI[T],
    ) -> tuple[
        StreamBuilder[T],
        StreamBuilder[T],
        StreamBuilder[T],
        StreamBuilder[T],
    ]: ...

    @overload
    def broadcast(
        self,
        name: str,
        ctor: BroadcastCtor[T],
        *,
        n_outputs: Literal[5],
        config: BaseBroadcastStageConfigI[T],
    ) -> tuple[
        StreamBuilder[T],
        StreamBuilder[T],
        StreamBuilder[T],
        StreamBuilder[T],
        StreamBuilder[T],
    ]: ...

    @overload
    def broadcast(
        self,
        name: str,
        ctor: BroadcastCtor[T],
        *,
        n_outputs: int,
        config: BaseBroadcastStageConfigI[T],
    ) -> tuple[StreamBuilder[T], ...]: ...

    def broadcast(
        self,
        name: str,
        ctor: BroadcastCtor[T],
        *,
        n_outputs: int,
        config: BaseBroadcastStageConfigI[T],
    ) -> tuple[StreamBuilder[T], ...]:
        upstream = self._consume()
        outputs = self._pipeline._registry.add_broadcast(  # pyright: ignore[reportPrivateUsage]
            upstream,
            name,
            ctor,
            n_outputs=n_outputs,
            config=config,
        )
        return tuple(StreamBuilder(self._pipeline, output) for output in outputs)

    def apply[R](self, fragment: Callable[[StreamBuilder[T]], R]) -> R:
        return fragment(self)


class PipelinePhase(StrEnum):
    """Execution phase for one instantiated pipeline."""

    CREATED = "created"
    RUNNING = "running"
    TERMINATED = "terminated"


class PipelineOutcome(StrEnum):
    """Terminal completion outcome for a pipeline run."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPED = "stopped"


type LiveStep = (
    ProducerLike[Any]
    | ProcessorLike[Any, Any]
    | ConsumerLike[Any]
    | SplitLike[Any, Any, Any]
    | BroadcastLike[Any]
)


class Pipeline:
    """Single-use instantiated pipeline built from a reusable specification."""

    def __init__(self, spec: PipelineSpec) -> None:
        self._spec = spec
        self._phase = PipelinePhase.CREATED
        self._outcome: PipelineOutcome | None = None
        self._steps: dict[str, LiveStep] = {}
        self._tasks: dict[str, ThreadTask] = {}
        self._error: BaseException | None = None
        self._stop_event = threading.Event()
        self._stop_requested_at: float | None = None
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._done_event = threading.Event()
        self._materialize()

    @classmethod
    def create(cls, spec: PipelineSpec) -> Self:
        return cls(spec)

    @property
    def spec(self) -> PipelineSpec:
        return self._spec

    @property
    def phase(self) -> PipelinePhase:
        return self._phase

    @property
    def outcome(self) -> PipelineOutcome | None:
        return self._outcome

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    @property
    def stop_requested_at(self) -> float | None:
        return self._stop_requested_at

    @property
    def started_at(self) -> float | None:
        return self._started_at

    @property
    def finished_at(self) -> float | None:
        return self._finished_at

    @property
    def done(self) -> bool:
        return self._done_event.is_set()

    @property
    def done_event(self) -> threading.Event:
        return self._done_event

    @property
    def is_alive(self) -> bool:
        return any(task.is_alive for task in self._tasks.values())

    @property
    def steps(self) -> Mapping[str, LiveStep]:
        return MappingProxyType(self._steps)

    @property
    def tasks(self) -> Mapping[str, ThreadTask]:
        return MappingProxyType(self._tasks)

    @property
    def error(self) -> BaseException | None:
        return self._error

    def step(self, name: str) -> LiveStep:
        return self._steps[name]

    def task(self, name: str) -> ThreadTask:
        return self._tasks[name]

    def _materialize(self) -> None:
        instances_by_id: dict[StepId, LiveStep] = {}
        outputs_by_ref: dict[StreamRef, BufferQ[Any]] = {}

        for step in self._spec.steps:
            instance = self._instantiate_step(step)
            instances_by_id[step.id] = instance
            self._steps[step.name] = instance
            outputs_by_ref.update(self._collect_outputs(step, instance))

        for step in self._spec.steps:
            instance = instances_by_id[step.id]
            task = self._build_task(step, instance, outputs_by_ref)
            self._tasks[step.name] = task

    def _instantiate_step(self, step: StepSpec) -> LiveStep:
        if step.kind == StepKind.SOURCE:
            return step.ctor(config=step.config)
        if step.kind == StepKind.PROCESSOR:
            return step.ctor(config=step.config)
        if step.kind == StepKind.SINK:
            return step.ctor(config=step.config)
        if step.kind == StepKind.SPLIT:
            return step.ctor(config=step.config)
        return step.ctor(config=step.config)

    def _collect_outputs(
        self,
        step: StepSpec,
        instance: LiveStep,
    ) -> dict[StreamRef, BufferQ[Any]]:
        if step.kind in {StepKind.SOURCE, StepKind.PROCESSOR}:
            return {StreamRef(step.id, 0): cast(BufferQ[Any], getattr(instance, "output"))}
        if step.kind == StepKind.SPLIT:
            return {
                StreamRef(step.id, 0): cast(BufferQ[Any], getattr(instance, "output_a")),
                StreamRef(step.id, 1): cast(BufferQ[Any], getattr(instance, "output_b")),
            }
        if step.kind == StepKind.BROADCAST:
            outputs = cast(tuple[BufferQ[Any], ...], getattr(instance, "outputs"))
            return {StreamRef(step.id, index): output for index, output in enumerate(outputs)}
        return {}

    def _build_task(
        self,
        step: StepSpec,
        instance: LiveStep,
        outputs_by_ref: Mapping[StreamRef, BufferQ[Any]],
    ) -> ThreadTask:
        if step.kind == StepKind.SOURCE:
            source = cast(ProducerLike[Any], instance)
            return source.threaded(name=f"{step.name}-worker", stop=self._stop_event)

        upstream = self._spec.upstream_of(step.id)
        if len(upstream) != 1:
            raise ValueError(f"Step {step.name!r} must have exactly one upstream edge.")

        edge = upstream[0]
        input_q = outputs_by_ref[StreamRef(edge.from_step, edge.from_port)]

        if step.kind == StepKind.PROCESSOR:
            processor = cast(ProcessorLike[Any, Any], instance)
            return processor.threaded(
                input_q,
                name=f"{step.name}-worker",
                stop=self._stop_event,
            )
        if step.kind == StepKind.SINK:
            sink = cast(ConsumerLike[Any], instance)
            return sink.threaded(input_q, name=f"{step.name}-worker", stop=self._stop_event)
        if step.kind == StepKind.SPLIT:
            split = cast(SplitLike[Any, Any, Any], instance)
            return split.threaded(
                input_q,
                name=f"{step.name}-worker",
                stop=self._stop_event,
            )
        broadcast = cast(BroadcastLike[Any], instance)
        return broadcast.threaded(
            input_q,
            name=f"{step.name}-worker",
            stop=self._stop_event,
        )

    def _stop_lifecycle_steps(self) -> None:
        for step in self._steps.values():
            running = getattr(step, "running", None)
            stop = getattr(step, "stop", None)
            if isinstance(running, bool) and running and callable(stop):
                stop()

    def _has_lifecycle_steps(self) -> bool:
        return any(callable(getattr(step, "stop", None)) for step in self._steps.values())

    def start(self) -> None:
        """Start worker threads once and transition the pipeline to running."""
        if self._phase is not PipelinePhase.CREATED:
            raise RuntimeError("Pipeline is single-use and has already been started.")
        for task in self._tasks.values():
            task.start()
        self._phase = PipelinePhase.RUNNING
        self._started_at = time.time()

    def request_stop(self) -> None:
        """Signal workers to stop gracefully without waiting for termination."""
        if not self._stop_event.is_set():
            self._stop_requested_at = time.time()
        self._stop_event.set()

    def wait(self, timeout: float | None = None, *, raise_on_error: bool = True) -> None:
        """Wait for worker termination and finalize the terminal outcome."""
        if self._phase is PipelinePhase.CREATED:
            return
        if self._phase is PipelinePhase.TERMINATED:
            if raise_on_error and self._error is not None:
                raise RuntimeError("Pipeline failed.") from self._error
            return

        deadline = (time.monotonic() + timeout) if timeout is not None else None
        worker_error: BaseException | None = None
        pending_lifecycle_stop = self._has_lifecycle_steps()

        if pending_lifecycle_stop:
            self._stop_lifecycle_steps()

        for task in self._tasks.values():
            try:
                remaining = max(deadline - time.monotonic(), 0) if deadline else None
                task.join(timeout=remaining)
                if deadline is not None and time.monotonic() >= deadline:
                    break
            except BaseException as exc:  # noqa: BLE001
                if worker_error is None:
                    worker_error = exc
                    self._stop_event.set()

        if worker_error is not None and not pending_lifecycle_stop:
            self._stop_lifecycle_steps()
            for task in self._tasks.values():
                if task.is_alive:
                    task.join(timeout=timeout)

        if worker_error is not None:
            self._error = worker_error.__cause__ or worker_error
            self._outcome = PipelineOutcome.FAILED
            self._phase = PipelinePhase.TERMINATED
            self._finished_at = time.time()
            self._done_event.set()
            if raise_on_error:
                raise worker_error
            return

        self._outcome = PipelineOutcome.STOPPED if self._stop_event.is_set() else PipelineOutcome.SUCCEEDED
        self._phase = PipelinePhase.TERMINATED
        self._finished_at = time.time()
        self._done_event.set()

    def run(self) -> None:
        """Run the pipeline synchronously from start through termination."""
        self.start()
        try:
            self.wait()
        except BaseException:
            self.raise_for_error()
            raise

    def cancel(self, timeout: float | None = None) -> None:
        """Request stop and block until the run reaches termination."""
        self.request_stop()
        self.wait(timeout=timeout, raise_on_error=False)

    def raise_for_error(self) -> None:
        if self._error is not None:
            raise self._error


class PipelineError(Exception):
    """Base for all PipelineManager domain errors."""


class PipelineRunNotFound(PipelineError):
    """Raised when a run_id does not match any known run."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"Pipeline run {run_id!r} not found.")
        self.run_id = run_id


class PipelineNotStarted(PipelineError):
    """Raised when an operation requires a started run but the run is not started."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"Pipeline run {run_id!r} has not been started.")
        self.run_id = run_id


class PipelineRunActive(PipelineError):
    """Raised when an operation requires a completed run but the run is active."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"Pipeline run {run_id!r} is still active.")
        self.run_id = run_id


class PipelineRunCompleted(PipelineError):
    """Raised when an operation requires an active run but the run is completed."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"Pipeline run {run_id!r} is already completed.")
        self.run_id = run_id


class DuplicateRunId(PipelineError):
    """Raised when an explicit run_id collides with an existing one."""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"Pipeline run {run_id!r} already exists.")
        self.run_id = run_id


class ResourceLimitExceeded(PipelineError):
    """Raised when the max active runs limit is hit."""

    def __init__(self, max_active: int) -> None:
        super().__init__(f"Max active runs ({max_active}) exceeded.")
        self.max_active = max_active


@dataclass(frozen=True, slots=True)
class RunInfo:
    """Serializable lifecycle snapshot for one managed pipeline run."""

    run_id: str
    spec_name: str
    phase: PipelinePhase
    outcome: PipelineOutcome | None
    stop_requested: bool
    done: bool
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    stop_requested_at: float | None = None
    error: str | None = None


@dataclass(slots=True)
class _RunRecord:
    """Internal manager record for one run."""

    pipeline: Pipeline
    created_at: float


class PipelineManager:
    """Manage multiple single-use pipeline runs built from reusable specs."""

    def __init__(
        self,
        max_active_runs: int = 0,
        completed_run_ttl: float | None = None,
    ) -> None:
        self._max_active = max_active_runs
        self._ttl = completed_run_ttl
        self._lock = threading.RLock()
        self._runs: dict[str, _RunRecord] = {}
        self._next_run_number: dict[str, int] = {}
        self._stop_cleanup: threading.Event | None = None
        self._cleanup_thread: threading.Thread | None = None
        if self._ttl is not None:
            self._start_cleanup()

    @property
    def created(self) -> Mapping[str, Pipeline]:
        with self._lock:
            self._reconcile_runs()
            return MappingProxyType(
                {
                    rid: rec.pipeline
                    for rid, rec in self._runs.items()
                    if rec.pipeline.phase is PipelinePhase.CREATED
                }
            )

    @property
    def running(self) -> Mapping[str, Pipeline]:
        with self._lock:
            self._reconcile_runs()
            return MappingProxyType(
                {
                    rid: rec.pipeline
                    for rid, rec in self._runs.items()
                    if rec.pipeline.phase is PipelinePhase.RUNNING
                }
            )

    @property
    def completed(self) -> Mapping[str, Pipeline]:
        with self._lock:
            self._reconcile_runs()
            return MappingProxyType(
                {
                    rid: rec.pipeline
                    for rid, rec in self._runs.items()
                    if rec.pipeline.phase is PipelinePhase.TERMINATED
                }
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generated_run_id(self, spec: PipelineSpec) -> str:
        with self._lock:
            next_number = self._next_run_number.get(spec.name, 0) + 1
            self._next_run_number[spec.name] = next_number
            return f"{spec.name}-run-{next_number}"

    def _ensure_run_id_available(self, run_id: str) -> None:
        if run_id in self._runs:
            raise DuplicateRunId(run_id)

    def _ensure_resource_limit(self) -> None:
        if self._max_active and self.active_count >= self._max_active:
            raise ResourceLimitExceeded(self._max_active)

    def _reconcile_runs(self) -> None:
        for record in self._runs.values():
            pipeline = record.pipeline
            if pipeline.phase is not PipelinePhase.RUNNING:
                continue
            if pipeline.is_alive:
                continue
            pipeline.wait(timeout=0, raise_on_error=False)

    def _run_info_from_record(self, run_id: str, record: _RunRecord) -> RunInfo:
        pipeline = record.pipeline
        return RunInfo(
            run_id=run_id,
            spec_name=pipeline.spec.name,
            phase=pipeline.phase,
            outcome=pipeline.outcome,
            stop_requested=pipeline.stop_requested,
            done=pipeline.done,
            created_at=record.created_at,
            started_at=pipeline.started_at,
            finished_at=pipeline.finished_at,
            stop_requested_at=pipeline.stop_requested_at,
            error=str(pipeline.error) if pipeline.error is not None else None,
        )

    def _start_cleanup(self) -> None:
        if self._ttl is None:
            return
        ttl: float = self._ttl  # type: ignore[assignment]  # guarded by caller
        interval = max(min(ttl / 4, 1.0), 0.05)
        stop_event = threading.Event()
        self._stop_cleanup = stop_event
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            args=(interval, ttl, stop_event),
            name="pipeline-cleanup",
            daemon=True,
        )
        self._cleanup_thread.start()

    def _cleanup_loop(
        self,
        interval: float,
        ttl: float,
        stop_event: threading.Event,
    ) -> None:
        while not stop_event.wait(timeout=interval):
            with self._lock:
                self._reconcile_runs()
            now = time.time()
            with self._lock:
                expired = [
                    rid
                    for rid, rec in self._runs.items()
                    if rec.pipeline.finished_at is not None
                    and rec.pipeline.phase is PipelinePhase.TERMINATED
                    and (now - rec.pipeline.finished_at) >= ttl
                ]
                for rid in expired:
                    del self._runs[rid]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create_run(self, spec: PipelineSpec) -> RunInfo:
        """Create a new run in created phase and return its snapshot."""
        with self._lock:
            self._reconcile_runs()
            self._ensure_resource_limit()
            resolved_run_id = self._generated_run_id(spec)
            self._ensure_run_id_available(resolved_run_id)
        pipeline = Pipeline.create(spec)
        with self._lock:
            self._reconcile_runs()
            self._ensure_run_id_available(resolved_run_id)
            self._ensure_resource_limit()
            record = _RunRecord(
                pipeline=pipeline,
                created_at=time.time(),
            )
            self._runs[resolved_run_id] = record
            return self._run_info_from_record(resolved_run_id, record)

    def start_run(self, run_id: str) -> RunInfo:
        """Start a created run by id and return its updated snapshot."""
        with self._lock:
            self._reconcile_runs()
            record = self._runs.get(run_id)
            if record is None:
                raise PipelineRunNotFound(run_id)
            if record.pipeline.phase is PipelinePhase.RUNNING:
                raise PipelineRunActive(run_id)
            if record.pipeline.phase is PipelinePhase.TERMINATED:
                raise PipelineRunCompleted(run_id)
            record.pipeline.start()
            return self._run_info_from_record(run_id, record)

    def start(self, spec: PipelineSpec) -> RunInfo:
        """Create and start a run from a specification."""
        created = self.create_run(spec)
        return self.start_run(created.run_id)

    def run_by_id(self, run_id: str) -> RunInfo:
        """Run an existing created run synchronously to termination."""
        with self._lock:
            self._reconcile_runs()
            record = self._runs.get(run_id)
            if record is None:
                raise PipelineRunNotFound(run_id)
            if record.pipeline.phase is PipelinePhase.RUNNING:
                raise PipelineRunActive(run_id)
            if record.pipeline.phase is PipelinePhase.TERMINATED:
                raise PipelineRunCompleted(run_id)
            pipeline = record.pipeline
        pipeline.run()
        with self._lock:
            return self._run_info_from_record(run_id, record)

    def run(self, spec: PipelineSpec) -> RunInfo:
        """Create, start, and wait for a run to terminate."""
        created = self.create_run(spec)
        return self.run_by_id(created.run_id)

    def request_stop(self, run_id: str) -> None:
        """Request stop on a running run without waiting for termination."""
        with self._lock:
            self._reconcile_runs()
            record = self._runs.get(run_id)
            if record is None:
                raise PipelineRunNotFound(run_id)
            if record.pipeline.phase is PipelinePhase.TERMINATED:
                raise PipelineRunCompleted(run_id)
            if record.pipeline.phase is PipelinePhase.CREATED:
                raise PipelineNotStarted(run_id)
            record.pipeline.request_stop()

    def wait(
        self,
        run_id: str,
        *,
        timeout: float | None = None,
        raise_on_error: bool = True,
    ) -> RunInfo:
        """Wait for a run to terminate and return its current snapshot."""
        with self._lock:
            self._reconcile_runs()
            record = self._runs.get(run_id)
            if record is None:
                raise PipelineRunNotFound(run_id)
        record.pipeline.wait(timeout=timeout, raise_on_error=raise_on_error)
        with self._lock:
            return self._run_info_from_record(run_id, record)

    def pipeline(self, run_id: str) -> Pipeline:
        with self._lock:
            self._reconcile_runs()
            record = self._runs.get(run_id)
            if record is not None:
                return record.pipeline
            raise PipelineRunNotFound(run_id)

    def discard(self, run_id: str) -> None:
        with self._lock:
            self._reconcile_runs()
            record = self._runs.get(run_id)
            if record is None:
                raise PipelineRunNotFound(run_id)
            if record.pipeline.phase is not PipelinePhase.TERMINATED:
                raise PipelineRunActive(run_id)
            del self._runs[run_id]

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def run_info(self, run_id: str) -> RunInfo:
        """Return a snapshot for one run."""
        with self._lock:
            self._reconcile_runs()
            record = self._runs.get(run_id)
            if record is None:
                raise PipelineRunNotFound(run_id)
            return self._run_info_from_record(run_id, record)

    def list_runs(self) -> list[RunInfo]:
        """Return snapshots for all tracked runs."""
        with self._lock:
            self._reconcile_runs()
            ids = list(self._runs)
            return [self.run_info(rid) for rid in ids]

    @property
    def created_count(self) -> int:
        with self._lock:
            self._reconcile_runs()
            return sum(
                1 for record in self._runs.values() if record.pipeline.phase is PipelinePhase.CREATED
            )

    @property
    def running_count(self) -> int:
        with self._lock:
            self._reconcile_runs()
            return sum(
                1 for record in self._runs.values() if record.pipeline.phase is PipelinePhase.RUNNING
            )

    @property
    def completed_count(self) -> int:
        with self._lock:
            self._reconcile_runs()
            return sum(
                1 for record in self._runs.values() if record.pipeline.phase is PipelinePhase.TERMINATED
            )

    @property
    def active_count(self) -> int:
        with self._lock:
            self._reconcile_runs()
            return sum(
                1 for record in self._runs.values() if record.pipeline.phase is not PipelinePhase.TERMINATED
            )

    @property
    def oldest_running(self) -> RunInfo | None:
        with self._lock:
            self._reconcile_runs()
            oldest_run_id: str | None = None
            oldest_record: _RunRecord | None = None
            for run_id, record in self._runs.items():
                started_at = record.pipeline.started_at
                if record.pipeline.phase is not PipelinePhase.RUNNING or started_at is None:
                    continue
                if oldest_record is None or cast(float | int, started_at) < cast(
                    float | int, oldest_record.pipeline.started_at
                ):
                    oldest_run_id = run_id
                    oldest_record = record
            if oldest_run_id is None or oldest_record is None:
                return None
            return self._run_info_from_record(oldest_run_id, oldest_record)

    def runs_by_phase(self) -> dict[PipelinePhase, int]:
        counts: dict[PipelinePhase, int] = {}
        with self._lock:
            self._reconcile_runs()
            for record in self._runs.values():
                phase = record.pipeline.phase
                counts[phase] = counts.get(phase, 0) + 1
        return counts

    def runs_by_outcome(self) -> dict[PipelineOutcome, int]:
        counts: dict[PipelineOutcome, int] = {}
        with self._lock:
            self._reconcile_runs()
            for record in self._runs.values():
                outcome = record.pipeline.outcome
                if outcome is None:
                    continue
                counts[outcome] = counts.get(outcome, 0) + 1
        return counts

    def health(self) -> dict[str, object]:
        with self._lock:
            self._reconcile_runs()
            oldest = self.oldest_running
            now = time.time()
            return {
                "created_count": self.created_count,
                "running_count": self.running_count,
                "completed_count": self.completed_count,
                "active_count": self.active_count,
                "oldest_running_seconds": ((now - oldest.created_at) if oldest is not None else None),
                "oldest_running_run_id": oldest.run_id if oldest is not None else None,
                "runs_by_phase": self.runs_by_phase(),
                "runs_by_outcome": self.runs_by_outcome(),
            }

    # ------------------------------------------------------------------
    # Bulk lifecycle
    # ------------------------------------------------------------------

    def stop_all(self) -> None:
        """Request stop for all currently running runs."""
        with self._lock:
            self._reconcile_runs()
            for record in self._runs.values():
                if record.pipeline.phase is PipelinePhase.RUNNING:
                    record.pipeline.request_stop()

    def cancel(self, run_id: str, *, timeout: float | None = None) -> RunInfo:
        """Request stop for a run and wait for it to terminate."""
        with self._lock:
            self._reconcile_runs()
            record = self._runs.get(run_id)
            if record is None:
                raise PipelineRunNotFound(run_id)
            if record.pipeline.phase is PipelinePhase.CREATED:
                raise PipelineNotStarted(run_id)
            if record.pipeline.phase is PipelinePhase.TERMINATED:
                return self._run_info_from_record(run_id, record)
            record.pipeline.request_stop()

        record.pipeline.wait(timeout=timeout, raise_on_error=False)
        with self._lock:
            return self._run_info_from_record(run_id, record)

    def discard_completed(self, older_than: float | None = None) -> int:
        with self._lock:
            self._reconcile_runs()
            if older_than is None:
                terminated_ids = [
                    rid for rid, rec in self._runs.items() if rec.pipeline.phase is PipelinePhase.TERMINATED
                ]
                for rid in terminated_ids:
                    del self._runs[rid]
                count = len(terminated_ids)
                return count
            now = time.time()
            expired = [
                rid
                for rid, rec in self._runs.items()
                if rec.pipeline.phase is PipelinePhase.TERMINATED
                and rec.pipeline.finished_at is not None
                and (now - rec.pipeline.finished_at) >= older_than
            ]
            for rid in expired:
                del self._runs[rid]
            return len(expired)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        """Stop managed runs, wait for termination, and clear run records."""
        if self._stop_cleanup is not None:
            self._stop_cleanup.set()
        if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=2.0)
        self.stop_all()

        with self._lock:
            records = [
                (run_id, record)
                for run_id, record in self._runs.items()
                if record.pipeline.phase is PipelinePhase.RUNNING
            ]

        for run_id, record in records:
            try:
                record.pipeline.wait(raise_on_error=False)
            except BaseException as exc:
                msg = f"Warning: pipeline run {run_id!r} did not stop within timeout."
                msg += f" Exception: {exc}"
                warnings.warn(msg)
        with self._lock:
            self._runs.clear()
