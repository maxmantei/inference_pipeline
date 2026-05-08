"""Pipeline builder, specification, and runtime API surfaces."""

from __future__ import annotations

import threading
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
        )  # pyright: ignore[reportUnnecessaryCast]

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
        self._edges.append(
            EdgeSpec(
                from_step=upstream.step_id, from_port=upstream.port, to_step=step_id
            )
        )

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

        outputs = tuple(
            StreamRef(step_id=step_id, port=index) for index in range(n_outputs)
        )
        self._open_streams.update(outputs)
        return outputs

    def freeze(self, *, name: str) -> PipelineSpec:
        """Return an immutable spec for the current graph.

        Freezing is idempotent: repeated calls return the same effective graph
        definition.
        """
        if self._frozen_spec is not None:
            return self._frozen_spec
        if not self._steps or not any(
            step.kind == StepKind.SOURCE for step in self._steps
        ):
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
        self._pipeline._ensure_mutable()
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
        stream = self._pipeline._registry.add_processor(
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
        self._pipeline._registry.add_sink(upstream, name, ctor, config=config)
        return self._pipeline

    def split[OutAT, OutBT, SConfT: BaseSplitStageConfigI](
        self,
        name: str,
        ctor: SplitCtor[T, OutAT, OutBT, SConfT],
        *,
        config: SConfT,
    ) -> tuple[StreamBuilder[OutAT], StreamBuilder[OutBT]]:
        upstream = self._consume()
        left, right = self._pipeline._registry.add_split(
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
        outputs = self._pipeline._registry.add_broadcast(
            upstream,
            name,
            ctor,
            n_outputs=n_outputs,
            config=config,
        )
        return tuple(StreamBuilder(self._pipeline, output) for output in outputs)

    def apply[R](self, fragment: Callable[[StreamBuilder[T]], R]) -> R:
        return fragment(self)


class PipelineState(StrEnum):
    """Lifecycle state for one instantiated ephemeral pipeline."""

    CREATED = "created"
    RUNNING = "running"
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
        self._state = PipelineState.CREATED
        self._steps: dict[str, LiveStep] = {}
        self._tasks: dict[str, ThreadTask] = {}
        self._error: BaseException | None = None
        self._stop_event = threading.Event()
        self._materialize()

    @classmethod
    def create(cls, spec: PipelineSpec) -> Self:
        return cls(spec)

    @property
    def spec(self) -> PipelineSpec:
        return self._spec

    @property
    def state(self) -> PipelineState:
        return self._state

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
            return {
                StreamRef(step.id, 0): cast(BufferQ[Any], getattr(instance, "output"))
            }
        if step.kind == StepKind.SPLIT:
            return {
                StreamRef(step.id, 0): cast(
                    BufferQ[Any], getattr(instance, "output_a")
                ),
                StreamRef(step.id, 1): cast(
                    BufferQ[Any], getattr(instance, "output_b")
                ),
            }
        if step.kind == StepKind.BROADCAST:
            outputs = cast(tuple[BufferQ[Any], ...], getattr(instance, "outputs"))
            return {
                StreamRef(step.id, index): output
                for index, output in enumerate(outputs)
            }
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
            return sink.threaded(
                input_q, name=f"{step.name}-worker", stop=self._stop_event
            )
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
        return any(
            callable(getattr(step, "stop", None)) for step in self._steps.values()
        )

    def start(self) -> None:
        if self._state is not PipelineState.CREATED:
            raise RuntimeError("Pipeline is single-use and has already been started.")
        for task in self._tasks.values():
            task.start()
        self._state = PipelineState.RUNNING

    def stop(self) -> None:
        self._stop_event.set()
        if self._state is PipelineState.RUNNING:
            self._state = PipelineState.STOPPED

    def join(self, timeout: float | None = None) -> None:
        del timeout
        worker_error: BaseException | None = None
        pending_lifecycle_stop = self._has_lifecycle_steps() and self._state in {
            PipelineState.RUNNING,
            PipelineState.STOPPED,
        }

        if pending_lifecycle_stop:
            self._stop_lifecycle_steps()

        for task in self._tasks.values():
            try:
                task.join()
            except BaseException as exc:  # noqa: BLE001
                if worker_error is None:
                    worker_error = exc
                    self._stop_event.set()

        if worker_error is not None and not pending_lifecycle_stop:
            self._stop_lifecycle_steps()
            for task in self._tasks.values():
                if task.is_alive:
                    task.join()

        if worker_error is not None:
            self._state = PipelineState.FAILED
            self._error = worker_error.__cause__ or worker_error
            raise worker_error

        if self._state is PipelineState.RUNNING:
            self._state = (
                PipelineState.STOPPED
                if self._stop_event.is_set()
                else PipelineState.SUCCEEDED
            )

    def run(self) -> None:
        self.start()
        try:
            self.join()
        except BaseException:
            self.raise_for_error()
            raise

    def raise_for_error(self) -> None:
        if self._error is not None:
            raise self._error


class PipelineManager:
    """Manage multiple single-use pipeline runs built from reusable specs."""

    def __init__(self) -> None:
        self._active: dict[str, Pipeline] = {}
        self._completed: dict[str, Pipeline] = {}
        self._next_run_number: dict[str, int] = {}

    @property
    def active(self) -> Mapping[str, Pipeline]:
        return MappingProxyType(self._active)

    @property
    def completed(self) -> Mapping[str, Pipeline]:
        return MappingProxyType(self._completed)

    def _generated_run_id(self, spec: PipelineSpec) -> str:
        next_number = self._next_run_number.get(spec.name, 0) + 1
        self._next_run_number[spec.name] = next_number
        return f"{spec.name}-run-{next_number}"

    def _ensure_run_id_available(self, run_id: str) -> None:
        if run_id in self._active or run_id in self._completed:
            raise ValueError(f"Pipeline run {run_id!r} already exists.")

    def _move_to_completed(self, run_id: str, pipeline: Pipeline) -> Pipeline:
        self._active.pop(run_id, None)
        self._completed[run_id] = pipeline
        return pipeline

    def create(self, spec: PipelineSpec, *, run_id: str | None = None) -> Pipeline:
        resolved_run_id = run_id or self._generated_run_id(spec)
        self._ensure_run_id_available(resolved_run_id)
        pipeline = Pipeline.create(spec)
        self._active[resolved_run_id] = pipeline
        return pipeline

    def start(self, spec: PipelineSpec, *, run_id: str | None = None) -> Pipeline:
        pipeline = self.create(spec, run_id=run_id)
        pipeline.start()
        return pipeline

    def run(self, spec: PipelineSpec, *, run_id: str | None = None) -> Pipeline:
        resolved_run_id = run_id or self._generated_run_id(spec)
        self._ensure_run_id_available(resolved_run_id)
        pipeline = Pipeline.create(spec)
        self._active[resolved_run_id] = pipeline
        try:
            pipeline.run()
        except BaseException:
            self._move_to_completed(resolved_run_id, pipeline)
            raise
        return self._move_to_completed(resolved_run_id, pipeline)

    def stop(self, run_id: str) -> None:
        if run_id in self._completed:
            raise RuntimeError(f"Pipeline run {run_id!r} is already completed.")
        pipeline = self._active.get(run_id)
        if pipeline is None:
            raise KeyError(run_id)
        pipeline.stop()

    def join(self, run_id: str, timeout: float | None = None) -> Pipeline:
        if run_id in self._completed:
            return self._completed[run_id]
        pipeline = self._active.get(run_id)
        if pipeline is None:
            raise KeyError(run_id)
        try:
            pipeline.join(timeout=timeout)
        except BaseException:
            self._move_to_completed(run_id, pipeline)
            raise
        return self._move_to_completed(run_id, pipeline)

    def pipeline(self, run_id: str) -> Pipeline:
        pipeline = self._active.get(run_id)
        if pipeline is not None:
            return pipeline
        pipeline = self._completed.get(run_id)
        if pipeline is not None:
            return pipeline
        raise KeyError(run_id)

    def discard(self, run_id: str) -> None:
        if run_id in self._active:
            raise RuntimeError(f"Pipeline run {run_id!r} is still active.")
        if run_id not in self._completed:
            raise KeyError(run_id)
        del self._completed[run_id]
