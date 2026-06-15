"""Sequential pipeline orchestration."""
from __future__ import annotations

import logging
from collections.abc import Callable

from pydantic import BaseModel, Field

from agent_harness.agent.base import AgentResult, BaseAgent
from agent_harness.core.config import HarnessConfig
from agent_harness.hooks import DefaultHooks, resolve_hooks

logger = logging.getLogger(__name__)


class PipelineStep(BaseModel, arbitrary_types_allowed=True):
    """A single step in a pipeline."""
    agent: BaseAgent
    name: str = ""
    condition: Callable[[str], bool] | None = None
    transform: Callable[[str], str] | None = None

    def model_post_init(self, __context: object) -> None:
        if not self.name:
            self.name = getattr(self.agent, 'name', 'unnamed')


class PipelineResult(BaseModel):
    """Result from a pipeline execution."""
    output: str
    step_results: dict[str, AgentResult] = Field(default_factory=dict)
    skipped_steps: list[str] = Field(default_factory=list)


class Pipeline:
    """Sequential agent pipeline.

    Each agent's output becomes the next agent's input.
    Supports conditional execution and input transformation.

    Example:
        pipeline = Pipeline(steps=[
            PipelineStep(agent=researcher),
            PipelineStep(agent=writer, transform=lambda x: f"Write article based on: {x}")],
            hooks=TracingHooks / MyPipelineHooks())
        result = await pipeline.run("Latest AI news")
    """
    def __init__(
        self,
        steps: list[PipelineStep],
        hooks: DefaultHooks | None = None,
        config: HarnessConfig | None = None,
    ) -> None:
        seen: dict[str, int] = {}
        for step in steps:
            if step.name in seen:
                seen[step.name] += 1
                step.name = f"{step.name}_{seen[step.name]}"
            else:
                seen[step.name] = 1
        self.steps = steps
        self.hooks = resolve_hooks(hooks, config)

    async def run(self, input: str) -> PipelineResult:
        current_input = input
        step_results: dict[str, AgentResult] = {}
        skipped: list[str] = []

        if hasattr(self.hooks, "on_pipeline_start"):
            await self.hooks.on_pipeline_start("pipeline")

        for step in self.steps:
            # Check condition
            if step.condition and not step.condition(current_input):
                logger.info("Pipeline: skipping step '%s' (condition not met)", step.name)
                skipped.append(step.name)
                continue

            # Apply input transform
            step_input = step.transform(current_input) if step.transform else current_input

            logger.info("Pipeline: running step '%s'", step.name)
            result = await step.agent.run(step_input)
            step_results[step.name] = result
            current_input = result.output

        if hasattr(self.hooks, "on_pipeline_end"):
            await self.hooks.on_pipeline_end("pipeline")

        return PipelineResult(
            output=current_input,
            step_results=step_results,
            skipped_steps=skipped,
        )
