"""DAG-based orchestration with parallel execution."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from pydantic import BaseModel, Field

from agent_harness.agent.base import AgentResult, BaseAgent
from agent_harness.core.config import HarnessConfig
from agent_harness.core.errors import CyclicDependencyError, OrchestrationError
from agent_harness.hooks import DefaultHooks, resolve_hooks

logger = logging.getLogger(__name__)


class DAGNode(BaseModel, arbitrary_types_allowed=True):
    """A node in the execution DAG."""
    id: str
    agent: BaseAgent
    dependencies: list[str] = Field(default_factory=list)
    input_transform: Callable[[dict[str, AgentResult]], str] | None = None


class DAGResult(BaseModel):
    """Result from DAG orchestration."""
    outputs: dict[str, AgentResult] = Field(default_factory=dict)
    execution_order: list[list[str]] = Field(default_factory=list)  # batches of parallel nodes


class DAGOrchestrator:
    """DAG-based orchestration with parallel branch execution.

    Nodes without dependencies execute in parallel.
    A node executes once all its dependencies complete.

    Example:
        dag = DAGOrchestrator(nodes=[
            DAGNode(id="search", agent=searcher),
            DAGNode(id="analyze", agent=analyzer, dependencies=["search"]),
            DAGNode(id="summarize", agent=summarizer, dependencies=["search"]),
            DAGNode(id="report", agent=reporter, dependencies=["analyze", "summarize"])]
            fail_fast=False)
        result = await dag.run("Research AI safety")
    """

    def __init__(
        self,
        nodes: list[DAGNode],
        fail_fast: bool = False,
        hooks: DefaultHooks | None = None,
        config: HarnessConfig | None = None,
    ) -> None:
        self.nodes = {n.id: n for n in nodes}
        self._fail_fast = fail_fast
        self.hooks = resolve_hooks(hooks, config)
        self._validate_dag()

    def _validate_dag(self) -> None:
        """Validate no cycles exist using topological sort."""
        visited: set[str] = set()
        in_stack: set[str] = set()

        def dfs(node_id: str) -> None:
            if node_id in in_stack:
                raise CyclicDependencyError(f"Cycle detected involving node '{node_id}'")
            if node_id in visited:
                return
            in_stack.add(node_id)
            node = self.nodes.get(node_id)
            if node:
                for dep in node.dependencies:
                    if dep not in self.nodes:
                        raise OrchestrationError(
                            f"Node '{node_id}' depends on unknown node '{dep}'"
                        )
                    dfs(dep)
            in_stack.discard(node_id)
            visited.add(node_id)

        for nid in self.nodes:
            dfs(nid)

    async def run(self, input: str) -> DAGResult:
        """Execute the DAG with maximum parallelism."""
        results: dict[str, AgentResult] = {}
        completed: set[str] = set()
        execution_order: list[list[str]] = []

        if hasattr(self.hooks, "on_dag_start"):
            await self.hooks.on_dag_start("dag")

        while len(completed) < len(self.nodes):
            # Find ready nodes (all deps completed)
            ready = [
                nid for nid, node in self.nodes.items()
                if nid not in completed
                and all(dep in completed for dep in node.dependencies)
            ]

            if not ready:
                raise OrchestrationError("DAG deadlock: no nodes ready but not all completed")

            execution_order.append(ready)
            logger.info("DAG: executing batch %s", ready)

            # Execute ready nodes in parallel
            async def run_node(node_id: str) -> tuple[str, AgentResult]:
                if hasattr(self.hooks, "on_dag_node_start"):
                    await self.hooks.on_dag_node_start(node_id)
                node = self.nodes[node_id]
                if node.input_transform:
                    node_input = node.input_transform(results)
                elif node.dependencies:
                    # Default: concatenate dependency outputs
                    dep_outputs = [results[d].output for d in node.dependencies if d in results]
                    node_input = "\n\n".join(dep_outputs)
                else:
                    node_input = input
                result = await node.agent.run(node_input)
                if hasattr(self.hooks, "on_dag_node_end"):
                    await self.hooks.on_dag_node_end(node_id)
                return node_id, result

            batch_results = await asyncio.gather(
                *(run_node(nid) for nid in ready),
                return_exceptions=True,
            )

            for nid, result_or_exc in zip(ready, batch_results):
                if isinstance(result_or_exc, BaseException):
                    if self._fail_fast:
                        raise OrchestrationError(
                            f"DAG node '{nid}' failed: {result_or_exc}"
                        ) from result_or_exc
                    logger.warning("DAG node '%s' failed: %s", nid, result_or_exc)
                    results[nid] = AgentResult(
                        output=f"Error: {result_or_exc}", messages=[], steps=[],
                    )
                else:
                    _, result = result_or_exc
                    results[nid] = result
                completed.add(nid)

        if hasattr(self.hooks, "on_dag_end"):
            await self.hooks.on_dag_end("dag")

        return DAGResult(outputs=results, execution_order=execution_order)
