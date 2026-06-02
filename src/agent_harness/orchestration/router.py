"""Agent router for intent-based dispatching."""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import BaseModel

from agent_harness.agent.base import AgentResult, BaseAgent
from agent_harness.core.config import HarnessConfig
from agent_harness.core.errors import OrchestrationError
from agent_harness.core.message import Message
from agent_harness.hooks import DefaultHooks, resolve_hooks
from agent_harness.utils.json_utils import parse_json_lenient

if TYPE_CHECKING:
    from agent_harness.llm.base import BaseLLM

logger = logging.getLogger(__name__)

ROUTER_PROMPTS: dict[str, str] = {
    "route.select": (
        "## Role\n"
        "You are a high-precision routing coordinator for a multi-agent system.\n\n"
        "## Inputs\n"
        "Available agents:\n"
        "{route_descriptions}\n\n"
        "User request:\n"
        "{user_request}\n\n"
        "## Objectives\n"
        "- Select the most appropriate agent set for this request.\n"
        "- Maximize relevance while minimizing unnecessary parallel dispatch.\n"
        "- Avoid selecting agents that add no clear value.\n\n"
        "## Rules\n"
        "- Choose agent names exactly as listed in available agents.\n"
        "- Select one agent when one clearly dominates.\n"
        "- Select multiple agents only when complementary expertise is required.\n"
        "- If none are suitable, return an empty list.\n"
        "- Do not include explanations or extra keys.\n\n"
        "## Output Format\n"
        "Return ONLY JSON:\n"
        "{{\"agents\": [\"agent_name_1\", \"agent_name_2\"]}}"
    ),
    "route.synthesize": (
        "## Role\n"
        "You are a synthesis coordinator combining multiple agent outputs.\n\n"
        "## Inputs\n"
        "Original user request:\n"
        "{user_request}\n\n"
        "Agent outputs:\n"
        "{agent_outputs}\n\n"
        "## Objectives\n"
        "- Produce one coherent, decision-ready response.\n"
        "- Preserve high-value insights and remove redundancy.\n"
        "- Resolve conflicts with explicit, practical reasoning.\n\n"
        "## Rules\n"
        "- Prioritize correctness, completeness, and actionability.\n"
        "- Keep the final answer concise but sufficiently detailed.\n"
        "- Avoid mentioning internal routing mechanics.\n\n"
        "## Output Format\n"
        "Return a single final answer in plain text."
    ),
}


class Route(BaseModel, arbitrary_types_allowed=True):
    """A routing rule."""
    agent: BaseAgent
    name: str = ""
    condition: Callable[[str], bool] | str
    description: str = ""
    _compiled_re: re.Pattern[str] | None = None

    def model_post_init(self, __context: object) -> None:
        if not self.name:
            self.name = getattr(self.agent, 'name', 'unnamed')
        if isinstance(self.condition, str):
            self._compiled_re = re.compile(self.condition, re.IGNORECASE)


class AgentRouter:
    """Routes requests to different agents based on rules and optional LLM routing.

    Routing strategies:
    - Callable condition: function(input) -> bool
    - String condition: regex pattern match
    - LLM-based: Let an LLM choose the best route(s) from descriptions

    Example:
        router = AgentRouter(
            routes=[
                Route(agent=coder, condition=lambda x: "code" in x.lower(), name="coder"),
                Route(agent=researcher, condition=r"search|find|look up", name="researcher"),
            ],
            fallback=general_agent,
            llm=openai_provider,       # Enable LLM routing
            llm_first=False,           # Try rules first, LLM as fallback
        )
        result = await router.run("Write a Python function")
    """

    def __init__(
        self,
        routes: list[Route],
        fallback: BaseAgent | None = None,
        llm: BaseLLM | None = None,
        llm_first: bool = False,
        hooks: DefaultHooks | None = None,
        config: HarnessConfig | None = None,
    ) -> None:
        if llm_first and llm is None:
            raise ValueError("llm is required when llm_first=True")
        self.routes = routes
        self.fallback = fallback
        self.llm = llm
        self.llm_first = llm_first
        self.hooks = resolve_hooks(hooks, config)
        self.last_routed_to: list[str] = []

    async def run(self, input: str) -> AgentResult:
        """Route the input to the matching agent."""
        self.last_routed_to = []
        if self.llm_first and self.llm:
            # LLM routing first
            routes = await self._llm_select_routes(input)
            if routes:
                return await self._execute_routes(input, routes)

            # Fall back to rule matching
            for route in self.routes:
                if self._matches(route, input):
                    logger.info("Router: matched route '%s' (rule fallback)", route.name)
                    self.last_routed_to = [route.name]
                    return await self._run_agent(route.agent, input)
        else:
            # Rule matching first
            for route in self.routes:
                if self._matches(route, input):
                    logger.info("Router: matched route '%s'", route.name)
                    self.last_routed_to = [route.name]
                    return await self._run_agent(route.agent, input)

            # LLM routing fallback
            if self.llm:
                routes = await self._llm_select_routes(input)
                if routes:
                    return await self._execute_routes(input, routes)

        if self.fallback:
            logger.info("Router: no match, using fallback")
            self.last_routed_to = [getattr(self.fallback, "name", "fallback")]
            return await self._run_agent(self.fallback, input)

        from agent_harness.core.errors import OrchestrationError
        raise OrchestrationError(
            f"No route matched and no fallback configured. Input: {input[:100]}"
        )

    async def _llm_select_routes(self, input: str) -> list[Route]:
        """Use LLM to select the best route(s). Returns matched Route objects."""
        if self.llm is None:
            raise OrchestrationError(
                "LLM required for route selection but not configured"
            )

        route_descriptions = "\n".join(
            f"- {r.name}: {r.description}" if r.description else f"- {r.name}"
            for r in self.routes
        )

        prompt = ROUTER_PROMPTS["route.select"].format(
            route_descriptions=route_descriptions,
            user_request=input,
        )

        try:
            response = await self.llm.generate(
                [Message.system(prompt)],
                tools=None,
            )
            content = response.message.content or ""
            data = parse_json_lenient(content)

            if isinstance(data, dict):
                agent_names = data.get("agents", [])
            elif isinstance(data, list):
                agent_names = data
            else:
                return []

            if not agent_names:
                return []

            # Map names to routes
            route_map = {r.name: r for r in self.routes}
            matched_routes = [route_map[name] for name in agent_names if name in route_map]

            return matched_routes

        except Exception as e:
            logger.warning("Router: LLM route selection failed: %s", e)
            return []

    async def _execute_routes(self, input: str, routes: list[Route]) -> AgentResult:
        """Execute the selected agents and synthesize results."""
        self.last_routed_to = [r.name for r in routes]
        if len(routes) == 1:
            logger.info("Router: LLM selected route '%s'", routes[0].name)
            return await self._run_agent(routes[0].agent, input)

        # Multiple agents — run in parallel and synthesize
        logger.info("Router: LLM selected %d routes, running in parallel", len(routes))
        tasks = [self._run_agent(r.agent, input) for r in routes]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successful results
        outputs: list[str] = []
        first_result: AgentResult | None = None
        for route, result in zip(routes, results):
            if isinstance(result, BaseException):
                logger.warning("Router: agent '%s' failed: %s", route.name, result)
                continue
            outputs.append(f"[{route.name}]: {result.output}")
            if first_result is None:
                first_result = result

        if not outputs:
            return AgentResult(output="All routes failed", messages=[], steps=[])

        if len(outputs) == 1 and first_result is not None:
            return first_result

        if self.llm is None:
            raise OrchestrationError(
                "LLM required for multi-route synthesis but not configured"
            )
        synthesis_prompt = ROUTER_PROMPTS["route.synthesize"].format(
            user_request=input,
            agent_outputs="\n\n".join(outputs),
        )
        synthesis_response = await self.llm.generate(
            [Message.user(synthesis_prompt)], tools=None,
        )
        return AgentResult(
            output=synthesis_response.message.content or outputs[0],
            messages=[],
            steps=[],
            usage=synthesis_response.usage,
        )

    @staticmethod
    async def _run_agent(agent: BaseAgent, input: str) -> AgentResult:
        """Run an agent."""
        return await agent.run(input)

    @staticmethod
    def _matches(route: Route, input: str) -> bool:
        if callable(route.condition):
            return route.condition(input)
        if route._compiled_re is not None:
            return bool(route._compiled_re.search(input))
        return False
