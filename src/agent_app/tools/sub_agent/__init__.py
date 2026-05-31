"""Runtime sub-agent delegation tool."""
from agent_app.tools.sub_agent.sub_agent import sub_agent
from agent_harness.tool.base import BaseTool

SUB_AGENT_TOOLS: list[BaseTool] = [sub_agent]

__all__ = ["SUB_AGENT_TOOLS", "sub_agent"]
