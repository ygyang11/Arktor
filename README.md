<p align="center">
  <img src="docs/images/banner.png" alt="Arktor" width="100%">
</p>

<h2 align="center">Arktor: An Agent Harness for the CLI and Your Research</h2>

<p align="center">
  <a href="https://github.com/ygyang11/Arktor/actions/workflows/ci.yml"><img src="https://github.com/ygyang11/Arktor/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.11%2B-blue" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License: MIT"></a>
  <a href="README_ZH.md"><img src="https://img.shields.io/badge/%F0%9F%93%84_%E4%B8%AD%E6%96%87%E6%96%87%E6%A1%A3-click-lightgrey" alt="中文文档"></a>
</p>

Arktor is an extensible agent harness. Use it directly through the **CLI**, or
build and adapt agents with the **Python SDK** for your own projects.

<p align="center">
  <img src="docs/images/cli.png" alt="Arktor terminal interface with command completion." width="100%">
</p>

## Features

### Agent capabilities

- **Agents and workflows.** Use ReAct, Plan-and-Execute, and Conversational agents;
  built-in agent pipelines, DAGs, routers, and teams.

- **Tools.** Built-in tools cover file system, terminal execution, web and
  paper search/fetch, skills, todo lists and document parsing.

- **Delegation and background work.** Delegate tasks to sub-agents with separate
  contexts and configurable tools and models. Run commands or sub-agents in the
  background.

### Harness design

- **Context and continuity.** Load multiple instructions, compact conversations,
  and restore sessions with their working state. Keep project or global memories
  across sessions, and use persistent goals for longer tasks.

- **Execution controls.** Configure approval modes, resource-specific rules,
  tool concurrency, and timeouts. Use an optional Docker sandbox for terminal
  execution.

- **Models and multimodal input.** Use OpenAI-compatible or Anthropic providers,
  custom endpoints, and separately configured sub-models. Attach images and PDFs
  where supported by the provider and model.

- **Observability.** Follow tool activity and token usage, add lifecycle hooks,
  and enable console or JSONL tracing.

## Getting started

### Installation

Install the CLI and built-in tools from source:

```bash
git clone https://github.com/ygyang11/Arktor.git
cd Arktor
conda create -n arktor python=3.11
conda activate arktor
python -m pip install -e ".[cli,app]"
```

<details>
<summary>Optional dependencies</summary>

| Extra | Includes |
| --- | --- |
| `cli` | Terminal interface dependencies. |
| `app` | For tools, including web search, web extraction, and document parsing dependencies. |
| `sandbox` | Docker client for isolated terminal execution; requires a Docker daemon. |
| `dev` | Test, lint, and type-checking tools. |

</details>

### Configuration

Keep your CLI configuration in `~/.arktor/arktor.yaml` to reuse it across projects.
From the repository root:

```bash
mkdir -p ~/.arktor
cp arktor_example.yaml ~/.arktor/arktor.yaml
```

For project-specific settings, place `arktor.yaml` in the working directory;
the CLI gives this file priority.

Set the model and credentials under `llm`. Choose `openai` for OpenAI-compatible
services or `anthropic` for Anthropic:

```yaml
llm:
  provider: openai
  model: "__SET_MODEL__"
  api_key: "__SET_API_KEY__"
  base_url: "__SET_BASE_URL__"
```

See [arktor_example.yaml](arktor_example.yaml) for available settings and
[.env_example](.env_example) for environment variables.

## CLI

### Interactive use

Start an interactive session:

```bash
arktor
```

Describe a task, then follow the agent's activity in the
terminal. Use `@path` to attach files, `!command` to run a shell command, and
`AGENTS.md` or `CLAUDE.md` to supply project instructions.

| Task | Commands |
| --- | --- |
| Inspect context and usage | `/context`, `/usage`, `/compact` |
| Select a model | `/model`, `/provider`, `/effort` |
| Continue work toward an objective | `/goal <objective>` |
| Choose an approval mode | `/permissions` |
| Plan and review changes | `/plan`, `/review`, `/diff` |
| Manage sessions | `/resume`, `/new` |

Use `/help` to browse all interactive commands.

### Non-interactive runs

Run a task and exit with `-p`:

```bash
arktor -p "Inspect this repository and summarize its architecture."
```

Add `--output-format json` for newline-delimited JSON containing step events and
a final result. Run `arktor --help` for startup options.

## Python SDK

### Build an agent

Use the SDK to build agents and workflows for your own projects and experiments.
This example gives an agent a tool for summarizing measurements and uses the
user configuration created above:

```python
import asyncio
import json
from pathlib import Path

from agent_harness import HarnessConfig, ReActAgent, tool


@tool
def summarize_measurements(values: list[float]) -> str:
    """Summarize numeric measurements.

    Args:
        values: Measurements to summarize.
    """
    if not values:
        return "No measurements provided."
    return json.dumps({
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    })


async def main():
    agent = ReActAgent(
        name="analyst",
        tools=[summarize_measurements],
        config=HarnessConfig.load(Path.home() / ".arktor" / "arktor.yaml"),
    )
    result = await agent.run(
        "Use summarize_measurements to summarize [1.8, 2.1, 2.0, 2.3]."
    )
    print(result.output)
    print(f"Tokens used: {result.usage.total_tokens}")


asyncio.run(main())
```

`@tool` derives the tool schema from type hints and the docstring for simple use. The returned `AgentResult` contains the final output, execution steps, and usage.

### Compose and extend

Arktor separates design, capabilities, and terminal interaction
into three clear layers:

| Package | Role |
| --- | --- |
| [agent_harness](src/agent_harness/) | Core abstractions and runtime for defining agent behavior and underlying design. |
| [agent_app](src/agent_app/) | Reusable tools and resources that equip agents for tasks. |
| [agent_cli](src/agent_cli/) | A terminal application that integrates the design and capabilities for interactive and scripted use. |

Build on these layers or adapt them to your own work, from individual
components to how they fit together. Explore the examples and source as a
starting point for your own methods and experiments.

### Examples

The example scripts read `arktor.yaml` from the repository root. To use your
user configuration, copy it there before running an example:

```bash
cp ~/.arktor/arktor.yaml arktor.yaml
python examples/agents/react_agent.py
```

- **Agent patterns:** [ReAct](examples/agents/react_agent.py) and
  [Plan-and-Execute](examples/agents/plan_and_execute.py).
- **Multi-agent workflows:** [pipelines and DAGs](examples/agents/multi_agent_pipeline.py),
  [teams](examples/agents/agent_team.py), and
  [deep research](examples/agents/deep_research.py).
- **Tools and state:** [coding tools](examples/features/coding_demo.py),
  [sessions](examples/features/session_demo.py), and
  [skills](examples/features/skill_demo.py).

## Citation

To cite Arktor, use the entry below or [CITATION.cff](CITATION.cff).

```bibtex
@software{arktor,
  author = {{ygyang11}},
  title  = {Arktor},
  url    = {https://github.com/ygyang11/Arktor}
}
```

## License

Released under the [MIT License](LICENSE).
