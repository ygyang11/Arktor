<p align="center">
  <img src="docs/images/banner.png" alt="Arktor" width="100%">
</p>

<p align="center">
  <b>可作为CLI直接使用，或者SDK 二次构建的 Agent 框架。</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License MIT">
  <img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python 3.11+">
  <a href="README.md"><img src="https://img.shields.io/badge/📄_English-click-lightgrey" alt="English"></a>
</p>

Arktor 是一个轻量、可改造的 Agent Harness，覆盖 Agent，以及围绕它的工具、上下文与编排等。
同一个包提供两种使用方式：

- **`arktor`** —— 终端中的交互式 Agent：流式输出、工具调用、Plan 模式、会话持久化、多模态输入。
- **`arktor-sdk`** —— 其底层的 Python SDK：Agent、工具、编排、记忆、Tracing 与 Hook，
  每一处都为继承与替换而设计。

---

## 特性

- **内置工具开箱即用** —— 文件系统与可沙箱化的终端、网页搜索与抓取、子 Agent、todo list、跨会话记忆
  与 Skill等，还有面向科研的工具 —— arXiv 与 Semantic Scholar 论文检索，以及把 PDF / 扫描件 /
  图片解析为干净 markdown 的 文档解析工具。图片与 PDF 全程作为一等 attachment 流转。
- **内置可观测性** —— 每一次 LLM 调用、工具执行与推理步骤都是一个 trace span，
  导出至控制台与 JSONL，并通过丰富的 Hook 对外暴露（CLI 的实时进度即由此驱动）。
- **Agent 与编排** —— ReAct、Plan-and-Execute、Conversational 三类 Agent，
  可组合为 Pipeline、DAG、Router 或 Team。
- **处处可改** —— 不绑定供应商的 LLM 层（内置重试与降级）、三层记忆（自动压缩）、
  审批策略、可选的 Docker 沙箱，以及预写调优的内置 Prompt —— 清晰可扩展。

---

## 安装

```bash
git clone https://github.com/ygyang11/Arktor.git
cd Arktor

# 创建虚拟环境（任选其一）
conda create -n arktor python=3.11 && conda activate arktor
# 或：python -m venv .venv && source .venv/bin/activate
# 或：uv venv && source .venv/bin/activate

pip install -e ".[cli,app]"     # arktor CLI + 内置工具
```

仅作为 SDK 使用（不含 REPL）：

```bash
pip install -e ".[dev,app]"
```

若需 Docker 隔离的工具执行，追加 `sandbox` 扩展。

### 配置

仅需填写三项：模型、API key 与 base URL。可在项目目录放置 `arktor.yaml`
（复制 [`arktor_example.yaml`](arktor_example.yaml)），或首次运行 `arktor`，
它会自动在 `~/.arktor/arktor.yaml` 生成一份模板供你编辑：

```yaml
llm:
  provider: openai           # 或：anthropic
  model: gpt-5.5
  api_key: sk-...
  base_url: ...
```

其余配置项均有合理默认且可自由定制；常用字段亦可用 `ARKTOR_` 前缀的环境变量覆盖（见 [`.env_example`](.env_example)）。

---

## 快速开始

### 命令行

<p align="center">
  <img src="docs/images/cli.png" alt="arktor CLI" width="100%">
</p>

```bash
arktor
```

`arktor` 会在当前项目目录启动一个交互式 Agent。以 `@path` 引入文件，以 `!cmd` 执行 shell，
以 `/` 调用命令。会话自动持久化，可随时 `/resume` 续接；审批模式可在
**Ask · Auto · Yolo** 之间切换。

斜杠命令覆盖完整工作流，以下为常用部分：

- **规划与审查** —— `/plan`（只读规划）、`/review`（审查当前改动）、`/diff`、
  `/init`（生成 `AGENTS.md`）
- **会话** —— `/resume`、`/new`、`/compact`（压缩上下文）、`/export`、`/status`、`/context`
- **模型与运行时** —— `/model`、`/effort`、`/provider`、`/permissions`、`/skills`、`/tasks`

`/help` 可列出全部命令。

### SDK

十行代码，一个会调用工具的 Agent：

```python
import asyncio
from agent_harness import ReActAgent, tool, HarnessConfig

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression.

    Args:
        expression: A Python math expression like '2 + 3 * 4'.
    """
    return str(eval(expression))

async def main():
    agent = ReActAgent(
        name="assistant",
        tools=[calculate],
        config=HarnessConfig.load("arktor.yaml"),
    )
    result = await agent.run("What is (42 * 37 + 15) / 3?")
    print(result.output, "·", result.step_count, "steps")

asyncio.run(main())
```

`@tool` 直接从类型注解与 docstring 生成 JSON Schema。
核心类型从 `agent_harness` 导入，内置工具位于 `agent_app`
（如 `from agent_app.tools import WEB_TOOLS, FILESYSTEM_TOOLS`）。

---

## 示例

[`examples/`](examples/) 下提供可直接运行的脚本：

- **agents/** —— [`react_agent`](examples/agents/react_agent.py)、
  [`plan_and_execute`](examples/agents/plan_and_execute.py)、
  [`multi_agent_pipeline`](examples/agents/multi_agent_pipeline.py)、[`agent_team`](examples/agents/agent_team.py)、
  [`deep_research`](examples/agents/deep_research.py)。
- **features/** —— [`coding_demo`](examples/features/coding_demo.py)、
  [`session_demo`](examples/features/session_demo.py)、
  [`skill_demo`](examples/features/skill_demo.py)。

```bash
python examples/agents/react_agent.py        # ReAct 循环 + 自定义工具
```

---

## 参与贡献

欢迎贡献。请先开 issue 讨论方案，并保持与现有代码风格一致。

本项目以 [MIT License](LICENSE) 开源。
