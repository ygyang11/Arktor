"""@tool decorator for automatic tool creation from functions.

The decorator inspects function signatures and docstrings to automatically
generate ToolSchema (JSON Schema) — providing a FastAPI-like developer experience.
"""
from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Callable
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

from agent_harness.core.message import ToolOutput
from agent_harness.tool.base import BaseTool, ToolSchema
from agent_harness.utils.async_utils import ensure_async

logger = logging.getLogger(__name__)

# Python type -> JSON Schema type mapping
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(py_type: type) -> dict[str, Any]:
    """Convert a Python type hint to a JSON Schema type descriptor."""
    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_json_schema(non_none[0])
        logger.warning("Union type %s not fully supported, defaulting to string", py_type)
        return {"type": "string"}

    if origin is Literal:
        return {"type": "string", "enum": list(args)}

    if origin is list:
        items = _python_type_to_json_schema(args[0]) if args else {}
        return {"type": "array", "items": items}

    if origin is dict:
        return {"type": "object"}

    json_type = _TYPE_MAP.get(py_type)
    if json_type is None:
        logger.warning("Unknown type %s, defaulting to string", py_type)
        json_type = "string"
    return {"type": json_type}


def _parse_docstring_params(docstring: str | None) -> dict[str, str]:
    """Extract parameter descriptions from Google-style docstring.

    Supports:
        Args:
            param_name: Description text
            param_name (type): Description text
    """
    if not docstring:
        return {}

    params: dict[str, str] = {}
    in_args = False

    for line in docstring.split("\n"):
        stripped = line.strip()

        if stripped.lower().startswith("args:"):
            in_args = True
            continue

        if in_args:
            if stripped and not stripped[0].isspace() and ":" not in stripped[:20]:
                # New section started
                if stripped.lower().startswith(("returns:", "raises:", "yields:", "example")):
                    break

            # Match "param_name: description" or "param_name (type): description"
            match = re.match(r"(\w+)(?:\s*\([^)]*\))?\s*:\s*(.+)", stripped)
            if match:
                params[match.group(1)] = match.group(2).strip()

    return params


def _build_schema_from_function(
    fn: Callable,
    name: str | None,
    description: str | None,
) -> ToolSchema:
    """Build a ToolSchema from function signature and docstring."""
    fn_name = name or fn.__name__
    fn_doc = description or (fn.__doc__ or "").split("\n")[0].strip() or fn_name

    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    param_docs = _parse_docstring_params(fn.__doc__)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        py_type = hints.get(param_name, str)
        # Skip return type
        if param_name == "return":
            continue

        prop = _python_type_to_json_schema(py_type)
        prop["description"] = param_docs.get(param_name, "")

        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(param_name)

        properties[param_name] = prop

    return ToolSchema(
        name=fn_name,
        description=fn_doc,
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
        },
    )


class FunctionTool(BaseTool):
    """A tool created from a function via the @tool decorator."""

    def __init__(
        self,
        fn: Callable[..., Any],
        name: str,
        description: str,
        schema: ToolSchema,
        executor_timeout: float | None = None,
        approval_resource_key: str | None = None,
    ) -> None:
        super().__init__(
            name=name, description=description,
            executor_timeout=executor_timeout,
            approval_resource_key=approval_resource_key,
        )
        self._fn = ensure_async(fn)
        self._schema = schema

    async def execute(self, **kwargs: Any) -> str | ToolOutput:
        result = await self._fn(**kwargs)
        if isinstance(result, ToolOutput):
            return result
        return str(result) if result is not None else ""

    def get_schema(self) -> ToolSchema:
        return self._schema


def tool(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    executor_timeout: float | None = None,
    approval_resource_key: str | None = None,
) -> Any:
    """Decorator to convert a function into a Tool.

    Can be used with or without arguments:

        @tool
        async def my_tool(query: str) -> str:
            ...

        @tool(name="custom_name", description="Custom description")
        async def my_tool(query: str) -> str:
            ...
    """
    def decorator(f: Callable) -> FunctionTool:
        schema = _build_schema_from_function(f, name, description)
        return FunctionTool(
            fn=f,
            name=schema.name,
            description=schema.description,
            schema=schema,
            executor_timeout=executor_timeout,
            approval_resource_key=approval_resource_key,
        )

    if fn is not None:
        # @tool without arguments
        return decorator(fn)

    # @tool(...) with arguments
    return decorator
