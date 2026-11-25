"""
Tool registry and validation engine.
Manages tool registration, parameter type checking, and execution dispatch.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import json

from mobile_agent.tools.schemas import ToolDefinition, SecurityLevel, ALL_CORE_TOOLS


class ToolValidationError(Exception):
    """Raised when tool arguments fail schema validation."""
    pass


class ToolRegistry:
    """Central registry and executor for on-device mobile tools."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable[..., Dict[str, Any]]] = {}

        # Register standard core tools
        for tool_def in ALL_CORE_TOOLS:
            self.register_tool(tool_def)

    def register_tool(self, tool_def: ToolDefinition, handler: Optional[Callable[..., Dict[str, Any]]] = None) -> None:
        """Register a tool definition and optional handler."""
        self._tools[tool_def.name] = tool_def
        if handler:
            self._handlers[tool_def.name] = handler

    def bind_handler(self, tool_name: str, handler: Callable[..., Dict[str, Any]]) -> None:
        """Bind an execution handler to an existing tool definition."""
        if tool_name not in self._tools:
            raise ValueError(f"Cannot bind handler: tool '{tool_name}' is not registered.")
        self._handlers[tool_name] = handler

    def get_tool(self, tool_name: str) -> Optional[ToolDefinition]:
        """Retrieve tool definition by name."""
        return self._tools.get(tool_name)

    def list_tools(self) -> List[ToolDefinition]:
        """Return list of all registered tool definitions."""
        return list(self._tools.values())

    def validate_call(self, tool_name: str, parameters: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate whether the given tool name and parameters conform to the schema.
        Returns (is_valid, error_message).
        """
        tool_def = self.get_tool(tool_name)
        if not tool_def:
            return False, f"Unknown tool: '{tool_name}'."

        schema = tool_def.parameters_schema
        required_fields = schema.get("required", [])
        properties = schema.get("properties", {})

        # Check required fields
        for field_name in required_fields:
            if field_name not in parameters or parameters[field_name] is None:
                return False, f"Missing required parameter '{field_name}' for tool '{tool_name}'."

        # Check property types and enums
        for key, val in parameters.items():
            if key not in properties:
                # Disallow unexpected parameters to prevent hallucinated kwargs
                return False, f"Unexpected parameter '{key}' for tool '{tool_name}'."

            # Allow optional parameters to be None
            if val is None:
                if key in required_fields:
                    return False, f"Required parameter '{key}' cannot be null."
                continue

            prop_spec = properties[key]
            expected_type = prop_spec.get("type")

            if expected_type == "string" and not isinstance(val, str):
                return False, f"Parameter '{key}' must be a string, got {type(val).__name__}."
            elif expected_type == "integer" and not isinstance(val, int):
                return False, f"Parameter '{key}' must be an integer, got {type(val).__name__}."
            elif expected_type == "boolean" and not isinstance(val, bool):
                return False, f"Parameter '{key}' must be a boolean, got {type(val).__name__}."

            if "enum" in prop_spec and val not in prop_spec["enum"]:
                return False, f"Parameter '{key}' value '{val}' not in allowed values: {prop_spec['enum']}."

        return True, None

    def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        user_confirmed: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a tool call with validation and security policy enforcement.
        """
        # Step 1: Validate parameters
        is_valid, error_msg = self.validate_call(tool_name, parameters)
        if not is_valid:
            return {
                "status": "ERROR",
                "error_type": "VALIDATION_ERROR",
                "message": error_msg,
                "tool": tool_name
            }

        tool_def = self.get_tool(tool_name)

        # Step 2: Enforce security confirmation gate
        # Safe read-only inspection (such as read_screen) does not mutate system state
        is_read_screen = tool_name == "automate_ui" and parameters.get("action") == "read_screen"
        if tool_def.security_level == SecurityLevel.REQUIRES_CONFIRMATION and not user_confirmed and not is_read_screen:
            return {
                "status": "CONFIRMATION_REQUIRED",
                "tool": tool_name,
                "parameters": parameters,
                "message": f"Action '{tool_name}' requires explicit user confirmation before execution."
            }

        # Step 3: Dispatch to registered handler
        handler = self._handlers.get(tool_name)
        if not handler:
            return {
                "status": "ERROR",
                "error_type": "HANDLER_NOT_BOUND",
                "message": f"Tool '{tool_name}' has no active runtime handler bound.",
                "tool": tool_name
            }

        try:
            result = handler(**parameters)
            return {
                "status": "SUCCESS",
                "tool": tool_name,
                "result": result
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "error_type": "EXECUTION_EXCEPTION",
                "message": str(e),
                "tool": tool_name
            }

    def format_tools_for_prompt(self) -> str:
        """Generate concise markdown documentation for the system prompt."""
        doc_lines = []
        for tool in self._tools.values():
            req = tool.parameters_schema.get("required", [])
            props = tool.parameters_schema.get("properties", {})
            param_str = ", ".join([f"{k}: {props[k].get('type')}{' (required)' if k in req else ''}" for k in props])
            security_tag = "[REQUIRES USER CONFIRMATION]" if tool.security_level == SecurityLevel.REQUIRES_CONFIRMATION else "[SAFE]"
            doc_lines.append(f"- **`{tool.name}({param_str})`** {security_tag}: {tool.description}")
        return "\n".join(doc_lines)
