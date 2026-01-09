"""
JSON Schema validation for Layer 6 tool call arguments.

Provides lightweight validation without requiring the jsonschema library.
Falls back gracefully if validation cannot be performed.

This validator checks:
- Required properties presence
- Type constraints (string, number, integer, boolean, array, object, null)
- Recursive validation for nested objects

Does NOT check (for simplicity):
- patterns, formats, enum values
- min/max, minLength/maxLength
- additionalProperties restrictions
- anyOf, oneOf, allOf

For production use cases requiring full JSON Schema validation,
consider using the jsonschema library with optional integration.
"""

from typing import Any

from .types import ToolCallValidationError


def validate_type(value: Any, expected_type: str) -> bool:
    """Validate a value against a JSON Schema type.

    Args:
        value: The value to check
        expected_type: JSON Schema type name

    Returns:
        True if value matches the expected type
    """
    type_map: dict[str, type | tuple[type, ...]] = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    expected = type_map.get(expected_type)
    if expected is None:
        # Unknown type, allow it
        return True

    # Special case: in JSON Schema, integers are also valid numbers
    if expected_type == "number" and isinstance(value, bool):
        return False  # Booleans are not numbers in JSON Schema

    return isinstance(value, expected)


def get_type_name(value: Any) -> str:
    """Get JSON Schema type name for a Python value.

    Args:
        value: Any Python value

    Returns:
        JSON Schema type name string
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def validate_against_schema(
    arguments: dict[str, Any],
    schema: dict[str, Any],
    path: str = "",
) -> list[ToolCallValidationError]:
    """Validate arguments against a JSON Schema.

    This is a lightweight validator that checks:
    - Required properties presence
    - Type constraints
    - Basic nested object validation

    Args:
        arguments: The arguments dict to validate
        schema: JSON Schema dict to validate against
        path: Current path in the schema (for error reporting)

    Returns:
        List of ToolCallValidationError for any validation failures
    """
    errors: list[ToolCallValidationError] = []

    if not isinstance(schema, dict):
        return errors

    if not isinstance(arguments, dict):
        errors.append(
            ToolCallValidationError(
                path=path or "(root)",
                message="Expected object, got " + get_type_name(arguments),
                expected_type="object",
                actual_type=get_type_name(arguments),
            )
        )
        return errors

    # Check required properties
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    for req in required:
        if req not in arguments:
            req_path = f"{path}.{req}" if path else req
            errors.append(
                ToolCallValidationError(
                    path=req_path,
                    message=f"Missing required property: {req}",
                    expected_type=None,
                    actual_type="undefined",
                )
            )

    # Check types for provided properties
    for prop_name, prop_value in arguments.items():
        if prop_name not in properties:
            # Allow additional properties by default
            continue

        prop_schema = properties[prop_name]
        if not isinstance(prop_schema, dict):
            continue

        prop_path = f"{path}.{prop_name}" if path else prop_name

        # Check type constraint
        expected_type = prop_schema.get("type")
        if expected_type:
            # Handle union types (type as array)
            if isinstance(expected_type, list):
                if not any(validate_type(prop_value, t) for t in expected_type):
                    errors.append(
                        ToolCallValidationError(
                            path=prop_path,
                            message=f"Type mismatch: expected one of {expected_type}, got {get_type_name(prop_value)}",
                            expected_type=str(expected_type),
                            actual_type=get_type_name(prop_value),
                        )
                    )
            elif not validate_type(prop_value, expected_type):
                errors.append(
                    ToolCallValidationError(
                        path=prop_path,
                        message=f"Type mismatch: expected {expected_type}, got {get_type_name(prop_value)}",
                        expected_type=expected_type,
                        actual_type=get_type_name(prop_value),
                    )
                )

        # Recurse into nested objects
        nested_type = prop_schema.get("type")
        if nested_type == "object" and isinstance(prop_value, dict):
            nested_errors = validate_against_schema(prop_value, prop_schema, prop_path)
            errors.extend(nested_errors)

        # Validate array items
        if nested_type == "array" and isinstance(prop_value, list):
            items_schema = prop_schema.get("items")
            if isinstance(items_schema, dict):
                item_type = items_schema.get("type")
                for i, item in enumerate(prop_value):
                    item_path = f"{prop_path}[{i}]"
                    if item_type:
                        if not validate_type(item, item_type):
                            errors.append(
                                ToolCallValidationError(
                                    path=item_path,
                                    message=f"Array item type mismatch: expected {item_type}, got {get_type_name(item)}",
                                    expected_type=item_type,
                                    actual_type=get_type_name(item),
                                )
                            )
                    # Recurse into object array items
                    if item_type == "object" and isinstance(item, dict):
                        nested_errors = validate_against_schema(item, items_schema, item_path)
                        errors.extend(nested_errors)

    return errors


def find_tool_schema(
    tool_name: str,
    tools: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Find the schema for a tool by name.

    Handles both OpenAI and Anthropic tool formats.

    Args:
        tool_name: Name of the tool to find
        tools: List of tool definitions from the request

    Returns:
        The parameters/input_schema for the tool, or None if not found
    """
    if not tools:
        return None

    for tool in tools:
        # OpenAI format: { type: "function", function: { name, parameters } }
        if tool.get("type") == "function":
            func = tool.get("function", {})
            if func.get("name") == tool_name:
                return func.get("parameters", {})

        # Anthropic format: { name, input_schema }
        if tool.get("name") == tool_name:
            return tool.get("input_schema", {})

    return None
