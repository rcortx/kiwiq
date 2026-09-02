"""
Structured validation-issue extraction for workflow graph node validation.

These helpers convert the already-structured information raised by Pydantic
ValidationError and jsonschema.ValidationError into a uniform, additive issue
dictionary. They are kept as pure functions so they can be unit-tested in
isolation without a database or registry.

The existing string-based error messages are preserved unchanged; this module
only adds parallel structured metadata for issue #8.
"""

from typing import Any, Dict, List, Optional, Tuple

from jsonschema import ValidationError as JsonSchemaValidationError
from pydantic_core import ValidationError as PydanticValidationError
from workflow_service.registry.node_docs import get_node_docs_url

# Mapping of Pydantic error `type` -> short human label for "expected" value.
_PYDANTIC_TYPE_EXPECTED: Dict[str, str] = {
    "string_type": "string",
    "string_too_short": "string (min length)",
    "string_too_long": "string (max length)",
    "int_type": "integer",
    "int_parsing": "integer",
    "float_type": "number",
    "float_parsing": "number",
    "bool_type": "boolean",
    "list_type": "list",
    "dict_type": "object/dict",
    "missing": "required field",
    "model_attributes_type": "object",
    "enum": "one of the allowed enum values",
    "literal_error": "one of the allowed literal values",
    "url_parsing": "valid URL",
    "email_type": "valid email",
    "extra_forbidden": "no extra fields",
}

# Mapping of Pydantic error `type` -> human suggested-fix text.
_PYDANTIC_TYPE_FIX: Dict[str, str] = {
    "string_type": "Provide a string value.",
    "string_too_short": "Provide a value meeting the minimum length.",
    "string_too_long": "Provide a value within the maximum length.",
    "int_type": "Provide an integer value.",
    "int_parsing": "Provide an integer value (digits only).",
    "float_type": "Provide a numeric value.",
    "float_parsing": "Provide a numeric value.",
    "bool_type": "Provide a boolean (true/false) value.",
    "list_type": "Provide a list value.",
    "dict_type": "Provide an object/dictionary value.",
    "missing": "Provide a value for this required field.",
    "model_attributes_type": "Provide an object/dictionary value.",
    "enum": "Provide one of the allowed enum values.",
    "literal_error": "Provide one of the allowed literal values.",
    "url_parsing": "Provide a valid URL string.",
    "email_type": "Provide a valid email address.",
    "extra_forbidden": "Remove the unknown/extra field.",
}

# Mapping of jsonschema validator keyword -> short human label for "expected".
_JSONSCHEMA_EXPECTED_TYPE: Dict[str, str] = {
    "type": "value of the declared type",
    "required": "required field",
    "enum": "one of the allowed enum values",
    "minimum": "value >= declared minimum",
    "maximum": "value <= declared maximum",
    "minLength": "string meeting min length",
    "maxLength": "string within max length",
    "pattern": "string matching the declared pattern",
    "additionalProperties": "no extra fields",
}

# Mapping of jsonschema validator keyword -> human suggested-fix text.
_JSONSCHEMA_FIX: Dict[str, str] = {
    "type": "Provide a value matching the declared type.",
    "required": "Provide a value for this required field.",
    "enum": "Provide one of the allowed enum values.",
    "minimum": "Provide a value at or above the declared minimum.",
    "maximum": "Provide a value at or below the declared maximum.",
    "minLength": "Provide a string meeting the declared minimum length.",
    "maxLength": "Provide a string within the declared maximum length.",
    "pattern": "Provide a string matching the declared pattern.",
    "additionalProperties": "Remove the unknown/extra field.",
}

# Pydantic error types that indicate a missing field rather than a wrong value.
_MISSING_TYPES = ("missing", "required")


def _field_path_from_loc(loc: Tuple[Any, ...]) -> str:
    """Convert a Pydantic error loc tuple to a dot-separated field path."""
    return ".".join(str(part) for part in loc) if loc else ""


def _safe_actual_label(value: Any) -> Tuple[Any, str]:
    """Return (repr-safe actual value, actual type name)."""
    if value is None:
        return None, "None"
    # Truncate / summarize large containers so the message is digestible.
    if isinstance(value, (dict, list)):
        label = type(value).__name__
        try:
            summary = str(value)
            if len(summary) > 200:
                summary = summary[:200] + "..."
            return summary, label
        except Exception:
            return label, label
    actual_type = type(value).__name__
    try:
        return value, actual_type
    except Exception:
        return repr(value), actual_type


def pydantic_error_to_issue(
    err: Dict[str, Any],
    *,
    node_id: Optional[str] = None,
    node_name: Optional[str] = None,
    message_prefix: str = "Pydantic validation error",
) -> Dict[str, Any]:
    """
    Convert a single Pydantic ValidationError error entry into a structured issue dict.

    Args:
        err: An item from ``ValidationError.errors()`` (contains ``loc``, ``type``,
            ``msg``, ``input``, and optionally ``ctx``).
        node_id: The node instance id the error belongs to (if known).
        node_name: The node type name (if known) used to derive the docs URL.
        message_prefix: Human label for the error source (informational).

    Returns:
        A dict with keys: field, node_id, node_name, expected, actual, actual_type,
        suggested_fix, documentation_url, message.
    """
    error_type = err.get("type", "")
    expected = _PYDANTIC_TYPE_EXPECTED.get(error_type, error_type or "valid value")
    suggested_fix = _PYDANTIC_TYPE_FIX.get(error_type)

    # Use context info when available (e.g. enum allowed values, min/max constraints).
    ctx = err.get("ctx") or {}

    # Constraint-based errors (greater_than_equal, less_than, min_length, ...) carry the
    # actual constraint in `ctx`; derive both `expected` and a precise suggested fix from it.
    field_path = _field_path_from_loc(err.get("loc", ()))
    constraint_hint = _constraint_hint_from_context(error_type, ctx, field_path)
    if constraint_hint:
        expected = constraint_hint["expected"]
        suggested_fix = constraint_hint["suggested_fix"]

    if not suggested_fix:
        suggested_fix = _suggested_fix_from_context(
            error_type, ctx, f"Fix the value for '{field_path}'."
        )

    actual_value, actual_type = _safe_actual_label(err.get("input"))

    message = f"{message_prefix} for '{field_path}': {err.get('msg', '')}".strip()

    return _build_issue(
        field=_field_path_from_loc(err.get("loc", ())),
        node_id=node_id,
        node_name=node_name,
        expected=expected,
        actual=actual_value,
        actual_type=actual_type,
        suggested_fix=suggested_fix,
        message=message,
    )


def jsonschema_error_to_issue(
    err: JsonSchemaValidationError,
    *,
    node_id: Optional[str] = None,
    node_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convert a jsonschema.ValidationError into a structured issue dict.

    Args:
        err: The jsonschema ValidationError (has ``path``, ``message``, ``validator``,
            ``validator_value``, ``instance``).
        node_id: The node instance id the error belongs to (if known).
        node_name: The node type name (if known) used to derive the docs URL.

    Returns:
        A dict with keys: field, node_id, node_name, expected, actual, actual_type,
        suggested_fix, documentation_url, message.
    """
    field = ".".join(map(str, err.path)) if err.path else ""
    validator = err.validator or ""
    actual_value, actual_type = _safe_actual_label(err.instance)

    expected = _JSONSCHEMA_EXPECTED_TYPE.get(validator)
    if not expected:
        expected = (
            str(err.validator_value)
            if err.validator_value is not None
            else "valid value"
        )

    suggested_fix = _JSONSCHEMA_FIX.get(validator)
    if not suggested_fix:
        suggested_fix = (
            f"Fix the value for '{field or 'config'}' to satisfy the schema."
        )

    message = f"JSONSchema validation failed for '{field or 'config'}': {err.message}"

    return _build_issue(
        field=field,
        node_id=node_id,
        node_name=node_name,
        expected=expected,
        actual=actual_value,
        actual_type=actual_type,
        suggested_fix=suggested_fix,
        message=message,
    )


def generic_error_to_issue(
    error_message: str,
    *,
    node_id: Optional[str] = None,
    node_name: Optional[str] = None,
    field: Optional[str] = None,
    expected: Optional[str] = None,
    actual: Optional[str] = None,
    actual_type: Optional[str] = None,
    suggested_fix: Optional[str] = None,
    message_label: str = "Validation error",
) -> Dict[str, Any]:
    """
    Build a structured issue dict for a non-Pydantic/JSON-Schema validation error
    (e.g. graph-level structural errors or unexpected errors). Preserves the
    original error string while adding best-effort structured metadata.
    """
    return _build_issue(
        field=field or "",
        node_id=node_id,
        node_name=node_name,
        expected=expected,
        actual=actual,
        actual_type=actual_type,
        suggested_fix=suggested_fix,
        message=f"{message_label}: {error_message}",
    )


def _build_issue(
    *,
    field: str,
    node_id: Optional[str],
    node_name: Optional[str],
    expected: Any,
    actual: Any,
    actual_type: Optional[str],
    suggested_fix: Optional[str],
    message: str,
) -> Dict[str, Any]:
    """Assemble a uniform issue dict and attach the node documentation URL."""
    issue: Dict[str, Any] = {
        "field": field,
        "node_id": node_id or "",
        "node_name": node_name or "",
        "expected": expected,
        "actual": actual,
        "actual_type": actual_type or "",
        "suggested_fix": suggested_fix or "",
        "documentation_url": get_node_docs_url(node_name),
        "message": message,
    }
    return issue


# Pydantic error `type` values that represent a bound/comparison constraint. For these the
# actual boundary live in ctx under Pydantic's constraint keys (ge/gt/le/lt/min_length/...),
# on which we can build a precise "expected" and suggested fix generically.
_CONSTRAINT_ERROR_HINTS = {
    "greater_than_equal": ("ge", "integer/number >= {value}"),
    "greater_than": ("gt", "integer/number > {value}"),
    "less_than_equal": ("le", "integer/number <= {value}"),
    "less_than": ("lt", "integer/number < {value}"),
    "min_length": ("min_length", "string with at least {value} characters"),
    "max_length": ("max_length", "string with at most {value} characters"),
}


def _constraint_hint_from_context(
    error_type: str, ctx: Dict[str, Any], field_path: str
) -> Optional[Dict[str, str]]:
    """
    Build a precise ``expected``/``suggested_fix`` pair for constraint-style Pydantic errors.

    Returns ``None`` when the error type is not a recognised constraint or when the
    boundary value is missing from the context.
    """
    hint = _CONSTRAINT_ERROR_HINTS.get(error_type)
    if not hint:
        return None
    key, expected_template = hint
    value = ctx.get(key)
    if value is None:
        return None
    expected = expected_template.format(value=value)
    fix_template = _CONSTRAINT_ERROR_FIX_TEMPLATES[error_type]
    suggested_fix = fix_template.format(field=field_path or "this field", value=value)
    return {"expected": expected, "suggested_fix": suggested_fix}


_CONSTRAINT_ERROR_FIX_TEMPLATES = {
    "greater_than_equal": "Set {field} to {value} or greater.",
    "greater_than": "Set {field} to a value greater than {value}.",
    "less_than_equal": "Set {field} to {value} or less.",
    "less_than": "Set {field} to a value less than {value}.",
    "min_length": "Set {field} to at least {value} characters.",
    "max_length": "Set {field} to at most {value} characters.",
}


def _suggested_fix_from_context(
    error_type: str, ctx: Dict[str, Any], fallback: str
) -> str:
    """Derive a suggested fix from Pydantic error context fields when available."""
    for key, template in (
        ("expected", "Provide one of the allowed values: {value}."),
        ("allowed", "Provide one of the allowed values: {value}."),
        ("enum_values", "Provide one of the allowed values: {value}."),
        ("min_length", "Provide a value with at least {value} characters."),
        ("max_length", "Provide a value with at most {value} characters."),
        ("min_value", "Provide a value >= {value}."),
        ("max_value", "Provide a value <= {value}."),
    ):
        val = ctx.get(key)
        if val is not None:
            return template.format(value=val)
    return fallback


def harness_errors_to_issues(
    errors: Dict[str, List[str]],
    node_id: Optional[str] = None,
    node_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convert a ``Dict[str, List[str]]`` of existing error strings into a flat list
    of structured issues, grouping generic messages under the aware node context.

    This is a compatibility helper used when only string errors are available
    (e.g. graph-level / GraphBuilder failures).
    """
    issues: List[Dict[str, Any]] = []
    for scope, msgs in errors.items():
        for msg in msgs:
            issues.append(
                generic_error_to_issue(
                    msg,
                    node_id=node_id or scope,
                    node_name=node_name,
                    field=scope,
                    message_label="Validation error",
                )
            )
    return issues
