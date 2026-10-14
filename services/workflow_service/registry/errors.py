"""
Node validation error classes and utilities for workflow nodes.

This module provides structured, descriptive validation errors for workflow nodes,
fulfilling the requirements of clear diagnostic feedback:
- Which field is invalid
- Expected vs. actual values
- Suggested fixes (including closest field name matching)
- Links to node documentation
"""

import difflib
from typing import Any, Dict, List, Optional, Sequence, Type, Union


NODE_DOCS_BASE_PATH = "docs/design_docs/workflow_service_docs/workflow_builder_guides/nodes"
DEFAULT_WORKFLOW_DOC_URL = "docs/design_docs/workflow_service_docs/workflow_builder_guides/workflow_building_guide.md"

# Mapping of known node type names to their documentation guide files
NODE_DOCS_MAPPING: Dict[str, str] = {
    "ai_answer_engine_scraper": f"{NODE_DOCS_BASE_PATH}/ai_answer_engine_scraper_node_guide.md",
    "code_runner": f"{NODE_DOCS_BASE_PATH}/code_runner_node_guide.md",
    "core_dynamic_nodes": f"{NODE_DOCS_BASE_PATH}/core_dynamic_nodes_guide.md",
    "input_node": f"{NODE_DOCS_BASE_PATH}/core_dynamic_nodes_guide.md",
    "output_node": f"{NODE_DOCS_BASE_PATH}/core_dynamic_nodes_guide.md",
    "crawler_scraper": f"{NODE_DOCS_BASE_PATH}/crawler_scraper_node_guide.md",
    "data_join_data": f"{NODE_DOCS_BASE_PATH}/data_join_node_guide.md",
    "delete_customer_data": f"{NODE_DOCS_BASE_PATH}/delete_customer_data_node_guide.md",
    "edit_document": f"{NODE_DOCS_BASE_PATH}/document_crud_tools_indepth_guide.md",
    "view_documents": f"{NODE_DOCS_BASE_PATH}/document_crud_tools_indepth_guide.md",
    "search_documents": f"{NODE_DOCS_BASE_PATH}/document_crud_tools_indepth_guide.md",
    "list_documents": f"{NODE_DOCS_BASE_PATH}/document_crud_tools_indepth_guide.md",
    "dynamic_router": f"{NODE_DOCS_BASE_PATH}/dynamic_router_node_guide.md",
    "router_node": f"{NODE_DOCS_BASE_PATH}/dynamic_router_node_guide.md",
    "filter_data": f"{NODE_DOCS_BASE_PATH}/filter_node_guide.md",
    "hitl_node": f"{NODE_DOCS_BASE_PATH}/hitl_node_guide.md",
    "if_else_condition": f"{NODE_DOCS_BASE_PATH}/if_else_node_guide.md",
    "linkedin_scraping": f"{NODE_DOCS_BASE_PATH}/linkedin_scraping_node_guide.md",
    "llm": f"{NODE_DOCS_BASE_PATH}/llm_node_guide.md",
    "tool_executor": f"{NODE_DOCS_BASE_PATH}/llm_tools_executor_and_document_crud_guide.md",
    "load_customer_data": f"{NODE_DOCS_BASE_PATH}/load_customer_data_node_guide.md",
    "load_multiple_customer_data": f"{NODE_DOCS_BASE_PATH}/load_multiple_customer_data_node_guide.md",
    "map_list_router_node": f"{NODE_DOCS_BASE_PATH}/map_list_router_node_guide.md",
    "merge_aggregate": f"{NODE_DOCS_BASE_PATH}/merge_aggregate_node_guide.md",
    "prompt_constructor": f"{NODE_DOCS_BASE_PATH}/prompt_constructor_node_guide.md",
    "load_prompt_templates": f"{NODE_DOCS_BASE_PATH}/prompt_loader_node_guide.md",
    "store_customer_data": f"{NODE_DOCS_BASE_PATH}/store_customer_data_node_guide.md",
    "transform_data": f"{NODE_DOCS_BASE_PATH}/transform_node_guide.md",
    "workflow_runner": f"{NODE_DOCS_BASE_PATH}/workflow_runner_node_guide.md",
}


def get_node_doc_url(node_name: Optional[str]) -> str:
    """
    Retrieve the documentation URL or file path for a given node type.

    Args:
        node_name: The registered node type name (e.g. 'llm', 'code_runner').

    Returns:
        Path or URL to the documentation guide for this node.
    """
    if not node_name:
        return DEFAULT_WORKFLOW_DOC_URL
    return NODE_DOCS_MAPPING.get(node_name, DEFAULT_WORKFLOW_DOC_URL)


def suggest_similar_fields(field_name: str, valid_fields: Sequence[str]) -> Optional[str]:
    """
    Suggest the closest matching valid field name using difflib.

    Args:
        field_name: The invalid field name provided by the user.
        valid_fields: List or sequence of valid field names.

    Returns:
        A suggestion string like "Did you mean 'field'?", or None if no close match.
    """
    if not field_name or not valid_fields:
        return None
    matches = difflib.get_close_matches(field_name, list(valid_fields), n=1, cutoff=0.5)
    if matches:
        return f"Did you mean '{matches[0]}'?"
    return None


class NodeValidationError(ValueError):
    """
    Descriptive validation error for workflow nodes.

    Provides rich, structured diagnostics when a node's configuration,
    inputs, or schema mappings fail validation:
    - node_id: Identifier of the node instance in the workflow graph.
    - node_name: Registered type name of the node.
    - field_name: Which specific field failed validation.
    - expected: What type, value, or format was expected.
    - actual: What value or type was actually received.
    - suggested_fix: Practical recommendation on how to resolve the error.
    - doc_url: Link or path to the node's documentation.
    - error_type: Machine-readable error code category.
    """

    def __init__(
        self,
        message: Optional[str] = None,
        node_id: Optional[str] = None,
        node_name: Optional[str] = None,
        field_name: Optional[str] = None,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None,
        suggested_fix: Optional[str] = None,
        doc_url: Optional[str] = None,
        error_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.node_id = node_id
        self.node_name = node_name
        self.field_name = field_name
        self.expected = expected
        self.actual = actual
        self.suggested_fix = suggested_fix
        self.doc_url = doc_url or get_node_doc_url(node_name)
        self.error_type = error_type
        self.details = details or {}
        self.raw_message = message

        constructed_message = self._build_formatted_message(message)
        super().__init__(constructed_message)

    def _build_formatted_message(self, base_message: Optional[str] = None) -> str:
        """Construct the descriptive multi-line error message."""
        target = []
        if self.node_id:
            target.append(f"node '{self.node_id}'")
        if self.node_name:
            target.append(f"type '{self.node_name}'")
        target_str = f" for {' ('.join(target) + (')' if len(target) > 1 else '')}" if target else ""

        lines = [f"Node validation error{target_str}:"]

        if base_message:
            lines.append(f"  Summary: {base_message}")
        if self.field_name is not None:
            lines.append(f"  • Invalid Field: {self.field_name}")
        if self.expected is not None:
            lines.append(f"  • Expected: {self.expected}")
        if self.actual is not None:
            lines.append(f"  • Actual: {self.actual}")
        if self.suggested_fix is not None:
            lines.append(f"  • Suggested Fix: {self.suggested_fix}")
        if self.doc_url is not None:
            lines.append(f"  • Documentation: {self.doc_url}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the validation error into a structured dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "field_name": self.field_name,
            "expected": str(self.expected) if self.expected is not None else None,
            "actual": str(self.actual) if self.actual is not None else None,
            "suggested_fix": self.suggested_fix,
            "doc_url": self.doc_url,
            "error_type": self.error_type,
            "details": self.details,
            "message": str(self),
        }


def format_node_validation_error(
    node_id: Optional[str] = None,
    node_name: Optional[str] = None,
    field_name: Optional[str] = None,
    expected: Optional[Any] = None,
    actual: Optional[Any] = None,
    suggested_fix: Optional[str] = None,
    doc_url: Optional[str] = None,
    error_type: Optional[str] = None,
    message: Optional[str] = None,
) -> NodeValidationError:
    """
    Convenience factory to create a NodeValidationError.
    """
    return NodeValidationError(
        message=message,
        node_id=node_id,
        node_name=node_name,
        field_name=field_name,
        expected=expected,
        actual=actual,
        suggested_fix=suggested_fix,
        doc_url=doc_url,
        error_type=error_type,
    )


def wrap_pydantic_validation_error(
    error: Any,
    node_id: Optional[str] = None,
    node_name: Optional[str] = None,
    schema_cls: Optional[Type] = None,
    context_desc: str = "node configuration",
) -> NodeValidationError:
    """
    Transform a Pydantic ValidationError into a descriptive NodeValidationError.

    Extracts the failing field, expected constraint, received value, and provides
    targeted suggestions along with documentation.
    """
    valid_fields = list(schema_cls.model_fields.keys()) if (schema_cls and hasattr(schema_cls, "model_fields")) else []

    errors_list = getattr(error, "errors", lambda: [])()
    if errors_list:
        first_err = errors_list[0]
        loc = ".".join(str(p) for p in first_err.get("loc", [])) or None
        err_msg = first_err.get("msg", str(error))
        err_type = first_err.get("type", "validation_error")
        input_val = first_err.get("input")

        # Formulate suggested fix
        suggestion = None
        if loc and valid_fields and loc not in valid_fields:
            similar = suggest_similar_fields(loc, valid_fields)
            if similar:
                suggestion = f"{similar} Available fields: {', '.join(valid_fields)}"
            else:
                suggestion = f"Field '{loc}' is not recognized. Valid fields: {', '.join(valid_fields)}"
        elif "missing" in err_type:
            suggestion = f"Field '{loc}' is required for {context_desc}. Please provide a valid value."
        else:
            suggestion = f"Update '{loc}' to satisfy the expected format ({err_msg})."

        return NodeValidationError(
            message=f"Invalid {context_desc}: {err_msg}",
            node_id=node_id,
            node_name=node_name,
            field_name=loc,
            expected=f"Valid value matching schema ({err_type})",
            actual=repr(input_val) if input_val is not None else "missing/empty",
            suggested_fix=suggestion,
            doc_url=get_node_doc_url(node_name),
            error_type=err_type,
            details={"pydantic_errors": errors_list},
        )

    return NodeValidationError(
        message=f"Validation failed for {context_desc}: {str(error)}",
        node_id=node_id,
        node_name=node_name,
        expected="Valid schema payload",
        actual=str(error),
        suggested_fix=f"Review the schema requirements in the documentation.",
        doc_url=get_node_doc_url(node_name),
    )
