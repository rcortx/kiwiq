"""
Unit tests for descriptive node validation errors.

Tests the enhancements introduced for Issue #8:
- Clear identification of invalid fields
- Expected vs. actual values
- Suggested fixes (including fuzzy matching)
- Documentation links
- Serialization to dictionary
- Backwards compatibility with ValueError
"""

import unittest
from typing import ClassVar, Dict, Any, Type, Optional
from pydantic import Field, BaseModel
from pydantic.json_schema import SkipJsonSchema

from workflow_service.registry.errors import (
    NodeValidationError,
    get_node_doc_url,
    suggest_similar_fields,
    format_node_validation_error,
    wrap_pydantic_validation_error,
    NODE_DOCS_MAPPING,
    DEFAULT_WORKFLOW_DOC_URL,
)
from workflow_service.registry.nodes.core.base import BaseNode
from workflow_service.registry.schemas.base import BaseSchema
from workflow_service.graph.graph import GraphSchema, EdgeSchema, NodeConfig
from workflow_service.graph.builder import GraphBuilder
from workflow_service.registry.registry import DBRegistry
from workflow_service.config.constants import INPUT_NODE_NAME, OUTPUT_NODE_NAME


# Test schemas and node for validation testing
class SampleInputSchema(BaseSchema):
    text: str = Field(description="Input text")
    count: int = Field(default=1, description="Count multiplier")


class SampleOutputSchema(BaseSchema):
    result: str = Field(description="Resulting text")


class SampleConfigSchema(BaseSchema):
    model: str = Field(description="Model name")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    secret_key: SkipJsonSchema[str] = Field(default="hidden", description="Non-editable internal key")


class MockValidationNode(BaseNode):
    node_name: ClassVar[str] = "llm"
    node_version: ClassVar[str] = "1.0.0"

    input_schema_cls: ClassVar[Type[SampleInputSchema]] = SampleInputSchema
    output_schema_cls: ClassVar[Type[SampleOutputSchema]] = SampleOutputSchema
    config_schema_cls: ClassVar[Type[SampleConfigSchema]] = SampleConfigSchema

    async def process(self, input_data: SampleInputSchema, config: Dict[str, Any], *args: Any, **kwargs: Any) -> SampleOutputSchema:
        return SampleOutputSchema(result=input_data.text * input_data.count)


class TestNodeValidationErrors(unittest.TestCase):
    """Test suite for NodeValidationError and associated diagnostic utilities."""

    def test_node_validation_error_is_value_error(self):
        """NodeValidationError must inherit from ValueError for backwards compatibility."""
        err = NodeValidationError("Validation failed")
        self.assertIsInstance(err, ValueError)

    def test_node_validation_error_attributes_and_formatting(self):
        """Error must contain field, expected, actual, fix, and doc link in string and dict."""
        err = NodeValidationError(
            message="Invalid temperature setting",
            node_id="llm_node_1",
            node_name="llm",
            field_name="temperature",
            expected="float between 0.0 and 2.0",
            actual=10.0,
            suggested_fix="Set temperature within [0.0, 2.0] range",
            error_type="value_out_of_range",
        )

        msg = str(err)
        self.assertIn("node 'llm_node_1'", msg)
        self.assertIn("type 'llm'", msg)
        self.assertIn("• Invalid Field: temperature", msg)
        self.assertIn("• Expected: float between 0.0 and 2.0", msg)
        self.assertIn("• Actual: 10.0", msg)
        self.assertIn("• Suggested Fix: Set temperature within [0.0, 2.0] range", msg)
        self.assertIn("• Documentation: docs/design_docs/workflow_service_docs/workflow_builder_guides/nodes/llm_node_guide.md", msg)

        # Verify serialization
        data = err.to_dict()
        self.assertEqual(data["node_id"], "llm_node_1")
        self.assertEqual(data["node_name"], "llm")
        self.assertEqual(data["field_name"], "temperature")
        self.assertEqual(data["expected"], "float between 0.0 and 2.0")
        self.assertEqual(data["actual"], "10.0")
        self.assertEqual(data["suggested_fix"], "Set temperature within [0.0, 2.0] range")
        self.assertEqual(data["doc_url"], NODE_DOCS_MAPPING["llm"])
        self.assertEqual(data["error_type"], "value_out_of_range")

    def test_node_docs_mapping(self):
        """Known nodes must map to specific guides; unknown nodes fallback to workflow building guide."""
        self.assertIn("llm_node_guide.md", get_node_doc_url("llm"))
        self.assertIn("code_runner_node_guide.md", get_node_doc_url("code_runner"))
        self.assertIn("filter_node_guide.md", get_node_doc_url("filter_data"))
        self.assertIn("dynamic_router_node_guide.md", get_node_doc_url("dynamic_router"))
        self.assertEqual(get_node_doc_url("unknown_custom_node"), DEFAULT_WORKFLOW_DOC_URL)
        self.assertEqual(get_node_doc_url(None), DEFAULT_WORKFLOW_DOC_URL)

    def test_suggest_similar_fields(self):
        """Fuzzy matching should detect slight typos in field names."""
        valid_fields = ["temperature", "model", "max_tokens", "prompt"]
        suggestion = suggest_similar_fields("temperatue", valid_fields)
        self.assertEqual(suggestion, "Did you mean 'temperature'?")

        suggestion = suggest_similar_fields("modl", valid_fields)
        self.assertEqual(suggestion, "Did you mean 'model'?")

        self.assertIsNone(suggest_similar_fields("completely_unrelated", valid_fields))

    def test_wrap_pydantic_validation_error(self):
        """Wrapping Pydantic ValidationError produces structured NodeValidationError with suggestions."""
        try:
            SampleConfigSchema.model_validate({"model": "gpt-4o", "temperatue": 0.5})  # typo in temperature
        except Exception as pydantic_err:
            wrapped = wrap_pydantic_validation_error(
                pydantic_err,
                node_id="test_node",
                node_name="llm",
                schema_cls=SampleConfigSchema,
                context_desc="node configuration",
            )
            self.assertIsInstance(wrapped, NodeValidationError)
            self.assertEqual(wrapped.node_id, "test_node")
            self.assertEqual(wrapped.node_name, "llm")
            self.assertIn("temperatue", str(wrapped))
            self.assertIn("Did you mean 'temperature'?", str(wrapped))
            self.assertIn("llm_node_guide.md", str(wrapped))

    def test_base_node_non_editable_field_raises_descriptive_error(self):
        """Passing non-user-editable field to node with allow_non_user_editable_fields_in_config=False raises NodeValidationError."""
        with self.assertRaises(NodeValidationError) as ctx:
            MockValidationNode(
                node_id="llm_1",
                allow_non_user_editable_fields_in_config=False,
                config={"model": "gpt-4o", "secret_key": "override_attempt"},
            )

        err = ctx.exception
        self.assertEqual(err.node_id, "llm_1")
        self.assertEqual(err.node_name, "llm")
        self.assertEqual(err.field_name, "secret_key")
        self.assertIn("Non-editable field", str(err))
        self.assertIn("secret_key", str(err))
        self.assertIn("Suggested Fix", str(err))
        self.assertIn("llm_node_guide.md", str(err))

    def test_base_node_validate_node_config_helper(self):
        """BaseNode.validate_node_config validates against schema and provides rich error."""
        # Valid config
        MockValidationNode.validate_node_config({"model": "gpt-4o", "temperature": 0.7}, node_id="test_llm")

        # Invalid config: missing required field 'model'
        with self.assertRaises(NodeValidationError) as ctx:
            MockValidationNode.validate_node_config({"temperature": 0.5}, node_id="test_llm")

        err = ctx.exception
        self.assertEqual(err.node_id, "test_llm")
        self.assertEqual(err.node_name, "llm")
        self.assertIn("model", str(err))
        self.assertIn("Documentation", str(err))

    def test_base_node_build_input_state_missing_config(self):
        """build_input_state with missing configurable raises descriptive NodeValidationError."""
        node = MockValidationNode(node_id="llm_1")
        with self.assertRaises(NodeValidationError) as ctx:
            node.build_input_state(state={}, config={"configurable": None})

        err = ctx.exception
        self.assertEqual(err.node_id, "llm_1")
        self.assertEqual(err.field_name, "configurable")
        self.assertIn("Suggested Fix", str(err))
        self.assertIn("llm_node_guide.md", str(err))

    def test_base_node_build_input_state_missing_input_mappings(self):
        """build_input_state with missing input mappings for required fields raises descriptive NodeValidationError."""
        node = MockValidationNode(node_id="llm_1")
        with self.assertRaises(NodeValidationError) as ctx:
            node.build_input_state(state={}, config={"configurable": {"inputs": {}}})

        err = ctx.exception
        self.assertEqual(err.node_id, "llm_1")
        self.assertEqual(err.field_name, "inputs")
        self.assertIn("text", str(err))
        self.assertIn("Suggested Fix", str(err))

    def test_base_node_build_input_state_invalid_field_type(self):
        """build_input_state failing input schema validation raises wrapped NodeValidationError."""
        from workflow_service.utils.utils import get_node_output_state_key
        node = MockValidationNode(node_id="llm_1")
        # 'count' expects an integer, pass an invalid string
        config = {
            "configurable": {
                "inputs": {
                    "llm_1": {
                        "text": [["src_1", "text"]],
                        "count": [["src_1", "count"]],
                    }
                }
            }
        }
        state = {
            get_node_output_state_key("src_1"): {
                "text": "sample text",
                "count": "not_a_valid_integer",
            }
        }

        with self.assertRaises(NodeValidationError) as ctx:
            node.build_input_state(state=state, config=config)

        err = ctx.exception
        self.assertEqual(err.node_id, "llm_1")
        self.assertEqual(err.field_name, "count")
        self.assertIn("count", str(err))
        self.assertIn("Suggested Fix", str(err))
        self.assertIn("llm_node_guide.md", str(err))

    def test_graph_builder_missing_input_mappings_raises_node_validation_error(self):
        """GraphBuilder raises NodeValidationError when a node has required fields but no input mappings."""
        from workflow_service.registry.nodes.core.dynamic_nodes import (
            InputNode, OutputNode, ConstructDynamicSchema, DynamicSchemaFieldConfig
        )
        registry = DBRegistry()
        registry.register_node(InputNode)
        registry.register_node(OutputNode)
        registry.register_node(MockValidationNode)

        # Graph with input_node, output_node, and MockValidationNode without any incoming edges
        schema = GraphSchema(
            nodes={
                INPUT_NODE_NAME: NodeConfig(
                    node_id=INPUT_NODE_NAME,
                    node_name=INPUT_NODE_NAME,
                    node_version=InputNode.node_version,
                    dynamic_output_schema=ConstructDynamicSchema(fields={
                        "text": DynamicSchemaFieldConfig(type="str", required=True)
                    }),
                ),
                OUTPUT_NODE_NAME: NodeConfig(
                    node_id=OUTPUT_NODE_NAME,
                    node_name=OUTPUT_NODE_NAME,
                    node_version=OutputNode.node_version,
                    dynamic_input_schema=ConstructDynamicSchema(fields={
                        "result": DynamicSchemaFieldConfig(type="str", required=True)
                    }),
                ),
                "llm_1": NodeConfig(node_id="llm_1", node_name="llm", node_version="1.0.0"),
            },
            edges=[
                EdgeSchema(src_node_id=INPUT_NODE_NAME, dst_node_id="llm_1"),
                EdgeSchema(src_node_id="llm_1", dst_node_id=OUTPUT_NODE_NAME),
            ],
            input_node_id=INPUT_NODE_NAME,
            output_node_id=OUTPUT_NODE_NAME,
        )

        builder = GraphBuilder(registry=registry)
        with self.assertRaises(NodeValidationError) as ctx:
            builder.build_graph_entities(schema)

        err = ctx.exception
        self.assertEqual(err.node_id, "llm_1")
        self.assertIn("text", str(err))
        self.assertIn("Suggested Fix", str(err))
        self.assertIn("llm_node_guide.md", str(err))


if __name__ == "__main__":
    unittest.main()
