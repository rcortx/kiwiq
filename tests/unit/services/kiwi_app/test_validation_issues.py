import json
import unittest

from jsonschema import ValidationError as JsonschemaValidationError
from kiwi_app.workflow_app.validation_issues import (
    generic_error_to_issue,
    jsonschema_error_to_issue,
    pydantic_error_to_issue,
)
from pydantic import BaseModel, Field, ValidationError
from workflow_service.registry.node_docs import (
    DOCUMENTATION_GITHUB_BASE_URL,
    NODE_DOCS_BASE_URL,
    get_node_docs_url,
)


class _DemoConfig(BaseModel):
    name: str
    count: int = Field(ge=0)


class _DemoBoundedConfig(BaseModel):
    count: int = Field(ge=1, le=10)


class TestPydanticErrorToIssue(unittest.TestCase):

    def _first_error(self, payload):
        try:
            _DemoConfig(**payload)
        except ValidationError as e:
            return e.errors()[0]
        raise AssertionError("expected ValidationError")

    def test_string_type_error(self):
        err = self._first_error({"name": 123})
        issue = pydantic_error_to_issue(err, node_id="n1", node_name="llm")
        self.assertEqual(issue["field"], "name")
        self.assertEqual(issue["node_id"], "n1")
        self.assertEqual(issue["node_name"], "llm")
        self.assertEqual(issue["expected"], "string")
        self.assertEqual(issue["actual"], 123)
        self.assertEqual(issue["actual_type"], "int")
        self.assertEqual(issue["suggested_fix"], "Provide a string value.")
        self.assertIn("Pydantic validation error for 'name'", issue["message"])
        self.assertIn("llm_node_guide.md", issue["documentation_url"])

    def test_dict_nested_error(self):
        err = self._first_error({"name": "x", "count": -1})
        self.assertEqual(err.get("type"), "greater_than_equal")
        issue = pydantic_error_to_issue(err, node_id="n1", node_name="llm")
        self.assertEqual(issue["field"], "count")
        self.assertTrue(issue["suggested_fix"])

    def test_greater_than_equal_suggested_fix_uses_actual_constraint(self):
        try:
            _DemoBoundedConfig(count=0)
        except ValidationError as e:
            err = e.errors()[0]
        self.assertEqual(err.get("type"), "greater_than_equal")
        issue = pydantic_error_to_issue(err, node_id="n1", node_name="llm")
        # Expected/actual/recommended fix must be derived from the real constraint (ge=1),
        # generically, not hard-coded per-field.
        self.assertEqual(issue["expected"], "integer/number >= 1")
        self.assertEqual(issue["actual"], 0)
        self.assertEqual(issue["suggested_fix"], "Set count to 1 or greater.")
        self.assertEqual(issue["field"], "count")

    def test_greater_than_equal_field_referenced_in_fix(self):
        try:
            _DemoBoundedConfig(count=0)
        except ValidationError as e:
            err = e.errors()[0]
        issue = pydantic_error_to_issue(err, node_id="n1", node_name="llm")
        # The fix text should reference the specific field ("count") generically.
        self.assertIn("count", issue["suggested_fix"])

    def test_actual_label_for_dict(self):
        from pydantic import ValidationError as VE

        class Cfg(BaseModel):
            tags: dict

        try:
            Cfg(tags="notdict")
        except VE as e:
            issue = pydantic_error_to_issue(e.errors()[0])
            self.assertEqual(issue["actual_type"], "str")

    def test_unknown_type_has_fallback(self):
        err = {"type": "some_unknown_type", "loc": ("x",), "msg": "boom", "input": 1}
        issue = pydantic_error_to_issue(err)
        self.assertEqual(issue["expected"], "some_unknown_type")
        self.assertNotEqual(issue["suggested_fix"], "")


class TestJsonSchemaErrorToIssue(unittest.TestCase):

    def _validate(self, schema, instance):
        import jsonschema

        try:
            jsonschema.validate(instance, schema)
        except JsonschemaValidationError as e:
            return e
        raise AssertionError("expected jsonschema ValidationError")

    def test_type_violation(self):
        schema = {"type": "object", "properties": {"model": {"type": "string"}}}
        err = self._validate(schema, {"model": 123})
        issue = jsonschema_error_to_issue(err, node_id="n2", node_name="llm")
        self.assertEqual(issue["node_id"], "n2")
        self.assertEqual(issue["node_name"], "llm")
        self.assertIn("llm_node_guide.md", issue["documentation_url"])
        self.assertEqual(issue["actual"], 123)
        self.assertEqual(issue["actual_type"], "int")

    def test_minimum_violation(self):
        schema = {
            "type": "object",
            "properties": {"temp": {"type": "number", "minimum": 0}},
        }
        err = self._validate(schema, {"temp": -5})
        issue = jsonschema_error_to_issue(err, node_id="n2", node_name="llm")
        self.assertEqual(issue["expected"], "value >= declared minimum")
        self.assertEqual(
            issue["suggested_fix"], "Provide a value at or above the declared minimum."
        )


class TestGenericErrorToIssue(unittest.TestCase):

    def test_basic(self):
        issue = generic_error_to_issue(
            "Could not find node template for foo v:latest error: nope",
            node_id="n3",
            node_name="foo",
            field="node_name",
        )
        self.assertEqual(issue["node_id"], "n3")
        self.assertEqual(issue["node_name"], "foo")
        self.assertEqual(issue["field"], "node_name")


class TestNodeDocsMapping(unittest.TestCase):

    def test_known_node_resolves_to_absolute_github_url(self):
        url = get_node_docs_url("router_node")
        self.assertIsNotNone(url)
        self.assertTrue(url.startswith("https://github.com/"))
        self.assertIn(DOCUMENTATION_GITHUB_BASE_URL, url)
        # URL must be "clickable": host + blob + branch + repo-relative path + filename.
        self.assertIn("/blob/main/", url)
        self.assertIn(NODE_DOCS_BASE_URL, url)
        self.assertTrue(url.endswith(".md"))

    def test_unknown_node_returns_none(self):
        self.assertIsNone(get_node_docs_url("not_a_real_node"))

    def test_hitl_node_resolves(self):
        self.assertIsNotNone(get_node_docs_url("hitl_node__default"))


class TestJsonSerializableIssues(unittest.TestCase):
    """Issues must be JSON-serializable so they can be returned by the API."""

    def test_pydantic_issue_serializable(self):
        try:
            _DemoConfig(name=123)
        except ValidationError as e:
            issue = pydantic_error_to_issue(e.errors()[0])
        json.dumps(issue)  # must not raise

    def test_generic_issue_serializable(self):
        issue = generic_error_to_issue("boom", node_id="n", node_name="x")
        json.dumps(issue)


if __name__ == "__main__":
    unittest.main()
