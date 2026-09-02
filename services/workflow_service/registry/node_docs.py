"""
Centralized mapping from node names to their documentation guides.

The guides live in ``docs/design_docs/workflow_service_docs/workflow_builder_guides/nodes/``
under the repository. Because guide filenames do not always match ``node_name``
(e.g. tool nodes are grouped under ``document_crud_tools_indepth_guide.md``), the
mapping is maintained explicitly here rather than derived from filenames.
"""

from typing import Dict, Optional

# Base path for node documentation guides, relative to the repository docs folder.
NODE_DOCS_BASE_URL = (
    "docs/design_docs/workflow_service_docs/workflow_builder_guides/nodes"
)

# Verified via GitHub API: sonaji94/kiwiq is a fork of rcortx/kiwiq (the upstream).
# Default branch: main (from `git branch -a` -> remotes/origin/HEAD -> origin/main).
DOCUMENTATION_GITHUB_BASE_URL = "https://github.com/rcortx/kiwiq/blob/main"

# NOTE: There is deliberately NO generic fallback guide. Unknown nodes return None
# unless a real generic node documentation page exists (none does in this folder).
# Every entry below is verified against an actual registered node_name declaration.

# Mapping of registered node_name -> documentation guide filename.
# Verified against node_name declarations under services/workflow_service/registry/nodes/.
NODE_DOCS_GUIDE_FILES: Dict[str, str] = {
    # LLM (llm_node.py:940 -> "llm")
    "llm": "llm_node_guide.md",
    # Prompt constructor (llm/prompt.py:401 -> "prompt_constructor")
    "prompt_constructor": "prompt_constructor_node_guide.md",
    # Prompt template loader (db/prompt_template_loader.py:247 -> "load_prompt_templates")
    "load_prompt_templates": "prompt_loader_node_guide.md",
    # Tool executor (tools/tool_executor_node.py:196 -> "tool_executor")
    "tool_executor": "llm_tools_executor_and_document_crud_guide.md",
    # Document CRUD tools (tools/documents/document_crud_tools.py -> grouped guide)
    "edit_document": "document_crud_tools_indepth_guide.md",
    "view_documents": "document_crud_tools_indepth_guide.md",
    "search_documents": "document_crud_tools_indepth_guide.md",
    "list_documents": "document_crud_tools_indepth_guide.md",
    # Data ops
    "transform_data": "transform_node_guide.md",  # data_ops/transform_node.py:189
    "data_join_data": "data_join_node_guide.md",  # data_ops/transform_node.py:438
    "merge_aggregate": "merge_aggregate_node_guide.md",  # data_ops/merge_aggregate_node.py:923
    # Core
    "router_node": "dynamic_router_node_guide.md",  # core/router_node.py:139
    "map_list_router_node": "map_list_router_node_guide.md",  # core/map_list_router_node.py:150
    "filter_data": "filter_node_guide.md",  # core/flow_nodes.py:899
    "if_else_condition": "if_else_node_guide.md",  # core/flow_nodes.py:1308
    "workflow_runner": "workflow_runner_node_guide.md",  # core/workflow_runner_node.py:357
    # Dynamic core nodes (core/dynamic_nodes.py:456/498 -> input_node/output_node)
    "input_node": "core_dynamic_nodes_guide.md",
    "output_node": "core_dynamic_nodes_guide.md",
    # HITL (core/dynamic_nodes.py:553 -> f"{HITL_NODE_NAME_PREFIX}default")
    "hitl_node__default": "hitl_node_guide.md",
    # DB / customer data
    "load_customer_data": "load_customer_data_node_guide.md",  # db/customer_data.py:696
    "load_multiple_customer_data": "load_multiple_customer_data_node_guide.md",  # db/load_multiple_customer_node.py:185
    "store_customer_data": "store_customer_data_node_guide.md",  # db/customer_data.py:1200
    "delete_customer_data": "delete_customer_data_node_guide.md",  # db/delete_customer_data_node.py:367
    # Scraping
    "linkedin_scraping": "linkedin_scraping_node_guide.md",  # scraping/linkedin_scraping.py:266
    "crawler_scraper": "crawler_scraper_node_guide.md",  # scraping/crawler_scraper_node.py:527
    "ai_answer_engine_scraper": "ai_answer_engine_scraper_node_guide.md",  # scraping/ai_answer_engine_scraper_node.py:889
    # Code runner
    "code_runner": "code_runner_node_guide.md",  # code_runner/code_runner_node.py:522
}

# Node names declared but intentionally NOT mapped (abstract / no dedicated guide):
# - "dynamic_router" (core/dynamic_nodes.py:622) is an abstract base class; the
#   concrete RouterNode is "router_node" and is what dynamic_router_node_guide.md documents.
# - "llm_models_guide.md" is a reference (models & internal tools), not a node guide.


def get_node_docs_url(
    node_name: Optional[str], node_category: Optional[str] = None
) -> Optional[str]:
    """
    Return a clickable GitHub documentation URL for a node name, or ``None`` if unknown.

    The returned URL points at the guide file on the upstream repository's default
    branch (``main``), verified via GitHub API against ``https://github.com/rcortx/kiwiq``.

    Args:
        node_name: The node type name (e.g. ``llm``, ``filter_data``).
        node_category: Optional category; reserved for future disambiguation.

    Returns:
        An absolute GitHub documentation URL string, or ``None`` when no guide is known.
    """
    if not node_name:
        return None
    guide_file = NODE_DOCS_GUIDE_FILES.get(node_name)
    if not guide_file:
        return None
    return f"{DOCUMENTATION_GITHUB_BASE_URL}/{NODE_DOCS_BASE_URL}/{guide_file}"
