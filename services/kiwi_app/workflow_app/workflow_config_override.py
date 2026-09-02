"""
Workflow Configuration Override Utilities.

This module provides utilities for applying partial graph schema overrides to a base
GraphSchema. It includes Pydantic models for defining the override structure,
logic for merging configurations, and optional validation of the resulting schema.
"""
import copy
from typing import Any, Dict, List, Optional, Tuple, Self
from collections import defaultdict

from pydantic import BaseModel, Field, model_validator, ConfigDict, ValidationError as PydanticValidationError
import jsonschema
from jsonschema.validators import Draft202012Validator
from jsonschema import ValidationError as JsonSchemaValidationError

from workflow_service.graph.graph import GraphSchema, NodeConfig # Assuming EdgeSchema is not directly part of override models but handled as dict
from workflow_service.registry.registry import DBRegistry
from workflow_service.graph.builder import GraphBuilder
from workflow_service.config.constants import INPUT_NODE_NAME, OUTPUT_NODE_NAME
from kiwi_app.utils import get_kiwi_logger # Added
from kiwi_app.workflow_app.validation_issues import (
    pydantic_error_to_issue,
    jsonschema_error_to_issue,
    generic_error_to_issue,
)

workflow_logger = get_kiwi_logger(name="kiwi_app.workflow_config_override") # Added logger

class NodeOverrideCriteria(BaseModel):
    """
    Defines the criteria for matching a node and the configuration to apply.
    
    Attributes:
        node_id (Optional[str]): If provided, the specific ID of the node to target.
                                 If `node_name` is also provided, it will be used for verification.
        node_name (str): The name of the node type. This is required for matching,
                         either as the primary matcher (if `node_id` is None) or for
                         verification (if `node_id` is provided).
        node_version (Optional[str]): The version of the node. If None, the override
                                      applies to all versions of the specified `node_name`.
                                      If provided, `node_name` must also be present.
        node_config (Dict[str, Any]): The dictionary of configuration key-value pairs
                                      to merge into the target node's `node_config`.
                                      This field must not be an empty dictionary.
        replace_mode (Optional[bool]): If True, the entire `node_config` of the matched node
                                       for the keys present in `node_config_override` will be
                                       replaced, not merged. Defaults to global `replace_mode`.
        list_replace_mode (Optional[bool]): If True, lists within `node_config` will be
                                            replaced entirely by lists from `node_config_override`,
                                            instead of element-wise merging. Only applies if
                                            the effective `replace_mode` for the node is False.
                                            Defaults to global `list_replace_mode`.
    """
    node_id: Optional[str] = Field(None, description="If provided, node_id to match. Verification with node_name might apply.")
    node_name: str = Field(..., description="Name of the node type to match. Always required for matching.")
    node_version: Optional[str] = Field(None, description="Version of the node. If None, applies to all versions of node_name.")
    
    node_config_override: Dict[str, Any] = Field(..., alias="node_config", description="Configuration to merge. Must not be empty.")

    replace_mode: Optional[bool] = Field(None, description="Node-specific replace mode. Overrides global if set. If True, override values replace base values entirely for this node's config.")
    list_replace_mode: Optional[bool] = Field(None, description="Node-specific list replace mode. Overrides global if set. If True, override lists replace base lists entirely. Applies only if node's effective replace_mode is False.")

    @model_validator(mode='after')
    def _check_node_config_override_not_empty(self) -> Self:
        """
        Validates that 'node_config_override' is not an empty dictionary.
        """
        if not self.node_config_override: # Checks if the dictionary itself is empty
            raise ValueError("'node_config' in an override rule cannot be an empty dictionary.")
        return self

    @model_validator(mode='after')
    def _check_version_requires_name(self) -> Self:
        """
        Ensures that if node_version is provided, node_name is also present.
        This is implicitly handled as node_name is mandatory.
        """
        # node_name is already mandatory (Field(..., ...)).
        # The condition "if node_version is present, node_name must be present" is thus met.
        return self


class GraphOverridePayload(BaseModel):
    """
    Defines the structure of the graph override configuration payload.
    
    This model allows specifying overrides for individual node configurations
    and for other top-level properties of a GraphSchema.
    
    Attributes:
        node_configs (Optional[List[NodeOverrideCriteria]]): A list of specific node
            configurations to override. Each item defines matching criteria and the
            configuration changes to apply.
        edges (Optional[List[Optional[Dict[str, Any]]]]): Overrides for graph edges.
            List items can be dictionaries representing partial or full EdgeSchema data,
            or None to indicate no change for an edge at that specific index if merging lists.
        metadata (Optional[Dict[str, Any]]): Overrides for the graph's metadata dictionary.
        input_node_id (Optional[str]): Override for the graph's input node ID.
        output_node_id (Optional[str]): Override for the graph's output node ID.
        replace_mode (bool): Global flag. If True, override values for top-level graph
                             properties (and node configs if not specified at node-level)
                             will replace base values entirely, instead of deep merging.
                             Defaults to False.
        list_replace_mode (bool): Global flag for list merging. If True, lists in the
                                  override will replace corresponding lists in the base
                                  schema entirely. If False, lists are merged element-wise.
                                  This applies only if `replace_mode` is False for the
                                  given scope. Defaults to False.
    
    eg: Global Config:

    {
        "override_graph_schema": {
            "node_configs": [
                {
                    "node_id": "generate_content",
                    "node_name": "llm",
                    "node_config": {
                        "llm_config": {
                            "model_spec": {
                                "provider": "openai",
                                "model": "gpt-4o-mini"
                            }
                        }
                    }
                },
                {
                    "node_id": "construct_prompt",
                    "node_name": "prompt_constructor",
                    "node_config": {
                        "prompt_templates": {
                            "user_prompt": {
                                "user_custom_instructions": "Generate a content strategy for the user's business."
                            }
                        }
                    }
                }
            ]
        }
    }

    # Payload for specific user / org; across all workflows (global override):

    {
        "override_graph_schema": {
            "node_configs": [
                        {
                            "node_name": "llm",
                            "node_config": {
                                "llm_config": {
                                    "model_spec": {
                                        "provider": "openai",
                                        "model": "gpt-4.1-nano"
                                    },
                                    "reasoning_tokens_budget": null,
                                    "reasoning_effort_class": null
                                }
                            }
                        }]
        },
        "is_system_entity": false,
        "user_id": "69bb8628-b83d-411e-a970-7d25a7cfaca9",
        "org_id": "0b5e5a20-a7bd-44ea-aab1-a23325a9eed0",
        "is_active": true,
        "description": "frontend testing override to reduce workflow latency"
    }


    {
        "override_graph_schema": {
            "node_configs": [
                        {
                            "node_id": "google_research_llm",
                            "node_name": "llm",
                            "node_config": {
                                "llm_config": {
                                    "model_spec": {
                                        "provider": "perplexity",
                                        "model": "sonar"
                                    }
                                }
                            }
                        }]
        },
        "is_system_entity": false,
        "user_id": "69bb8628-b83d-411e-a970-7d25a7cfaca9",
        "org_id": "0b5e5a20-a7bd-44ea-aab1-a23325a9eed0",
        "is_active": true,
        "description": "frontend testing override to reduce workflow latency"
    }

    {
        "override_graph_schema": {
            "node_configs": [
                        {
                            "node_id": "reddit_research_llm",
                            "node_name": "llm",
                            "node_config": {
                                "llm_config": {
                                    "model_spec": {
                                        "provider": "perplexity",
                                        "model": "sonar"
                                    }
                                }
                            }
                        }]
        },
        "is_system_entity": false,
        "user_id": "69bb8628-b83d-411e-a970-7d25a7cfaca9",
        "org_id": "0b5e5a20-a7bd-44ea-aab1-a23325a9eed0",
        "is_active": true,
        "description": "frontend testing override to reduce workflow latency"
    }

    {
        "override_graph_schema": {
            "node_configs": [
                        {
                            "node_id": "research_llm",
                            "node_name": "llm",
                            "node_config": {
                                "llm_config": {
                                    "model_spec": {
                                        "provider": "perplexity",
                                        "model": "sonar"
                                    }
                                }
                            }
                        }]
        },
        "workflow_name": "blog_content_calendar_entry",
        "is_system_entity": false,
        "user_id": "69bb8628-b83d-411e-a970-7d25a7cfaca9",
        "org_id": "0b5e5a20-a7bd-44ea-aab1-a23325a9eed0",
        "is_active": true,
        "description": "frontend testing override to reduce workflow latency"
    }

    {
        "override_graph_schema": {
            "node_configs": [
                        {
                            "node_id": "content_gap_finder_llm",
                            "node_name": "llm",
                            "node_config": {
                                "llm_config": {
                                    "model_spec": {
                                        "provider": "perplexity",
                                        "model": "sonar"
                                    }
                                }
                            }
                        }]
        },
        "workflow_name": "blog_content_optimisation_workflow",
        "is_system_entity": false,
        "user_id": "69bb8628-b83d-411e-a970-7d25a7cfaca9",
        "org_id": "0b5e5a20-a7bd-44ea-aab1-a23325a9eed0",
        "is_active": true,
        "description": "frontend testing override to reduce workflow latency"
    }





    # TOOL calling for backup

    {
        "override_graph_schema": {
            "node_configs": [
                        {
                            "node_name": "llm",
                            "node_config": {
                                "llm_config": {
                                    "model_spec": {
                                        "provider": "openai",
                                        "model": "gpt-4.1-nano"
                                    },
                                    "reasoning_tokens_budget": null,
                                    "reasoning_effort_class": null
                                }
                            }
                        }]
        },
        "workflow_name": "blog_content_playbook_workflow",
        "is_system_entity": false,
        "user_id": "69bb8628-b83d-411e-a970-7d25a7cfaca9",
        "org_id": "0b5e5a20-a7bd-44ea-aab1-a23325a9eed0",
        "is_active": true,
        "description": "frontend testing override to reduce workflow latency"
    }

    {
        "override_graph_schema": {
            "node_configs": [
                        {
                            "node_name": "llm",
                            "node_config": {
                                "llm_config": {
                                    "model_spec": {
                                        "provider": "openai",
                                        "model": "gpt-4.1-nano"
                                    }
                                }
                            }
                        }]
        },
        "workflow_name": "linkedin_content_playbook_workflow",
        "is_system_entity": false,
        "user_id": "69bb8628-b83d-411e-a970-7d25a7cfaca9",
        "org_id": "0b5e5a20-a7bd-44ea-aab1-a23325a9eed0",
        "is_active": true,
        "description": "frontend testing override to reduce workflow latency"
    }


    Web search:
        workflow id: blog_brief_from_calendar_topic
        node id:
        google_research_llm
        reddit_research_llm

        workflow id: blog_content_calendar_entry
        node id:
        research_llm

        workflow id : blog_content_optimisation_workflow
        node id:
        content_gap_finder_llm

        workflow id: blog_user_input_to_brief_workflow
        node id:
        google_research_llm
        reddit_research_llm

    Tool calling workflows
        blog_content_playbook_workflow
        linkedin_content_playbook_workflow

    """
    node_configs: Optional[List[NodeOverrideCriteria]] = Field(None, description="List of specific node configurations to override.")
    
    # Optional overrides for other GraphSchema fields
    # These fields are explicitly listed for clarity and type checking during parsing,
    # matching common overridable parts of GraphSchema.
    edges: Optional[List[Optional[Dict[str, Any]]]] = Field(None, description="Overrides for graph edges. List items can be None to skip corresponding original edge.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Overrides for graph metadata.")
    input_node_id: Optional[str] = Field(None, description="Override for the graph's input node ID.")
    output_node_id: Optional[str] = Field(None, description="Override for the graph's output node ID.")
    
    replace_mode: bool = Field(False, description="Global replace mode. If True, override values replace base values entirely.")
    list_replace_mode: bool = Field(False, description="Global list replace mode. If True, override lists replace base lists. Applies if replace_mode is False.")

    # Allow other fields not explicitly defined here to be part of the override.
    # These will be handled by the generic deep merge logic if they match GraphSchema fields.
    model_config = ConfigDict(extra='allow')


def _deep_merge_dicts(
    base_dict: Dict[str, Any],
    override_dict: Dict[str, Any],
    current_replace_mode: bool,
    current_list_replace_mode: bool
) -> Dict[str, Any]:
    """
    Recursively merges two dictionaries, respecting replace and list_replace modes.
    
    - If `current_replace_mode` is True for a key, `override_dict`'s value for that key
      replaces `base_dict`'s value.
    - Otherwise (if `current_replace_mode` is False):
        - Keys from `override_dict` will overwrite keys in `base_dict` for scalar values.
        - If both `base_dict` and `override_dict` have the same key and both values are dicts,
          it recursively merges them (passing down the modes).
        - If both values are lists:
            - If `current_list_replace_mode` is True, `base_dict`'s list is replaced by
              `override_dict`'s list.
            - If `current_list_replace_mode` is False, it attempts to merge them element-wise:
                - If an element in `override_dict`'s list is `None`, the corresponding element
                  in `base_dict`'s list is unchanged.
                - If an element in `override_dict`'s list is a dict and the corresponding
                  `base_dict` element is also a dict, they are recursively merged.
                - Otherwise, the `base_dict` element is replaced by the `override_dict` element.
                - If `override_dict`'s list is longer, its additional elements are appended.
        - If `override_dict` has a list for a key where `base_dict` does not (and base is not None),
          and `current_replace_mode` is False, a TypeError is raised.
    - New keys from `override_dict` are added to `base_dict`.

    Args:
        base_dict (Dict[str, Any]): The base dictionary to merge into.
        override_dict (Dict[str, Any]): The dictionary with override values.
        current_replace_mode (bool): If True, override values replace base values entirely.
        current_list_replace_mode (bool): If True and not in replace_mode, override lists
                                          replace base lists.

    Returns:
        Dict[str, Any]: The merged dictionary.
        
    Raises:
        TypeError: If there's a type mismatch (e.g., override is list, base is not a list)
                   and not in `current_replace_mode`.
    """
    merged_dict = base_dict.copy() # Start with a copy of the base dictionary

    for key, override_item_value in override_dict.items():
        base_item_value = merged_dict.get(key)

        if current_replace_mode: # If replace mode is active for this level/key
            merged_dict[key] = override_item_value
            continue

        # --- If not in current_replace_mode, proceed with deep merge logic ---
        if key not in merged_dict: # Key is new, add from override
            merged_dict[key] = override_item_value
            continue

        # Key exists in both base and override, proceed with merging logic
        if isinstance(base_item_value, dict) and isinstance(override_item_value, dict):
            merged_dict[key] = _deep_merge_dicts(
                base_item_value,
                override_item_value,
                current_replace_mode, # Should be False here, but pass for consistency
                current_list_replace_mode
            )
        elif isinstance(base_item_value, list) and isinstance(override_item_value, list):
            if current_list_replace_mode: # Replace entire list
                merged_dict[key] = override_item_value
            else: # Element-wise list merging
                new_list = list(base_item_value) 
                len_base_list = len(new_list)
                
                for i, ov_list_entry in enumerate(override_item_value):
                    if i < len_base_list: 
                        if ov_list_entry is not None: 
                            if isinstance(new_list[i], dict) and isinstance(ov_list_entry, dict):
                                new_list[i] = _deep_merge_dicts(
                                    new_list[i],
                                    ov_list_entry,
                                    current_replace_mode, # Should be False
                                    current_list_replace_mode
                                )
                            else:
                                new_list[i] = ov_list_entry 
                    else: 
                        new_list.append(ov_list_entry)
                merged_dict[key] = new_list
        elif isinstance(override_item_value, list) and not isinstance(base_item_value, list) and base_item_value is not None:
            # This error condition applies if not in replace_mode
            raise TypeError(
                f"Type mismatch for key '{key}': Override provides a list, but the base schema value is not a list (and not None), and replace_mode is False. "
                f"Base type: {type(base_item_value).__name__}, Override type: list."
            )
        else:
            # Default behavior for scalars or mismatched types (when not list vs non-list error):
            # override scalar values.
            merged_dict[key] = override_item_value
            
    return merged_dict


async def _validate_updated_graph_schema(
    graph_config_dict: Dict[str, Any],
    db_registry: DBRegistry,
    perform_detailed_validation: bool = True,
    is_superuser: bool = False,
    create_override_mode: bool = False,

) -> Tuple[bool, bool, bool, Dict[str, List[str]], Dict[str, List[Dict[str, Any]]]]:
    """
    Validates the merged graph configuration dictionary.
    
    This function adapts and consolidates validation logic from 
    `services/kiwi_app/workflow_app/routes.py validate_graph` and the previous
    version of this function.
    
    Args:
        graph_config_dict (Dict[str, Any]): The graph configuration dictionary to validate.
        db_registry (DBRegistry): An instance of DBRegistry for fetching node templates.
        perform_detailed_validation (bool): If True, performs node configuration
                                            and GraphBuilder validation. Defaults to True.
        is_superuser (bool): If True, the function will allow system entities to be validated.
            Defaults to False.
        create_override_mode (bool): If True, the function will operate in create override mode.
            Defaults to False.
    Returns:
        Tuple[bool, bool, bool, Dict[str, List[str]], Dict[str, List[Dict[str, Any]]]]:
            A tuple containing:
            - overall_is_valid (bool): True if all performed validations pass.
            - pydantic_graph_schema_is_valid (bool): True if the graph_config_dict
              conforms to the GraphSchema Pydantic model.
            - detailed_validation_is_valid (bool): True if node configurations and
              GraphBuilder validations passed, or if perform_detailed_validation was False.
            - all_errors (Dict[str, List[str]]): A dictionary of validation errors (strings,
              preserved for backward compatibility).
            - validation_errors (Dict[str, List[Dict[str, Any]]]): A parallel dictionary of
              structured validation issues (additive metadata for issue #8).
    """
    all_errors: Dict[str, List[str]] = defaultdict(list)
    validation_errors: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    pydantic_graph_schema_is_valid = False
    node_configs_are_valid = True  # Assume true initially, only relevant if perform_detailed_validation is true
    graph_builder_is_valid = True # Assume true initially, only relevant if perform_detailed_validation is true

    # 1. Validate against GraphSchema Pydantic model
    graph_schema: Optional[GraphSchema] = None
    try:
        workflow_logger.info("Attempting Pydantic GraphSchema validation.")
        graph_schema = GraphSchema.model_validate(graph_config_dict)
        pydantic_graph_schema_is_valid = True
        workflow_logger.info("Pydantic GraphSchema validation successful.")
    except PydanticValidationError as e:
        pydantic_graph_schema_is_valid = False
        # Extract Pydantic error messages
        for err in e.errors():
            err_loc = ".".join(map(str, err['loc'])) if err['loc'] else "graph_schema"
            all_errors[err_loc].append(err['msg'])
            validation_errors[err_loc].append(
                pydantic_error_to_issue(err, message_prefix="Pydantic GraphSchema validation error")
            )
        # It's also good to have the summary from the original logic:
        all_errors["graph_schema"].append(f"Overall Pydantic GraphSchema validation failed: {str(e)}")
        workflow_logger.warning(f"Pydantic GraphSchema validation failed: {str(e)}")
        # Critical failure, cannot proceed with detailed validation if base schema is wrong
        return False, pydantic_graph_schema_is_valid, True, all_errors, validation_errors

    if perform_detailed_validation and pydantic_graph_schema_is_valid and graph_schema:
        workflow_logger.info("Performing detailed validation: Node configurations and GraphBuilder.")
        # 2. Node configuration validation
        workflow_logger.info("Starting node configuration validation...")
        temp_node_configs_valid = True # Renamed from node_configs_valid in original to avoid conflict
        for node_id, node_obj in graph_schema.nodes.items(): # node_obj is node_config from original
            if node_obj.node_name == INPUT_NODE_NAME or node_obj.node_name == OUTPUT_NODE_NAME:
                workflow_logger.debug(f"Skipping validation for special node: {node_id} ({node_obj.node_name})")
                continue

            node_name = node_obj.node_name
            # Use node_obj.node_version directly; db_registry.get_node handles None as latest
            node_version_to_fetch = node_obj.node_version 
            node_version_display = node_obj.node_version or "latest" # For logging
            
            node_template = None # Equivalent to 'node' in original
            try:
                # Original used `None if node_version == "latest" else node_version`
                # but `node_version_to_fetch` (which is `node_obj.node_version`) directy is fine.
                node_template = db_registry.get_node(node_name=node_name, version=node_version_to_fetch)
                # Original updated a 'node_version' var here from node_template.node_version, used in error.
                # We use node_template.node_version for logging if needed, or node_version_display.
                workflow_logger.debug(f"Found template for node {node_id} ({node_name} v{node_template.node_version}).")
            except ValueError as e:
                # Original error message used the potentially updated 'node_version' from template if found,
                # or the initial 'latest'/specified if not found. node_version_display matches this.
                error_msg = f"Could not find node template for {node_name} v:{node_version_display} error: {str(e)}"
                all_errors[node_id].append(error_msg)
                validation_errors[node_id].append(
                    generic_error_to_issue(
                        error_msg,
                        node_id=node_id,
                        node_name=node_name,
                        field="node_name",
                        expected="a registered node template",
                        suggested_fix=f"Verify '{node_name}' is a valid registered node name/version.",
                    )
                )
                workflow_logger.warning(f"Node config validation error for {node_id}: {error_msg} ({str(e)})")
                temp_node_configs_valid = False
                continue 

            config_schema_class = node_template.config_schema_cls # Equivalent to original's config_schema then config_schema_cls
            
            # This logic matches original for handling nodes with no config schema
            if not config_schema_class:
                # node_obj.node_config can be None, {}, or {'key':'val'}
                if not node_obj.node_config: # True if node_obj.node_config is None or {}
                    workflow_logger.info(f"Node {node_id} ({node_name}) has no config schema defined and no config provided, skipping validation.")
                else: # Config is provided (e.g. {'key':'val'}), but no schema defined in template
                    temp_node_configs_valid = False
                    all_errors[node_id].append("Node has no config schema defined, but has a config in the graph schema")
                    validation_errors[node_id].append(
                        generic_error_to_issue(
                            "Node has no config schema defined, but has a config in the graph schema",
                            node_id=node_id,
                            node_name=node_name,
                            field="node_config",
                            expected="no config (node template has no config_schema_cls)",
                            actual=str(node_obj.node_config),
                            suggested_fix="Remove the node_config from the graph schema, or use a node that supports configuration.",
                        )
                    )
                    workflow_logger.warning(f"Node {node_id} ({node_name}) has no config schema defined, but has a config in the graph schema: {node_obj.node_config}")
                continue
            # else: # config_schema_class exists
            #    original converted to json_schema here, we do it inside try.

            # Use node_obj.node_config directly, which can be None.
            # Pydantic's default_factory ensures it's {} if not specified, but can be explicitly None.
            current_node_config_payload = node_obj.node_config

            try:
                json_schema_for_node = config_schema_class.model_json_schema()
                try:
                    jsonschema.validate(
                        instance=current_node_config_payload, # Use direct value
                        schema=json_schema_for_node,
                        format_checker=Draft202012Validator.FORMAT_CHECKER
                    )
                    workflow_logger.debug(f"Node {node_id} ({node_name}) JSONSchema configuration validated successfully.")
                except Exception as e:
                    workflow_logger.warning(f"Node {node_id} ({node_name}) JSONSchema configuration validation failed: {str(e)}")
                    if (not is_superuser) and create_override_mode:
                        raise e
                
                config_schema_class.model_validate(current_node_config_payload) # Use direct value
                workflow_logger.debug(f"Node {node_id} ({node_name}) Pydantic configuration validated successfully.")

            except JsonSchemaValidationError as e_json:
                temp_node_configs_valid = False
                error_path = ".".join(map(str, e_json.path)) if e_json.path else "config"
                error_msg = f"JSONSchema validation failed for '{error_path}': {e_json.message}"
                all_errors[node_id].append(error_msg)
                validation_errors[node_id].append(
                    jsonschema_error_to_issue(e_json, node_id=node_id, node_name=node_name)
                )
                workflow_logger.warning(f"Node config validation (JSONSchema) error for {node_id} ({node_name}): {error_msg}")
            except PydanticValidationError as e_pydantic:
                temp_node_configs_valid = False
                for err in e_pydantic.errors(): # Detailed Pydantic errors
                    err_loc = ".".join(map(str, err['loc'])) if err['loc'] else "config"
                    all_errors[node_id].append(f"Pydantic validation error for '{err_loc}': {err['msg']}")
                    validation_errors[node_id].append(
                        pydantic_error_to_issue(err, node_id=node_id, node_name=node_name)
                    )
                workflow_logger.warning(f"Node config validation (Pydantic) error for {node_id} ({node_name}): {str(e_pydantic)}")
            except Exception as e_other: 
                temp_node_configs_valid = False
                error_msg = f"Unexpected error during node config validation: {str(e_other)}"
                all_errors[node_id].append(error_msg)
                validation_errors[node_id].append(
                    generic_error_to_issue(error_msg, node_id=node_id, node_name=node_name, field="node_config")
                )
                workflow_logger.error(f"Unexpected node config validation error for {node_id} ({node_name}): {error_msg}", exc_info=True)
        
        node_configs_are_valid = temp_node_configs_valid # Set the flag based on loop results
        if node_configs_are_valid:
            workflow_logger.info("All node configurations validated successfully (or skipped where appropriate).")
        else:
            workflow_logger.warning("One or more node configurations failed validation.")

        # 3. GraphBuilder validation
        # Original condition: `if not all_errors:` after node validation.
        # This means if any error occurred (pydantic schema, or node config), it skips.
        # Our `node_configs_are_valid` captures the node config part.
        # `pydantic_graph_schema_is_valid` captures the schema part.
        if pydantic_graph_schema_is_valid and node_configs_are_valid and graph_schema:
            # The original logic for GraphBuilder re-fetched db_registry using get_external_context_manager_with_clients.
            # We will use the db_registry passed into this function, which is cleaner.
            workflow_logger.info("Starting GraphBuilder validation...")
            try:
                graph_builder = GraphBuilder(registry=db_registry)
                _ = graph_builder.build_graph_entities(graph_schema) # graph_schema is already validated GraphSchema model
                graph_builder_is_valid = True
                workflow_logger.info("GraphBuilder validation successful.")
            except Exception as e:
                # Original used last node_id for error key, this is corrected to "graph_builder"
                error_msg = f"Graph construction/validation by GraphBuilder failed: {str(e)}"
                all_errors["graph_builder"].append(error_msg)
                validation_errors["graph_builder"].append(
                    generic_error_to_issue(error_msg, field="graph_builder")
                )
                workflow_logger.error(f"GraphBuilder validation failed: {error_msg}", exc_info=True)
                graph_builder_is_valid = False # Set specific flag
                # Original set node_configs_valid = False here, which was less clear.
        else:
            if not (pydantic_graph_schema_is_valid and node_configs_are_valid):
                 workflow_logger.info("Skipping GraphBuilder validation due to prior errors in Pydantic schema or node configurations.")
            graph_builder_is_valid = True # Skipped, so not failed.

    # Determine overall validity
    detailed_validation_attempted_and_passed = (node_configs_are_valid and graph_builder_is_valid)
    detailed_validation_passed_or_skipped = (not perform_detailed_validation) or detailed_validation_attempted_and_passed
    
    # Overall valid if Pydantic schema is valid AND (either detailed validation was skipped OR it was attempted and passed)
    # AND there are no errors accumulated from any step (this last check is a safeguard).
    overall_is_valid = pydantic_graph_schema_is_valid and detailed_validation_passed_or_skipped and not any(err_list for err_list in all_errors.values())
    
    if overall_is_valid:
        workflow_logger.info("Overall graph validation successful.")
    else:
        final_error_count = sum(len(el) for el in all_errors.values())
        workflow_logger.warning(f"Overall graph validation failed with {final_error_count} errors. Errors: {dict(all_errors)}")
        
    return overall_is_valid, pydantic_graph_schema_is_valid, detailed_validation_passed_or_skipped, all_errors, validation_errors


async def apply_graph_override(
    base_graph_schema: GraphSchema,
    override_payload_dict: Dict[str, Any],
    validate_schema: bool = False,
    db_registry: Optional[DBRegistry] = None,
    create_override_mode: bool = False,
    is_superuser: bool = False
) -> GraphSchema:
    """
    Applies a partial graph schema override to a base GraphSchema.

    The function merges the `override_payload_dict` into the `base_graph_schema`.
    Node-specific configurations are applied based on matching criteria. Other graph
    properties (like edges, metadata) are merged deeply.

    Args:
        base_graph_schema (GraphSchema): The original GraphSchema object.
        override_payload_dict (Dict[str, Any]): A dictionary representing the override
            configuration. This should conform to GraphOverridePayload structure.
        validate_schema (bool): If True, the merged schema will be validated.
            Defaults to False.
        db_registry (Optional[DBRegistry]): An instance of DBRegistry, required if
            `validate_schema` is True.
        create_override_mode (bool): If True, the function will operate in create override mode.
            Defaults to False.
        is_superuser (bool): If True, the function will allow system entities to be validated.
            Defaults to False.
    Returns:
        GraphSchema: The updated GraphSchema object after applying overrides.

    Raises:
        ValueError: If `validate_schema` is True but `db_registry` is not provided,
                    or if parsing the override_payload_dict fails, or if validation fails.
        TypeError: For type mismatches during merging (e.g., list in override,
                   non-list in base schema for the same key).
        PydanticValidationError: If the final merged structure is invalid w.r.t GraphSchema.
    """
    if validate_schema and not db_registry:
        raise ValueError("DBRegistry must be provided if validate_schema is True.")

    try:
        override_model = GraphOverridePayload.model_validate(override_payload_dict)
    except PydanticValidationError as e:
        raise ValueError(f"Invalid override payload structure: {e}") from e

    # Convert base schema to dict for mutable operations
    # Use deepcopy to ensure the original base_graph_schema object is not modified.
    # model_dump ensures we get a dictionary representation.
    updated_graph_dict = copy.deepcopy(base_graph_schema.model_dump(mode='python') if isinstance(base_graph_schema, BaseModel) else base_graph_schema) # mode='python' for Pydantic models within

    # 1. Apply node-specific overrides
    if override_model.node_configs:
        if "nodes" not in updated_graph_dict or not isinstance(updated_graph_dict["nodes"], dict):
            # This case should ideally not happen if base_graph_schema is a valid GraphSchema
            # but good to be defensive.
            updated_graph_dict["nodes"] = {} 

        for node_override_rule in override_model.node_configs:
            nodes_to_update_ids: List[str] = []
            
            effective_replace_mode = node_override_rule.replace_mode if node_override_rule.replace_mode is not None else override_model.replace_mode
            effective_list_replace_mode = node_override_rule.list_replace_mode if node_override_rule.list_replace_mode is not None else override_model.list_replace_mode
            
            # If node-level replace_mode is True, list_replace_mode becomes irrelevant for its direct effect on node_config keys.
            # The entire node_config will be replaced by node_config_override content.
            # The effective_list_replace_mode would apply if node_config_override itself had nested lists
            # and effective_replace_mode was True for those nested keys (which is not how it's set up here).

            if node_override_rule.node_id:
                # Match by node_id, verify with node_name if also provided in rule
                if node_override_rule.node_id in updated_graph_dict["nodes"]:
                    target_node_candidate = updated_graph_dict["nodes"][node_override_rule.node_id]
                    # "if node_id and node_name both present, only node_ids with matching node_name will be applied against"
                    if target_node_candidate.get("node_name") == node_override_rule.node_name:
                        # Version check for the specific node_id
                        current_version = target_node_candidate.get("node_version")
                        if node_override_rule.node_version is None or current_version == node_override_rule.node_version:
                             nodes_to_update_ids.append(node_override_rule.node_id)
                        # Else: node_id and node_name matched, but version did not. Skip this rule for this node.
                    # Else: node_id matched, but node_name in rule did not match the node's actual name. Skip.
            else:
                # Match by node_name (node_id not provided in rule)
                for nid, n_data in updated_graph_dict["nodes"].items():
                    if n_data.get("node_name") == node_override_rule.node_name:
                        # "if node_version is None, it will be applied to all node_versions of that node_name"
                        # "if node_version is present, node_name must be present" (name is already matched here)
                        if node_override_rule.node_version is None or \
                           n_data.get("node_version") == node_override_rule.node_version:
                            nodes_to_update_ids.append(nid)
            
            # Apply node_config_override to all matched nodes
            for nid_to_update in nodes_to_update_ids:
                target_node_dict = updated_graph_dict["nodes"][nid_to_update]
                
                if effective_replace_mode:
                    # If node-level replace_mode is true, the entire node_config is replaced
                    # by the content of node_config_override.
                    target_node_dict["node_config"] = copy.deepcopy(node_override_rule.node_config_override)
                else:
                    # Otherwise, perform a deep merge.
                    if "node_config" not in target_node_dict or target_node_dict["node_config"] is None:
                        target_node_dict["node_config"] = {} # Ensure node_config exists for merging
                    
                    target_node_dict["node_config"] = _deep_merge_dicts(
                        target_node_dict["node_config"],
                        node_override_rule.node_config_override,
                        current_replace_mode=effective_replace_mode, # This will be False here
                        current_list_replace_mode=effective_list_replace_mode
                    )
    
    # 2. Apply general graph overrides (edges, metadata, etc.)
    # Exclude node_configs as they are handled separately and structure differs.
    # Exclude_none=True ensures we only merge fields explicitly set in the override.
    general_overrides = override_model.model_dump(
        exclude_none=True, 
        exclude={"node_configs", "replace_mode", "list_replace_mode"}, # Already processed
        mode='python'
    )
    # For general overrides, use the global replace_mode and list_replace_mode
    updated_graph_dict = _deep_merge_dicts(
        updated_graph_dict,
        general_overrides,
        current_replace_mode=override_model.replace_mode,
        current_list_replace_mode=override_model.list_replace_mode
    )

    # 3. Optionally validate the merged schema
    if validate_schema and db_registry:
        is_valid, pydantic_valid, detailed_valid, errors, _validation_errors = await _validate_updated_graph_schema(
            updated_graph_dict, 
            db_registry,
            perform_detailed_validation=True, # When applying override, always perform full validation if requested
            is_superuser=is_superuser,
            create_override_mode=create_override_mode,
        )
        if not is_valid:
            # Aggregate errors into a readable message
            error_summary = ["Validation failed for the merged graph schema:"]
            for scope, err_list in errors.items():
                error_summary.append(f"  Scope '{scope}':")
                for err_detail in err_list:
                    error_summary.append(f"    - {err_detail}")
            raise ValueError("\\n".join(error_summary))

    # 4. Convert back to GraphSchema Pydantic model
    # This also serves as a final structural validation against GraphSchema.
    try:
        final_graph_schema = GraphSchema.model_validate(updated_graph_dict)
        return final_graph_schema
    except PydanticValidationError as e:
        # This might happen if merging resulted in a structure not compliant with GraphSchema
        # even if individual validations passed or were skipped.
        raise ValueError(f"The final merged graph configuration is invalid against GraphSchema: {e}") from e

