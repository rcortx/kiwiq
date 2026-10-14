"""
Workflow graph builder module.

This module contains functionality for building and validating workflow graphs
based on graph schemas.
"""
from collections import defaultdict
from copy import copy
import json
from typing import Any, Dict, List, Optional, Set, Type, Tuple, cast, Union
from pydantic import ValidationError
from pydantic.fields import FieldInfo
from typing import Annotated, get_origin
from typing_extensions import TypedDict
from langchain_core.runnables import RunnableConfig
# from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode

# from workflow_service.graph.runtime.adapter import GraphRuntimeAdapter
from workflow_service.registry.schemas.reducers import ReducerRegistry, ReducerType
from workflow_service.graph.graph import EdgeSchema, GraphSchema
from workflow_service.registry.nodes.core.base import BaseNode
from workflow_service.registry.nodes.core.dynamic_nodes import BaseDynamicNode
from workflow_service.config.constants import GRAPH_STATE_SPECIAL_NODE_NAME, CONFIG_REDUCER_KEY, NODE_EXECUTION_ORDER_KEY, INPUT_NODE_NAME, OUTPUT_NODE_NAME
from workflow_service.utils.utils import get_central_state_field_key, get_node_output_state_key, is_central_state_special_node, is_dynamic_schema_node
from workflow_service.registry.registry import DBRegistry
from workflow_service.registry.errors import (
    NodeValidationError,
    get_node_doc_url,
    suggest_similar_fields,
)


class GraphEntities(TypedDict):
    """
    Container for all graph-related entities created during graph building.
    
    Attributes:
        node_instances: Dictionary mapping node IDs to node instances
        central_state: TypedDict class for the central state schema
        graph_state: TypedDict class for the complete graph state
        runtime_config: Configuration for runtime execution with input/output mappings
    """
    graph_schema: GraphSchema
    node_instances: Dict[str, BaseNode]
    central_state: Type
    graph_state: Type
    runtime_config: Dict[str, Dict[str, Any]]
    edges: Optional[List[EdgeSchema]]
    input_node_id: str
    output_node_id: str


class GraphBuilder:
    """
    Builder for workflow graphs.
    
    This class is responsible for building and validating workflow graphs
    based on graph schemas. It handles node instantiation, edge validation,
    and graph construction.
    
    Attributes:
        registry (Dict[str, Type[BaseNode]]): Registry of available node types.
        runtime_adapter (GraphRuntimeAdapter): Adapter for runtime execution.
    """
    
    def __init__(
        self, 
        registry: DBRegistry, 
        # runtime_adapter: Optional[GraphRuntimeAdapter] = None
    ):
        """
        Initialize the graph builder.
        
        Args:
            registry (Dict[str, Type[BaseNode]]): Registry of available node types.
            runtime_adapter (Optional[GraphRuntimeAdapter]): Adapter for runtime execution.
                If None, a LangGraphRuntimeAdapter will be created.
        """
        self.registry = registry
        # self.runtime_adapter = runtime_adapter or LangGraphRuntimeAdapter()
    def build_central_state_schema(self, graph_schema: GraphSchema, node_instances: Dict[str, BaseNode]) -> Dict[str, Dict[str, Any]]:
        """
        Build the central state schema for the graph.
        """
        # Build the central state!
        central_state_fields = {}
        for edge in graph_schema.edges:
            src_node_config = graph_schema.nodes.get(edge.src_node_id, None)
            src_node_name = src_node_config.node_name if src_node_config else None
            dst_node_config = graph_schema.nodes.get(edge.dst_node_id, None)
            dst_node_name = dst_node_config.node_name if dst_node_config else None
            if is_central_state_special_node(edge.dst_node_id) and is_central_state_special_node(edge.src_node_id):
                raise ValueError("Central state node cannot be connected to itself!")
            if is_central_state_special_node(edge.dst_node_id) and (not self.registry.is_dynamic_node(src_node_name)):
                # Edge from node to central state
                node_is_src = True
                node_id = edge.src_node_id
                node_name = src_node_name
                # node_config = dst_node_config
            elif is_central_state_special_node(edge.src_node_id) and (not self.registry.is_dynamic_node(dst_node_name)):
                # Edge from central state to node
                node_is_src = False
                node_id = edge.dst_node_id
                node_name = dst_node_name
                # node_config = src_node_config
            else:
                continue

            assert not is_central_state_special_node(node_name), "Central state node cannot be connected to itself!"

            for mapping in edge.mappings:
                # Get the field definition from the target node's input schema

                set_field_to_any_type = False
                src_field = mapping.src_field
                dst_field = mapping.dst_field
                if "." in mapping.src_field or "." in mapping.dst_field:
                    set_field_to_any_type = True
                    src_field = src_field.split(".")[0]
                    dst_field = dst_field.split(".")[0]

                if set_field_to_any_type:
                    field_info = FieldInfo(
                        default=None,
                        annotation=Any,
                    )
                    field_type_for_reducer = Any
                else:

                    node_cls = node_instances[node_id].__class__
                    node_schema = node_cls.output_schema_cls if node_is_src else node_cls.input_schema_cls
                    node_field_name = src_field if node_is_src else dst_field
                    field_info = node_schema.model_fields[node_field_name]

                    field_validation_result = node_schema._get_field_validation_result(node_field_name)
                    field_core_type_annotation = field_validation_result.core_type_annotation
                    field_core_type_class = field_validation_result.core_type_class
                    field_type_for_reducer = get_origin(field_core_type_annotation) if get_origin(field_core_type_annotation) else field_core_type_class

                central_state_field_name = dst_field if node_is_src else src_field
                central_state_field_key = get_central_state_field_key(central_state_field_name)
                
                # default reducer for type
                reducer = ReducerRegistry.get_reducer_for_type(field_type_for_reducer)
                
                # Check if field had preexisting reducer in metadata
                reducer_from_metadata = next((r for r in field_info.metadata if callable(r)), None)
                if reducer_from_metadata:
                    reducer = reducer_from_metadata
                
                annotation_to_set = field_info.annotation
                field_info_to_set = copy(field_info)
                # override reducer if specified in graph schema metadata
                if GRAPH_STATE_SPECIAL_NODE_NAME in graph_schema.metadata and CONFIG_REDUCER_KEY in graph_schema.metadata[GRAPH_STATE_SPECIAL_NODE_NAME]:
                    reducer_name = graph_schema.metadata[GRAPH_STATE_SPECIAL_NODE_NAME][CONFIG_REDUCER_KEY].get(central_state_field_name, None)
                    if reducer_name:
                        reducer = ReducerRegistry.get_reducer(reducer_name)
                        if ReducerType(reducer_name.lower()) == ReducerType.COLLECT_VALUES:
                            annotation_to_set = Any
                            field_info_to_set = FieldInfo(
                                default=None,
                                annotation=Any,
                            )
                field_to_set = (Annotated[annotation_to_set, reducer] if reducer else annotation_to_set, field_info_to_set)
                if central_state_field_key in central_state_fields:
                    if field_to_set[0] is Any:
                        field_to_set = central_state_fields[central_state_field_key]
                    assert central_state_fields[central_state_field_key][0] == field_to_set[0], f"Central state field '{central_state_field_key}' has multiple different types of edges to/from it!"
                    
                central_state_fields[central_state_field_key] = field_to_set
                # TODO: convert fields_info to Stategraph typeddict annotations with reducers!
        
        return central_state_fields

    def build_dynamic_nodes_from_schema_mappings(self, graph_schema: GraphSchema, central_state_fields: Dict[str, Dict[str, Any]], allow_non_user_editable_fields: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Build dynamic nodes from schema mappings.

        Algorithm:
        For each node:
            1. Identify which schemas a node needs constructed dynamically by checking 
               if its respective schema class has IS_DYNAMIC_SCHEMA set to True
            2. First create dictionary of node_id -> all dynamic schemas constructed via 
               ConstructDynamicSchema from provided node config fields; assert that for any 
               schema which is not marked dynamic, the schema config must not be provided in node config
            3. Iterate through all edges and for each edge, if the src/dst node is a dynamic node, 
               at least one field either src or dst has to be set by either non-dynamic node schema 
               or constructed dynamic schema from step 2
            4. Construct all remaining schemas using edge mappings and initialize the node class 
               with a combination of inferred schema fields and constructed dynamic schemas

        Notes:
            - Dynamic nodes can connect to each other if at least one of the fields in the edge is 
              dynamically defined in the graph schema
            - If a dynamic node connects to a central state field that doesn't exist yet, the 
              dynamic schema field will be used to create a field in the central state
        """
        dynamic_nodes = {}

        # Step 1 & 2: Identify nodes with dynamic schemas and use explicitly provided schemas
        explicit_dynamic_schemas = {}
        explicit_schema_fields = {}  # To store fields from explicitly defined schemas
        
        for node_id, node_config in graph_schema.nodes.items():
            # if is_central_state_special_node(node_config.node_name):
            #     continue
            node_name = node_config.node_name
            
            # Skip if not a dynamic node
            if not self.registry.is_dynamic_node(node_name):
                if node_config.dynamic_input_schema or node_config.dynamic_output_schema or node_config.dynamic_config_schema:
                    raise ValueError(f"Node {node_id} is not a dynamic node but has a dynamic schema!")
                continue
            
            # Get node class to check which schemas need to be dynamic
            node_cls = self.registry.get_node(node_name, node_config.node_version)
            
            # Initialize container for this node's explicit schemas
            explicit_dynamic_schemas[node_id] = {}
            explicit_schema_fields[node_id] = {}
            
            # Check for explicit dynamic input schema
            if is_dynamic_schema_node(node_cls.input_schema_cls):
                if node_config.dynamic_input_schema:
                    # Build the dynamic schema and store it
                    schema_name = f"{node_name}_{node_id}_InputSchema"
                    dynamic_schema = node_config.dynamic_input_schema.build_schema(schema_name=schema_name)
                    explicit_dynamic_schemas[node_id]['input_schema'] = dynamic_schema
                    explicit_schema_fields[node_id]['input_fields'] = {
                        field_name: (field_info.annotation, copy(field_info)) 
                        for field_name, field_info in node_cls.input_schema_cls.model_fields.items()
                    }
                    # NOTE: explicitly defined fields from graph config potentially override any pre-existing dynamic fields
                    explicit_schema_fields[node_id]['input_fields'].update({
                        field_name: (field_info.annotation, copy(field_info)) 
                        for field_name, field_info in dynamic_schema.model_fields.items()
                    })
                else:
                    # Mark that this node needs a dynamic input schema
                    explicit_dynamic_schemas[node_id]['input_schema'] = None
            elif node_config.dynamic_input_schema:
                raise ValueError(f"Node {node_id} has an explicitly provided dynamic_input_schema but its input_schema_cls is not marked as dynamic")
            
            # Check for explicit dynamic output schema
            if is_dynamic_schema_node(node_cls.output_schema_cls):
                if node_config.dynamic_output_schema:
                    # Build the dynamic schema and store it
                    schema_name = f"{node_name}_{node_id}_OutputSchema"
                    dynamic_schema = node_config.dynamic_output_schema.build_schema(schema_name=schema_name)
                    explicit_dynamic_schemas[node_id]['output_schema'] = dynamic_schema
                    explicit_schema_fields[node_id]['output_fields'] = {
                        field_name: (field_info.annotation, copy(field_info)) 
                        for field_name, field_info in node_cls.output_schema_cls.model_fields.items()
                    }
                    # NOTE: explicitly defined fields from graph config potentially override any pre-existing dynamic fields
                    explicit_schema_fields[node_id]['output_fields'].update({
                        field_name: (field_info.annotation, copy(field_info)) 
                        for field_name, field_info in dynamic_schema.model_fields.items()
                    })
                else:
                    # Mark that this node needs a dynamic output schema
                    explicit_dynamic_schemas[node_id]['output_schema'] = None
            elif node_config.dynamic_output_schema:
                raise ValueError(f"Node {node_id} has an explicitly provided dynamic_output_schema but its output_schema_cls is not marked as dynamic")
            
            # Check for explicit dynamic config schema
            if is_dynamic_schema_node(node_cls.config_schema_cls):
                if node_config.dynamic_config_schema:
                    # Build the dynamic schema and store it
                    schema_name = f"{node_name}_{node_id}_ConfigSchema"
                    dynamic_schema = node_config.dynamic_config_schema.build_schema(schema_name=schema_name)
                    explicit_dynamic_schemas[node_id]['config_schema'] = dynamic_schema
                    explicit_schema_fields[node_id]['config_fields'] = {
                        field_name: (field_info.annotation, copy(field_info)) 
                        for field_name, field_info in node_cls.config_schema_cls.model_fields.items()
                    }
                    # NOTE: explicitly defined fields from graph config potentially override any pre-existing dynamic fields
                    explicit_schema_fields[node_id]['config_fields'].update({
                        field_name: (field_info.annotation, copy(field_info)) 
                        for field_name, field_info in dynamic_schema.model_fields.items()
                    })
                else:
                    # Mark that this node needs a dynamic config schema
                    explicit_dynamic_schemas[node_id]['config_schema'] = None
            elif node_config.dynamic_config_schema:
                raise ValueError(f"Node {node_id} has an explicitly provided dynamic_config_schema but its config_schema_cls is not marked as dynamic")
            
            if node_id not in explicit_dynamic_schemas:
                explicit_dynamic_schemas[node_id] = {}
        
        
        
        # Step 3: Process edges to gather field information for dynamic schemas
        output_fields = defaultdict(dict)  # For source nodes with dynamic output schemas
        input_fields = defaultdict(dict)   # For target nodes with dynamic input schemas

        def field_src_validation_assert(field_src: str):
            assert field_src in ["output_fields", "input_fields"], "field_src must be either 'output_fields' or 'input_fields'"

        
        def get_field_info_if_field_explicitly_defined(node_id: str, field_name: str, field_src: str = "output_fields") -> bool:
            if (node_id in explicit_schema_fields and 
                field_src in explicit_schema_fields[node_id] and 
                field_name in explicit_schema_fields[node_id][field_src]):
                return explicit_schema_fields[node_id][field_src][field_name][1]
            return None
        
        def get_field_info_if_field_in_central_state(field_name: str) -> bool:
            central_state_field_key = get_central_state_field_key(field_name)
            if central_state_field_key in central_state_fields:
                _, field_info = central_state_fields[central_state_field_key]
                return field_info
            return None
        
        def get_field_info_from_normal_node(node_id: str, field_name: str, field_src: str = "output_fields") -> bool:
            field_src_validation_assert(field_src)
            node_config = graph_schema.nodes[node_id]
            # Edge from dynamic node to regular node
            node_cls = self.registry.get_node(node_config.node_name, node_config.node_version)
            if not node_cls:
                available_nodes = list(getattr(self.registry, "_nodes", {}).keys())
                similar = suggest_similar_fields(node_config.node_name, available_nodes)
                fix_msg = "Verify the node is registered with DBRegistry and check spelling."
                if similar:
                    fix_msg += f" {similar}"
                raise NodeValidationError(
                    message=f"Node '{node_config.node_name}' not found in registry",
                    node_id=node_id,
                    node_name=node_config.node_name,
                    expected=f"A registered node type (available: {available_nodes[:5]}...)" if available_nodes else "A registered node type",
                    actual=node_config.node_name,
                    suggested_fix=fix_msg,
                    doc_url=get_node_doc_url(node_config.node_name),
                    error_type="node_not_found",
                )
            
            node_schema = node_cls.input_schema_cls if field_src == "input_fields" else node_cls.output_schema_cls
            if node_schema is not None and field_name in node_schema.model_fields:  #  and (not is_dynamic_schema_node(node_schema)):
                field_info = node_schema.model_fields[field_name]
                return field_info
            return None
        
        def get_field_info_for_field_name(node_id: str, field_name: str, other_node_id: str, other_field_name: str, field_src: str, other_field_src: str, fields_dict: Dict[str, Dict[str, Any]], other_fields_dict: Dict[str, Dict[str, Any]]) -> Union[FieldInfo, None]:
            field_src_validation_assert(field_src)
            field_src_validation_assert(other_field_src)
            # Get previously set field from fields_dict
            # NOTE: this behaviour also doesn't let latter found field_info to replace previously found field_infos!
            # TODO: FIXME --> try to get field with most wider applicability!
            field_info = get_field_info_from_fields_dict(node_id, field_name, fields_dict)
            # get field info from self
            # NOTE: if self is normal schema and not dynamic, we can still fetch field_info for future use since during schema setup, dynamic schema setting will be checked before replacing the schema!
            field_info = field_info or (get_field_info_if_field_explicitly_defined(node_id, field_name, field_src) or get_field_info_from_normal_node(node_id, field_name, field_src))
            
            # get field info from central state if other node is central state
            if is_central_state_special_node(other_node_id):
                field_info = field_info or get_field_info_if_field_in_central_state(other_field_name)
            else:            
                # get field info from other node
                field_info = field_info or (
                    get_field_info_from_fields_dict(other_node_id, other_field_name, other_fields_dict) or
                    get_field_info_if_field_explicitly_defined(other_node_id, other_field_name, other_field_src) or
                    get_field_info_from_normal_node(other_node_id, other_field_name, other_field_src)
                )
            
            return field_info
        
        def get_field_info_from_fields_dict(node_id: str, field_name: str, fields_dict: Dict[str, Dict[str, Any]]):
            return fields_dict[node_id][field_name][1] if node_id in fields_dict and field_name in fields_dict[node_id] else None

        
        def try_recuperate_missing_field_info(node_id: str, field_name: str, other_node_id: str, other_field_name: str, fields_dict: Dict[str, Dict[str, Any]], other_fields_dict: Dict[str, Dict[str, Any]]) -> Union[FieldInfo, None]:
            if is_central_state_special_node(other_node_id):
                field_info = get_field_info_if_field_in_central_state(other_field_name)
            else:
                field_info = get_field_info_from_fields_dict(other_node_id, other_field_name, other_fields_dict)
            
            # hack to set Any type on unavailable field!
            if not field_info:
                can_set_any_type = False
                if not is_central_state_special_node(other_node_id):
                    other_node_config = graph_schema.nodes.get(other_node_id, None)
                    if self.registry.is_dynamic_node(other_node_config.node_name):
                        can_set_any_type = True
                else:
                    can_set_any_type = True
                if can_set_any_type:
                    field_info = FieldInfo(
                        default=None,
                        annotation=Any,
                    )

            if field_info:
                copy_field_info_to_fields_dict(node_id, field_name, fields_dict, field_info)
                return field_info
            return None
        
        def set_field_info_to_any_type(node_id: str, field_name: str, fields_dict: Dict[str, Dict[str, Any]]):
            field_info = FieldInfo(
                default=None,
                annotation=Any,
            )
            copy_field_info_to_fields_dict(node_id, field_name, fields_dict, field_info)
            return field_info
        
        def copy_field_info_to_fields_dict(node_id: str, field_name: str, fields_dict: Dict[str, Dict[str, Any]], field_info: FieldInfo):
            if node_id not in fields_dict:
                fields_dict[node_id] = {}
            fields_dict[node_id][field_name] = (field_info.annotation, copy(field_info))
        
        def copy_field_to_central_state_fields_no_replace(field_name: str, field_info: FieldInfo):
            central_state_field_key = get_central_state_field_key(field_name)
            # If field exists in central state, use it
            if central_state_field_key not in central_state_fields:
                # Add to central state fields
                reducer = ReducerRegistry.get_reducer_for_type(field_info.annotation)
                annotation_to_set = field_info.annotation
                field_info_to_set = copy(field_info)
                if GRAPH_STATE_SPECIAL_NODE_NAME in graph_schema.metadata and CONFIG_REDUCER_KEY in graph_schema.metadata[GRAPH_STATE_SPECIAL_NODE_NAME]:
                    reducer_name = graph_schema.metadata[GRAPH_STATE_SPECIAL_NODE_NAME][CONFIG_REDUCER_KEY].get(field_name, None)
                    if reducer_name:
                        reducer = ReducerRegistry.get_reducer(reducer_name)
                        if ReducerType(reducer_name.lower()) == ReducerType.COLLECT_VALUES:
                            annotation_to_set = Any
                            field_info_to_set = FieldInfo(
                                default=None,
                                annotation=Any,
                            )
                central_state_fields[central_state_field_key] = (
                    Annotated[annotation_to_set, reducer] if reducer else annotation_to_set, 
                    field_info_to_set
                )
        
        for edge in graph_schema.edges:
            assert not (is_central_state_special_node(edge.src_node_id) and is_central_state_special_node(edge.dst_node_id)), "Source or destination node both can't be central state special nodes!"
            src_node_id = edge.src_node_id
            dst_node_id = edge.dst_node_id
            src_node_config = graph_schema.nodes.get(src_node_id, None)
            dst_node_config = graph_schema.nodes.get(dst_node_id, None)

            if not ( ((src_node_config is not None) and self.registry.is_dynamic_node(src_node_config.node_name)) or 
                    ((dst_node_config is not None) and self.registry.is_dynamic_node(dst_node_config.node_name)) ):
                continue
            
            for mapping in edge.mappings:
                if "." in mapping.src_field or "." in mapping.dst_field:
                    continue
                if not is_central_state_special_node(src_node_id):
                    field_info = get_field_info_for_field_name(src_node_id, mapping.src_field, dst_node_id, mapping.dst_field, "output_fields", "input_fields", output_fields, input_fields)
                    if field_info:
                        copy_field_info_to_fields_dict(src_node_id, mapping.src_field, output_fields, field_info)
                        if is_central_state_special_node(dst_node_id):
                            copy_field_to_central_state_fields_no_replace(mapping.dst_field, field_info)
                
                if not is_central_state_special_node(dst_node_id):
                    field_info = get_field_info_for_field_name(dst_node_id, mapping.dst_field, src_node_id, mapping.src_field, "input_fields", "output_fields", input_fields, output_fields)
                    if field_info:
                        copy_field_info_to_fields_dict(dst_node_id, mapping.dst_field, input_fields, field_info)
                        if is_central_state_special_node(src_node_id):
                            copy_field_to_central_state_fields_no_replace(mapping.src_field, field_info)
        
        # Step 4: Recuperate missing field info
        for edge in graph_schema.edges:
            src_node_id = edge.src_node_id
            dst_node_id = edge.dst_node_id
            src_node_config = graph_schema.nodes.get(src_node_id, None)
            dst_node_config = graph_schema.nodes.get(dst_node_id, None)

            if not ( ((src_node_config is not None) and self.registry.is_dynamic_node(src_node_config.node_name)) or 
                    ((dst_node_config is not None) and self.registry.is_dynamic_node(dst_node_config.node_name)) ):
                continue
            
            for mapping in edge.mappings:
                set_field_to_any_type = False
                src_field = mapping.src_field
                dst_field = mapping.dst_field
                if "." in mapping.src_field or "." in mapping.dst_field:
                    set_field_to_any_type = True
                    src_field = src_field.split(".")[0]
                    dst_field = dst_field.split(".")[0]
                if not is_central_state_special_node(src_node_id):
                    src_node_cls = self.registry.get_node(src_node_config.node_name, src_node_config.node_version)
                    if self.registry.is_dynamic_node(src_node_config.node_name) and is_dynamic_schema_node(src_node_cls.output_schema_cls) and (not get_field_info_from_fields_dict(edge.src_node_id, src_field, output_fields)):
                        # print(f"\n\nRecuperating field `{mapping.src_field}` for node `{edge.src_node_id}` -- > `{src_node_config.node_name}` from `{edge.dst_node_id}`'s `{mapping.dst_field}`! ; \nis_dynamic_schema_node(src_node_cls.output_schema_cls): {is_dynamic_schema_node(src_node_cls.output_schema_cls)}\n\n")
                        if set_field_to_any_type:
                            recuperated_field_info = set_field_info_to_any_type(edge.src_node_id, src_field, output_fields)
                        else:
                            recuperated_field_info = try_recuperate_missing_field_info(edge.src_node_id, src_field, edge.dst_node_id, dst_field, output_fields, input_fields)
                        if not recuperated_field_info:
                            raise ValueError(f"Field `{mapping.src_field}` for node `{edge.src_node_id}` is not found in output fields and couldn't be recuperated from `{edge.dst_node_id}`'s `{mapping.dst_field}`!")
                        if is_central_state_special_node(dst_node_id):
                            copy_field_to_central_state_fields_no_replace(dst_field, recuperated_field_info)
                
                if not is_central_state_special_node(dst_node_id):
                    dst_node_cls = self.registry.get_node(dst_node_config.node_name, dst_node_config.node_version)
                    if self.registry.is_dynamic_node(dst_node_config.node_name) and is_dynamic_schema_node(dst_node_cls.input_schema_cls) and (not get_field_info_from_fields_dict(edge.dst_node_id, dst_field, input_fields)):
                        # print(f"\n\nRecuperating field `{mapping.dst_field}` for node `{edge.dst_node_id}` -- > `{dst_node_config.node_name}` from `{edge.src_node_id}`'s `{mapping.src_field}`! ; \nis_dynamic_schema_node(dst_node_cls.input_schema_cls): {is_dynamic_schema_node(dst_node_cls.input_schema_cls)}\n\n")
                        if set_field_to_any_type:
                            recuperated_field_info = set_field_info_to_any_type(edge.dst_node_id, dst_field, input_fields)
                        else:
                            recuperated_field_info = try_recuperate_missing_field_info(edge.dst_node_id, dst_field, edge.src_node_id, src_field, input_fields, output_fields)
                        if not recuperated_field_info:
                            raise ValueError(f"Field `{mapping.dst_field}` for node `{edge.dst_node_id}` is not found in input fields and couldn't be recuperated from `{edge.src_node_id}`'s `{mapping.src_field}`!")
                        if is_central_state_special_node(src_node_id):
                            copy_field_to_central_state_fields_no_replace(src_field, recuperated_field_info)
        
        # Step 5: Construct dynamic nodes with gathered schema information
        for node_id, schemas in explicit_dynamic_schemas.items():
            node_config = graph_schema.nodes[node_id]
            node_name = node_config.node_name
            node_version = node_config.node_version
            
            # Get base node class
            node_cls: BaseDynamicNode = self.registry.get_node(node_name, node_version)
            
            # Prepare kwargs for node creation
            kwargs = defaultdict(dict)
            
            # Input fields handling
            if 'input_schema' in schemas:
                if schemas['input_schema']:  # Explicit schema was built
                    kwargs['input_fields'] = explicit_schema_fields[node_id].get('input_fields', {})
                if node_id in input_fields and is_dynamic_schema_node(node_cls.input_schema_cls) and node_config.enable_dynamic_fields_from_edges:  # Fields gathered from connections
                    # Only add inferred fields if they haven't been explicitly overwritten by manual dynamic schema config!
                    for field_name, (field_type, field_info) in input_fields[node_id].items():
                        if field_name not in kwargs['input_fields']:
                            kwargs['input_fields'][field_name] = (field_type, copy(field_info))
            
            # Output fields handling
            if 'output_schema' in schemas:
                if schemas['output_schema']:  # Explicit schema was built
                    kwargs['output_fields'] = explicit_schema_fields[node_id].get('output_fields', {})
                if node_id in output_fields and is_dynamic_schema_node(node_cls.output_schema_cls) and node_config.enable_dynamic_fields_from_edges:  # Fields gathered from connections
                    # Only add inferred fields if they haven't been explicitly overwritten by manual dynamic schema config!
                    for field_name, (field_type, field_info) in output_fields[node_id].items():
                        if field_name not in kwargs['output_fields']:
                            kwargs['output_fields'][field_name] = (field_type, copy(field_info))
            
            if 'config_schema' in schemas:
                if schemas['config_schema']:  # Explicit schema was built
                    kwargs['config_fields'] = explicit_schema_fields[node_id].get('config_fields', {})
                # if node_id in explicit_schema_fields:  # Fields gathered from connections
                #     kwargs['config_fields'] = explicit_schema_fields[node_id].get('config_fields', {})
            
            # Skip if we couldn't determine required schemas

            # Propagate input fields to output fields and vice versa for input and output nodes
            # This copies input / output schema to each other
            # We wanna be careful with this behaviour and don't want it in every other node!
            if node_name in [INPUT_NODE_NAME, OUTPUT_NODE_NAME]:
                kwargs["propagate_input_fields_to_output_fields"] = True
                kwargs["propagate_output_fields_to_input_fields"] = True

            # print(kwargs)
            if not kwargs:
                # Could be that dynamic node doesn't require any schema since this is only dependency node!
                # raise ValueError(f"Dynamic schemas couldn't be determined for node `{node_id}` --> `{node_name}`!")
                dynamic_node_cls = node_cls
            else:
                # Create dynamic node class with appropriate schemas
                dynamic_node_cls = node_cls.create_node_with_input_output_schemas(**kwargs)
            
            # import ipdb; ipdb.set_trace()
            # Instantiate the node
            dynamic_nodes[node_id] = dynamic_node_cls(
                node_id=node_id,
                config=node_config.node_config,
                private_input_mode=node_config.private_input_mode,
                private_output_passthrough_data_to_central_state_keys=node_config.private_output_passthrough_data_to_central_state_keys,
                private_output_to_central_state_node_output_key=node_config.private_output_to_central_state_node_output_key,
                private_output_mode=node_config.private_output_mode,
                output_private_output_to_central_state=node_config.output_private_output_to_central_state,
                read_private_input_passthrough_data_to_input_field_mappings=node_config.read_private_input_passthrough_data_to_input_field_mappings,
                write_to_private_output_passthrough_data_from_output_mappings=node_config.write_to_private_output_passthrough_data_from_output_mappings,
                allow_non_user_editable_fields_in_config=allow_non_user_editable_fields
            )
        
        return dynamic_nodes
    
    @staticmethod
    def build_central_state_typeddict(input_schema: Type, central_state_fields: Dict[str, Dict[str, Any]]) -> Type:
        """
        Build the central state for the graph.
        """
        # Dynamically create a TypedDict for central state
        # Extract field names and annotations from central_state_fields
        fields = {
            field_name: field_annotation 
            for field_name, (field_annotation, _) in central_state_fields.items()
        }
        if input_schema is not None:
            for field_name, field_info in input_schema.model_fields.items():
                fields[get_central_state_field_key(field_name)] = field_info.annotation
        
        
        node_order_field_key = get_central_state_field_key(NODE_EXECUTION_ORDER_KEY)
        assert not any(field_name in [node_order_field_key] for field_name in fields), f"Central state field name {node_order_field_key} is reserved for internal use!"

        node_order_field = Annotated[List[str], ReducerRegistry.get_reducer(ReducerType.APPEND_LIST)]
        fields[node_order_field_key] = node_order_field

        # Create the TypedDict dynamically with the extracted fields
        CentralState = TypedDict('CentralState', fields)
        return CentralState
    
    
    def build_nodes_from_schema_mappings(self, graph_schema: GraphSchema, allow_non_user_editable_fields: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Build mappings of input and output schemas for each node in the graph.
        
        Args:
            graph_schema (GraphSchema): The graph schema to analyze.
            
        Returns:
            Dict[str, Dict[str, Any]]: Mapping of node IDs to schema information.
        """
        node_instances = {}
        input_node_id = graph_schema.input_node_id
        output_node_id = graph_schema.output_node_id
        
        # Extract schema information for each node
        for node_id, node_config in graph_schema.nodes.items():
            # if is_central_state_special_node(node_config.node_name): 
            #     continue
            assert node_id != GRAPH_STATE_SPECIAL_NODE_NAME, "Central state special node ID is reserved for internal use!"
            if self.registry.is_input_node(node_config.node_name):
                assert node_id == input_node_id, "Input node ID for graph config is different from input node config's node ID!"
            elif self.registry.is_output_node(node_config.node_name):
                assert node_id == output_node_id, "Output node ID for graph config is different from output node config's node ID!"
            else:
                assert node_id == node_config.node_id, "Node ID and node config node ID must be same!"
            if self.registry.is_non_dynamic_normal_node(node_config.node_name):            
                node_cls = self.registry.get_node(node_config.node_name, node_config.node_version)
                node_instances[node_id] = node_cls(node_id=node_id, 
                                                   config=node_config.node_config, 
                                                   private_input_mode=node_config.private_input_mode, 
                                                   private_output_mode=node_config.private_output_mode,
                                                   private_output_passthrough_data_to_central_state_keys=node_config.private_output_passthrough_data_to_central_state_keys,
                                                   private_output_to_central_state_node_output_key=node_config.private_output_to_central_state_node_output_key,
                                                   output_private_output_to_central_state=node_config.output_private_output_to_central_state,
                                                   read_private_input_passthrough_data_to_input_field_mappings=node_config.read_private_input_passthrough_data_to_input_field_mappings,
                                                   write_to_private_output_passthrough_data_from_output_mappings=node_config.write_to_private_output_passthrough_data_from_output_mappings,
                                                   allow_non_user_editable_fields_in_config=allow_non_user_editable_fields)
        
        return node_instances
    
    @staticmethod
    def build_graph_state(node_instances: Dict[str, BaseNode], central_state: Type) -> Type:
        """
        Build the graph state for the graph.
        """
        fields = copy(central_state.__annotations__)
        # if input_schema is not None:
        #     for field_name, field_info in input_schema.model_fields.items():
        #         fields[field_name] = field_info.annotation
        for node_id, node_instance in node_instances.items():
            # NOTE: this is only for debugging parallel branches outputs!
            if node_instance.private_input_mode and node_instance.output_private_output_to_central_state:
                node_output_field = Annotated[node_instance.output_schema_cls, ReducerRegistry.get_reducer(ReducerType.COLLECT_VALUES)]
            else:
                node_output_field = node_instance.output_schema_cls
            fields[get_node_output_state_key(node_id)] = node_output_field
        GraphState = TypedDict('GraphState', fields)
        return GraphState
    
    @staticmethod
    def build_runtime_config(graph_schema: GraphSchema, node_instances: Dict[str, BaseNode]) -> Dict[str, Dict[str, Any]]:
        """
        Build runtime configuration for node inputs and outputs based on graph edges.
        
        This method analyzes the graph edges to determine:
        1. Input mappings: How each node's input fields are connected to source nodes or central state
        2. Output mappings: How each node's output fields are connected to the central graph state
        
        Args:
            graph_schema (GraphSchema): The graph schema containing nodes and edges
            node_instances (Dict[str, BaseNode]): Dictionary of instantiated node objects
            
        Returns:
            Dict[str, Dict[str, Any]]: Runtime configuration with input and output mappings
            
        Structure:
        {
            "inputs": {
                "node_id": {
                    "input_field": ["source_node_id", "source_field"] or "central_state_field"
                }
            },
            "outputs": {
                "node_id": {
                    "output_field": "central_state_field"
                }
            }
        }
        """
        runtime_config = {
            "inputs": {},
            "outputs": {},
            "outgoing_edges": defaultdict(dict),
            "incoming_edges": defaultdict(dict),
        }
        edge_out_from_src = defaultdict(list)
        edge_in_to_dst = defaultdict(list)
        # Process each edge in the graph
        for edge_idx, edge in enumerate(graph_schema.edges):
            source_id = edge.src_node_id
            target_id = edge.dst_node_id
            edge_out_from_src[source_id].append(target_id)
            edge_in_to_dst[target_id].append(source_id)
            
            # Handle edges from central state to nodes (inputs)
            if is_central_state_special_node(source_id):
                # Edge from central state to a node
                if target_id not in runtime_config["inputs"]:
                    runtime_config["inputs"][target_id] = {}
                
                # Process each field mapping in this edge
                for mapping in edge.mappings:
                    # Map the target node's input field to the central state field
                    # assert mapping.dst_field not in runtime_config["inputs"][target_id], f"Input field '{mapping.dst_field}' for node '{target_id}' already has an incoming edge mapping!"
                    if mapping.dst_field in runtime_config["inputs"][target_id]:
                        runtime_config["inputs"][target_id][mapping.dst_field] = runtime_config["inputs"][target_id][mapping.dst_field] + [mapping.src_field]
                    else:
                        runtime_config["inputs"][target_id][mapping.dst_field] = [mapping.src_field]  # get_central_state_field_key
                    
            
            # Handle edges from nodes to central state (outputs)
            elif is_central_state_special_node(target_id):
                # Edge from a node to central state
                if source_id not in runtime_config["outputs"]:
                    runtime_config["outputs"][source_id] = {}
                
                # Process each field mapping in this edge
                for mapping in edge.mappings:
                    # Map the source node's output field to the central state field
                    runtime_config["outputs"][source_id][mapping.src_field] = mapping.dst_field
            
            # Handle regular edges between nodes (inputs)
            else:
                # Edge from one node to another
                if target_id not in runtime_config["inputs"]:
                    runtime_config["inputs"][target_id] = {}
                
                runtime_config["outgoing_edges"][source_id][target_id] = edge
                runtime_config["incoming_edges"][target_id][source_id] = edge
                
                # Process each field mapping in this edge
                for mapping in edge.mappings:
                    # Map the target node's input field to the source node and field
                    # assert mapping.dst_field not in runtime_config["inputs"][target_id], f"Input field '{mapping.dst_field}' for node '{target_id}' already has an incoming edge mapping!"
                    # TODO: FIXME: disabling this check for router node case where there could be FAN IN from multiple nodes after the router to a node further ahead!
                    # Also the case for loops!
                    if mapping.dst_field in runtime_config["inputs"][target_id]:
                        runtime_config["inputs"][target_id][mapping.dst_field] = runtime_config["inputs"][target_id][mapping.dst_field] + [[source_id, mapping.src_field]]
                    else:
                        runtime_config["inputs"][target_id][mapping.dst_field] = [[source_id, mapping.src_field]]
        
        # Add input node configuration to runtime config
        # This ensures the input node can receive data from the graph invocation
        if graph_schema.input_node_id and graph_schema.input_node_id in node_instances:
            input_node = node_instances[graph_schema.input_node_id]
            if input_node.input_schema_cls is not None:
                # Create input mappings for the input node based on its schema
                if graph_schema.input_node_id not in runtime_config["inputs"]:
                    runtime_config["inputs"][graph_schema.input_node_id] = {}
                
                # For each field in the input schema, create a direct mapping
                # This allows the input data passed to graph.invoke() to be properly routed
                for field_name in input_node.input_schema_cls.model_fields:
                    # Only add if not already mapped by an edge
                    if field_name not in runtime_config["inputs"][graph_schema.input_node_id]:
                        # Map directly from the input data to the input node's field
                        # get_central_state_field_key(field_name) not needed! Node base input builder func handles central state inputs!
                        runtime_config["inputs"][graph_schema.input_node_id][field_name] = [field_name]
        
        # Validate that all required input fields have mappings
        #     validate certain types of dynamic nodes like HITL / router have valid outgoing edges!
        for node_id, node_instance in node_instances.items():
            # Skip validation for special nodes
            if node_id in [graph_schema.input_node_id, graph_schema.output_node_id]:
                continue

            if DBRegistry.is_node_instance_hitl(node_instance) or DBRegistry.is_node_instance_router(node_instance):
                assert len(edge_out_from_src[node_id]) > 0, f"HITL or routing node '{node_id}' must have outgoing edges!"
                
            # Check if node has input schema
            if not node_instance.input_schema_cls:
                continue
                
            # Get required input fields
            required_fields = node_instance.input_schema_cls.get_required_fields()
            
            # Ensure all required fields have mappings
            if required_fields:
                if node_id not in runtime_config["inputs"]:
                    required_list = list(required_fields)
                    raise NodeValidationError(
                        message=f"Node '{node_id}' has required input fields but no input mappings",
                        node_id=node_id,
                        node_name=node_instance.node_name,
                        field_name="inputs",
                        expected=f"Input edge mappings for required fields: {required_list}",
                        actual="No input mappings found",
                        suggested_fix=f"Connect incoming edges to node '{node_id}' in the graph schema for required fields: {required_list}.",
                        doc_url=get_node_doc_url(node_instance.node_name),
                        error_type="missing_input_mappings",
                    )
                
                fields_with_edges = set(f.split(".")[0] for f in runtime_config["inputs"][node_id])
                for field in required_fields:
                    if field not in fields_with_edges:
                        similar = suggest_similar_fields(field, list(fields_with_edges))
                        fix_suggestion = f"Add an edge in the graph schema targeting '{node_id}.{field}'."
                        if similar:
                            fix_suggestion += f" {similar}"
                        raise NodeValidationError(
                            message=f"Required input field '{field}' for node '{node_id}' has no mapping",
                            node_id=node_id,
                            node_name=node_instance.node_name,
                            field_name=field,
                            expected=f"Mapping for required field '{field}'",
                            actual="Unmapped (available mapped fields: " + (", ".join(fields_with_edges) if fields_with_edges else "none") + ")",
                            suggested_fix=fix_suggestion,
                            doc_url=get_node_doc_url(node_instance.node_name),
                            error_type="missing_field_mapping",
                        )
        
        return runtime_config

    def build_graph_entities(
        self, 
        graph_schema: GraphSchema,
        allow_non_user_editable_fields: bool = True,
        prefect_mode: bool = False,
        runtime_metadata: Optional[Dict[str, Any]] = None
    ) -> GraphEntities:
        """
        Build all graph entities from a graph schema.
        
        This method creates all the necessary components for a workflow graph:
        - Instantiates all node instances
        - Builds the central state schema
        - Creates the graph state TypedDict
        - Generates the runtime configuration
        
        Args:
            graph_schema (GraphSchema): The graph schema to build from.
            
        Returns:
            GraphEntities: A container with all graph components including node instances,
                          central state, graph state, and runtime configuration.
            
        Raises:
            ValueError: If the graph schema is invalid.
        """
        
        # Build node schema mappings
        node_instances = self.build_nodes_from_schema_mappings(graph_schema, allow_non_user_editable_fields=allow_non_user_editable_fields)

        # Build input schema for input node based on outgoing edges

        # NOTE: there can be multiple HITL nodes or any type of other nodes apart from input / output!!!

        central_state_fields = self.build_central_state_schema(graph_schema, node_instances)

        dynamic_nodes = self.build_dynamic_nodes_from_schema_mappings(graph_schema, central_state_fields, allow_non_user_editable_fields=allow_non_user_editable_fields)
        node_instances.update(dynamic_nodes)

        input_schema = node_instances[graph_schema.input_node_id].input_schema_cls

        # self.node_instances = node_instances
        central_state = GraphBuilder.build_central_state_typeddict(input_schema, central_state_fields)
        # self.central_state = central_state
        
        graph_state = GraphBuilder.build_graph_state(node_instances, central_state)
        # self.graph_state = graph_state
        runtime_config = GraphBuilder.build_runtime_config(graph_schema, node_instances)
        # self.runtime_config = runtime_config

        # TODO: FIXME!
        # Add external objects like registry!
        if not runtime_config.get("external"):
            runtime_config["external"] = {}
        runtime_config["external"]["registry"] = self.registry

        # if prefect_mode:
        for node_id, node_instance in node_instances.items():
            node_instance.prefect_mode = prefect_mode
            node_instance.runtime_metadata = runtime_metadata
            if node_id == graph_schema.input_node_id:
                node_instance.is_input_node = True
            if node_id == graph_schema.output_node_id:
                node_instance.is_output_node = True
        
        # Instantiate the GraphEntities with all components
        graph_entities = GraphEntities(
            graph_schema=graph_schema,
            node_instances=node_instances,
            central_state=central_state,
            graph_state=graph_state,
            runtime_config=runtime_config,
            edges=graph_schema.edges,
            input_node_id=graph_schema.input_node_id,
            output_node_id=graph_schema.output_node_id
        )
        
        # Use the runtime adapter to build the graph
        return graph_entities
