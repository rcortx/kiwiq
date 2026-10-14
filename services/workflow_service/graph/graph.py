"""
Edge schema definitions for workflow connections.

This module contains the schema definitions for edges in the workflow system.
Edges connect nodes in the workflow graph and define how data flows between them.
"""
from typing import Any, Dict, List, Optional, Union, Set, Literal
from typing_extensions import Self
from pydantic import Field, BaseModel, model_validator, ConfigDict
from workflow_service.config.constants import INPUT_NODE_NAME, OUTPUT_NODE_NAME
from workflow_service.utils.utils import is_central_state_special_node
from workflow_service.registry.nodes.core.dynamic_nodes import ConstructDynamicSchema

class EdgeMapping(BaseModel):
    """
    Schema for mapping data from source node output to target node input.
    
    This schema defines a single mapping between a field in the source node's output
    and a field in the target node's input. Complex mappings can be created using
    dot notation for nested fields.
    
    Dot notation examples:
    - "field.nested_field" to access nested objects
    - "items.0.name" to access list elements by index
    - "data.user.profile.email" for deeply nested paths
    
    Attributes:
        src_field (str): Field name in the source node's output schema. Supports dot notation
                        for nested field access (e.g., "result.data.user_name").
        dst_field (str): Field name in the target node's input schema. Supports dot notation
                        for setting nested field values (e.g., "input.user_data.name").
    """
    model_config = ConfigDict(extra='forbid')  # Allow additional arguments during model init!
    src_field: str = Field(..., description="Field name in the source node's output. Supports dot notation for nested access (e.g., 'data.result.value')")
    dst_field: str = Field(..., description="Field name in the target node's input. Supports dot notation for nested assignment (e.g., 'input.nested.field')")
    description: Optional[str] = Field(None, description="Description of the edge mapping; this is not used anywhere currently!")
    override_type_validation: Optional[bool] = Field(False, description=" [DEPRECATED] Override type validation for the src and target field")
    # transform: Optional[str] = Field(None, description="Optional transformation to apply")


class BaseEdgeSchema(BaseModel):
    """
    Base schema for edges in the workflow graph.
    """
    model_config = ConfigDict(extra='forbid')  # Allow additional arguments during model init!
    dst_node_id: str = Field(..., description="ID of the target node")
    data_only_edge: Optional[bool] = Field(False, description="If True, the edge will only pass data from source to target and will not be a graph flow edge!")
    mappings: Optional[List[EdgeMapping]] = Field(default_factory=list, description="Field mappings from source to target")
    description: Optional[str] = Field(None, description="Description of the edge mapping; this is not used anywhere currently!")

class EdgeSchema(BaseEdgeSchema):
    """
    Schema for edges in the workflow graph.
    
    Edges connect nodes in the workflow graph and define how data flows between them.
    Each edge connects a source node's output to a target node's input, with optional
    field mappings to specify which data fields flow where.
    
    Attributes:
        src_node_id (str): ID of the source node.
        dst_node_id (str): ID of the target node.
        mappings (Optional[List[EdgeMapping]]): Optional list of field mappings from source to target.
    """
    src_node_id: str = Field(..., description="ID of the source node")

class NodeConfig(BaseModel):
    """
    Schema for node configuration in the workflow graph.
    
    This schema defines the configuration for a single node in the workflow graph.
    It includes the node's name, version, and any additional configuration parameters
    specific to that node type.
    
    Attributes:
        node_name (str): Name of the node type.
        node_version (str): Version of the node implementation.
        node_config (Dict[str, Any]): Configuration parameters specific to this node type.
    """
    model_config = ConfigDict(extra='forbid')  # Allow additional arguments during model init!
    node_id: str = Field(..., description="Unique ID of the node")
    node_name: str = Field(..., description="Name of the node type. There may be multiple nodes with same name / type!")
    node_category: Optional[str] = Field(None, description="Category of the node type. This is used to group nodes together in the UI.")
    positions: Optional[Dict[Any, Any]] = Field(None, description="Position of the node in the UI. This is used to sort nodes in the UI. The keys are x and y coordinates and the values are the position in the UI.")
    node_version: Optional[str] = Field(None, description="Version of the node implementation") 
    node_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Node-specific configuration parameters")
    private_input_mode: Optional[bool] = Field(False, description="Enable private input mode for the node which means the node will receive direct inputs instead of whole graph state, this is useful for map/reduce or branching patterns or maintaining private states.")
    private_output_mode: Optional[bool] = Field(False, description="Enable private output mode for the node which means the node will send direct outputs to connected nodes, this is useful for map/reduce or branching patterns or maintaining private states. NOTE: the branches will converge to the next node where this is unset and they are collapsing to it, and that node will receive the full graph state. For reducing, you have to use the central state field with the correct reducer!")
    output_private_output_to_central_state: Optional[bool] = Field(False, description="Enable output private output to central state for the node which means the node will send direct outputs to the central state too for debugging purposes.")
    private_output_passthrough_data_to_central_state_keys: Optional[List[str]] = Field(None, description="These keys are passed through the private output data to the central state directly! eg usecase: while using map list router node, preserve unique IDs of each element as passthrough data that gets collected in items collected in central state")
    private_output_to_central_state_node_output_key: Optional[str] = Field("output", description="This key is used to send the central state output to from the node output (for each mapped edge to central state) in cases when there's extra private_output_passthrough_data ...")
    edges: Optional[List[BaseEdgeSchema]] = Field(default_factory=list, description="List of edges connecting the node to other nodes. Either this node may declare edges here or the aggregated edges field from graph_schema but not both!")
    # Optional mappings for reading from and writing to passthrough data
    # Only used when private_input_mode and private_output_mode are True, respectively
    read_private_input_passthrough_data_to_input_field_mappings: Optional[Dict[str, str]] = Field(None, description="Maps passthrough data keys to input field names when private_input_mode is True. Supports dot notation for nested paths. Format: {'passthrough.nested.key': 'input_field_name'} or {'passthrough_key': 'nested.input.field'}")
    write_to_private_output_passthrough_data_from_output_mappings: Optional[Dict[str, str]] = Field(None, description="Maps output field names to passthrough data keys when private_output_mode is True. Supports dot notation for nested paths. Format: {'output.nested.field': 'passthrough_key'} or {'output_field_name': 'nested.passthrough.key'}")
    
    # TODO: implement support for JSON Schema which can be converted to Pydantic models using: https://github.com/koxudaxi/datamodel-code-generator
    #     https://koxudaxi.github.io/datamodel-code-generator/using_as_module/
    #     https://koxudaxi.github.io/datamodel-code-generator/using_as_module/
    #     https://koxudaxi.github.io/datamodel-code-generator/using_as_module/
    dynamic_input_schema: Optional[ConstructDynamicSchema] = Field(None, description="Dynamic schema for the node input")
    dynamic_output_schema: Optional[ConstructDynamicSchema] = Field(None, description="Dynamic schema for the node output")
    dynamic_config_schema: Optional[ConstructDynamicSchema] = Field(None, description="Dynamic schema for the node config")
    enable_dynamic_fields_from_edges: Optional[bool] = Field(True, description="Enable adding dynamic fields to input / output schemas from edges --> assuming the node is marked DynamicNode and it has a dynamic input / output schema!")
    enable_node_fan_in: Optional[bool] = Field(False, description="Enable node fan-in for all incoming edges except direct edges from router nodes -> since they are not directly added to the langgraph graph! This ensure the node is not executed multiple times for each incoming edge and is only executed after all nodes with direct incoming edges are executed before it!")
    defer_node: Optional[bool] = Field(False, description="Defer node execution until all nodes with direct incoming edges are executed before it! This is a better alternative to FAN IN so the node still executes if one of the incoming branch is skipped!")


"""

GraphSchema / GraphConfig
    - list of nodeconfigs
    - list of edgeconfigs
    # NOTE: each graph must have an input and output node by default, they neednlty be present in the above node configs
    - input edges (edges from graph input node to graph nodes)
    - output edges (edges from graph nodes to graph output node)
    - optional metadata
- custom validations (eg: all edges must reference nodes in list, all nodes are conected, graph validations etc!)
    - input is root and output is leaf
    - no dangling nodes!
    - all required inputs are fulfilled via edges across all nodes!
- build a graph (requires graph runtime adapter to build the graph with) not implemented yet!
"""


class RuntimeConfig(BaseModel):
    """
    Schema for the runtime configuration of the workflow graph.
    """
    model_config = ConfigDict(extra='forbid')  # Allow additional arguments during model init!
    db_concurrent_pool_tier: Literal["small", "medium", "large", "xlarge"] = Field(default="small", description="Tier of the worker pool")


class GraphSchema(BaseModel):
    """
    Schema for the workflow graph.
    
    The graph schema defines the entire workflow as a collection of nodes and edges.
    This is the top-level schema that is sent from the frontend and used to construct
    the workflow graph for execution.
    
    Attributes:
        nodes (Dict[str, Dict[str, Any]]): Map of node IDs to node configurations.
        edges (List[EdgeSchema]): List of edges connecting the nodes.
        input_node_id (str): ID of the input node for the workflow.
        output_node_id (str): ID of the output node for the workflow.
        metadata (Optional[Dict[str, Any]]): Optional metadata for the graph.
    """
    # TODO: test if langgraph allows nodes to not generate any output or have empty schemas, in case where
    #    there's no input / output data but still need these nodes for dependency edges to starter nodes!
    model_config = ConfigDict(extra='forbid')  # Allow additional arguments during model init!
    # workflow_id: str = Field(..., description="ID of the workflow")
    nodes: Dict[str, NodeConfig] = Field(..., description="Map of node IDs to node configurations")
    edges: Optional[List[EdgeSchema]] = Field(default_factory=list, description="List of edges connecting the nodes")
    input_node_id: str = Field(default=INPUT_NODE_NAME, description="ID of the input node")
    output_node_id: str = Field(default=OUTPUT_NODE_NAME, description="ID of the output node")
    runtime_config: Optional[RuntimeConfig] = Field(default_factory=RuntimeConfig, description="Runtime configuration for the graph")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata")
    # check builder.py for paths of the below keys
    # metadata={
    #     # GRAPH_STATE_SPECIAL_NODE_NAME: {  # $graph_state
    #     #     CONFIG_REDUCER_KEY: {  # reducer
    #     #         # central state key : reducer name
    #     #         "messages": "collect_values"  # Use collect_values reducer for collecting states not initialized as lists!
    #     #     }
    #     # },
    # },
    # TODO: implement central graph state reducer config or default reducers for each field type??!
    # TODO: a special graph central state fake node -> schema just defines fields, edges to fields and reducers! This is a fake node
    #    set in dyanmic config and node outputs in central channel instead of node specific channel!
    # LeT the graph state be a typed dict without validations! (validations handled by actual inputs and outputs!)
    # check for edges in between GRAPH_STATE_SPECIAL_NODE_NAME and node and vice versa!
    # TODO: validations: for any outgoing edge from graph state, there must be an incoming edge and come in execution order before the outgoing edge!
    #     NOTE: we will just pass empty default input in such a case and warn the user maybe ?!
    # TODO: validations: the graph state typed dict must have uniform types and all incoming edges to same field name must be of same type and vice versa for outgoing edges and fields!
    # config (runtime): {"outputs": "node_name" : "node_output_field_name" : "graph_state_field_name"  # by default, use same field name!}

    @model_validator(mode='after')
    def validate_graph(self) -> Self:
        """
        Validate the entire graph structure for correctness.
        
        This validation ensures that:
        1. Input and output nodes exist
        2. All edges reference existing nodes
        3. No nodes are disconnected (except for input/output which have special requirements)
        4. Input node has only outgoing edges
        5. Output node has only incoming edges
        
        Returns:
            Self: The validated graph schema
            
        Raises:
            ValueError: If validation fails
        """
        errors = []
        
        # Check that input and output nodes exist
        if self.input_node_id not in self.nodes:
            errors.append(f"Input node {self.input_node_id} not found in nodes")
        if self.input_node_id != INPUT_NODE_NAME:
            if INPUT_NODE_NAME in self.nodes:
                errors.append(f"Input node {INPUT_NODE_NAME} is not set as graph's input node!")
        if self.output_node_id not in self.nodes:
            errors.append(f"Output node {self.output_node_id} not found in nodes")
        if self.output_node_id != OUTPUT_NODE_NAME:
            if OUTPUT_NODE_NAME in self.nodes:
                errors.append(f"Output node {OUTPUT_NODE_NAME} is not set as graph's output node!")
        
        # If input or output nodes are missing, we can't continue with other validations
        if errors:
            raise ValueError("; ".join(errors))
            
        # Check that all nodes in edges exist in nodes
        node_ids = set(self.nodes.keys())

        for node_id in node_ids:
            assert not node_id.startswith("$"), f"Node ID can't start with `$`! {node_id}"
         
        # TODO: Registry checks! Do it graph builder!
        # Check that all nodes have a node_type that exists in the registry
        # for node_id, node_config in self.nodes.items():
        #     if "node_type" not in node_config:
        #         errors.append(f"Node {node_id} is missing node_type")
        #         continue
                
        #     node_type = node_config["node_type"]
        #     if node_type != "input_node" and node_type != "output_node" and node_type not in self.registry:
        #         errors.append(f"Node type {node_type} for node {node_id} not found in registry")

        nodes_with_edges = set()

        for node_id, node_config in self.nodes.items():
            if node_config.node_name == INPUT_NODE_NAME:
                if node_config.dynamic_input_schema:
                    errors.append(f"Input node {node_id} must not have dynamic input schema defined! Define output schema instead!")
            elif node_config.node_name == OUTPUT_NODE_NAME:
                if node_config.dynamic_output_schema:
                    errors.append(f"Output node {node_id} must not have dynamic output schema defined! Define input schema instead!")
            
            if node_config.defer_node and node_config.enable_node_fan_in:
                errors.append(f"Node {node_id} must not have fan-in enabled if defer is enabled! This is a contradiction!")
            
            if node_config.edges:
                nodes_with_edges.add(node_id)

        for edge in self.edges:
            if edge.src_node_id in nodes_with_edges:
                errors.append(f"Source node {edge.src_node_id} in edge is a node with edges declared in node config and aggregated edges in graph config both! This is not allowed, a node may declare edges in node config or in graph config but not both!")
        
        for node_id in nodes_with_edges:
            for edge in self.nodes[node_id].edges:
                edge_dump = edge.model_dump()
                edge_dump["src_node_id"] = node_id
                self.edges.append(EdgeSchema(**edge_dump))
            self.nodes[node_id].edges = []
        
        unique_edges = set()
        for edge in self.edges:
            if edge.src_node_id not in node_ids and (not is_central_state_special_node(edge.src_node_id)):
                errors.append(f"Source node {edge.src_node_id} in edge not found in nodes")
            if edge.dst_node_id not in node_ids and (not is_central_state_special_node(edge.dst_node_id)):
                errors.append(f"Target node {edge.dst_node_id} in edge not found in nodes")
            if (edge.src_node_id, edge.dst_node_id) in unique_edges:
                errors.append(f"Duplicate edge between nodes {edge.src_node_id} and {edge.dst_node_id}")
            else:
                unique_edges.add((edge.src_node_id, edge.dst_node_id))

        # TODO: ENSURE no input edges are to non-usereditable fields!
        # TODO: Ensure no output edges are from non-usereditable fields!
        # TODO: assert edge fields are actually present in the respective nodes!
        # If edges reference non-existent nodes, we should fail
        if errors:
            raise ValueError("; ".join(errors))
            
        # Calculate incoming and outgoing edges for each node
        incoming_edges: Dict[str, List[EdgeSchema]] = {node_id: [] for node_id in node_ids}
        outgoing_edges: Dict[str, List[EdgeSchema]] = {node_id: [] for node_id in node_ids}
        
        for edge in self.edges:
            if not is_central_state_special_node(edge.src_node_id):
                outgoing_edges[edge.src_node_id].append(edge)
            if not is_central_state_special_node(edge.dst_node_id):
                incoming_edges[edge.dst_node_id].append(edge)
        
        # Input node should not have incoming edges
        if incoming_edges[self.input_node_id]:
            errors.append(f"Input node {self.input_node_id} should not have incoming edges")
            
        # Output node should not have outgoing edges
        if outgoing_edges[self.output_node_id]:
            errors.append(f"Output node {self.output_node_id} should not have outgoing edges")
            
        # Every node (except input) should have at least one incoming edge
        for node_id, edges in incoming_edges.items():
            if node_id != self.input_node_id and not edges:
                errors.append(f"Node {node_id} has no incoming edges")
            
        
        # NOTE: we can have a node which ends without connecting with output node!
        # Every node (except output) should have at least one outgoing edge
        # for node_id, edges in outgoing_edges.items():
        #     if node_id != self.output_node_id and not edges:
        #         errors.append(f"Node {node_id} has no outgoing edges")
        
        # # Check for cycles in the graph
        # if self._has_cycles():
        #     errors.append("The graph contains cycles, which are not allowed")
            
        # Raise all validation errors at once
        if errors:
            raise ValueError("; \n".join(errors))
            
        return self
    
    # def _has_cycles(self) -> bool:
    #     """
    #     Check if the graph contains any cycles.
        
    #     Returns:
    #         bool: True if the graph contains cycles, False otherwise
    #     """
    #     # Get all node IDs
    #     node_ids = set(self.nodes.keys())
        
    #     # Build adjacency list
    #     adjacency: Dict[str, List[str]] = {node_id: [] for node_id in node_ids}
    #     for edge in self.edges:
    #         adjacency[edge.src_node_id].append(edge.dst_node_id)
        
    #     # DFS to detect cycles
    #     visited: Set[str] = set()
    #     rec_stack: Set[str] = set()
        
    #     def is_cyclic(node_id: str) -> bool:
    #         visited.add(node_id)
    #         rec_stack.add(node_id)
    #      # Additional validations could check for:
    #     # - No cycles in the graph
    #     # - All nodes are connected
    #     # - Required inputs are fulfilled
    #     # - Input node is root
    #     # - Output node is leaf
    #     # TODO: Implement additional graph structure validations    
    #         for neighbor in adjacency[node_id]:
    #             if neighbor not in visited:
    #                 if is_cyclic(neighbor):
    #                     return True
    #             elif neighbor in rec_stack:
    #                 return True
                    
    #         rec_stack.remove(node_id)
    #         return False
        
    #     # Try starting DFS from each unvisited node
    #     for node_id in node_ids:
    #         if node_id not in visited:
    #             if is_cyclic(node_id):
    #                 return True
                    
    #     return False

# def validate_edge_schema_compatibility(
#     source_output_schema: Dict[str, Any],
#     target_input_schema: Dict[str, Any],
#     edge: EdgeSchema
# ) -> List[str]:
#     """
#     Validate that the edge mappings are compatible with the source and target schemas.
    
#     Args:
#         source_output_schema (Dict[str, Any]): Output schema of the source node
#         target_input_schema (Dict[str, Any]): Input schema of the target node
#         edge (EdgeSchema): Edge to validate
        
#     Returns:
#         List[str]: List of validation errors, empty if valid
#     """
#     errors = []
    
#     # Get the properties from both schemas
#     source_properties = source_output_schema.get("properties", {})
#     target_properties = target_input_schema.get("properties", {})
    
#     # Check mappings
#     for mapping in edge.mappings:
#         # Check if source field exists in source output schema
#         src_field_parts = mapping.src_field.split('.')
#         current_source_schema = source_properties
        
#         for part in src_field_parts:
#             if part not in current_source_schema:
#                 errors.append(f"Source field '{mapping.src_field}' does not exist in source output schema")
#                 break
#             if isinstance(current_source_schema[part], dict) and "properties" in current_source_schema[part]:
#                 current_source_schema = current_source_schema[part]["properties"]
#             else:
#                 # We've reached a leaf field, so we're done with this path
#                 break
        
#         # Check if target field exists in target input schema
#         dst_field_parts = mapping.dst_field.split('.')
#         current_target_schema = target_properties
        
#         for part in dst_field_parts:
#             if part not in current_target_schema:
#                 errors.append(f"Target field '{mapping.dst_field}' does not exist in target input schema")
#                 break
#             if isinstance(current_target_schema[part], dict) and "properties" in current_target_schema[part]:
#                 current_target_schema = current_target_schema[part]["properties"]
#             else:
#                 # We've reached a leaf field, so we're done with this path
#                 break
        
#         # TODO: Add type compatibility checking here
#         # This would involve comparing the types of the source and target fields
#         # and ensuring they are compatible
        
#     return errors

        




# class GraphSchema(BaseSchema):
#     """
#     Schema for the workflow graph.
    
#     The graph schema defines the entire workflow as a collection of nodes and edges.
#     This is the top-level schema that is sent from the frontend and used to construct
#     the workflow graph for execution.
    
#     Attributes:
#         nodes (Dict[str, Dict[str, Any]]): Map of node IDs to node configurations.
#         edges (List[EdgeSchema]): List of edges connecting the nodes.
#         input_node_id (str): ID of the input node for the workflow.
#         output_node_id (str): ID of the output node for the workflow.
#         metadata (Optional[Dict[str, Any]]): Optional metadata for the graph.
#     """
#     nodes: Dict[str, Dict[str, Any]] = Field(..., description="Map of node IDs to node configurations")
#     edges: List[EdgeSchema] = Field(..., description="List of edges connecting the nodes")
#     input_node_id: str = Field(..., description="ID of the input node")
#     output_node_id: str = Field(..., description="ID of the output node")
#     metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata") 



# if __name__ == "__main__":
#     graph_config = {"nodes": {
#         # --- 1. Input Node ---
#         "input_node": {
#             "node_id": "input_node",
#             "node_name": "input_node",
#             "node_config": {},
#             "dynamic_output_schema": {
#                 "fields": {
#                     "leads_to_process": {
#                         "type": "list",
#                         "required": False,
#                         "default": [
#                             {
#                                 "linkedinUrl": "https://www.linkedin.com/in/ACoAAAGSbXUBOy1FuiRw9wcaJ-_24TnS_u4dgS8",
#                                 "Company": "Metadata.Io",
#                                 "firstName": "Dee", 
#                                 "lastName": "Acosta 🤖",
#                                 "companyWebsite": "metadata.io",
#                                 "emailId": "dee.acosta@metadata.io",
#                                 "jobTitle": "Ad AI, Sales, and GTM	"
#                             },
#                             {
#                                 "linkedinUrl": "https://www.linkedin.com/in/ACoAAAAMYK4BCyNT23Ui4i6ijdr7-Xu2s8H1pa4",
#                                 "Company": "Stacklok",
#                                 "firstName": "Christine",
#                                 "lastName": "Simonini", 
#                                 "companyWebsite": "stacklok.com",
#                                 "emailId": "csimonini@appomni.com",
#                                 "jobTitle": "Advisor"
#                             },
#                             {
#                                 "linkedinUrl": "https://www.linkedin.com/in/ACoAACngUhwBxcSAdAIis1EyHyGe0oSxoLg0lVE",
#                                 "Company": "Cresta",
#                                 "firstName": "Kurtis",
#                                 "lastName": "Wagner",
#                                 "companyWebsite": "cresta.com", 
#                                 "emailId": "kurtis@cresta.ai",
#                                 "jobTitle": "AI Agent Architect"
#                             }
#                         ],
#                         "description": "List of leads with LinkedIn URL, Company, First Name, Last Name, Company website, Email ID, Job Title"
#                     }
#                 }
#             },
#             "edges": [
#                 {"dst_node_id": "route_leads_to_qualification", "mappings": [
#                 {"src_field": "leads_to_process", "dst_field": "original_leads"}
#             ]},
        
#             ]
#         },

#         # --- 2. Map List Router Node - Routes each lead to Step 1 qualification ---
#         "route_leads_to_qualification": {
#             "node_id": "route_leads_to_qualification",
#             "node_name": "map_list_router_node",
#             "node_config": {
#                 "choices": ["step1_qualification_prompt"],
#                 "map_targets": [
#                     {
#                         "source_path": "leads_to_process",
#                         "destinations": ["step1_qualification_prompt"],
#                         "batch_size": 1
#                     }
#                 ]
#             }
#         },
#         "output_node": {
#             "node_id": "output_node",
#             "node_name": "output_node",
#             "enable_node_fan_in": True,
#             "node_config": {}
#         }
#     },

#     # --- Edges Defining Data Flow ---
#     "edges": [
#         # Input to state and router
        
#         # Input to Step 1 router
#         # {"src_node_id": "input_node", "dst_node_id": "route_leads_to_qualification", "mappings": [
#         #     {"src_field": "leads_to_process", "dst_field": "leads_to_process"}
#         # ]},

#         {"src_node_id": "route_leads_to_qualification", "dst_node_id": "output_node", "mappings": [
#             {"src_field": "leads_to_process", "dst_field": "leads_to_process"}
#         ]},
#     ]
#     }

#     graph_schema = GraphSchema(**graph_config)
#     # graph_schema.validate_graph()
#     print(graph_schema.model_dump_json(indent=2))

