import json
import asyncio
from logging import Logger
import random
import re
from typing import Any, Callable, ClassVar, Dict, Generic, Optional, Type, TypeVar, Union, get_type_hints, List, Tuple
import inspect
from abc import ABC, abstractmethod

# Prefect integration with graceful fallback
try:
    from prefect import task
    from prefect.cache_policies import NO_CACHE
    from prefect import runtime
    from prefect.context import get_run_context
except ImportError:
    NO_CACHE = None
    runtime = None
    def task(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator
    def get_run_context():
        return None

from langgraph.types import Command, Send, Interrupt

from workflow_service.config.constants import NODE_EXECUTION_ORDER_KEY
from workflow_service.registry.schemas.base import BaseSchema
from workflow_service.utils.utils import get_central_state_field_key, is_dynamic_schema_node

from kiwi_app.workflow_app.constants import LaunchStatus

from workflow_service.config.constants import GRAPH_STATE_SPECIAL_NODE_NAME, TEMP_STATE_UPDATE_KEY, ROUTER_CHOICE_KEY
from workflow_service.registry.errors import (
    NodeValidationError,
    get_node_doc_url,
    suggest_similar_fields,
    wrap_pydantic_validation_error,
    DEFAULT_WORKFLOW_DOC_URL,
)

# Define type variables for input, output, and config schemas
InputSchemaT = TypeVar('InputSchemaT', bound=BaseSchema)
OutputSchemaT = TypeVar('OutputSchemaT', bound=BaseSchema)
ConfigSchemaT = TypeVar('ConfigSchemaT', bound=BaseSchema)
StateT = TypeVar('StateT')  # For langgraph state type

from pydantic import BaseModel, ConfigDict

from workflow_service.utils.utils import get_node_output_state_key
# from workflow_service.config.constants import GRAPH_STATE_SPECIAL_NODE_NAME, STATE_KEY_DELIMITER

from typing import TYPE_CHECKING # Import ClassVar

# TYPE_CHECKING helps avoid runtime circular dependencies if Task hints itself
if TYPE_CHECKING:
    from prefect.tasks import Task


from global_config.logger import get_prefect_or_regular_python_logger


def _get_nested_obj(data: Any, field_path: str) -> Tuple[Any, bool]:
    """
    Retrieves a nested object or value at the specified path.

    Handles navigation through dictionaries and lists using dot notation.
    Adapted from customer_data.py for use in BaseNode passthrough data mappings.

    Args:
        data: The data structure (dict or list) to navigate.
        field_path: Dot-notation path (e.g., 'a.b.0.c').

    Returns:
        Tuple[Any, bool]: The retrieved object/value and a boolean indicating if the path was found.
                         Returns (None, False) if the path is invalid or not found.
    """
    current = data
    if isinstance(current, BaseSchema):
        current = current.model_dump()
    parts = field_path.split('.') if field_path else []

    if not field_path:
        return data, True  # Return the whole data if path is empty

    for part in parts:
        if isinstance(current, dict):
            if part in current:
                current = current[part]
            else:
                return None, False  # Key not found in dict
        elif isinstance(current, list):
            try:
                idx = int(part)
                # Check bounds for list index
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return None, False  # Index out of bounds
            except (ValueError, TypeError):
                # Invalid index format for list
                return None, False
        else:
            # Cannot navigate further (e.g., encountered a primitive type)
            return None, False

    return current, True


def _set_nested_obj(data: Dict[str, Any], field_path: str, value: Any, logger: Optional[Logger] = None) -> bool:
    """
    Sets a value in a nested dictionary or list structure.

    Creates necessary dictionaries if intermediate path segments for dictionaries do not exist.
    Assumes lists and their indices exist up to the point of setting the value for list elements.
    Adapted from customer_data.py for use in BaseNode passthrough data mappings.

    Args:
        data: The dictionary to modify.
        field_path: Dot-notation path (e.g., 'a.b.0.c') where the value should be set.
        value: The value to set at the specified path.
        logger: Optional logger instance for error reporting.

    Returns:
        bool: True if the value was successfully set, False otherwise (e.g., path is invalid).
    """
    logger = logger or get_prefect_or_regular_python_logger(f"{__name__}._set_nested_obj")
    parts = field_path.split('.')
    current_obj = data
    
    # Navigate to the parent object of the target key/index
    for i, part in enumerate(parts[:-1]):
        if isinstance(current_obj, dict):
            if part not in current_obj or not isinstance(current_obj[part], (dict, list)):
                # Create necessary dictionaries if intermediate path segments do not exist
                current_obj[part] = {}  # Default to creating a dictionary
            current_obj = current_obj[part]
        elif isinstance(current_obj, list):
            try:
                idx = int(part)
                if 0 <= idx < len(current_obj):
                    current_obj = current_obj[idx]
                else:
                    logger.error(f"Index {idx} out of bounds for path segment '{part}' in path '{field_path}'. Current list length: {len(current_obj)}.")
                    return False  # Index out of bounds
            except (ValueError, TypeError):
                logger.error(f"Invalid list index '{part}' in path '{field_path}'.")
                return False  # Invalid index format for list
        else:
            logger.error(f"Cannot traverse path '{field_path}': segment '{part}' leads to a non-container type ({type(current_obj)}).")
            return False  # Path leads to a non-container type

    # Set the value at the final key/index
    last_part = parts[-1]
    if isinstance(current_obj, dict):
        current_obj[last_part] = value
        return True
    elif isinstance(current_obj, list):
        try:
            idx = int(last_part)
            if 0 <= idx < len(current_obj):
                current_obj[idx] = value
                return True
            else:
                logger.error(f"Final index {idx} out of bounds for list at path '{'.'.join(parts[:-1])}' in path '{field_path}'. List length: {len(current_obj)}.")
                return False  # Index out of bounds for setting
        except (ValueError, TypeError):
            logger.error(f"Final path segment '{last_part}' is not a valid list index for path '{field_path}'.")
            return False  # last_part is not a valid index for a list
    else:
        logger.error(f"Cannot set value: parent for '{last_part}' in path '{field_path}' is not a dict or list, but {type(current_obj)}.")
        return False


class BaseNode(BaseModel, Generic[InputSchemaT, OutputSchemaT, ConfigSchemaT], ABC):
    """
    Abstract base class for workflow nodes.
    
    This class provides the foundation for all nodes in a workflow. It defines the interface
    and common functionality that all nodes must implement. Nodes process input data according
    to their configuration and produce output data that can be passed to downstream nodes.
    
    Key Features:
    - Input/Output/Config schema validation
    - Error code registration and handling
    - Custom event emission
    - Environment flagging (staging/experimental/prod)
    - Subnode composition support
    - Version tracking
    
    Attributes:
        input_schema_cls (ClassVar[Type[InputSchemaT]]): Class reference for input schema validation.
        output_schema_cls (ClassVar[Type[OutputSchemaT]]): Class reference for output schema validation.
        config_schema_cls (ClassVar[Type[ConfigSchemaT]]): Class reference for configuration schema validation.
        config (ConfigSchemaT): Configuration parameters for the node.
        node_name (str): Unique identifier for the node type
        node_version (str): Version identifier for this node implementation
        env_flag (str): Environment flag - one of: staging, experimental, production
        error_codes (Dict[str, str]): Mapping of error codes to descriptions
        custom_events (Dict[str, str]): Mapping of custom event types to descriptions
        has_subnodes (bool): Whether this node contains subnodes
    """
    # Node metadata
    node_name: ClassVar[str]  # Required unique identifier
    # node_description: # Comes from the docstring available via `__doc__`!
    node_version: ClassVar[str]  # Required version
    env_flag: ClassVar[LaunchStatus] = LaunchStatus.EXPERIMENTAL  # Default to experimental
    # error_codes: ClassVar[Dict[str, str]] = {}  # Error code registry
    # custom_events: ClassVar[Dict[str, str]] = {}  # Custom event registry
    has_subnodes: ClassVar[bool] = False  # Subnode flag
    
    dynamic_schemas: ClassVar[bool] = False
    node_is_tool: ClassVar[bool] = False
    node_default_timeout_seconds: ClassVar[int] = 3600
    node_default_retry_count: ClassVar[int] = 0
    # Schema class references to be overridden by subclasses
    input_schema_cls: ClassVar[Optional[Type[InputSchemaT]]] = None
    output_schema_cls: ClassVar[Optional[Type[OutputSchemaT]]] = None
    config_schema_cls: ClassVar[Optional[Type[ConfigSchemaT]]] = None

    runtime_postprocessor: ClassVar[Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]] = None
    runtime_preprocessor: ClassVar[Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]] = None
    logger: Optional[Logger] = None

    # FIXME: DEBUG: Prefect test!
    # --- The Fix ---
    # Annotate the task method attribute as ClassVar
    # Optionally, provide a more specific type hint for the Task object itself
    # run: ClassVar['Task[..., str]']

    # Instance configuration
    node_id: str  # Required unique identifier in the context of a graph run
    config: Optional[Union[ConfigSchemaT, Dict[str, Any]]] = None
    runtime_metadata: Optional[Dict[str, Any]] = None
    # Whether to run the node in Prefect mode for logging and tracking flow/tasks
    prefect_mode: bool = True
    billing_mode: bool = True
    
    # Whether to run the node in private input mode (directly accepting previous node's output)
    #   Useful for map/reduce or branching patterns or maintaining private states
    is_input_node: bool = False
    is_output_node: bool = False
    private_input_mode: bool = False
    private_output_mode: bool = False
    private_output_passthrough_data_to_central_state_keys: Optional[List[str]] = []  # These keys are passed through the private output data to the central state directly (if an edge from private node to central state exists!), also this data becomes passthrough data too and gets passed on to next private input node (if this is private output node)! eg usecase: while using map list router node, preserve unique IDs of each element as passthrough data that gets collected in items collected in central state
    private_output_to_central_state_node_output_key: Optional[str] = "output"  # This key is used to send the central state output to from the node output (for each mapped edge to central state) in cases when there's extra private_output_passthrough_data ...
    output_private_output_to_central_state: bool = False
    
    # Optional mappings for reading from and writing to passthrough data
    # Only used when private_input_mode and private_output_mode are True, respectively
    read_private_input_passthrough_data_to_input_field_mappings: Optional[Dict[str, str]] = None  # Maps passthrough data keys to input field names. Supports dot notation for nested paths: {"passthrough.nested.key": "input_field_name"} or {"passthrough_key": "nested.input.field"}
    write_to_private_output_passthrough_data_from_output_mappings: Optional[Dict[str, str]] = None  # Maps output field names to passthrough data keys. Supports dot notation for nested paths: {"output.nested.field": "passthrough_key"} or {"output_field_name": "nested.passthrough.key"}

    model_config = ConfigDict(arbitrary_types_allowed=True)  # Allow non-pydantic types like clients, etc!

    @classmethod
    def __pydantic_init_subclass__(cls, *args, **kwargs):
        """
        Validates schema field definitions during class creation.
        """
        super().__pydantic_init_subclass__(*args, **kwargs)
        # Check if class is abstract by looking for abstract methods
        if cls == BaseNode:
            return  # Skip validation for the base class itself
            
        # Check if class is abstract by looking for abstract methods
        if inspect.isabstract(cls):
            return  # Skip validation for abstract classes
        
        assert cls.node_name is not None and re.match(r"^[a-zA-Z0-9_\. \(\),]+$", cls.node_name), f"Valid characters for node_name are: a-z, A-Z, 0-9, _, ., (, ), , and space!"
        
        # cls.logger = get_prefect_or_regular_python_logger(f"{__name__}.{cls.__name__}")

        if cls.input_schema_cls is None:
            cls.input_schema_cls = None
        if cls.output_schema_cls is None:
            cls.output_schema_cls = None
        if cls.config_schema_cls is None:
            cls.config_schema_cls = None
        if not cls.node_name:
            raise TypeError(
                f"node_name must be set for {cls.__name__}! "
                f"Expected: non-empty string identifier (e.g. node_name: ClassVar[str] = 'my_node'). "
                f"Documentation: {DEFAULT_WORKFLOW_DOC_URL}"
            )
        if not cls.node_version:
            raise TypeError(
                f"node_version must be set for {cls.__name__}! "
                f"Expected: semantic version string (e.g. node_version: ClassVar[str] = '1.0.0'). "
                f"Documentation: {DEFAULT_WORKFLOW_DOC_URL}"
            )
    
    def __init__(self, allow_non_user_editable_fields_in_config: bool = True, **kwargs):
        super().__init__(**kwargs)
        # classes instantiated with configs during flow run!
        self.logger = get_prefect_or_regular_python_logger(f"{__name__}.{self.node_name}.{self.node_id}")
        if (not allow_non_user_editable_fields_in_config) and (self.__class__.config_schema_cls is not None) and (not inspect.isabstract(self.__class__.config_schema_cls)):
            is_valid, error_field = self.__class__.config_schema_cls.validate_only_user_editable_fields_provided_in_input(kwargs.get("config", {}))
            if not is_valid:
                raise NodeValidationError(
                    message=f"Non-editable field provided in configuration: `{error_field}`",
                    node_id=self.node_id,
                    node_name=self.node_name,
                    field_name=error_field,
                    expected="Only user-editable fields defined in the node configuration schema",
                    actual=f"Field '{error_field}' marked as non-user-editable",
                    suggested_fix=f"Remove '{error_field}' from the node config or pass it dynamically via state mapping.",
                    doc_url=get_node_doc_url(self.node_name),
                    error_type="non_user_editable_field",
                )

    @classmethod
    def validate_node_config(cls, config: Dict[str, Any], node_id: Optional[str] = None) -> None:
        """
        Validate node configuration data against the node's config schema.

        Raises:
            NodeValidationError: If validation fails, with descriptive field, expected/actual, and doc info.
        """
        if cls.config_schema_cls is None or inspect.isabstract(cls.config_schema_cls):
            return

        try:
            cls.config_schema_cls.model_validate(config)
        except Exception as e:
            raise wrap_pydantic_validation_error(
                e,
                node_id=node_id,
                node_name=cls.node_name,
                schema_cls=cls.config_schema_cls,
                context_desc="node configuration",
            ) from e
    
    @abstractmethod
    async def process(self, input_data: InputSchemaT, config: Dict[str, Any], *args: Any, **kwargs: Any) -> OutputSchemaT:
        """
        Process input data and produce output data.
        
        This is the primary execution method that subclasses must implement.
        
        Args:
            input_data (InputSchemaT): Input data conforming to the input schema.
            
        Returns:
            OutputSchemaT: Output data conforming to the output schema.
            
        Raises:
            Exception: If an unregistered error code is encountered
        """
        pass

    # def emit_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
    #     """
    #     Emit a custom event during node execution.
        
    #     Args:
    #         event_type (str): Type of event - must be registered in custom_events
    #         event_data (Dict[str, Any]): Event payload
            
    #     Raises:
    #         ValueError: If event_type is not registered
    #     """
    #     if event_type not in self.custom_events:
    #         raise ValueError(f"Unregistered event type: {event_type}")
    #     # TODO: Implement event emission logic
    #     pass

    def build_input_state(self, state: StateT, config: Dict[str, Any], only_fetch_central_state: bool = False, build_input_schema_obj: bool = True) -> Dict[str, Any]:
        """
        Build the input state for the node.
        """
        # print(f"\n\n\n\n#### build_input_state --  > #############################  {self.node_id} --> {self.node_name}  ###############################\n\n\n\n")
        # print("\n\n\n\n#### build_input_state")
        # print("\n\n\n\n#### state (in build input state)", state, "\n\n\n\n")
        
        # import ipdb; ipdb.set_trace()
        configurable = config.get("configurable", {})
        # print("\n\n\n\n#### configurable (in build input state)", configurable, "\n\n\n\n")  # json.dumps(configurable, indent=4))
        if configurable is None:
            raise NodeValidationError(
                message="No config provided to node",
                node_id=self.node_id,
                node_name=self.node_name,
                field_name="configurable",
                expected="A non-empty configuration dictionary containing runtime settings and input mappings",
                actual="None",
                suggested_fix="Ensure the workflow runtime passes a valid configuration dictionary with 'configurable' settings.",
                doc_url=get_node_doc_url(self.node_name),
                error_type="missing_configuration",
            )
        if self.input_schema_cls is None or (not self.input_schema_cls.model_fields):
            return {}
        input_dict = {}
        if self.input_schema_cls.get_required_fields() and (("inputs" not in configurable) or (self.node_id not in configurable["inputs"])):
            required = list(self.input_schema_cls.get_required_fields())
            raise NodeValidationError(
                message=f"Missing input mappings for node '{self.node_id}'",
                node_id=self.node_id,
                node_name=self.node_name,
                field_name="inputs",
                expected=f"Configurable inputs dictionary mapping required fields: {required}",
                actual="Missing from configurable['inputs']",
                suggested_fix=f"Connect incoming edges to node '{self.node_id}' in the workflow graph to map required fields: {required}.",
                doc_url=get_node_doc_url(self.node_name),
                error_type="missing_input_mappings",
            )
        field_mappings = configurable["inputs"][self.node_id]
        # print("\n\n\n\n#### field_mappings", field_mappings, "\n\n\n\n")
        parents_run_status = {}
        
        node_execution_order = {}
        for idx, node_id in enumerate(state.get(get_central_state_field_key(NODE_EXECUTION_ORDER_KEY), [])):
            node_execution_order[node_id] = max(node_execution_order.get(node_id, 0), idx)

        ordered_field_mappings = {}
        for input_field, state_key in field_mappings.items():
            node_state_keys = []
            central_state_keys = []
            for state_key_instance in state_key:
                if isinstance(state_key_instance, list):
                    node_id = state_key_instance[0]
                    node_state_keys.append((node_execution_order.get(node_id, -1), state_key_instance))
                else:
                    central_state_keys.append(state_key_instance)
            # sort node state keys by execution order (latest executed is first!)
            node_state_keys = sorted(node_state_keys, key = lambda x: x[0], reverse=True)
            # remove idx
            node_state_keys = [x[1] for x in node_state_keys]
            ordered_field_mappings[input_field] = central_state_keys + node_state_keys

        for input_field, state_key in ordered_field_mappings.items():  # field_mappings.items()
            # Get value from state using the mapped state key
            for state_key_instance in state_key:
                if isinstance(state_key_instance, list):
                    if only_fetch_central_state:
                        continue
                
                    # Get value from state using the mapped state key
                    # NOTE: this handles the case when a router node creates multiple fan outs and a subsequent node receives multiple fan ins from router nodes at runtime!
                    node_id = state_key_instance[0]
                    node_state_key = get_node_output_state_key(node_id)
                    source_field_path = state_key_instance[1]
                    
                    if node_state_key not in state:
                        # This could mean the the parent node outputs are uninitialized and this node could be part of a loop!
                        #     This means that the input must be marked as optional!
                        parents_run_status[node_id] = False
                        continue
                    
                    # Use nested path support for extracting the source field value
                    input_received_value, found = _get_nested_obj(state[node_state_key], source_field_path)
                    if not found:
                        self.warning(f"Source field '{source_field_path}' not found in node '{node_id}' output state!")
                        continue
                    
                    parents_run_status[node_id] = True
                    
                    # Use nested path support for setting the destination field value
                    _, found = _get_nested_obj(input_dict, input_field)
                    if not found:
                        if not _set_nested_obj(input_dict, input_field, input_received_value, self):
                            self.warning(f"Failed to set nested input field '{input_field}' from source field '{source_field_path}' in node '{self.node_id}'")
                    # break
                else:
                    # Get value from CENTRAL state using the mapped state key!
                    
                    central_state_source_field_path = get_central_state_field_key(state_key_instance)

                    central_state_value, found = _get_nested_obj(state, central_state_source_field_path)
                    if not found:  # if central_state_key not in state
                        # This could mean the the parent node outputs are uninitialized and this node could be part of a loop!
                        #     This means that the input must be marked as optional!
                        self.info(f"Source field '{state_key_instance}' not found in node central state!")
                        continue
                        # raise ValueError(f"State key {state_key} not found in state!")
                    
                    # Use nested path support for setting the destination field value
                    if not _set_nested_obj(input_dict, input_field, central_state_value, self):
                        self.warning(f"Failed to set nested input field '{input_field}' from source field '{state_key_instance}' in node '{self.node_id}'")
                # break
        # if self.node_id == "review":
        #     import ipdb; ipdb.set_trace()
        # print("\n\n\n\n#### input_dict (in build input state)", input_dict, "\n\n\n\n")
        
        # This generates a weird error in Langgraph!
        # Best to block multiple incoming edges on frontend -> use central state for data passing from multiple edges, use edges as part of a loop or explicitly handled FAN IN!
        # # TODO: FIXME: hack for multiple fan Ins and Loops!
        # for fname, field_info in self.input_schema_cls.model_fields.items():
        #     if fname not in input_dict:
        #         if field_info.is_required() and any(p == False for p in parents_run_status.values()):
        #             print(f"fname: {fname} is required but not found in input_dict: {input_dict}")
        #             return None
        # if self.private_input_mode:
        #     self.warning(f"{self.node_id} --> {self.node_name} private input mode is enabled! input_dict: {input_dict} from state: {json.dumps(state, indent=4)}")
        if not build_input_schema_obj:
            return input_dict

        try:
            input_obj = self.input_schema_cls(**input_dict)
            return input_obj
        except Exception as e:
            raise wrap_pydantic_validation_error(
                e,
                node_id=self.node_id,
                node_name=self.node_name,
                schema_cls=self.input_schema_cls,
                context_desc="node input state",
            ) from e

    def build_output_state_update(self, output_data: OutputSchemaT, config: Dict[str, Any], state_private_output_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Build the state update for the node.
        """
        from workflow_service.registry.registry import DBRegistry
        configurable = config.get("configurable", {})
        state_update = {}
        output_schema = self.__class__.output_schema_cls

        # state_private_output_data = {}
        # if self.private_input_mode and self.private_output_passthrough_data_to_central_state_keys and state:
        #     # NOTE: this is to pass the private output data to the central state!
        #     for key in self.private_output_passthrough_data_to_central_state_keys:
        #         _data = state.get(key, None)
        #         # if _data:
        #         state_private_output_data[key] = _data

        if output_schema and isinstance(output_data, output_schema):
            if not self.is_output_node:
                output_data = output_data.model_dump()
            # We will let node outputs be outputed even in private input mode for debugging purpsoes by adding collect values reducer for node output keys!
            if (not self.private_input_mode) or self.output_private_output_to_central_state:
                state_update = {get_node_output_state_key(self.node_id): output_data}
        
            # Ensure that this node is configured to send outputs to the global central state
            #     And it has a set output schema and the generated output is actually an instance of the schema object
            #     The latter helps avoiding issues like when output is actually interrupts in HITL node, etc!
            if isinstance(state_update, dict) and configurable.get("outputs", None) and configurable["outputs"].get(self.node_id, None):
                # This pushes output to global central state if configured to do so!
                # TODO: check if output field name is valid!
                central_state_output_field_mapping = configurable["outputs"].get(self.node_id, {})
                central_state_output = {}
                for output_field_name, central_state_field_name in central_state_output_field_mapping.items():
                    central_state_output_field_name = get_central_state_field_key(central_state_field_name)
                    
                    # Use nested path support for extracting the output field value
                    output_field_value, found = _get_nested_obj(output_data, output_field_name)
                    if found:
                        if not _set_nested_obj(central_state_output, central_state_output_field_name, output_field_value, self):
                            self.warning(f"Failed to set nested central state field '{central_state_field_name}' from source field '{output_field_name}' in node '{self.node_id}'")
                    else:
                        self.warning(f"Output field '{output_field_name}' not found in output data for mapping to central state field '{central_state_field_name}' in node '{self.node_id}'")
                    if state_private_output_data:
                        _output = {
                            self.private_output_to_central_state_node_output_key: central_state_output[central_state_output_field_name]
                        }
                        if self.private_output_to_central_state_node_output_key in state_private_output_data:
                            self.warning(f"Passthrough data to central state being overwritten by node output key: `{self.private_output_to_central_state_node_output_key}`")
                        central_state_output[central_state_output_field_name] = dict(state_private_output_data) | _output
                state_update.update(central_state_output)
        else:
            # NOTE: may wanna copy this potentially!
            state_update = output_data if output_data and isinstance(output_data, dict) else state_update
            if DBRegistry.is_node_instance_router(self) and state_update and self.private_output_mode:
                # NOTE: this is a hack to ensure that the router node output is a dict!
                state_update = {get_node_output_state_key(self.node_id): output_data}

        # Add node execution order to state update
        state_update[get_central_state_field_key(NODE_EXECUTION_ORDER_KEY)] = [self.node_id]

        

        return state_update
    
    # def _prepare_input_data(self, input_data: Union[InputSchemaT, Dict[str, Any]], config: Dict[str, Any]) -> InputSchemaT:
    #     """
    #     Prepare the input data for the node.
    #     """
    #     return self.input_schema_cls(**input_data) if isinstance(input_data, dict) else input_data

    async def apply_passthrough_data_to_input_data_in_private_input_mode(self, input_data_dict: Dict[str, Any], state: StateT):
        """
        Apply passthrough data to input data in private input mode.
        """
        # Apply passthrough data mappings if configured
        if self.read_private_input_passthrough_data_to_input_field_mappings:
            for passthrough_key, input_field in self.read_private_input_passthrough_data_to_input_field_mappings.items():
                # Get passthrough value using dot notation
                passthrough_value, found = _get_nested_obj(state, passthrough_key)
                if found:
                    # Check if target field already exists (for simple fields only, nested paths may create new structure)
                    if '.' not in input_field and input_field in input_data_dict:
                        self.warning(f"Input field '{input_field}' already exists in input data dict! Overwriting with passthrough data!")
                    
                    # Set the value using dot notation support
                    if '.' in input_field:
                        # For nested paths, use _set_nested_obj
                        if not _set_nested_obj(input_data_dict, input_field, passthrough_value, self):
                            self.warning(f"Failed to set nested input field '{input_field}' from passthrough data key '{passthrough_key}'")
                        else:
                            self.debug(f"Mapped passthrough data key '{passthrough_key}' to nested input field '{input_field}' with value: {passthrough_value}")
                    else:
                        # For simple fields, set directly
                        input_data_dict[input_field] = passthrough_value
                        self.debug(f"Mapped passthrough data key '{passthrough_key}' to input field '{input_field}' with value: {passthrough_value}")
                else:
                    self.debug(f"Passthrough data key '{passthrough_key}' not found in state, skipping mapping to input field '{input_field}'")
    
    async def build_passthrough_data_in_private_mode(self, output_data: Dict[str, Any], state: StateT) -> Dict[str, Any]:
        """
        Apply passthrough data to output data in private output mode.
        """
        # Build Private Mode Passthrough Data
        private_mode_passthrough_data = {}
        if self.private_input_mode or self.private_output_mode:
            if self.private_output_passthrough_data_to_central_state_keys:
                private_mode_passthrough_data = {key: state.get(key, None) for key in self.private_output_passthrough_data_to_central_state_keys if key in state}
            
            # Add mappings from output fields to passthrough data if configured
            if self.write_to_private_output_passthrough_data_from_output_mappings:
                for output_field, passthrough_key in self.write_to_private_output_passthrough_data_from_output_mappings.items():
                    # Get value from output_data using dot notation
                    # output_value = None
                    output_value, found = _get_nested_obj(output_data, output_field)

                    
                    if output_value is not None:
                        # Set the passthrough value using dot notation support
                        if '.' in passthrough_key:
                            # For nested passthrough keys, ensure we have a dict structure
                            if not _set_nested_obj(private_mode_passthrough_data, passthrough_key, output_value, self):
                                self.warning(f"Failed to set nested passthrough data key '{passthrough_key}' from output field '{output_field}'")
                            else:
                                self.debug(f"Mapped output field '{output_field}' to nested passthrough data key '{passthrough_key}' with value: {output_value}")
                        else:
                            # Simple key assignment
                            private_mode_passthrough_data[passthrough_key] = output_value
                            self.debug(f"Mapped output field '{output_field}' to passthrough data key '{passthrough_key}' with value: {output_value}")
                    else:
                        self.debug(f"Output field '{output_field}' not found or is None, skipping mapping to passthrough data key '{passthrough_key}'")
        return private_mode_passthrough_data

    # FIXME: DEBUG: Prefect test!
    # @task
    async def run(self, state: StateT, config: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        LangGraph-compatible execution method.
        
        This method adapts the node's processing logic to be compatible with LangGraph's
        execution model. It extracts input data from the state, processes it, and returns
        the updated state.
        
        Args:
            state (StateT): The current state of the workflow graph. This is a TypedDict.
            config (Dict[str, Any]): Runtime configuration overrides.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Dict[str, Any]: The updated state after processing.
            
        Raises:
            Exception: For any unregistered error codes encountered during processing
        """
        from workflow_service.registry.registry import DBRegistry
        
        try:
            if self.private_input_mode:
                # fetch central state inputs!
                input_data_dict = self.build_input_state(state, config, only_fetch_central_state=True, build_input_schema_obj=False)
                # assume the received input is entire input!
                #   for conflicting keys, overwrite central state values with private values!
                input_data_dict.update(state)
                
                await self.apply_passthrough_data_to_input_data_in_private_input_mode(input_data_dict, state)
                
                # TODO: Potentially use model_validate(...)?
                input_data = input_data_dict
                # NOTE: TODO: FIXME: We don't initialize the input schema in private input mode! Be careful and handle this behaviour more gracefully!
                #     This is mainly the case due to dynamic schemas and missing field types leading to empty input schema cls!
                # input_data = self.input_schema_cls(**input_data_dict)
            else:
                input_data = self.build_input_state(state, config)
            if input_data is None:
                # print("\n\n\n\n NODE EARLY EXIT -> REQUIRED NOT FULFILLED!!!!!: ", "="*100, "\n\n\n\n")
                # TODO: FIXME:
                # This is exceptional case when not all required fields were found in input!
                #     It could either be a bug or the node is prematurely called due to FAN IN bug!
                # We will exit the node without running the node since required field is missing 
                #     and hope this node is re-executed when all parents are run and requried field is found in input!
                return {}

            if self.__class__.runtime_preprocessor:
                # Generally used for interupts to get HITL!
                # print("\n\n\n\n#### self.__class__.runtime_preprocessor (in run)", "\n\n\n\n")
                preprocessed_input_data = self.__class__.runtime_preprocessor(self, input_data, config, *args, **kwargs)
                input_data = preprocessed_input_data

            # Build Central State
            central_state_data = {
                field_name: value 
                for field_name, value in state.items() 
                if field_name.startswith(GRAPH_STATE_SPECIAL_NODE_NAME)
            }
            kwargs["central_state_data"] = central_state_data

            # Process the input data using node's process method to compute output data
            config_node_retry_count = getattr(self.config, "node_retry_count", None)
            node_retry_count = config_node_retry_count if config_node_retry_count is not None else self.__class__.node_default_retry_count
            retry_delay_seconds = 10
            retry_jitter_factor = 0.5
            retry_config = {
                "retry_delay_seconds": retry_delay_seconds,
                "retries": node_retry_count,
                "retry_jitter_factor": retry_jitter_factor,
            }
            if self.prefect_mode:
                prefect_kwargs = {
                    "name": f"Node Name: `{self.node_name}` - Node ID: `{self.node_id}`", 
                    "cache_policy":NO_CACHE, 
                    "timeout_seconds":self.__class__.node_default_timeout_seconds,
                }
                if node_retry_count:
                    prefect_kwargs.update(retry_config)
                
                async def process_retry_wrapper(input_data: InputSchemaT, config: Dict[str, Any], *args: Any, **kwargs: Any) -> OutputSchemaT:
                    attempt = runtime.task_run.run_count or 1          # 1 = first try, 2 = first retry, etc.
                    max_attempts = get_run_context().task.retries + 1  # retries + initial try
                    if attempt > 1:
                        self.warning(f"Retry attempt {attempt-1}/{max_attempts-1}")
                    return await self.process(input_data, config, *args, **kwargs)

                output_data = await task(**prefect_kwargs)(process_retry_wrapper)(input_data, config, *args, **kwargs)  # self.process   process_retry_wrapper
            else:
                i = 0
                node_retry_count = node_retry_count or 0
                while i <= node_retry_count:
                    try:
                        output_data = await self.process(input_data, config, *args, **kwargs)
                        break
                    except Exception as e:
                        if node_retry_count > i:
                            self.error(f"Retry #{i} of {node_retry_count} for Node {self.node_id} failed with error: {e}", exc_info=True)
                        else:
                            raise e
                        i += 1
                        await asyncio.sleep(retry_delay_seconds * (1 + random.random() * retry_jitter_factor))
            # print("\n\n\n\n#### output_data (in run)", output_data, "\n\n\n\n")
            if DBRegistry.is_node_instance_router(self) and self.private_output_mode:
                router_choices = output_data.get(ROUTER_CHOICE_KEY, None)
                output_data = output_data.get(TEMP_STATE_UPDATE_KEY, None)

            private_mode_passthrough_data = await self.build_passthrough_data_in_private_mode(output_data, state)
            
            # Convert output to dict and return as state update
            # NOTE: in case a Command / Send is generated in the process(...) method, this method to build state update should be called from there directly since otehrwise
            #       the Command / Send will not make it to langgraph and state will be built with atleast the node order update!
            #       Also if the Command / Send don't include the central state update, the node order won't be captured!
            if isinstance(output_data, (Command, Send, Interrupt)):
                state_update = output_data
            else:
                state_update = self.build_output_state_update(output_data, config, private_mode_passthrough_data, )

            state_update = state_update if state_update else output_data   # state_update could be None if no output schema defined but output_data may be a runtime command / interupt!

            if self.__class__.runtime_postprocessor and (not self.private_output_mode) and (not isinstance(state_update, (Command, Send, Interrupt))) and (not isinstance(output_data, (Command, Send, Interrupt))):
                state_update = self.__class__.runtime_postprocessor(self, state_update, config, *args, **kwargs)
            
            if (isinstance(output_data, dict) or isinstance(output_data, BaseSchema)) and (not isinstance(state_update, (Command, Send, Interrupt))) and (not isinstance(output_data, (Command, Send, Interrupt))):
                # assume this is standard state_update and not Command / Send / Interrupts for eg
                configurable = config.get("configurable", {})

                if self.private_output_mode:
                    # update central state with private output!
                    
                    output_to_nodes = {}
                    for out_node_id, out_node_edge in configurable.get("outgoing_edges", {}).get(self.node_id, {}).items():
                        if DBRegistry.is_node_instance_router(self):
                            if out_node_id not in router_choices:
                                continue

                        output_to_node = {}
                        for mapping in out_node_edge.mappings:
                            # Get source field value from output_data using nested path support
                            src_value, found = _get_nested_obj(output_data, mapping.src_field)
                            
                            # Set the destination field in output_data using nested path support
                            if found and src_value is not None:
                                if not _set_nested_obj(output_to_node, mapping.dst_field, src_value, self):
                                    self.warning(f"Failed to set nested destination field '{mapping.dst_field}' from source field '{mapping.src_field}' for output to node '{out_node_id}'")
                            elif not found:
                                self.warning(f"Private mode: Source field '{mapping.src_field}' not found in output data for mapping to destination field '{mapping.dst_field}' for output to node '{out_node_id}'")
                        
                        # Central state hack!
                        # TODO: FIXME
                        if (not isinstance(output_to_node, dict)) and central_state_data:
                            self.warning(f"Output to node is not a dict in {self.node_id}. Creating a dict with 'data' key to wrap object and adding central state data directly.")
                            output_to_node = {"data": output_to_node}
                        output_to_node.update(central_state_data)
                        
                        for key, value in private_mode_passthrough_data.items():
                            if key in output_to_node:
                                self.warning(f"Passthrough data to central state being overwritten by node output key: `{key}`")
                            else:
                                output_to_node[key] = value
                        # output_to_node.update(private_mode_passthrough_data)

                        output_to_nodes[out_node_id] = output_to_node
                    # Ignore any other state update apart from a dictionary, potentialy ignore Command / Send / Interrupt!
                    state_update = state_update if isinstance(state_update, dict) else None
                    # TODO: langgraph dependency!
                    response = Command(goto=[Send(node_id, node_input) for node_id, node_input in output_to_nodes.items()], update=state_update, )
                    return response

            return state_update
        except Exception as e:
            # TODO: raise custom error codes which are registered!
            raise e
        # except Exception as e:
        #     # Check if error code is registered
        #     error_code = getattr(e, 'code', str(type(e).__name__))
        #     if error_code not in self.error_codes:
        #         # Unregistered errors should trigger alerts
        #         # TODO: Implement alert mechanism
        #         raise
        #     raise  # Re-raise the original error
        # pass
    def set_run_annotations(
        self,
        state_type: Type[Any],
        config_type: Optional[Type[Any]] = None,
        return_type: Optional[Type[Any]] = None
    ) -> None:
        """
        Updates the run method's type annotations directly on the instance.

        This method modifies the run method's type hints to use specific types for state and config parameters.
        This is useful when integrating with systems that rely on type annotations for validation.

        Args:
            state_type (Type[Any]): The type to use for the state parameter.
            config_type (Optional[Type[Any]]): The type to use for the config parameter.
                Defaults to Dict[str, Any] if not provided.

        Note:
            This modifies the instance's run method annotations directly, rather than creating a new method.
            The actual implementation remains unchanged - only the type hints are updated.
        """
        if config_type is None:
            config_type = Dict[str, Any]
        if return_type is None:
            return_type = Dict[str, Any]

        # Update the run method's annotations directly
        self.run.__func__.__annotations__ = {
            'state': state_type,
            'config': Optional[config_type],
            'return': return_type
        }
    
    @classmethod
    def with_typed_signature(
        cls, 
        state_type: Type[Any], 
        config_type: Optional[Type[Any]] = None
    ) -> Callable[[StateT, Optional[Dict[str, Any]], Any, Any], Dict[str, Any]]:
        """
        Create a version of the run method with specific type annotations.

        This helper method allows for more precise type hints when integrating with
        LangGraph or other systems that rely on type annotations for validation.

        Args:
            state_type (Type[Any]): The type to use for the state parameter.
            config_type (Optional[Type[Any]]): The type to use for the config parameter.
                
        Returns:
            Callable: A function with the same behavior as run but with updated type annotations.
        """
        if config_type is None:
            config_type = Dict[str, Any]
        
        # Create a new function with the same implementation but different annotations
        def typed_run(
            self, 
            state: state_type, 
            config: Optional[config_type] = None, 
            *args: Any, 
            **kwargs: Any
        ) -> Dict[str, Any]:
            return cls.run(self, state, config, *args, **kwargs)
        
        # Update the function's signature to reflect the new types
        typed_run.__annotations__ = {
            'state': state_type,
            'config': Optional[config_type],
            'return': Dict[str, Any]
        }
        
        return typed_run
    
    @classmethod
    def dump_node_signature(cls) -> Dict[str, Any]:
        """
        Dumps the node's signature including name, version and schemas.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - node_name: Name of the node class
                - version: Version of the node (if defined)
                - input_schema: Schema for input (None if not defined) 
                - output_schema: Schema for output (None if not defined)
                - config_schema: Schema for config (None if not defined)
        """
        # Get generic type args for input/output/config schemas
        
        # Extract schemas, defaulting to None if not specified

        signature = {
            "node_name": cls.node_name,
            "version": cls.node_version,
            "input_schema": cls.input_schema_cls.get_schema_for_db() if cls.input_schema_cls else None,
            "output_schema": cls.output_schema_cls.get_schema_for_db() if cls.output_schema_cls else None,
            "config_schema": cls.config_schema_cls.get_schema_for_db() if cls.config_schema_cls else None,
            "input_schema_is_dynamic": is_dynamic_schema_node(cls.input_schema_cls),
            "output_schema_is_dynamic": is_dynamic_schema_node(cls.output_schema_cls),
            "config_schema_is_dynamic": is_dynamic_schema_node(cls.config_schema_cls),
        }
        
        return signature

    @classmethod
    def diff_from_provided_signature(cls, provided_signature: Dict[str, Any], self_is_base_for_diff:bool = False) -> Dict[str, Any]:
        """
        Compare current node signature against a provided signature and return differences.
        Uses BaseSchema diff for schema comparisons.

        Args:
            provided_signature (Dict[str, Any]): Node signature to compare against

        Returns:
            Dict[str, Any]: Dictionary containing:
                - version_changed: True if version differs
                - schema_diffs: Dict containing diffs for each schema type:
                    - input_schema: Diff results if input schema changed
                    - output_schema: Diff results if output schema changed  
                    - config_schema: Diff results if config schema changed
        """
        # current_sig = cls.dump_node_signature()
        
        # Name changed
        # NOTE: This should not happen!
        name_changed = cls.node_name != provided_signature.get("node_name")

        # Compare versions
        version_changed = cls.node_version != provided_signature.get("version")
        
        # Compare schemas
        signature_schema_types = ["input_schema", "output_schema", "config_schema"]
        current_schema_types = [cls.input_schema_cls, cls.output_schema_cls, cls.config_schema_cls]

        diff = {
            "name_changed": name_changed,
            "version_changed": version_changed,
            "schema_added": [],
            "schema_removed": [],
            "schema_diffs": {},
            "dynamic_schema_changes": {},
        }
        if self_is_base_for_diff:
            current_added_key = "schema_removed"
            provided_added_key = "schema_added"
            provided_dynamic_change_key = "changed_to_dynamic"
            current_dynamic_change_key = "changed_to_static"
        else:
            current_added_key = "schema_added"
            provided_added_key = "schema_removed"
            provided_dynamic_change_key = "changed_to_static"
            current_dynamic_change_key = "changed_to_dynamic"
        for schema_type, current_schema_cls in zip(signature_schema_types, current_schema_types):
            provided_schema = provided_signature.get(schema_type)
            
            # Skip if both schemas are None
            if current_schema_cls is None and provided_schema is None:
                continue
                
            # Record change if one schema is None and other isn't
            if current_schema_cls is None or provided_schema is None:
                if current_schema_cls is not None:
                    diff[current_added_key].append(schema_type)
                else:
                    diff[provided_added_key].append(schema_type)
                continue
            
            # Check for dynamic schema changes
            dynamic_schema_key = f"{schema_type}_is_dynamic"
            current_is_dynamic = is_dynamic_schema_node(current_schema_cls)
            provided_is_dynamic = provided_signature.get(dynamic_schema_key, False)
            
            if current_is_dynamic != provided_is_dynamic:
                diff["dynamic_schema_changes"][schema_type] = {
                    provided_dynamic_change_key: provided_is_dynamic,
                    current_dynamic_change_key: current_is_dynamic
                }
            
            if current_schema_cls is not None:
                # Use BaseSchema diff to compare schemas
                diff["schema_diffs"][schema_type] = current_schema_cls.diff_from_provided_schema(provided_schema, self_is_base_for_diff=self_is_base_for_diff)
        return diff

    def _format_log_message(self, msg: str) -> str:
        """
        Format a log message with node identification prefix.
        
        Args:
            msg: The original message to format
            
        Returns:
            Formatted message with node_id and node_name prefix
        """
        if self.runtime_metadata:
            workflow_name = self.runtime_metadata.get("workflow_name", None)
            is_sub_workflow = self.runtime_metadata.get("is_sub_workflow", None)
            if is_sub_workflow and workflow_name:
                return f"{workflow_name}: {self.node_id}: {self.node_name} - {msg}"
        return f"{self.node_id}: {self.node_name} - {msg}"
    
    # Logger convenience methods
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Log a debug message through the node's logger with node identification prefix.
        
        Args:
            msg: The message to log
            *args: Additional positional arguments for the logger
            **kwargs: Additional keyword arguments for the logger
        """
        if self.logger:
            self.logger.debug(self._format_log_message(msg), *args, **kwargs)
    
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Log an info message through the node's logger with node identification prefix.
        
        Args:
            msg: The message to log
            *args: Additional positional arguments for the logger
            **kwargs: Additional keyword arguments for the logger
        """
        if self.logger:
            self.logger.info(self._format_log_message(msg), *args, **kwargs)
    
    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Log a warning message through the node's logger with node identification prefix.
        
        Args:
            msg: The message to log
            *args: Additional positional arguments for the logger
            **kwargs: Additional keyword arguments for the logger
        """
        if self.logger:
            self.logger.warning(self._format_log_message(msg), *args, **kwargs)
    
    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Log an error message through the node's logger with node identification prefix.
        
        Args:
            msg: The message to log
            *args: Additional positional arguments for the logger
            **kwargs: Additional keyword arguments for the logger
        """
        if self.logger:
            self.logger.error(self._format_log_message(msg), *args, **kwargs)
    
    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Log a critical message through the node's logger with node identification prefix.
        
        Args:
            msg: The message to log
            *args: Additional positional arguments for the logger
            **kwargs: Additional keyword arguments for the logger
        """
        if self.logger:
            self.logger.critical(self._format_log_message(msg), *args, **kwargs)
