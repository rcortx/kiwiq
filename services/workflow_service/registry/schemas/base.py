"""
# TODO: Create a special schema for prompt management (prompt template, prompt variables and version with validations (all the prompt variables are actually present in the prompt template))
# NOTE: CHANGING config version probably must change node version too! Except certain change types -> like changing user visible fields?? in that case, all instances must change!
"""
from abc import ABC
from copy import copy
from datetime import datetime, date
import annotated_types
from pydantic import BaseModel, model_validator, Field, ConfigDict, Discriminator
from pydantic_core import PydanticUndefined
from pydantic.json_schema import SkipJsonSchema
from pydantic import TypeAdapter

from pydantic import create_model
# from pydantic.main import FieldInfo, create_model

import json
from typing import Any, Dict, List, Optional, Tuple, ClassVar, Union, get_origin, get_args, Annotated, Type, Literal
import inspect
from enum import Enum
from langchain_core.messages import AnyMessage
from langchain_core.messages.utils import _get_type


# from workflow_service.registry.schemas.test import *

# Primitive types


def create_dynamic_schema_with_fields(
    cls, 
    fields: Dict[str, Any],
    schema_name: str = None,
    # module_name: Optional[str] = None
) -> Type['BaseSchema']:
    """
    Create a new schema class based on the current class with the provided fields.
    
    This method dynamically creates a new schema class that inherits from the current class
    and includes the specified fields. Fields can be provided either as field names that already
    exist in the current class, or as tuples containing (annotation, field_info).
    
    Args:
        schema_name (str): Name for the new schema class
        fields (Dict[str, Any]): Mapping of field names to either:
            - str: Field name that exists in the current class's model_fields
            - Tuple[Type, Field]: Tuple of (annotation, field_info)
        module_name (Optional[str]): Module name for the new class. Defaults to the current class's module.
    
    Returns:
        Type[BaseSchema]: A new schema class inheriting from the current class with the specified fields.
        
    Example:
        ```python
        # Create a new schema with selected fields from the original schema
        NewUserSchema = UserSchema.create_schema_with_fields(
            "NewUserSchema",
            {
                "username": None,  # Reuse existing field
                "email": None,        # Reuse existing field
                "role": (str, Field(description="User role"))  # New field
            }
        )
        ```
    
    Raises:
        ValueError: If a field name is provided as a string but doesn't exist in the current class's model_fields.
    """
    field_definitions: Dict[str, Any] = {}
    
    # Process each field
    for field_name, field_def in fields.items():
        if field_def is None:
            # Field is specified by name, copy from current class
            if field_name not in cls.model_fields:
                raise ValueError(f"Field '{field_name}' does not exist in {cls.__name__}")
            
            # Get the field from the current class
            existing_field = cls.model_fields[field_name]
            field_definitions[field_name] = (existing_field.annotation, existing_field)
        else:
            if not isinstance(field_def, tuple):
                raise ValueError(f"Field '{field_name}' must be specified as a tuple of (annotation, field_info)")
            # Field is specified as (annotation, field_info) tuple
            field_definitions[field_name] = field_def
    
    # Create the new schema class
    new_schema_class = create_model(
        schema_name or f"{cls.__name__}DynamicSchema",
        __base__=BaseSchema,
        __doc__=cls.__doc__,
        __module__=cls.__module__,  # module_name or 
        **field_definitions
    )

    new_schema_class.IS_DYNAMIC_SCHEMA = True
    
    return new_schema_class


class FieldValidationResult:
    def __init__(self, valid, error_message=None, core_type_annotation=None, core_type_class=None, core_type_object_iterator=None, is_optional=False):
        self.valid = valid
        self.error_message = error_message
        self.core_type_annotation = core_type_annotation  # Type annotation without Optional / Union eg: `Dict[str, BaseSchema]`
        self.core_type_class = core_type_class  # Primitive | Enum | BaseSchema
        self.core_type_object_iterator = core_type_object_iterator
        self.is_optional = is_optional


_CORE_PRIMITIVE_TYPES: Tuple[type, ...] = (str, int, float, bool, bytes, datetime, date)
# NOTE since AnyMessage is Annotated type with annotations, also add its base type without surface annotation (which is the Union of messages)
# Due to some hidden bug, sometimes when defining AnyMessage fields, only the base type makes it pass!
_PRIMITIVE_TYPES_EXTENDED: Tuple[type, ...] = tuple(list(_CORE_PRIMITIVE_TYPES) + [AnyMessage, get_args(AnyMessage)[0], Any, BaseModel])
# # UNION allows any primitive type to be set and all are JSON serializable, 
# #     allowing multi type dicts / lists eg: [int, str, ...] or Dict[str, ]
# _PRIMITIVE_TYPES_UNION = Annotated[
#     Union[
#         str, int, float, bool, bytes, datetime, date
#         # Union.__getitem__(_CORE_PRIMITIVE_TYPES)
#     ]
#     ,
#     Field(discriminator=Discriminator(_get_type)),
# ]
# # _PRIMITIVE_TYPES_UNION: type = Union.__getitem__(_PRIMITIVE_TYPES_EXTENDED)
# _PRIMITIVE_TYPES_UNION_TYPES: Tuple[type, ...] = (_PRIMITIVE_TYPES_UNION, )
_PRIMITIVE_TYPES = _PRIMITIVE_TYPES_EXTENDED


def _cache_classmethod(func):
    """
    A decorator that caches the output of a classmethod function.
    The cache is stored in the class's _CACHE dictionary under a key specific to the function.
    """
    cache_attr = f'_cache_{func.__name__}'

    def wrapper(cls, *args, **kwargs):
        if getattr(cls, 'GLOBAL_DISABLE_METHOD_CACHE', False):
            return func(cls, *args, **kwargs)
        
        if cache_attr not in cls._CACHE:
            cls._CACHE[cache_attr] = {}
        
        key = (args, tuple(sorted(list(kwargs.items()))))
        cache = cls._CACHE[cache_attr]
        
        if key not in cache:
            cache[key] = func(cls, *args, **kwargs)
        
        return cache[key]
        
    return classmethod(wrapper)


class BaseSchema(BaseModel, ABC):
    model_config = ConfigDict(extra='ignore')  # Allow additional arguments during model init!
    PRIMITIVE_TYPES: ClassVar[Tuple[type, ...]] = _PRIMITIVE_TYPES
    # CORE_PRIMITIVE_TYPES: ClassVar[Tuple[type, ...]] = _CORE_PRIMITIVE_TYPES
    """
    Represents input, config, output base schemas for a node template and also an HITL (human input required) schema.
    NOTE: just like BaseModel, BaseSchema fields can be recursively nested and point to other BaseSchema models.
    TODO: implement single select and multi select fields potentially
    NOTE: schemas can be inherited to create new schemas.

    Add following validations:
    1. all json_extra fields must be in EXTRA_FIELD_KEYS_WITH_DEFAULTS
    2. in callable validation during model instantiation, input json to instantiate model may not contain any non-user editable fields

    NOTE: Allowed types are: 
    - Base Objects
        - Primitives (str | int | float | bool | bytes)
        - Enums (str Enum or IntEnum)
        - BaseSchema
    - Nested Objects
        - List [primitives | enums | BaseSchema]
        - Dict [primitive keys, allowed List | Base Objects]
        - Nested Lists or Dicts not allowed!
    - Optional
        - Union [None, Base Objects | Nested Objects]
        - Nested Optional not allowed!
    
    # NOTE: datetime can be passed via milliseconds since epoch or isoformat string!
    #   UUID as string or bytes

    For single select and multi select fields, the type must Enum and List[Enum] respectively, never Literal!

    In the future; there could be backward compatibility layer to load older instances in newer Schema versions.
    Especially true for trivial changes like Enum to str, int to float etc.
    It could just be a custom init function for eg! Schema registry seems like an overkill for now.
    {
        "backward_schema_version": "1.0.0",
        "_type_converters" = [{"old_type": <type>, "new_type": <type>, "converter": lambda x: x}, ...],
        "field_converters" = ["field_name": {"old_type": <type>, "new_type": <type> # same as current field type, "converter": lambda x: x}, ...],
    }

    2 kinds of traversals needed:
    1. Declared Type traversals -> useful to find nested BaseSchema / primitive / Enum
    2.     For diffing
    3. For dumping object values which are user visible -> (Enum: <enum_instance>.value)

    Other diff types 
    1. Enum -> Enum values changed; select to multiselect and vice versa
    2. Optional




    """
    # Keys used for converting schemas into JSON or field definitions in BaseSchemas
    DEPRECATED_FIELD_KEY: ClassVar[str] = "_deprecated"
    FOR_LLM_TOOL_CALL_FIELD_KEY: ClassVar[str] = "_for_llm_tool_call"
    OPTIONAL_FIELD_KEY: ClassVar[str] = "_optional"  # NOTE: optional fields must be provided with a default value
    USER_EDITABLE_FIELD_KEY: ClassVar[str] = "_user_editable"  # NOTE: marked as such via SkipJsonSchema Annotation!
    DEFAULT_FIELD_KEY: ClassVar[str] = "_default"
    TYPE_FIELD_KEY: ClassVar[str] = "_type"
    NESTED_SCHEMA_KEY: ClassVar[str] = "_nested_schema"
    # EXTRA_FIELD_KEY: ClassVar[str] = "_extra"
    # USER_EDITABLE_FIELD_KEY: ClassVar[str] = "_user_editable"  # default value True if not mentioned in Field
    EXTRA_FIELD_KEYS_WITH_DEFAULTS: ClassVar[Dict[str, Any]] = {
        DEPRECATED_FIELD_KEY: False,
        FOR_LLM_TOOL_CALL_FIELD_KEY: True,
        # USER_EDITABLE_FIELD_KEY: True,
    }
    _CACHE: ClassVar[Optional[Dict[str, Any]]] = None  # TODO: cache class processing so recursively, each class is not called too frequently!
    _CACHE_FIELD_VALIDATION_RESULTS_KEY: ClassVar[str] = "_field_validation_results"
    GLOBAL_DISABLE_METHOD_CACHE: ClassVar[bool] = True
    DTYPE_LIST_KEY: ClassVar[str] = "list"
    DTYPE_DICT_KEY: ClassVar[str] = "dict"

    IS_DYNAMIC_SCHEMA: ClassVar[bool] = False

    # TODO: FIXME: add support for langgraph serializable types to use powerful messaging capabilities and state management with OOB reducers!
    # AnyMessage: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/messages/utils.py#L64
    # BaseMessage: https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/messages/base.py#L18
    # add_messages: https://langchain-ai.github.io/langgraph/how-tos/state-reducers/#messagesstate

    @classmethod
    def __pydantic_init_subclass__(cls, *args, **kwargs):
        """
        Validates schema field definitions during class creation.
        
        Performs validation checks on field definitions:
        1. All json_schema_extra fields must be defined in EXTRA_FIELD_KEYS_WITH_DEFAULTS
        2. Field types must be one of:
           - Primitives (str, int, float, bool)
           - Enums
           - Lists of primitives or enums
           - Dicts with primitive keys and allowed value types
           - Nested BaseSchema classes
           - Optional versions of above types
        
        Raises:
            TypeError: If field definitions violate the rules
        """
        super().__pydantic_init_subclass__(*args, **kwargs)
        
        # # Get the class's MRO (Method Resolution Order) to find all ancestors
        # mro = cls.mro()
        # # Store the ancestor history, excluding the class itself and object/BaseModel
        # ancestors = [ancestor.__name__ for ancestor in mro 
        #             if ancestor not in (object, BaseModel)]
        # class_id = "__".join(ancestors)
        # assert class_id not in cls._CACHE, f"Class has duplicate ancestor history / name! {class_id}"
        # cls._CACHE[class_id] = cls
        # print("cls._CACHE:", cls._CACHE)
        # # Store ancestor history in class variable for reference
        
        # NOTE: instantiating cache here means this classvar is not shared across diff 
        #     subclass / ancestor inheritance hierarchies, only this specific class, 
        #     but shared across its instances!
        cls._CACHE = {}
        # Validate all field type annotations when class is defined
        # annotations = cls.__annotations__
            
        
        # Get all fields defined in the schema
        for field_name, field in cls.model_fields.items():
            # Validate json_schema_extra keys
            if field.json_schema_extra:
                invalid_keys = set(field.json_schema_extra.keys()) - set(cls.EXTRA_FIELD_KEYS_WITH_DEFAULTS.keys())
                if invalid_keys:
                    raise TypeError(
                        f"Field '{field_name}' contains invalid json_schema_extra keys: {invalid_keys}. "
                        f"Allowed keys are: {list(cls.EXTRA_FIELD_KEYS_WITH_DEFAULTS.keys())}"
                    )
            # Validate field type
            result = BaseSchema._validate_type(field.annotation)
            if not result.valid:
                raise TypeError(
                    f"Invalid type annotation for field '{field_name}' in {cls.__name__}: "
                    f"{result.error_message}"
                )

            # # Validate Optional fields have default value or are not required
            # if result.is_optional:
            #     if field.is_required():
            #         raise TypeError(
            #             f"Field '{field_name}' is marked as Optional but has no default value "
            #             f"and is required. Optional fields must either have a default value "
            #             f"or be marked as not required."
            #         )

            # Cache validation result
            if cls._CACHE_FIELD_VALIDATION_RESULTS_KEY not in cls._CACHE:
                cls._CACHE[cls._CACHE_FIELD_VALIDATION_RESULTS_KEY] = {}
            cls._CACHE[cls._CACHE_FIELD_VALIDATION_RESULTS_KEY][field_name] = result
    
    @classmethod
    def _get_field_validation_result(cls, field_name: str) -> FieldValidationResult:
        """
        Validate a type annotation and cache the result.
        """
        if cls._CACHE_FIELD_VALIDATION_RESULTS_KEY in cls._CACHE:
            if field_name in cls._CACHE[cls._CACHE_FIELD_VALIDATION_RESULTS_KEY]:
                return cls._CACHE[cls._CACHE_FIELD_VALIDATION_RESULTS_KEY][field_name]
        raise ValueError(f"Field '{field_name}' not found in cache")

    
    @staticmethod
    def _validate_type(type_annotation, path=[], check_literal_instance_are_core_primitive_types: bool = False, only_primitives_allowed: bool = False, union_type_allowed: bool = True, list_type_allowed: bool = True, dict_type_allowed: bool = True):
        """
        Validate a type annotation.
        Returns FieldValidationResult with valid flag and error message.

        union_type_allowed: bool = True --> this is set to False in recursive calls to disallow 
            nested Union types
        list_type_allowed: bool = True --> this is set to False in recursive calls to disallow 
            nested types eg: List[List[X]] or Dict[K, Dict[K, V]]
        dict_type_allowed: bool = True --> this is set to False in recursive calls to disallow 
            nested types eg: Dict[K, Dict[K, V]]
        """
        def create_iterator(path):
            def iterator(field_value):
                if path is None:
                    yield field_value
                    return
                # Initialize stack with current field_value and path index
                stack = [(field_value, 0, None)]  # (value, path_idx, iterator)
                
                while stack:
                    current_value, path_idx, iterator = stack[-1]
                    
                    # If we've processed all path items, yield the current value
                    if path_idx >= len(path):
                        yield current_value
                        stack.pop()
                        continue
                        
                    item = path[path_idx]
                    
                    # If no iterator exists yet, create one based on type
                    if iterator is None:
                        if item == BaseSchema.DTYPE_LIST_KEY and isinstance(current_value, list):
                            iterator = iter(current_value)
                        elif item == BaseSchema.DTYPE_DICT_KEY and isinstance(current_value, dict):
                            iterator = iter(current_value.values())
                        else:
                            # Type mismatch - pop current value
                            stack.pop()
                            continue
                        # Update stack entry with new iterator
                        stack[-1] = (current_value, path_idx, iterator)
                        
                    # Try to get next value from iterator
                    try:
                        next_value = next(iterator)
                        # Add next value to stack with next path level
                        stack.append((next_value, path_idx + 1, None))
                    except StopIteration:
                        # Iterator exhausted, pop current level
                        stack.pop()
            return iterator
        
        if check_literal_instance_are_core_primitive_types:
            if type_annotation is None or any(isinstance(type_annotation, t) for t in _CORE_PRIMITIVE_TYPES):
                return FieldValidationResult(True, core_type_annotation=type_annotation, core_type_class=type_annotation, core_type_object_iterator=create_iterator(path))
            else:
                return FieldValidationResult(
                    False, 
                    f"Only core primitive types are allowed, got {type_annotation}"
                )
        # Check primitive types directly
        if type_annotation in BaseSchema.PRIMITIVE_TYPES:
            return FieldValidationResult(True, core_type_annotation=type_annotation, core_type_class=type_annotation, core_type_object_iterator=create_iterator(path))
        
        if only_primitives_allowed:
            return FieldValidationResult(
                False, 
                f"Only primitive types are allowed, got {type_annotation}"
            )
        
        # Check if it's BaseSchema or it's subclass | or Enum
        if inspect.isclass(type_annotation):
            if issubclass(type_annotation, BaseSchema) or issubclass(type_annotation, Enum) or issubclass(type_annotation, BaseModel):
                return FieldValidationResult(True, core_type_annotation=type_annotation, core_type_class=type_annotation, core_type_object_iterator=create_iterator(path))
        
        # Get origin and args for generic types
        origin = get_origin(type_annotation)
        args = get_args(type_annotation)

        if origin is Literal:
            first_result = None
            for arg in args:
                result = BaseSchema._validate_type(arg, check_literal_instance_are_core_primitive_types=True, path=path)
                if first_result is None:
                    first_result = result
                if not result.valid:
                    return FieldValidationResult(
                        False, 
                        f"Invalid Literal type: {result.error_message}"
                    )
            first_result.is_optional = None in args
            first_result.core_type_class = type_annotation
            first_result.core_type_annotation = type_annotation
            first_result.core_type_object_iterator = create_iterator(path)
            return first_result

        
        # Handle Optional (Union with None)
        if origin is Union and union_type_allowed:
            # Check if None is in the union arguments
            is_optional = False
            non_none_args = args
            if type(None) in args:
                # Get all non-None arguments
                non_none_args = [arg for arg in args if arg is not type(None)]
                is_optional = True
                
            # For Optional[X], just validate X
            if len(non_none_args) == 1:
                result = BaseSchema._validate_type(non_none_args[0], path=path)
                result.is_optional = is_optional
                # TODO: FIXME: lot of type related algos in this class assume unnested Union types, etc! FIX that assumption!
                # # NOTE: the below 3 were only copied to this branch because nested Union Types are allowed now! Commenting these out as they fail tests!
                # result.core_type_class = type_annotation
                # result.core_type_annotation = type_annotation
                # result.core_type_object_iterator = create_iterator(path)
                return result
            else:
                # For Union[X, Y, None], validate all non-None types; only_primitives_allowed!
                first_result = None
                for arg in non_none_args:
                    result = BaseSchema._validate_type(arg, path=path)
                    if first_result is None:
                        first_result = result
                    if not result.valid:
                        return FieldValidationResult(
                            False, 
                            f"Invalid Union type: {result.error_message}"
                        )
                first_result.is_optional = is_optional
                first_result.core_type_class = type_annotation
                first_result.core_type_annotation = type_annotation
                first_result.core_type_object_iterator = create_iterator(path)
                return first_result
            
            # # Unions without None are not supported
            # return FieldValidationResult(
            #     False, 
            #     f"Union types are only supported as Optional (Union with None)"
            #     f"received Args to Union: {args}"
            # )
        
        # Handle List[X]
        if origin is list and list_type_allowed:
            if len(args) != 1:
                return FieldValidationResult(
                    False, 
                    f"List requires exactly one type argument"
                )
            
            # Validate the list item type
            item_result = BaseSchema._validate_type(args[0], path=path + [BaseSchema.DTYPE_LIST_KEY], union_type_allowed=True, list_type_allowed=True, dict_type_allowed=True)
            if not item_result.valid:
                return FieldValidationResult(
                    False, 
                    f"Invalid List item type: {item_result.error_message}"
                )
            item_result.core_type_annotation = type_annotation
            return item_result
        
        # Handle Dict[K, V]
        if origin is dict and dict_type_allowed:
            if len(args) != 2:
                return FieldValidationResult(
                    False, 
                    f"Dict requires exactly two type arguments"
                )
            
            key_type, value_type = args
            
            # Keys must be primitives
            if key_type not in BaseSchema.PRIMITIVE_TYPES:
                return FieldValidationResult(
                    False, 
                    f"Dict keys must be primitives, got {key_type}"
                )
            
            # Validate the value type
            value_result = BaseSchema._validate_type(value_type, path=path + [BaseSchema.DTYPE_DICT_KEY], union_type_allowed=True, list_type_allowed=True, dict_type_allowed=True)
            if not value_result.valid:
                return FieldValidationResult(
                    False, 
                    f"Invalid Dict value type: {value_result.error_message}"
                )
            value_result.core_type_annotation = type_annotation
            return value_result
        
        # If we get here, the type is not supported
        return FieldValidationResult(
            False, 
            f"Unsupported type: {type_annotation}. Must be a primitive, BaseSchema, Enum, "
            f"or collection (List, Dict, Optional) of these types."
        )
    
    cache_classmethod = staticmethod(_cache_classmethod)

    @staticmethod
    def _is_field_deprecated(model_field: Any) -> bool:
        """
        Check whether a given model field is marked as deprecated based on its extra metadata.

        Args:
            model_field (Any): The Pydantic model field to inspect.

        Returns:
            bool: True if the field is deprecated, False otherwise.
        """
        return ((model_field.json_schema_extra is not None) and 
                model_field.json_schema_extra.get(BaseSchema.DEPRECATED_FIELD_KEY, False))

    @staticmethod
    def _is_field_for_llm_tool_call(model_field: Any) -> bool:
        """
        Check whether a given model field is marked as for llm tool call based on its extra metadata.

        Args:
            model_field (Any): The Pydantic model field to inspect.

        Returns:
            bool: True if the field is for llm tool call, False otherwise.
        """
        return ((model_field.json_schema_extra is None) or 
                model_field.json_schema_extra.get(BaseSchema.FOR_LLM_TOOL_CALL_FIELD_KEY, True))
    
    @staticmethod
    def _is_field_user_editable(model_field: Any) -> bool:
        """
        Check whether a given model field is marked as user editable based on its extra metadata.
        """
        return (model_field.metadata is None) or \
                all(not isinstance(metadata, SkipJsonSchema) for metadata in model_field.metadata)

    # @classmethod
    # @cache_classmethod
    # def get_user_editable_fields(cls, include_deprecated: bool = False) -> Dict[str, Any]:
    #     """
    #     Retrieve a list of user-editable field names.
    #     Fields are considered user editable based on the 'user_editable' flag in their JSON schema extra.
    #     For fields whose type is a subclass of BaseSchema, this function is recursively called
    #     to retrieve nested user editable fields. The nested fields are prefixed with the parent's field name,
    #     using dot notation (e.g., "parent.nested_field").
    #     Optionally, fields marked as deprecated are excluded unless include_deprecated is True.
        
    #     Args:
    #         include_deprecated (bool, optional): If False, deprecated fields are excluded. Defaults to False.
        
    #     Returns:
    #         List[str]: List of user editable field names, including nested ones in dot notation.
            
    #     Caveats:
    #         - This implementation only recurses for fields whose type is directly a subclass of BaseSchema.
    #           It does not handle cases where fields are Optional[BaseSchema] or list types containing BaseSchema.
    #         - Any changes to the nested schema structure could impact the resulting naming conventions.
        
    #     TODO: handle cases where fields are Optional[BaseSchema] or list types containing BaseSchema.
    #     Or Dict[str, BaseSchema]
        
    #     """
    #     print(cls, include_deprecated)
    #     editable_fields: Dict[str, Any] = {}
    #     # Iterate over each defined model field in the class.
    #     for field_name, model_field in cls.model_fields.items():
    #         # Exclude deprecated fields if include_deprecated is False.
    #         if not include_deprecated and BaseSchema._is_field_deprecated(model_field):
    #             continue

    #         # Retrieve the type of the current field.
    #         field_type = model_field.annotation
    #         # Check if the field's type is a subclass of BaseSchema to handle nested schemas.
    #         if BaseSchema._is_field_user_editable(model_field):
    #             if isinstance(field_type, type) and issubclass(field_type, BaseSchema):
    #                 # Recursively obtain the user-editable fields from the nested BaseSchema.
    #                 nested_editable_fields: Dict[str, Any] = field_type.get_user_editable_fields(include_deprecated=include_deprecated)
    #                 # Prefix each nested field name with the parent's field name using dot notation.
    #                 editable_fields[f"{field_name}"] = nested_editable_fields
    #                 # for nested_field in nested_editable_fields:
    #                 #     editable_fields[f"{field_name}.{nested_field}"] = nested_editable_fields[nested_field]
    #             else:
    #                 editable_fields[f"{field_name}"] = True
                        
    #     return editable_fields

    def model_dump_only_user_editable(self, include_deprecated: bool = True, serialize_values: bool = False, *args, **kwargs) -> str:
        """
        Serialize the model instance to a JSON string, including only user-editable fields.

        NOTE: use the dumped JSON schema if dump needs additional field metadata, eg: alias, name ,descriptions etc!

        NOTE: model_dump from pydantic can be used with *args, **kwargs to achive things such as dumping in JSON serializable format, include/exclude fields etc!
        https://docs.pydantic.dev/latest/api/base_model/#pydantic.BaseModel.model_dump
        """
        model_dump = {}
        if serialize_values:
            dump_str = self.model_dump_json(*args, **kwargs)
            full_model_dump = json.loads(dump_str)
        else:
            full_model_dump = self.model_dump(*args, **kwargs)
        for (field_name, field), (_, field_dump) in zip(self.model_fields.items(), full_model_dump.items()):
            # Check if field is marked as user_editable
            if (BaseSchema._is_field_deprecated(field) and not include_deprecated) or \
                (not BaseSchema._is_field_user_editable(field)):
                continue
            value = getattr(self, field_name)
            # Handle different value types:
            # 1. BaseSchema - Recursively get user editable fields
            # 2. List - Process each element 
            # 3. Dict - Process each value
            # 4. Enum - Get the value
            # 5. Primitive types - Use directly
            def _process_value(value: Any, value_dump: Any) -> Any:
                """Helper function to process individual values based on their type"""
                if isinstance(value, BaseSchema):
                    return value.model_dump_only_user_editable(include_deprecated=include_deprecated, serialize_values=serialize_values, *args, **kwargs)
                elif isinstance(value, list):
                    return [_process_value(item, item_dump) for item, item_dump in zip(value, value_dump)]
                elif isinstance(value, dict):
                    return {k: _process_value(val, val_dump) for (k, val), (_, val_dump) in zip(value.items(), value_dump.items())}
                # elif value is None:
                #     return value_dump
                # elif isinstance(value, Enum):
                #     return value_dump  # value.value
                else:
                    return value_dump
            model_dump[field_name] = _process_value(value, field_dump)
        return model_dump
    
    # @model_validator(mode='before')
    @classmethod
    def validate_only_user_editable_fields_provided_in_input(cls, data: Dict[str, Any]) -> bool:
        """
        Validates that the input data used to instantiate the model does not contain any non-user editable fields.
        
        This validator runs before model creation and checks that all fields in the input data are marked as 
        user editable. For nested BaseSchema fields, it recursively validates their fields as well.
        
        Args:
            data (Dict[str, Any]): The input dictionary used to instantiate the model
            
        Returns:
            Dict[str, Any]: The validated input data if all fields are user editable
            
        Raises:
            ValueError: If any non-user editable fields are found in the input data
            
        Example:
            >>> class MySchema(BaseSchema):
            ...     editable: str  # User editable by default
            ...     internal: SkipJsonSchema[str]  # Non-user editable
            ...     nested: "NestedSchema"  # Nested schema
            ...
            >>> MySchema(editable="ok", nested={"editable_nested": "ok"})  # Works
            >>> MySchema(internal="bad")  # Raises ValueError
            >>> MySchema(nested={"non_editable_nested": "bad"})  # Raises ValueError
        """
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")

        # Check each field in input data against model fields
        for field_name, field_value in data.items():
            if field_name not in cls.model_fields:
                raise ValueError(f"Unknown field '{field_name}'")
                
            model_field = cls.model_fields[field_name]
            
            # Check if current field is user editable
            if not cls._is_field_user_editable(model_field):
                return False, field_name
                
            # Handle nested BaseSchema fields
            # Get field validation result to determine core type class and annotation
            field_validation_result = cls._get_field_validation_result(field_name)
            core_type_class = field_validation_result.core_type_class
            # core_type_annotation = field_validation_result.core_type_annotation
            core_type_object_iterator = field_validation_result.core_type_object_iterator

            if issubclass(core_type_class, BaseSchema):
                for item in core_type_object_iterator(field_value):
                    if item is None:
                        continue
                    is_valid, nested_field_name = core_type_class.validate_only_user_editable_fields_provided_in_input(item)
                    if not is_valid:
                        return False, f"{field_name}.{nested_field_name}"

        return True, None

    @_cache_classmethod
    def get_schema_for_db(cls, include_deprecated: bool = True) -> Dict[str, Any]:
        """
        Generate a dictionary representing the complete schema of the model for database storage.

        This method constructs a schema dictionary that includes all fields defined in the model,
        capturing both user-visible and non-user-visible fields. It records default values, explicit
        type information, and any additional schema metadata specified via 'json_schema_extra'.
        Deprecated fields are included if 'include_deprecated' is True; otherwise, they are omitted.

        For any field whose type is a subclass of BaseSchema or Enum, the method recursively dumps the nested
        schema structure, ensuring that the entire hierarchical definition is preserved. For Enum fields,
        the possible values are stored in the nested schema.

        Args:
            include_deprecated (bool, optional): Whether to include deprecated fields in the schema. 
                                               Defaults to True.

        Caveats:
            - Recursion is only performed for fields whose core type is directly a subclass of BaseSchema.
              Fields declared as Optional[BaseSchema] or as lists containing BaseSchema are also deeply inspected and 
                the nested schema is also generated.
            - Default values are taken directly from the field definitions; if no default is provided,
              the value defaults to None. PydanticUndefined is converted to None.
            - Changes in nested BaseSchema structures or extra metadata may impact the resulting schema representation.

        Returns:
            Dict[str, Any]: A dictionary describing the model's schema. Each key corresponds to a field name,
            and each value is a dictionary containing:
                - The field's type annotation as a string
                - Nested schema for BaseSchema fields or enum values for Enum fields
                - Default value
                - Deprecation status
                - User editability flag
                - Optional status
                - Any extra metadata from json_schema_extra
        NOTE: this doesn't work with multi-hop circular references between BaseSchema types, 
            only handles the case where a BaseSchema type is circular referenced by itself!
        """
        schema: Dict[str, Any] = {}

        # Iterate over each model field defined in the class.
        for field_name, model_field in cls.model_fields.items():
            # Skip deprecated fields if they should not be included.
            if (not include_deprecated) and cls._is_field_deprecated(model_field):
                continue

            # Build the schema representation for this field.
            field_schema: Dict[str, Any] = {}
            field_schema[BaseSchema.DEPRECATED_FIELD_KEY] = cls._is_field_deprecated(model_field)
            field_schema[BaseSchema.USER_EDITABLE_FIELD_KEY] = cls._is_field_user_editable(model_field)
            # NOTE: there's a difference between default set to None and default set to Pydantic Undefined; 
            #     it has to be explicity set to None!
            field_schema[BaseSchema.DEFAULT_FIELD_KEY] = getattr(model_field, "default", None)
            if field_schema[BaseSchema.DEFAULT_FIELD_KEY] is PydanticUndefined:
                field_schema[BaseSchema.DEFAULT_FIELD_KEY] = None
            field_schema[BaseSchema.OPTIONAL_FIELD_KEY] = not model_field.is_required()

            # Include any extra schema metadata for effective diff comparisons.
            extra_metadata = getattr(model_field, "json_schema_extra", None)
            if extra_metadata is not None:
                field_schema.update(extra_metadata)

            field_validation_result = cls._get_field_validation_result(field_name)
            # This is the full annotation without the Optional part i.e. List[int] or Dict[str, BaseSchema] etc
            annotation = field_validation_result.core_type_annotation
            field_schema[BaseSchema.TYPE_FIELD_KEY] = getattr(annotation, '__name__', str(annotation)) if annotation in cls.PRIMITIVE_TYPES else str(annotation)
            # The core type class either Primitive | BaseSchema | Enum
            core_type_class = field_validation_result.core_type_class
            field_schema[BaseSchema.NESTED_SCHEMA_KEY] = None
            # print("field_type:", field_type)
            # Check if the field is a nested BaseSchema.
            if isinstance(core_type_class, type):
                if issubclass(core_type_class, BaseSchema):
                # Recursively retrieve the nested schema.
                    field_schema[BaseSchema.NESTED_SCHEMA_KEY] = "$self" if core_type_class is cls else core_type_class.get_schema_for_db(include_deprecated=include_deprecated)
                elif issubclass(core_type_class, Enum):
                    field_schema[BaseSchema.NESTED_SCHEMA_KEY] = {
                        "$type": "enum",
                        "$values": [e.value for e in core_type_class]
                    }

            schema[field_name] = field_schema
            # print("field_name:", field_name, "field_schema:", field_schema)

        return schema
    
    @classmethod
    def get_required_fields(cls) -> List[str]:
        """
        Get a list of required field names from the schema.
        
        This method traverses the schema and returns a list of field names that are required 
        (not optional) and not deprecated. This is useful for validating that all required 
        fields are provided when using the schema.

        Returns:
            List[str]: List of required field names
        """
        required_fields = []
        
        # Check each field
        for field_name, field_info in cls.model_fields.items():
            # Add field if it's required (not optional) 
            if field_info.is_required():
                required_fields.append(field_name)
                
        return required_fields

    @classmethod
    def diff_from_provided_schema(cls, provided_schema: Dict[str, Any], include_deprecated: bool = True, self_is_base_for_diff:bool = False) -> Dict[str, List[str]]:
        """
        Compare the current schema against a provided older schema dump and return a detailed diff.

        The differences are categorized as follows:
          - "added": Fields present in current schema but absent in provided schema (new fields added).
          - "removed": Fields present in provided schema but missing in current schema (fields removed).
          - "modified_deprecated": Fields where the deprecation flag has changed.
          - "modified_default": Fields where the default value has changed.
          - "modified_editable": Fields where the 'is_user_editable' attribute has changed.
          - "modified_type": Fields where the field type or nested schema has changed.
          - "modified_optional": Fields where the optional status has changed.
          - "modified_enum_values": Fields where enum values have changed, with subcategories:
            - "backward_compatible": Only new values added to current schema
            - "non_backward_compatible": Values removed or modified from provided schema
          - "modified_other": Any other modifications (e.g., differences in extra metadata).

        Args:
            provided_schema (Dict[str, Any]): A dictionary representing the older schema for comparison.
            include_deprecated (bool, optional): If False, deprecated fields are filtered out before comparison.
                                               Defaults to True.

        Returns:
            Dict[str, List[str]]: A dictionary with categorized differences, each containing a list of 
                                 dot-notated paths where differences were found.
        """
        # Get current schema from the class
        current_schema: Dict[str, Any] = cls.get_schema_for_db(include_deprecated=include_deprecated)
        if self_is_base_for_diff:
            provided_schema, current_schema = current_schema, provided_schema

        
        def filter_deprecated(schema: Dict[str, Any]) -> Dict[str, Any]:
            """Helper to recursively remove deprecated fields from schema."""
            filtered: Dict[str, Any] = {}
            for key, field in schema.items():
                if field.get(BaseSchema.DEPRECATED_FIELD_KEY, False):
                    continue
                new_field: Dict[str, Any] = field.copy()
                if isinstance(new_field.get(BaseSchema.NESTED_SCHEMA_KEY), dict):
                    new_field[BaseSchema.NESTED_SCHEMA_KEY] = filter_deprecated(new_field[BaseSchema.NESTED_SCHEMA_KEY])
                filtered[key] = new_field
            return filtered
        
        if not include_deprecated:
            current_schema = filter_deprecated(current_schema)
            provided_schema = filter_deprecated(provided_schema)
        
        def compare_enum_values(current_values: List[Any], provided_values: List[Any]) -> Dict[str, bool]:
            """Helper to compare enum values and determine compatibility."""
            provided_set = set(provided_values)
            current_set = set(current_values)
            
            removed_values = provided_set - current_set
            added_values = current_set - provided_set
            
            return {
                "backward_compatible": len(removed_values) == 0 and len(added_values) > 0,
                "non_backward_compatible": len(removed_values) > 0
            }

        def recursive_diff(d1: Any, d2: Any, path: Optional[List[str]] = None) -> Dict[str, List[str]]:
            """
            Recursively calculate differences between current (d1) and provided (d2) schema segments (Both must be dicts!).
            """
            def add_change_modified_type(diff: Dict[str, List[str]], new_path: List[str]):
                full_path = ".".join(new_path + [BaseSchema.TYPE_FIELD_KEY])
                if full_path not in diff["modified_type"]:
                    diff["modified_type"].append(full_path)
            
            if path is None:
                path = []
            
            diff_categories = ["added", "removed", "modified_deprecated", "modified_default", 
                             "modified_editable", "modified_type", "modified_optional", 
                             "modified_enum_values_backward_compatible",
                             "modified_enum_values_non_backward_compatible", "modified_other"]
            diff: Dict[str, List[str]] = {cat: [] for cat in diff_categories}
            current_path: str = '.'.join(path) + ('.' if path else '')
            
            if (not isinstance(d1, dict)) or (not isinstance(d2, dict)):
                raise ValueError(f"Invalid schema structure! \nd1: {json.dumps(d1, indent=4)} \nd2: {json.dumps(d2, indent=4)}")

            keys1 = set(d1.keys())
            keys2 = set(d2.keys())
            
            # Fields added to current schema
            for key in keys1 - keys2:
                diff["added"].append(f"{current_path}{key}")
            
            # Fields removed from current schema
            for key in keys2 - keys1:
                diff["removed"].append(f"{current_path}{key}")
            
            # Compare common fields
            for key in keys1 & keys2:
                new_path = path + [key]
                if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    # Check other field attributes
                    field_attributes = {
                        BaseSchema.DEPRECATED_FIELD_KEY: "modified_deprecated",
                        BaseSchema.DEFAULT_FIELD_KEY: "modified_default",
                        BaseSchema.USER_EDITABLE_FIELD_KEY: "modified_editable",
                        BaseSchema.TYPE_FIELD_KEY: "modified_type",
                        BaseSchema.OPTIONAL_FIELD_KEY: "modified_optional"
                    }
                    
                    for attr, diff_key in field_attributes.items():
                        if d1[key].get(attr) != d2[key].get(attr):
                            full_path = ".".join(new_path + [attr])
                            diff[diff_key].append(full_path)
                    
                    
                    nested_schema1 = d1[key].get(BaseSchema.NESTED_SCHEMA_KEY, {})
                    nested_schema2 = d2[key].get(BaseSchema.NESTED_SCHEMA_KEY, {})
                    
                    if (isinstance(nested_schema1, dict) and isinstance(nested_schema2, dict)): 
                        if nested_schema1.get("$type") == "enum" and nested_schema2.get("$type") == "enum":
                            # Check for enum value changes
                            enum_diff = compare_enum_values(
                                nested_schema1.get("$values", []),
                                nested_schema2.get("$values", [])
                            )
                            if enum_diff["backward_compatible"]:
                                diff["modified_enum_values_backward_compatible"].append(".".join(new_path))
                            if enum_diff["non_backward_compatible"]:
                                diff["modified_enum_values_non_backward_compatible"].append(".".join(new_path))
                        elif nested_schema1.get("$type", False) != nested_schema2.get("$type", False):
                            # type modified!
                            add_change_modified_type(diff, new_path)
                        else:
                            # nested schema is BaseSchema!
                            # Recursively check nested schemas
                            sub_diff = recursive_diff(nested_schema1, nested_schema2, new_path)
                            for cat in diff_categories:
                                diff[cat].extend(sub_diff.get(cat, []))
                    elif (nested_schema1 == "$self") or (nested_schema2 == "$self"):
                        if nested_schema1 != nested_schema2:
                            add_change_modified_type(diff, new_path)
                    elif (nested_schema1 is not None) or (nested_schema2 is not None):
                        add_change_modified_type(diff, new_path)
            return diff
        
        return recursive_diff(current_schema, provided_schema)

    # def get_deprecated_fields(self) -> List[str]:
    #     """
    #     Retrieve a list of field names that are marked as deprecated.

    #     Returns:
    #         List[str]: A list of deprecated field names.
    #     """
    #     deprecated_fields: List[str] = []
    #     for field_name, model_field in self.model_fields.items():
    #         if self._is_field_deprecated(model_field):
    #             deprecated_fields.append(field_name)
    #     return deprecated_fields

    @staticmethod
    def _remove_skip_json_schema_recursive_from_field(model_field: Any) -> Any:
        """
        Recursively remove SkipJsonSchema instances from the field's metadata and rebuild its type annotation 
        by processing nested BaseSchema types. Nested BaseSchema types (including those found within generic 
        containers such as List or Dict) are re-created using pydantic's create_model function via a copy strategy.
        
        The cleaning strategy is as follows:
          1. For direct BaseSchema types, iterate over its model_fields, remove SkipJsonSchema from each field’s 
             metadata, and create a new model using create_model.
          2. For generic annotations (e.g. List[OldBaseSchemaNested]), recursively clean each type parameter so that 
             any BaseSchema types are replaced by their cleaned (metadata-free) version.
        
        Args:
            model_field: The model field to process.
            
        Returns:
            A new model field with updated metadata, where SkipJsonSchema elements are removed, and with an updated 
            type annotation referencing cleaned BaseSchema types.
        """
        if model_field is None:
            return None

        # Remove SkipJsonSchema objects from the field's metadata.
        new_metadata = (
            [m for m in model_field.metadata if not isinstance(m, SkipJsonSchema)]
            if model_field.metadata
            else model_field.metadata
        )

        def clean_type(annotation: Any) -> Any:
            """
            Recursively clean a type annotation by replacing BaseSchema types with new models that have cleared metadata.
            
            Args:
                annotation: The original type annotation.
                
            Returns:
                The cleaned type annotation with any BaseSchema types (or generic containers thereof) replaced by a new type.
            """
            origin = get_origin(annotation)
            if origin is not None:
                # Process generic types (e.g. List[InnerSchema], Dict[str, InnerSchema], etc.)
                args = get_args(annotation)
                new_args = tuple(clean_type(arg) for arg in args)
                try:
                    # Attempt to reconstruct the generic type using the cleaned arguments.
                    return origin[new_args]
                except TypeError:
                    # If reconstruction fails, revert to the original annotation.
                    # TODO: FIXME: Fix Print and log instead!
                    print(f"ERROR: Failed to reconstruct generic type {annotation} with origin: {origin} and args: {args}")
                    return annotation
            elif isinstance(annotation, type) and issubclass(annotation, BaseSchema):
                # For direct BaseSchema types, create a new fields copy removing SkipJsonSchema metadata.
                copy_fields = {}
                for field_name, fld in annotation.model_fields.items():
                    # Recursively clean each nested field.
                    copy_fields[field_name] = BaseSchema._remove_skip_json_schema_recursive_from_field(fld)
                # Use create_model to build a new BaseSchema subclass with the cleaned fields.
                new_model = create_model(
                    f"Clean{annotation.__name__}",
                    __base__=BaseSchema,
                    __module__=annotation.__module__,
                    __doc__=f"Copy of {annotation.__name__} with cleared metadata",
                    **copy_fields
                )
                return new_model
            else:
                # For all other types, return the annotation unchanged.
                return annotation

        # Clean the field's type annotation recursively.
        new_annotation = clean_type(model_field.annotation)

        # Create and return a new copy of the model_field with updated metadata and cleaned annotation.
        new_field = copy(model_field)
        new_field.metadata = new_metadata
        new_field.annotation = new_annotation

        # NOTE: The below code sample is in case the new annotation is not properly instantiated within new_field! But should probably not be the case!
        # merged_new_field = FieldInfo.merge_field_infos(new_field, FieldInfo.from_annotation(new_annotation))

        return new_annotation, new_field

    @classmethod
    def model_json_schema_with_skipped_fields(cls, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Generate a complete JSON schema that includes skipped fields which are normally excluded due to 
        SkipJsonSchema metadata.

        This method rebuilds the model by processing all model fields using the _remove_skip_json_schema_recursive 
        function. It ensures that:
          - All SkipJsonSchema markers are removed from field metadata.
          - Any nested BaseSchema types (including those wrapped as generics) are re-created using pydantic's create_model.
        
        Args:
            *args: Positional arguments to be passed to model_json_schema().
            **kwargs: Keyword arguments to be passed to model_json_schema().
        
        Returns:
            Dict[str, Any]: The full JSON schema including all internal fields.
        
        Potential usecase: Let admin see and modify internal fields too which are not visible to users!

        NOTE: this doesn't work with circular references between BaseSchema types!
        """
        # Build a dictionary of cleaned fields.
        temp_fields: Dict[str, Any] = {}
        for field_name, field in cls.model_fields.items():
            temp_fields[field_name] = BaseSchema._remove_skip_json_schema_recursive_from_field(field)
            if cls._get_field_validation_result(field_name).core_type_class is cls:
                raise ValueError(f"Field {field_name} is a circular reference to BaseSchema of type {cls.__name__}! Cannot be skipped!")

        # Create a temporary model using the cleaned fields.
        temp_cls = create_model(
            f"Temp{cls.__name__}",
            __base__=BaseSchema,
            __module__=cls.__module__,
            __doc__=cls.__doc__,  # f"Temporary copy of {cls.__name__} with cleared metadata",
            **temp_fields
        )

        # Generate the JSON schema from the temporary model.
        return temp_cls.model_json_schema(*args, **kwargs)

    @classmethod
    def create_dynamic_schema_with_fields(
        cls, 
        fields: Dict[str, Any],
        schema_name: str = None,
        # module_name: Optional[str] = None
    ) -> Type['BaseSchema']:
        return create_dynamic_schema_with_fields(cls, fields, schema_name)


class BaseNodeConfig(BaseSchema, ABC):
    model_config = ConfigDict(extra='forbid')
    
    node_retry_count: SkipJsonSchema[Optional[int]] = Field(
        None,
        description="Default retry count for the node"
    )


# class InnerSchema(BaseSchema):
#     """Inner schema with mix of editable and non-editable fields."""
#     editable_str: str = Field(default="test", description="Editable string field")
#     non_editable_int: SkipJsonSchema[int] = Field(default=42, description="Non-editable integer field")
#     editable_str_2: str = Field(default="test", description="Editable string field")

# print(json.dumps(InnerSchema.model_json_schema(), indent=4))

# print(json.dumps(InnerSchema.create_dynamic_schema_with_fields(fields={"editable_str_2": None, "non_editable_int": None}).model_json_schema_with_skipped_fields(), indent=4))

# print(json.dumps(InnerSchema.create_dynamic_schema_with_fields(fields={}).model_json_schema_with_skipped_fields(), indent=4))

# class InnerSchema(BaseSchema):
#     """Inner schema with mix of editable and non-editable fields."""
#     editable_str: str = Field(default="test")
#     editable_int: int = Field(default=42)
#     editable_date: datetime = Field(default_factory=datetime.now)
#     deprecated_field: str = Field(default="old", **{BaseSchema.DEPRECATED_FIELD_KEY: True})
#     internal_id: SkipJsonSchema[str] = Field(default="id123") 
#     # test_field: _PRIMITIVE_TYPES_UNION = None
#     # internal_metadata = Annotated[SkipJsonSchema.__getitem__(Union.__getitem__(_PRIMITIVE_TYPES_EXTENDED)), Field(default_factory=dict)]  # Union.__getitem__((int, str))
#     # internal_metadata: SkipJsonSchema[Dict[str, _PRIMITIVE_TYPES_UNION]] = Field(default_factory=dict)
#     internal_metadata: SkipJsonSchema[Dict[str, Union[str, int]]] = Field(default_factory=dict)
#     normal_dict_field: Dict[str, str] = Field(default_factory=dict)
#     # Added list and dict fields
#     str_list: List[str] = Field(default_factory=list)
#     int_dict: Dict[str, int] = Field(default_factory=dict)

# print("\n\n" + "-"*100 + "\n\n")
# print("## model_json_schema")
# print(json.dumps(InnerSchema.model_json_schema(), indent=4))
# print("\n\n" + "-"*100 + "\n\n")
# print("## model_json_schema_with_skipped_fields")
# print(json.dumps(InnerSchema.model_json_schema_with_skipped_fields(), indent=4))
# print("\n\n" + "-"*100 + "\n\n")
# print("## get_schema_for_db")
# print(json.dumps(InnerSchema.get_schema_for_db(), indent=4))
