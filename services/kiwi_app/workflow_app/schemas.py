"""Pydantic schemas for Workflow Service API interaction."""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Union, Sequence, Literal
from pydantic import BaseModel, Field, validator, ConfigDict, field_validator, model_validator
from pydantic import HttpUrl
from enum import Enum

from kiwi_app.workflow_app.constants import LaunchStatus, WorkflowRunStatus, NotificationType, HITLJobStatus, SchemaType
# Import base User/Org schemas if needed for nesting, or use full models if simpler
# from kiwi_app.auth.schemas import UserRead, OrganizationRead
# Import GraphSchema for type hinting if needed, but graph_config will be stored as Dict
from workflow_service.graph.graph import GraphSchema
# Import event schemas for detailing run results
from workflow_service.services import events as event_schemas

# --- NodeTemplate Schemas --- #

class NodeTemplateBase(BaseModel):
    """Base schema for NodeTemplate."""
    name: str = Field(..., description="Unique name identifying the node type (e.g., 'llm_generator')")
    version: str = Field(..., description="Version string (e.g., '1.0.0', 'latest')")
    description: Optional[str] = Field(None, description="Description of the node template")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for inputs")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for outputs")
    config_schema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for node configuration")
    node_is_tool: Optional[bool] = Field(default=None, description="True if this is a tool node meant to be used with LLMs tool calling.")
    launch_status: LaunchStatus = Field(..., description="Deployment status of the node template")

class NodeTemplateCreate(NodeTemplateBase):
    """Schema for creating a NodeTemplate (Admin only)."""
    pass # Inherits all fields from base

class NodeTemplateUpdate(BaseModel):
    """Schema for updating a NodeTemplate (Admin only). Allows partial updates."""
    description: Optional[str] = None
    node_is_tool: Optional[bool] = Field(default=None, description="True if this is a tool node meant to be used with LLMs tool calling.")
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    config_schema: Optional[Dict[str, Any]] = None
    launch_status: Optional[LaunchStatus] = None

class NodeTemplateRead(NodeTemplateBase):
    """Schema for reading a NodeTemplate."""
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --- Workflow Schemas --- #

class WorkflowBase(BaseModel):
    """Base schema for Workflow."""
    name: str = Field(..., min_length=1, description="User-defined name for the workflow")
    description: Optional[str] = Field(None, description="Description of the workflow")
    graph_config: Dict[str, Any] = Field(..., description="The core definition of the workflow graph (nodes, edges, etc.)")
    version_tag: Optional[str] = Field(None, description="User-defined tag for versioning (e.g., 'v1.2-stable')")
    is_template: Optional[bool] = Field(default=False, description="Indicates if this workflow can be used as a template within the org")
    launch_status: Optional[LaunchStatus] = Field(default=LaunchStatus.DEVELOPMENT, description="Deployment status of the workflow")
    is_public: Optional[bool] = Field(default=False, description="Indicates if this workflow is publicly accessible")
    is_system_entity: Optional[bool] = Field(default=False, description="Indicates if this workflow is a system entity. Only admins can create system workflows.")

class WorkflowCreate(WorkflowBase):
    """Schema for creating a new Workflow."""
    parent_base_id: Optional[uuid.UUID] = None
    # is_system_entity: Optional[bool] = Field(default=False, description="Indicates if this workflow is a system entity. Only admins can create system workflows.")

class WorkflowUpdate(WorkflowBase):
    """Schema for updating an existing Workflow. Allows partial updates."""
    # set both the below fields as Optional!
    name: Optional[str] = Field(None, min_length=1, description="User-defined name for the workflow")
    graph_config: Optional[Dict[str, Any]] = Field(None, description="The core definition of the workflow graph (nodes, edges, etc.)")

class WorkflowRead(WorkflowBase):
    """Schema for reading a Workflow, including owner info."""
    id: uuid.UUID
    owner_org_id: Optional[uuid.UUID]
    created_at: datetime
    updated_at: datetime
    created_by_user_id: Optional[uuid.UUID] = None
    parent_base_id: Optional[uuid.UUID] = None

    model_config = ConfigDict(from_attributes=True)


# -- Graph Validation --
class WorkflowGraphValidationResult(BaseModel):
    """Schema for graph validation results."""
    is_valid: bool = Field(..., description="Whether the graph is valid overall")
    graph_schema_valid: bool = Field(..., description="Whether the graph schema structure is valid")
    node_configs_valid: bool = Field(..., description="Whether all node configurations are valid")
    errors: Dict[str, Any] = Field(default_factory=dict, description="Validation errors by category/node")
    validation_errors: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Structured validation issues by category/node: each issue includes field, node_id, node_name, expected, actual, actual_type, suggested_fix, documentation_url, and message")


# --- WorkflowRun Schemas --- #

class WorkflowRunBase(BaseModel):
    """Base schema for WorkflowRun."""
    status: WorkflowRunStatus
    inputs: Optional[Dict[str, Any]] = Field(None, description="Inputs provided when the run was triggered")
    outputs: Optional[Dict[str, Any]] = Field(None, description="High-level final outputs or summary")
    error_message: Optional[str] = None
    detailed_results_ref: Optional[str] = Field(None, description="Reference to detailed logs/results (e.g., NoSQL collection/path)")
    thread_id: Optional[uuid.UUID] = Field(None, description="Associated thread ID (e.g., from LangGraph)")
    retry_count: Optional[int] = Field(default=0, ge=0, description="Number of times this workflow run has been retried. Defaults to 0.")
    # resume_from_checkpoint: Optional[bool] = Field(default=False, description="Whether to resume from a checkpoint")


class WorkflowRunCreate(BaseModel):
    """Schema used internally or by other services to create a run record."""
    run_id: Optional[uuid.UUID] = None
    workflow_id: Optional[uuid.UUID] = None
    workflow_name: Optional[str] = Field(None, description="Name of the workflow to run, this is optional and is only used for debugging/logging purposes; workflow ID is used to fetch the workflow instance")
    inputs: Optional[Dict[str, Any]] = Field(None, description="Inputs to provide to the workflow run")
    # TODO: add checkpoint ID as well for resume!
    thread_id: Optional[uuid.UUID] = Field(None, description="Optional existing thread ID to reuse")
    parent_run_id: Optional[uuid.UUID] = Field(None, description="Optional parent run ID to reuse")
    graph_schema: Optional[GraphSchema] = None
    resume_after_hitl: Optional[bool] = False
    force_resume_experimental_option: Optional[bool] = Field(default=False, description="Experimental option to force resume after HITL even if not in WAITING_HITL state or without pending HITL jobs! (Use with caution!)")
    on_behalf_of_user_id: Optional[uuid.UUID] = Field(None, description="User ID to act on behalf of (requires superuser privileges)")
    tag: Optional[str] = Field(None, description="Optional tag to mark this run for experimentation tracking")
    applied_workflow_config_overrides: Optional[str] = Field(None, description="Comma-separated list of override IDs that were applied to this run")
    applied_workflow_config_override_tags: Optional[str] = Field(None, description="Comma-separated list of override tags that were applied to this run")
    retry_count: Optional[int] = Field(default=0, ge=0, description="Number of times this workflow run has been retried. Defaults to 0.")
    # Override configs
    include_active_overrides: Optional[bool] = Field(default=True, description="Whether to include active overrides")
    include_override_tags: Optional[List[str]] = Field(default=None, description="List of override tags to include")
    reset_overrides_on_hitl_resume: Optional[bool] = Field(default=False, description="Whether to reset overrides on HITL resume. If this is False, on resume, the same overrides applied in previous session of this run will be reapplied.")
    streaming_mode: Optional[bool] = Field(default=True, description="Whether to stream the LLM tokens")
    is_manual_retry: Optional[bool] = Field(default=False, description="Whether this is a manual retry of a failed/cancelled workflow run")
    manual_retry_search_by_workflow_name: Optional[bool] = Field(default=False, description="Whether to search for the workflow by name when retrying a failed/cancelled workflow run; so it will retry a newly ingested workflow even though workflow ID changed.")


class WorkflowRunJobCreate(WorkflowRunCreate):
    """Schema used specifically to trigger a new run."""
    owner_org_id: uuid.UUID
    triggered_by_user_id: Optional[uuid.UUID] = None
    status: Optional[WorkflowRunStatus] = Field(default=WorkflowRunStatus.SCHEDULED)

    # @model_validator(mode='before')
    # @classmethod
    # def check_workflow_or_graph(cls, values):
    #     workflow_id, graph_schema = values.get('workflow_id'), values.get('graph_schema')
    #     if workflow_id is None and graph_schema is None:
    #         raise ValueError('Either workflow_id or graph_schema must be provided.')
    #     if workflow_id is not None and graph_schema is not None:
    #         raise ValueError('Provide either workflow_id or graph_schema, not both.')
    #     return values

class WorkflowRunUpdate(BaseModel):
    """Schema for internal updates to a WorkflowRun (e.g., status, outputs)."""
    run_id: uuid.UUID
    status: Optional[WorkflowRunStatus] = None
    outputs: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    detailed_results_ref: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    thread_id: Optional[uuid.UUID] = None
    retry_count: Optional[int] = Field(None, ge=0, description="Number of times this workflow run has been retried")

class LogEntry(BaseModel):
    """Schema for a log entry with simplified fields."""
    level: str = Field(..., description="Log level as string (e.g., 'INFO', 'ERROR')")
    message: str = Field(..., description="The log message")
    timestamp: datetime = Field(..., description="The log timestamp")
    flow_run_id: uuid.UUID | None = Field(None, description="The Prefect flow run ID")

class WorkflowRunLogs(BaseModel):
    """Schema for reading a WorkflowRun logs."""
    logs: List[LogEntry] = Field(..., description="List of log entries with level, message, and timestamp")

class WorkflowRunRead(WorkflowRunBase):
    """Schema for reading a WorkflowRun summary (SQL data mainly)."""
    id: uuid.UUID
    workflow_id: Optional[uuid.UUID] = None
    workflow_name: Optional[str] = Field(None, description="Name of the workflow this run belongs to")
    owner_org_id: uuid.UUID
    triggered_by_user_id: Optional[uuid.UUID] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    run_ids: Optional[str] = Field(None, description="Comma-separated list of P run IDs that are part of this run")
    tag: Optional[str] = Field(None, description="Optional tag marking this run for experimentation tracking")
    applied_workflow_config_overrides: Optional[str] = Field(None, description="Comma-separated list of override IDs that were applied to this run")
    applied_workflow_config_override_tags: Optional[str] = Field(None, description="Comma-separated list of override tags that were applied to this run")
    parent_run_id: Optional[uuid.UUID] = Field(None, description="Optional parent run ID to reuse")
    model_config = ConfigDict(from_attributes=True)

class WorkflowRunState(BaseModel):
    """Schema for reading a WorkflowRun state (SQL data mainly)."""
    central_state: Dict[str, Any]
    node_outputs: Dict[str, Any]
    run_id: uuid.UUID
    thread_id: uuid.UUID

    model_config = ConfigDict(from_attributes=True)

# Define a type for the detailed_results (list of events from MongoDB)
WorkflowRunEventDetail = Union[event_schemas.WorkflowRunNodeOutputEvent, event_schemas.WorkflowRunStatusUpdateEvent, event_schemas.MessageStreamChunk, event_schemas.HITLRequestEvent, event_schemas.ToolCallEvent, event_schemas.NodeStatusEvent, Dict[str, Any]]

class WorkflowRunDetailRead(WorkflowRunRead):
    """Schema for reading detailed WorkflowRun results (combines SQL + NoSQL data)."""
    detailed_results: Optional[List[WorkflowRunEventDetail]] = Field(None, description="Detailed run events fetched from external store (MongoDB)")


# --- PromptTemplate Schemas --- #

class PromptTemplateBase(BaseModel):
    """Base schema for PromptTemplate."""
    name: str = Field(..., description="Name of the prompt template (unique within org/system context)")
    version: Optional[str] = Field(None, description="Version string (e.g., '1.0', 'latest')")
    description: Optional[str] = Field(None, description="Description of the prompt template")
    template_content: str = Field(..., description="The prompt template string (e.g., Jinja2 format)")
    input_variables: Optional[Dict[str, Any]] = Field(None, description="Dictionary of expected input variables and optional defaults")
    is_public: bool = Field(default=False, description="Whether the template is public (system templates only)")

class PromptTemplateCreate(PromptTemplateBase):
    """Schema for creating an organization-specific PromptTemplate."""
    is_system_entity: Optional[bool] = Field(default=False, description="Indicates if this prompt template is a system entity. Only system admins can create system templates.")

class PromptTemplateUpdate(BaseModel):
    """Schema for updating an organization-specific PromptTemplate."""
    name: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None
    template_content: Optional[str] = None
    input_variables: Optional[Dict[str, Any]] = None
    # parent_base_id: Optional[uuid.UUID] = None

class PromptTemplateRead(PromptTemplateBase):
    """Schema for reading a PromptTemplate."""
    id: uuid.UUID
    owner_org_id: Optional[uuid.UUID] = None
    is_system_entity: bool
    # parent_base_id: Optional[uuid.UUID] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --- SchemaTemplate Schemas --- #

class SchemaTemplateBase(BaseModel):
    """Base schema for SchemaTemplate."""
    name: str = Field(..., description="Name of the schema template (unique within org/system context)")
    version: str = Field(..., description="Version string")
    description: Optional[str] = Field(None, description="Description of the schema template")
    schema_definition: Optional[Dict[str, Any]] = Field(None, description="The JSON schema definition")
    schema_type: SchemaType = Field(default=SchemaType.JSON_SCHEMA, description="Type of schema")
    is_public: bool = Field(default=False, description="Whether the schema is public (system templates only)")

class SchemaTemplateCreate(SchemaTemplateBase):
    """Schema for creating an organization-specific SchemaTemplate."""
    is_system_entity: Optional[bool] = Field(default=False, description="Indicates if this schema template is a system entity. Only system admins can create system templates.")

class SchemaTemplateUpdate(BaseModel):
    """Schema for updating an organization-specific SchemaTemplate."""
    name: Optional[str] = None
    version: Optional[str] = None
    description: Optional[str] = None
    schema_definition: Optional[Dict[str, Any]] = None
    schema_type: Optional[SchemaType] = None

class SchemaTemplateRead(SchemaTemplateBase):
    """Schema for reading a SchemaTemplate."""
    id: uuid.UUID
    owner_org_id: Optional[uuid.UUID] = None
    is_system_entity: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# --- UserNotification Schemas --- #

class UserNotificationBase(BaseModel):
    """Base schema for user notifications."""
    notification_type: NotificationType
    message: Dict[str, Any] = Field(..., description="Content of the notification")
    related_run_id: Optional[uuid.UUID] = None

class UserNotificationCreate(UserNotificationBase):
    """Schema used internally to create a notification."""
    user_id: uuid.UUID
    org_id: Optional[uuid.UUID]

class UserNotificationRead(UserNotificationBase):
    """Schema for reading a user notification."""
    id: uuid.UUID
    user_id: uuid.UUID
    org_id: Optional[uuid.UUID]
    is_read: bool
    created_at: datetime
    read_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# --- HITLJob Schemas --- #

class HITLJobBase(BaseModel):
    """Base schema for HITL jobs."""
    assigned_user_id: Optional[uuid.UUID] = None
    request_details: Dict[str, Any] = Field(..., description="Information presented to the user")
    response_schema: Optional[Dict[str, Any]] = Field(None, description="Optional expected response schema")
    expires_at: Optional[datetime] = None

class HITLJobCreate(HITLJobBase):
    """Schema used internally to create an HITL job (triggered by a workflow)."""
    requesting_run_id: Optional[uuid.UUID]
    org_id: Optional[uuid.UUID]
    status: HITLJobStatus = Field(default=HITLJobStatus.PENDING)

class HITLJobRespond(BaseModel):
    """Schema for a user responding to an HITL job."""
    response_data: Dict[str, Any] = Field(..., description="The user's response data")

class HITLJobRead(HITLJobBase):
    """Schema for reading an HITL job."""
    id: uuid.UUID
    requesting_run_id: Optional[uuid.UUID]
    org_id: Optional[uuid.UUID]
    status: HITLJobStatus
    response_data: Optional[Dict[str, Any]] = None
    created_at: datetime
    responded_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)

class HITLJobCancel(BaseModel):
    """Schema for cancelling an HITL job (no body needed, ID from path)."""
    pass


# --- List Query Parameter Schemas --- #


class CommonListQuery(BaseModel):
    """Base query parameters for list endpoints."""
    skip: int = Field(0, ge=0, description="Number of items to skip")
    limit: int = Field(100, ge=1, le=200, description="Maximum number of items to return")


class WorkflowListQuery(CommonListQuery):
    """Query parameters for listing workflows."""
    owner_org_id: Optional[uuid.UUID] = Field(None, description="Filter by owning organization ID (Superuser only)")
    only_public: Optional[bool] = Field(True, description="Include public workflows in the results")
    only_system_entities: bool = Field(True, description="Include system entities (superuser only)")
    return_graphs_schemas: Optional[bool] = Field(True, description="Return graphs schemas with the results")
    workflow_name_search: Optional[str] = Field(None, description="Search workflows by name (case-insensitive partial match)")


class WorkflowRunListQuery(CommonListQuery):
    """Query parameters for listing workflow runs."""
    workflow_id: Optional[uuid.UUID] = Field(None, description="Filter runs by workflow ID")
    status: Optional[WorkflowRunStatus] = Field(None, description="Filter runs by status")
    triggered_by_user_id: Optional[uuid.UUID] = Field(None, description="Filter runs by triggering user ID (Superuser only)")
    owner_org_id: Optional[uuid.UUID] = Field(None, description="Filter by owning organization ID (Superuser only)")
    workflow_name: Optional[str] = Field(None, description="Name of the workflow this run belongs to")
    tag: Optional[str] = Field(None, description="Filter runs by experiment tag")
    parent_run_id: Optional[uuid.UUID] = Field(None, description="Filter runs by parent run ID to get child runs")

class HITLJobListQuery(CommonListQuery):
    """Query parameters for listing HITL jobs."""
    run_id: Optional[uuid.UUID] = Field(None, description="Filter by requesting run ID")
    assigned_user_id: Optional[uuid.UUID] = Field(None, description="Filter by assigned user ID ('me' for current user, uuid.UUID for specific user)")
    status: Optional[HITLJobStatus] = Field(None, description="Filter by job status")
    pending_only: Optional[bool] = Field(None, description="If true, only returns PENDING jobs")
    exclude_cancelled: Optional[bool] = Field(True, description="If true (default), excludes CANCELLED jobs")
    owner_org_id: Optional[uuid.UUID] = Field(None, description="Filter by owning organization ID (Superuser only)")


class NotificationListQuery(CommonListQuery):
    """Query parameters for listing notifications."""
    is_read: Optional[bool] = Field(None, description="Filter by read status (true=read, false=unread, null=all)")
    sort_by: str = Field("created_at", description="Field to sort by")
    sort_order: str = Field("desc", description="Sort order ('asc' or 'desc')")
    get_notifications_for_all_user_orgs: Optional[bool] = Field(False, description="If true, get notifications for all user organizations")


class NodeTemplateListQuery(CommonListQuery):
    """Query parameters for listing node templates."""
    launch_status: Optional[List[LaunchStatus]] = Field(
        default=[LaunchStatus.STAGING, LaunchStatus.PRODUCTION], 
        description="Filter by one or more launch statuses"
    )


class PromptTemplateListQuery(CommonListQuery):
    """Query parameters for listing prompt templates."""
    owner_org_id: Optional[uuid.UUID] = Field(None, description="Filter by owning organization ID (Superuser only)")
    include_system: bool = Field(False, description="Include system templates.")


class SchemaTemplateListQuery(CommonListQuery):
    """Query parameters for listing schema templates."""
    owner_org_id: Optional[uuid.UUID] = Field(None, description="Filter by owning organization ID (Superuser only)")
    include_system: bool = Field(False, description="Include system templates.")


class CustomerDataSortBy(str, Enum):
    """Enum for sorting customer data results."""
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"


class SearchSortBy(str, Enum):
    """Enum for sorting search results."""
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"
    SELF_OWNED_FIRST = "self_owned_first"


class SortOrder(str, Enum):
    """Enum for sorting order."""
    ASC = "asc"
    DESC = "desc"


class BaseSearchQuery(BaseModel):
    """Base query parameters for searching workflows, prompt templates, and schema templates."""
    name: str = Field(..., description="Name of the Entity to search for")
    include_public: bool = Field(False, description="Include public entities in the results")
    include_system_entities: bool = Field(False, description="Include system entities (superuser only)")
    include_public_system_entities: bool = Field(True, description="Include public system entities")
    sort_by: SearchSortBy = Field(SearchSortBy.CREATED_AT, description="Field to sort by")
    sort_order: SortOrder = Field(SortOrder.DESC, description="Sort order ('asc' or 'desc'). Note: SELF_OWNED_FIRST sort order is always OWNED entities first.")


class WorkflowSearchQuery(BaseSearchQuery):
    """Query parameters for searching workflows by name and version."""
    version_tag: Optional[str] = Field(None, description="Optional version tag to filter by")


class PromptTemplateSearchQuery(BaseSearchQuery):
    """Query parameters for searching prompt templates by name and version."""
    version: Optional[str] = Field(None, description="Optional version to filter by")


class SchemaTemplateSearchQuery(BaseSearchQuery):
    """Query parameters for searching schema templates by name and version."""
    version: Optional[str] = Field(None, description="Optional version to filter by")


# --- Customer Data Schemas ---

class CustomerDataVersionedInitialize(BaseModel):
    """
    Schema for initializing a customer data document.
    
    When is_system_entity=True, document is stored in system paths instead of 
    organization-specific paths. The is_shared parameter still applies normally 
    to determine if it's shared with the organization or user-specific.
    
    The on_behalf_of_user_id parameter only works with user-specific documents 
    and is ignored for system entities or shared documents.
    """
    is_shared: bool = Field(False, description="Set to true to create a document shared within the organization, false for a user-specific document.")
    initial_version: Optional[str] = Field(default="default", description="The initial version name for the document. Defaults to 'default'.")
    schema_template_name: Optional[str] = Field(None, description="Optional name of a SchemaTemplate to enforce.")
    schema_template_version: Optional[str] = Field(None, description="Optional version of the SchemaTemplate. Defaults to latest if name is provided but version is not.")
    initial_data: Optional[Any] = Field({}, description="Optional initial data for the document. Defaults to an empty object.")
    is_complete: Optional[bool] = Field(False, description="Mark the initial data as complete for validation.")
    is_system_entity: bool = Field(False, description="Whether this is a system entity (superusers only). When True, data is stored in system paths instead of organization-specific paths. The is_shared parameter still applies normally to determine if it's shared with the org or user-specific.")
    on_behalf_of_user_id: Optional[uuid.UUID] = Field(None, description="Optional user ID to act on behalf of (superusers only). Note: This parameter only works with user-specific documents and is ignored for system entities or shared documents.")


class CustomerDataVersionedUpdate(BaseModel):
    """
    Schema for updating customer data.
    
    When is_system_entity=True, document is updated in system paths instead of 
    organization-specific paths. The is_shared parameter still applies normally 
    to determine if it's shared with the organization or user-specific.
    
    The on_behalf_of_user_id parameter only works with user-specific documents 
    and is ignored for system entities or shared documents.
    """
    is_shared: bool = Field(False, description="Set to true to update a shared document, false for a user-specific document.")
    data: Any = Field(..., description="The data to update the document with. Can be a partial update for JSON objects or a full replacement for primitive types.")
    version: Optional[str] = Field(None, description="Specific version to update. If None, updates the active version.")
    is_complete: Optional[bool] = Field(None, description="Mark the document as complete after this update (for validation). Leave as None to keep current status.")
    schema_template_name: Optional[str] = Field(None, description="Optional: Name of a SchemaTemplate to update the document's schema with before applying the data update.")
    schema_template_version: Optional[str] = Field(None, description="Optional: Version of the SchemaTemplate. Defaults to latest if name is provided.")
    is_system_entity: bool = Field(False, description="Whether this is a system entity (superusers only). When True, data is accessed from system paths instead of organization-specific paths. The is_shared parameter still applies normally to determine if it's shared with the org or user-specific.")
    on_behalf_of_user_id: Optional[uuid.UUID] = Field(None, description="Optional user ID to act on behalf of (superusers only). Note: This parameter only works with user-specific documents and is ignored for system entities or shared documents.")
    create_only_fields: List[str] = Field(default_factory=list, description="List of fields in data which should be removed since this operation is an update rather than creation")
    keep_create_fields_if_missing: bool = Field(default=False, description="If True, keep create_only_fields in data if they don't exist in `existing object` during this update operation")


class CustomerDataRead(BaseModel):
    """Schema for reading customer data. Returns the data itself."""
    data: Any = Field(..., description="The customer data.")

    model_config = ConfigDict(from_attributes=True) # Allow creation from ORM/dict


class CustomerDataVersionInfo(BaseModel):
    """Schema representing metadata about a specific version."""
    version: str
    is_active: bool
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    is_complete: bool
    edit_count: int

    model_config = ConfigDict(from_attributes=True)


class CustomerDataVersionHistoryItem(BaseModel):
    """Schema representing an item in the version history."""
    timestamp: datetime
    sequence: int
    patch: str = Field(..., description="JSON Patch string representing the change.")
    is_primitive: bool = Field(..., description="Indicates if the change was a replacement of a primitive type.")

    model_config = ConfigDict(from_attributes=True)


class CustomerDataCreateVersion(BaseModel):
    """
    Schema for creating a new version (branching).
    
    When is_system_entity=True, document versions are created in system paths instead of 
    organization-specific paths. The is_shared parameter still applies normally 
    to determine if it's shared with the organization or user-specific.
    
    The on_behalf_of_user_id parameter only works with user-specific documents 
    and is ignored for system entities or shared documents.
    """
    is_shared: bool = Field(False, description="Set to true to create a new version for a shared document, false for a user-specific document.")
    new_version: str = Field(..., description="The name for the new version.")
    from_version: Optional[str] = Field(None, description="The version to branch from. If None, branches from the active version.")
    is_system_entity: bool = Field(False, description="Whether this is a system entity (superusers only). When True, data is accessed from system paths instead of organization-specific paths. The is_shared parameter still applies normally to determine if it's shared with the org or user-specific.")
    on_behalf_of_user_id: Optional[uuid.UUID] = Field(None, description="Optional user ID to act on behalf of (superusers only). Note: This parameter only works with user-specific documents and is ignored for system entities or shared documents.")


class CustomerDataSetActiveVersion(BaseModel):
    """
    Schema for setting the active version.
    
    When is_system_entity=True, document versions are accessed in system paths instead of 
    organization-specific paths. The is_shared parameter still applies normally 
    to determine if it's shared with the organization or user-specific.
    
    The on_behalf_of_user_id parameter only works with user-specific documents 
    and is ignored for system entities or shared documents.
    """
    is_shared: bool = Field(False, description="Set to true to set the active version for a shared document, false for a user-specific document.")
    version: str = Field(..., description="The version name to set as active.")
    is_system_entity: bool = Field(False, description="Whether this is a system entity (superusers only). When True, data is accessed from system paths instead of organization-specific paths. The is_shared parameter still applies normally to determine if it's shared with the org or user-specific.")
    on_behalf_of_user_id: Optional[uuid.UUID] = Field(None, description="Optional user ID to act on behalf of (superusers only). Note: This parameter only works with user-specific documents and is ignored for system entities or shared documents.")


class CustomerDataVersionedRestore(BaseModel):
    """
    Schema for restoring a document to a previous state.
    
    When is_system_entity=True, document is restored in system paths instead of 
    organization-specific paths. The is_shared parameter still applies normally 
    to determine if it's shared with the organization or user-specific.
    
    The on_behalf_of_user_id parameter only works with user-specific documents 
    and is ignored for system entities or shared documents.
    """
    is_shared: bool = Field(False, description="Set to true to restore a shared document, false for a user-specific document.")
    sequence: int = Field(..., ge=0, description="The sequence number to restore to.")
    version: Optional[str] = Field(None, description="The version to restore within. If None, uses the active version.")
    is_system_entity: bool = Field(False, description="Whether this is a system entity (superusers only). When True, data is accessed from system paths instead of organization-specific paths. The is_shared parameter still applies normally to determine if it's shared with the org or user-specific.")
    on_behalf_of_user_id: Optional[uuid.UUID] = Field(None, description="Optional user ID to act on behalf of (superusers only). Note: This parameter only works with user-specific documents and is ignored for system entities or shared documents.")


class CustomerDataSchemaUpdate(BaseModel):
    """
    Schema for explicitly updating the schema of a document.
    
    When is_system_entity=True, document schema is updated in system paths instead of 
    organization-specific paths. The is_shared parameter still applies normally 
    to determine if it's shared with the organization or user-specific.
    
    The on_behalf_of_user_id parameter only works with user-specific documents 
    and is ignored for system entities or shared documents.
    """
    is_shared: bool = Field(False, description="Set to true to update a shared document, false for a user-specific document.")
    schema_template_name: str = Field(..., description="Name of the SchemaTemplate to apply.")
    schema_template_version: Optional[str] = Field(None, description="Optional version of the SchemaTemplate. Defaults to latest if name is provided but version is not.")
    is_system_entity: bool = Field(False, description="Whether this is a system entity (superusers only). When True, data is accessed from system paths instead of organization-specific paths. The is_shared parameter still applies normally to determine if it's shared with the org or user-specific.")
    on_behalf_of_user_id: Optional[uuid.UUID] = Field(None, description="Optional user ID to act on behalf of (superusers only). Note: This parameter only works with user-specific documents and is ignored for system entities or shared documents.")


# --- Unversioned Customer Data Schemas ---

class CustomerDataUnversionedCreateUpdate(BaseModel):
    """
    Schema for creating or updating unversioned customer data.
    
    When is_system_entity=True, document is stored in system paths instead of 
    organization-specific paths. The is_shared parameter still applies normally 
    to determine if it's shared with the organization or user-specific.
    
    The on_behalf_of_user_id parameter only works with user-specific documents 
    and is ignored for system entities or shared documents.
    """
    is_shared: bool = Field(False, description="Set to true to create/update a shared document, false for user-specific.")
    data: Any = Field(..., description="The data for the document.")
    schema_template_name: Optional[str] = Field(None, description="Optional name of a SchemaTemplate to validate against.")
    schema_template_version: Optional[str] = Field(None, description="Optional version of the SchemaTemplate. Defaults to latest if name is provided.")
    is_system_entity: bool = Field(False, description="Whether this is a system entity (superusers only). When True, data is stored in system paths instead of organization-specific paths. The is_shared parameter still applies normally to determine if it's shared with the org or user-specific.")
    on_behalf_of_user_id: Optional[uuid.UUID] = Field(None, description="Optional user ID to act on behalf of (superusers only). Note: This parameter only works with user-specific documents and is ignored for system entities or shared documents.")
    create_only_fields: List[str] = Field(default_factory=list, description="List of fields in data which should be removed if the operation is an update rather than creation")
    keep_create_fields_if_missing: bool = Field(default=False, description="If True, keep create_only_fields in data if they don't exist in `existing object` during update")


class CustomerDataUnversionedRead(BaseModel):
    """Schema for reading unversioned customer data."""
    data: Any = Field(..., description="The unversioned customer data.")
    # Include metadata if needed, e.g., last updated timestamp
    # updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# --- Document Listing Schema ---

class CustomerDocumentMetadata(BaseModel):
    """
    Schema representing metadata about a customer document (versioned or unversioned).
    
    For system entities (is_system_entity=True), the document is stored in system paths
    instead of organization-specific paths. The is_shared flag indicates whether the document
    is shared within the organization or specific to a user.
    """
    versionless_path: Optional[str] = None
    id: Optional[str] = None
    org_id: Optional[uuid.UUID] = None
    # scope: str # e.g., 'shared', 'user' - Replaced by user_id + is_shared logic
    user_id_or_shared_placeholder: Optional[str] = Field(None, description="The user ID or '_shared_' placeholder.")
    namespace: str
    docname: str
    is_versioned: bool = Field(..., description="Indicates if this corresponds to a versioned document entry")
    is_shared: bool = Field(..., description="Indicates if this is a shared document path accessible by all users in the organization.")
    is_system_entity: bool = Field(False, description="Whether this is a system entity. When True, document is stored in system paths instead of organization-specific paths. The is_shared flag still determines if it's shared within the organization or user-specific.")
    version: Optional[str] = Field(None, description="The version of the document if it is versioned and this document is not just versioning metadata.")
    is_active_version: Optional[bool] = Field(None, description="Indicates if this is the active version of the document.")
    # active_version: Optional[str] = Field(None, description="The active version name of the document if it is versioned. Only available if this document is the versioning metadata and not the actual document data.")
    # Add other relevant metadata like updated_at if available from the base client document
    # updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class CustomerDocumentSearchResultMetadata(CustomerDocumentMetadata):
    """
    Schema representing metadata about a customer document (versioned or unversioned).
    This is the metadata for the document that is returned from the search results.
    
    For system entities (is_system_entity=True), the document is stored in system paths
    instead of organization-specific paths. The is_shared flag indicates whether the document
    is shared within the organization or specific to a user.
    """
    is_versioning_metadata: bool = Field(..., description="Indicates if this is the versioning metadata document for a versioned document.")

    model_config = ConfigDict(from_attributes=True)


class CustomerDocumentSearchResult(BaseModel):
    """
    Schema for the result of a customer document search.
    """
    metadata: CustomerDocumentSearchResultMetadata = Field(..., description="The metadata for the document.")
    document_contents: Any = Field(..., description="The data for the document.")

    model_config = ConfigDict(from_attributes=True)


class CustomerDataVersionedUpsert(BaseModel):
    """
    Schema for upserting a versioned customer data document.
    Updates the document if it exists, otherwise initializes it.
    Can target a specific version or the active version.
    If updating a specific version that doesn't exist, it attempts to create it first.
    """
    is_shared: bool = Field(False, description="Target a shared document (true) or user-specific document (false).")
    data: Any = Field(..., description="The data to upsert into the document.")
    version: Optional[str] = Field(None, description="Specific version to target. If None, targets the active version for updates, or uses 'default' for initialization.")
    from_version: Optional[str] = Field(None, description="If 'version' is specified and doesn't exist, use this version to branch from when creating it. Defaults to active version if None.")
    is_complete: Optional[bool] = Field(None, description="Mark the document state as complete/incomplete after the operation. Applies to both updates and initializations. If None during update, keeps current status; if None during init, defaults to False.")
    schema_template_name: Optional[str] = Field(None, description="Optional name of a SchemaTemplate to enforce/apply.")
    schema_template_version: Optional[str] = Field(None, description="Optional version of the SchemaTemplate. Defaults to latest.")
    # schema_definition: Optional[Dict[str, Any]] = Field(None, description="Optional explicit schema definition (takes precedence over template). NOTE: Not typically exposed directly in API for security/simplicity, prefer templates.")
    is_system_entity: bool = Field(False, description="Target a system entity (superusers only).")
    on_behalf_of_user_id: Optional[uuid.UUID] = Field(None, description="Act on behalf of another user (superusers only, requires is_shared=False).")
    set_active_version: bool = Field(True, description="Set the active version after the operation.")
    create_only_fields: List[str] = Field(default_factory=list, description="List of fields in data which should be removed if the operation is an update rather than creation")
    keep_create_fields_if_missing: bool = Field(default=False, description="If True, keep create_only_fields in data if they don't exist in `existing object` during update")


class CustomerDocumentIdentifier(BaseModel):
    """
    Identifies a specific customer document, potentially including version info.
    Used to reconstruct the path or retrieve the document after an operation.
    """
    doc_path_segments: Dict[str, str] = Field(..., description="Core path segments (org/system, user/shared, namespace, docname).")
    operation_params: Dict[str, Any] = Field(..., description="Parameters used in the operation affecting the document (org_id, is_shared, etc.).")
    version: Optional[str] = Field(None, description="The specific version affected or created (can be 'active' or None).")


class CustomerDataVersionedUpsertResponse(BaseModel):
    """
    Response schema for the versioned document upsert operation.
    """
    operation_performed: str = Field(..., description="String indicating the action taken (e.g., 'updated_active', 'initialized_version_default').")
    document_identifier: CustomerDocumentIdentifier = Field(..., description="Identifier for the document that was upserted.")

    model_config = ConfigDict(from_attributes=True)


# --- File Upload Schemas ---

class FileUploadModeEnum(str, Enum):
    """Enum defining the modes for handling file uploads."""
    create = "create"
    upsert = "upsert"


class FileUploadVersionedConfig(BaseModel):
    """
    Configuration specific to versioned file uploads.
    These fields are only relevant if `is_versioned` is True in the main config.
    """
    version: Optional[str] = Field(None, description="Specific version to target for upsert/create. If None during upsert, targets active version. If None during create, uses 'default'.")
    from_version: Optional[str] = Field(None, description="Version to branch from if creating a new version during an 'upsert' operation when the target 'version' doesn't exist. Defaults to active version if None.")
    is_complete: Optional[bool] = Field(None, description="Mark the document state as complete/incomplete after the operation. Applies to both upserts and creations. If None during update, keeps current status; if None during create, defaults to False.")

    model_config = ConfigDict(extra='forbid') # Forbid extra fields


class FileUploadConfig(BaseModel):
    """
    Configuration for processing a single uploaded file and storing it as customer data.
    
    Note: Individual files cannot exceed 16MB due to MongoDB document size limitations.
    """
    namespace: str = Field("uploaded_files", description="Namespace where the document should be stored. Default is 'uploaded_files'.")
    docname: Optional[str] = Field(None, description="Document name. If None, it will be inferred from the uploaded filename.")
    description: Optional[str] = Field(None, description="Optional description for the uploaded file.")
    is_shared: bool = Field(False, description="Set to true to store as a document shared within the organization, false for a user-specific document.")
    is_system_entity: bool = Field(False, description="Whether this is a system entity (superusers only).")
    on_behalf_of_user_id: Optional[uuid.UUID] = Field(None, description="Act on behalf of another user (superusers only, requires is_shared=False).")

    mode: FileUploadModeEnum = Field(FileUploadModeEnum.upsert, description="Upload mode: 'create' (fail if exists) or 'upsert' (create or update).")
    is_versioned: bool = Field(False, description="Whether to store the file content as a versioned document.")
    save_as_raw: bool = Field(False, description="Whether to save the file as raw bytes instead of converting to markdown. Uses BSON Binary format for MongoDB storage.")

    versioned_config: Optional[FileUploadVersionedConfig] = Field(None, description="Configuration for versioned uploads. Required if is_versioned is True.")

    @model_validator(mode='after')
    def check_versioned_config(self) -> 'FileUploadConfig':
        """Validate that versioned_config is provided if is_versioned is True."""
        if self.is_versioned and self.versioned_config is None:
            raise ValueError("`versioned_config` must be provided when `is_versioned` is True.")
        if not self.is_versioned and self.versioned_config is not None:
            # Optionally, you could clear it or raise an error. Clearing seems more robust.
            # raise ValueError("`versioned_config` should not be provided when `is_versioned` is False.")
            self.versioned_config = None # Clear it if provided unnecessarily
        return self

    model_config = ConfigDict(extra='forbid') # Forbid extra fields


# Optional: If you want to allow passing global defaults and per-file overrides in one request
class FileUploadRequestPayload(BaseModel):
    """
    Optional schema to allow specifying global defaults and per-file overrides
    in a single API request payload, potentially alongside the file uploads
    using multipart/form-data encoding (requires careful handling in the endpoint).

    Eg Request Upload Payload:
    {
        "config_payload": {
            "global_defaults": {
            "namespace": "blog_uploaded_xyz"
            }
        }
    }

     (In admin portal): 
    {
        "global_defaults": {
            "namespace": "blog_uploaded_xyz"
        }
    }

    {
        "config_payload": {
            "global_defaults": {
                "description": "This is a description for the uploaded file",
                "save_as_raw": true
            }
        }
    }

    (In admin portal): 
    {
        "global_defaults": {
            "description": "Eg CSV",
            "save_as_raw": true
        }
    }
    """
    global_defaults: Optional[FileUploadConfig] = Field(default_factory=FileUploadConfig, description="Global default configuration to apply to all files unless overridden.")
    file_configs: Optional[Dict[str, FileUploadConfig]] = Field(None, description="Dictionary mapping original filenames to their specific configurations, overriding global defaults.")

    model_config = ConfigDict(extra='forbid')


class FileUploadValidationRequest(BaseModel):
    """
    Schema for the request of the file upload configuration validation endpoint.
    """
    payload: FileUploadRequestPayload = Field(default_factory=FileUploadRequestPayload, description="The payload containing global defaults and per-file configurations.")
    files: List[str] = Field(..., description="The files to upload.")


class FileUploadValidationResult(BaseModel):
    """
    Schema for the response of the file upload configuration validation endpoint.
    Indicates overall validity and lists any global or per-file errors found.
    """
    is_valid: bool = Field(..., description="True if all configurations and checks passed, False otherwise.")
    global_errors: List[str] = Field(default_factory=list, description="List of validation errors not specific to a single file (e.g., payload parsing). Empty if none.")
    file_errors: Dict[str, List[str]] = Field(default_factory=dict, description="Dictionary mapping filenames to a list of their specific validation errors. Empty if none.")

    model_config = ConfigDict(extra='forbid')


class CustomerDataSearchQuery(BaseModel):
    """Schema for searching customer documents."""
    namespace_filter: Optional[str] = Field(None, description="Filter by namespace (e.g., 'invoices', 'user_profiles'). Supports '*' as a wildcard for docname if namespace is specific.")
    text_search_query: Optional[str] = Field(None, description="Optional text search query to match against document content.")
    value_filter: Optional[Dict[str, Any]] = Field(None, description="Optional dictionary of field-value pairs to filter documents by specific data content.")
    include_shared: bool = Field(True, description="Include documents shared within the organization.")
    include_user_specific: bool = Field(True, description="Include documents specific to the calling user (or on_behalf_of_user_id).")
    skip: int = Field(0, ge=0, description="Number of documents to skip for pagination.")
    limit: int = Field(100, ge=1, le=200, description="Maximum number of documents to return.") # Max limit can be adjusted
    on_behalf_of_user_id: Optional[uuid.UUID] = Field(None, description="Optional user ID to search on behalf of (requires superuser privileges).")
    include_system_entities: bool = Field(False, description="Include system-level documents in the search (requires superuser privileges for non-shared system docs).")
    sort_by: Optional[CustomerDataSortBy] = Field(None, description="Field to sort results by (e.g., 'created_at', 'updated_at').")
    sort_order: Optional[SortOrder] = Field(SortOrder.DESC, description="Order to sort results (ASC or DESC).")

    model_config = ConfigDict(from_attributes=True)


class CustomerDataDeleteByPattern(BaseModel):
    """
    Schema for deleting customer data documents by pattern.
    
    When is_system_entity=True, documents are deleted from system paths instead of 
    organization-specific paths. The is_shared parameter still applies normally 
    to determine if it's shared with the organization or user-specific.
    
    The on_behalf_of_user_id parameter only works with user-specific documents 
    and is ignored for system entities or shared documents.
    """
    is_shared: bool = Field(False, description="Set to true to delete shared documents, false for user-specific documents.")
    namespace: str = Field(..., description="Namespace pattern for the documents to delete. Supports '*' as wildcard.")
    docname: str = Field(..., description="Document name pattern for the documents to delete. Supports '*' as wildcard.")
    is_system_entity: bool = Field(False, description="Whether to delete system entities (superusers only).")
    on_behalf_of_user_id: Optional[uuid.UUID] = Field(None, description="Optional user ID to act on behalf of (superusers only).")
    dry_run: bool = Field(False, description="If true, only returns the count of objects that would be deleted without actually deleting them.")
    
    model_config = ConfigDict(extra='forbid')  # Forbid extra fields
    
class CustomerDataDeleteResponse(BaseModel):
    """Response schema for delete operations that return a count."""
    deleted_count: int = Field(..., description="Number of documents deleted")
    dry_run: bool = Field(False, description="Whether this was a dry run (no actual deletion)")
    
    model_config = ConfigDict(from_attributes=True)

class WorkflowConfigOverrideBase(BaseModel):
    """Base schema for WorkflowConfigOverride."""
    # Either workflow_id OR workflow_name must be provided (not both)
    # If workflow_name is used, workflow_version is optional
    workflow_id: Optional[uuid.UUID] = Field(None, description="Reference to the workflow being overridden")
    workflow_name: Optional[str] = Field(None, description="Name of the workflow being overridden (alternative to workflow_id)")
    workflow_version: Optional[str] = Field(None, description="Version of the workflow to override (if null, applies to all versions)")
    
    # The override configuration
    override_graph_schema: Dict[str, Any] = Field(..., description="The graph schema override configuration")
    
    # At least one of is_system_entity, user_id, or org_id must be provided
    # If is_system_entity is True, user_id and org_id must be None
    is_system_entity: bool = Field(False, description="True if this is a system-wide override")
    user_id: Optional[uuid.UUID] = Field(None, description="User-specific override")
    org_id: Optional[uuid.UUID] = Field(None, description="Organization-specific override")
    
    is_active: bool = Field(True, description="Whether this override configuration is currently active")
    description: Optional[str] = Field(None, description="Description of what this override configuration does")
    tag: Optional[str] = Field(None, description="Optional tag to further categorize this override configuration")
    
    
    # @model_validator(mode='after')
    # def validate_scope_identifiers(self):
    #     """Ensure at least one scope identifier is provided with proper constraints."""
    #     is_system = self.is_system_entity
    #     user_id = self.user_id
    #     org_id = self.org_id
        
    #     # At least one scope identifier must be provided
    #     if not is_system and user_id is None and org_id is None:
    #         raise ValueError("At least one of is_system_entity, user_id, or org_id must be provided")
        
    #     # If is_system_entity is True, user_id and org_id must be None
    #     if is_system and (user_id is not None or org_id is not None):
    #         raise ValueError("If is_system_entity is True, user_id and org_id must be None")
        
    #     # Validate that workflow_version is only used with workflow_name
    #     if self.workflow_id is not None and self.workflow_version is not None:
    #         raise ValueError("workflow_version can only be provided when using workflow_name")
        
    #     return self

class WorkflowConfigOverrideCreate(WorkflowConfigOverrideBase):
    """Schema for creating a new WorkflowConfigOverride."""
    pass

class WorkflowConfigOverrideUpdate(BaseModel):
    """Schema for updating an existing WorkflowConfigOverride.

    Only certain fields can be updated (override_graph_schema, is_active, 
    description, tag). Core identifiers (workflow_id/name, scope) cannot be changed.
    """
    override_graph_schema: Optional[Dict[str, Any]] = Field(None, description="The updated graph schema override configuration")
    is_active: Optional[bool] = Field(None, description="Whether this override configuration is active")
    description: Optional[str] = Field(None, description="Description of what this override configuration does")
    tag: Optional[str] = Field(None, description="Optional tag to further categorize this override configuration")

class WorkflowConfigOverrideRead(WorkflowConfigOverrideBase):
    """Schema for reading a WorkflowConfigOverride."""
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class WorkflowSpecificOverrideListPayload(BaseModel):
    """Request body for listing workflow-specific configuration overrides with tag filtering."""
    include_tags: Optional[List[str]] = Field(None, description="Optional list of tags to filter by. Overrides matching ANY of these tags will be prioritized.")

class UserOverrideListQuery(CommonListQuery):
    """Query parameters for listing user-specific configuration overrides."""
    user_id: uuid.UUID = Field(None, description="The ID of the user whose overrides to list")
    is_active: Optional[bool] = Field(None, description="Filter by active status")
    
class OrgOverrideListQuery(CommonListQuery):
    """Query parameters for listing organization-specific configuration overrides."""
    org_id: uuid.UUID = Field(None, description="The ID of the organization whose overrides to list")
    is_active: Optional[bool] = Field(None, description="Filter by active status")
    
class SystemOverrideListQuery(CommonListQuery):
    """Query parameters for listing system-wide configuration overrides."""
    is_active: Optional[bool] = Field(None, description="Filter by active status")
    
class TagOverrideListQuery(CommonListQuery):
    """Query parameters for listing tag-specific configuration overrides."""
    tag: str = Field(..., description="The tag to filter by")
    is_active: Optional[bool] = Field(None, description="Filter by active status")

# --- Effective Workflow Config Schemas --- #

class WorkflowEffectiveConfigQuery(BaseModel):
    """Query parameters for retrieving effective workflow configuration."""
    include_active: bool = Field(True, description="Whether to include active overrides when calculating the effective config.")
    include_tags: Optional[List[str]] = Field(None, description="Optional list of override tags to specifically include.")

class WorkflowEffectiveConfigResponse(BaseModel):
    """Response schema for an effective workflow configuration."""
    applied_overrides: List[WorkflowConfigOverrideRead] = Field(..., description="List of workflow configuration overrides that were applied.")
    effective_graph_schema: GraphSchema = Field(..., description="The final effective graph schema after applying all relevant overrides.")

    model_config = ConfigDict(from_attributes=True)


# --- ChatThread Schemas --- #
class ChatThreadBase(BaseModel):
    """Base schema for ChatThread."""
    workflow_name: Optional[str] = Field(None, description="Name of the workflow this chat thread is associated with")
    workflow_version: Optional[str] = Field(None, description="Version of the workflow this chat thread is associated with")
    thread_name: Optional[str] = Field(None, description="Optional name/title for this chat thread")
    tag: Optional[str] = Field(None, description="Optional tag to categorize or filter chat threads")

class ChatThreadCreate(ChatThreadBase):
    """Schema for creating a new ChatThread."""
    user_id: Optional[uuid.UUID] = Field(None, description="User ID of the thread owner. Only superusers can specify this field. Non-superusers will always use their own user ID.")

class ChatThreadUpdate(BaseModel):
    """Schema for updating an existing ChatThread."""
    workflow_name: Optional[str] = None
    workflow_version: Optional[str] = None
    thread_name: Optional[str] = None
    tag: Optional[str] = None

class ChatThreadRead(ChatThreadBase):
    """Schema for reading a ChatThread."""
    id: uuid.UUID
    user_id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class ChatThreadListQuery(CommonListQuery):
    """Query parameters for listing chat threads."""
    workflow_name: Optional[str] = Field(None, description="Filter threads by workflow name")
    workflow_version: Optional[str] = Field(None, description="Filter threads by workflow version")
    user_id: Optional[uuid.UUID] = Field(None, description="Filter threads by owner (superuser only, others can only see their own threads)")
    tag: Optional[str] = Field(None, description="Filter threads by tag")

class DocumentOperationType(str, Enum):
    """Enum for document operation types."""
    MOVE = "move"
    COPY = "copy"


class DocumentOperation(BaseModel):
    """
    Schema for a single document operation (move or copy).
    """
    operation_type: DocumentOperationType = Field(..., description="Type of operation: 'move' or 'copy'")
    source_org_id: uuid.UUID = Field(..., description="Source organization ID")
    source_namespace: str = Field(..., description="Source document namespace")
    source_docname: str = Field(..., description="Source document name")
    source_is_shared: bool = Field(default=False, description="Whether source is a shared document")
    destination_org_id: uuid.UUID = Field(..., description="Destination organization ID")
    destination_namespace: str = Field(..., description="Destination document namespace")
    destination_docname: str = Field(..., description="Destination document name")
    destination_is_shared: bool = Field(default=False, description="Whether destination should be a shared document")
    source_is_system_entity: bool = Field(default=False, description="Whether source is a system entity")
    destination_is_system_entity: bool = Field(default=False, description="Whether destination should be a system entity")

    model_config = ConfigDict(from_attributes=True)


class DocumentOperationResult(BaseModel):
    """
    Schema for the result of a document operation (move or copy).
    """
    operation_type: DocumentOperationType = Field(..., description="Type of operation that was performed: 'move' or 'copy'")
    source_org_id: uuid.UUID = Field(..., description="Source organization ID")
    source_namespace: str = Field(..., description="Source document namespace")
    source_docname: str = Field(..., description="Source document name")
    destination_org_id: uuid.UUID = Field(..., description="Destination organization ID")
    destination_namespace: str = Field(..., description="Destination document namespace")
    destination_docname: str = Field(..., description="Destination document name")
    success: bool = Field(..., description="Whether the operation was successful")
    error_message: Optional[str] = Field(None, description="Error message if the operation failed")

    model_config = ConfigDict(from_attributes=True)


# --- Asset Schemas --- #

# --- Enums --- #

class AssetType(str, Enum):
    """Enum for asset types."""
    LINKEDIN_PROFILE = "linkedin_profile"
    BLOG_URL = "blog_url"


class AssetAppDataOperation(str, Enum):
    """Enum for asset app_data update operations."""
    ADD_OR_UPDATE = "add_or_update"
    DELETE = "delete"
    REPLACE = "replace"


# --- Asset App Data Schemas --- #

class Onboarding(BaseModel):
    isDocumentUpdateComplete: bool = False
    diagnosticsComplete: bool = False
    playbookComplete: bool = False
    contentCalendarCreatedAt: str = ""


class LinkedInProfileAppData(BaseModel):
    """Schema for LinkedIn Profile asset app_data."""
    profile_url: HttpUrl = Field(..., description="Full LinkedIn profile URL")
    # improve pattern for company too: "^https?://([a-z]{2,3}\\.)?linkedin\\.com/in/[a-zA-Z0-9_-]+/?$",
    onboarding: Onboarding = Field(default_factory=Onboarding)
    entity_name: Optional[str] = None
    # last_scraped: Optional[datetime] = Field(None, description="Last time the profile was scraped")
    # scrape_frequency: Optional[str] = Field("weekly", pattern="^(daily|weekly|monthly|manual)$", description="How often to scrape this profile")
    # extracted_data: Optional[Dict[str, Any]] = Field(None, description="Extracted profile data from last scrape")
    # monitoring_enabled: Optional[bool] = Field(True, description="Whether to monitor this profile for changes")
    
    model_config = ConfigDict(extra='allow')  # Allow additional fields for flexibility

    @field_validator('profile_url', mode='before')
    def validate_profile_url(cls, v):
        """Validate that the profile URL is a valid LinkedIn profile URL."""
        if (not v.startswith("https://www.linkedin.com/in/")) and (not v.startswith("https://www.linkedin.com/company/")):
            raise ValueError("Profile URL must start with 'https://www.linkedin.com/in/' or 'https://www.linkedin.com/company/'")
        return v


class BlogUrlAppData(BaseModel):
    """Schema for Blog URL asset app_data."""
    blog_url: HttpUrl = Field(..., description="Full URL of the blog or blog post")
    onboarding: Onboarding = Field(default_factory=Onboarding)
    company_name: Optional[str] = None
    # blog_type: Optional[str] = Field("unknown", pattern="^(wordpress|medium|substack|ghost|custom|unknown)$", description="Type of blog platform")
    # last_scraped: Optional[datetime] = Field(None, description="Last time the blog was scraped")
    # scrape_frequency: Optional[str] = Field("weekly", pattern="^(daily|weekly|monthly|manual)$", description="How often to scrape this blog")
    # extracted_content: Optional[Dict[str, Any]] = Field(None, description="Extracted blog content from last scrape")
    # rss_feed_url: Optional[str] = Field(None, description="RSS feed URL if available")
    # monitoring_enabled: Optional[bool] = Field(True, description="Whether to monitor this blog for new posts")
    
    model_config = ConfigDict(extra='allow')  # Allow additional fields for flexibility

class AssetBase(BaseModel):
    """Base schema for Asset."""
    asset_type: AssetType = Field(..., description="Type of the asset (linkedin_profile or blog_url)")
    asset_name: str = Field(..., description="Name of the asset (linkedin username for linkedin_profile, domain or domain+path for blog_url)")
    is_shared: bool = Field(True, description="Whether the asset is shared within the organization")
    is_active: bool = Field(True, description="Whether the asset is currently active")
    app_data: Optional[Dict[str, Any]] = Field(None, description="Application-specific data for the asset")


class AssetCreate(AssetBase):
    """Schema for creating a new Asset."""
    org_id: Optional[uuid.UUID] = Field(None, description="Organization ID (optional for regular users, uses current org)")
    managing_user_id: Optional[uuid.UUID] = Field(None, description="Managing user ID (optional, defaults to current user)")
    on_behalf_of_user_id: Optional[uuid.UUID] = Field(None, description="Act on behalf of another user (superusers only)")


class AssetUpdate(BaseModel):
    """Schema for updating an existing Asset. Allows partial updates."""
    asset_name: Optional[str] = Field(None, description="Name of the asset")
    is_shared: Optional[bool] = Field(None, description="Whether the asset is shared within the organization")
    is_active: Optional[bool] = Field(None, description="Whether the asset is currently active")
    app_data: Optional[Dict[str, Any]] = Field(None, description="Application-specific data for the asset")


class AssetAppDataUpdate(BaseModel):
    """Schema for updating asset app_data with specific operations."""
    operation: AssetAppDataOperation = Field(..., description="Operation to perform on app_data")
    path: Optional[List[str]] = Field(None, description="JSON path for add_or_update/delete operations (e.g., ['field1', 'subfield'])")
    value: Optional[Any] = Field(None, description="Value for add_or_update/replace operations")
    
    @model_validator(mode='after')
    def validate_operation(self):
        """Validate that required fields are present for each operation."""
        if self.operation == AssetAppDataOperation.ADD_OR_UPDATE and (self.path is None or self.value is None):
            raise ValueError(f"'path' and 'value' are required for '{self.operation.value}' operation")
        elif self.operation == AssetAppDataOperation.DELETE and self.path is None:
            raise ValueError("'path' is required for 'delete' operation")
        elif self.operation == AssetAppDataOperation.REPLACE and self.value is None:
            raise ValueError("'value' is required for 'replace' operation")
        return self


class AssetAppDataIncrement(BaseModel):
    """Schema for atomically incrementing a numeric field in asset app_data."""
    path: List[str] = Field(..., description="JSON path to the numeric field to increment (e.g., ['counter'])")
    increment: Union[int, float] = Field(1, description="Amount to increment by (default 1), can be int or float")
    
    @field_validator('path')
    @classmethod
    def path_not_empty(cls, v):
        """Ensure path is not empty."""
        if not v:
            raise ValueError("Path cannot be empty")
        return v


class AssetRead(AssetBase):
    """Schema for reading an Asset."""
    id: uuid.UUID
    org_id: uuid.UUID
    managing_user_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class AssetSortBy(str, Enum):
    """Enum for sorting asset results."""
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"
    ASSET_NAME = "asset_name"
    ASSET_TYPE = "asset_type"


class AssetListQuery(CommonListQuery):
    """Query parameters for listing assets."""
    asset_type: Optional[AssetType] = Field(None, description="Filter by asset type (linkedin_profile or blog_url)")
    is_active: Optional[bool] = Field(None, description="Filter by active status")
    is_shared: Optional[bool] = Field(None, description="Filter by shared status")
    managing_user_id: Optional[uuid.UUID] = Field(None, description="Filter by managing user (UUID or 'me')")
    org_id: Optional[uuid.UUID] = Field(None, description="Filter by organization (superuser only)")
    managed_only: bool = Field(False, description="Only show assets where current user is managing user")
    include_all_orgs: bool = Field(False, description="Include assets from all user's organizations (for managed_only=true)")
    app_data_fields: Optional[Union[str, List[str]]] = Field(None, description="Specific app_data fields to include in response (comma-separated string or list)")
    sort_by: AssetSortBy = Field(AssetSortBy.UPDATED_AT, description="Field to sort by")
    sort_order: SortOrder = Field(SortOrder.DESC, description="Sort order ('asc' or 'desc')")
    
    @field_validator('app_data_fields', mode='before')
    def parse_app_data_fields(cls, v):
        """Parse comma-separated string into list if needed."""
        if isinstance(v, str):
            return [field.strip() for field in v.split(',') if field.strip()]
        return v


class AssetReadQuery(BaseModel):
    """Query parameters for reading a single asset."""
    app_data_fields: Optional[Union[str, List[str]]] = Field(None, description="Specific app_data fields to include in response (comma-separated string or list)")
    
    @field_validator('app_data_fields', mode='before')
    def parse_app_data_fields(cls, v):
        """Parse comma-separated string into list if needed."""
        if isinstance(v, str):
            return [field.strip() for field in v.split(',') if field.strip()]
        return v


class AssetTypeInfo(BaseModel):
    """Information about an asset type and its schema."""
    asset_type: str = Field(..., description="Asset type identifier")
    display_name: str = Field(..., description="Human-readable name for the asset type")
    description: Optional[str] = Field(None, description="Description of the asset type")
    app_data_schema: Optional[Dict[str, Any]] = Field(None, description="JSON Schema for validating app_data")
    
    model_config = ConfigDict(from_attributes=True)


# --- UserAppResumeMetadata Schemas --- #

class UserAppResumeMetadataBase(BaseModel):
    """Base schema for UserAppResumeMetadata."""
    workflow_name: Optional[str] = Field(None, description="Associated workflow name")
    asset_id: Optional[uuid.UUID] = Field(None, description="Associated asset ID")
    entity_tag: Optional[str] = Field(None, description="Entity tag for grouping/filtering")
    frontend_stage: Optional[str] = Field(None, description="Frontend application stage/state")
    run_id: Optional[uuid.UUID] = Field(None, description="Associated workflow run ID")
    app_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for resuming application state")
    
    @model_validator(mode='after')
    def validate_identifiers(self):
        """Validate that at least one identifier and one data field are present."""
        identifiers = [self.workflow_name, self.asset_id, self.entity_tag, self.frontend_stage]
        data_fields = [self.run_id, self.app_metadata]
        
        if not any(field is not None for field in identifiers):
            raise ValueError("At least one of workflow_name, asset_id, entity_tag, or frontend_stage must be provided")
        
        if not any(field is not None for field in data_fields):
            raise ValueError("At least one of run_id or app_metadata must be provided")
        
        return self


class UserAppResumeMetadataCreate(UserAppResumeMetadataBase):
    """Schema for creating UserAppResumeMetadata."""
    org_id: Optional[uuid.UUID] = Field(None, description="Organization ID (optional for regular users, uses current org). Can be used by superusers to create metadata for any orgs.")
    on_behalf_of_user_id: Optional[uuid.UUID] = Field(None, description="Act on behalf of another user (superusers only)")


class UserAppResumeMetadataUpdate(BaseModel):
    """Schema for updating UserAppResumeMetadata. Allows partial updates."""
    workflow_name: Optional[str] = None
    asset_id: Optional[uuid.UUID] = None
    entity_tag: Optional[str] = None
    frontend_stage: Optional[str] = None
    run_id: Optional[uuid.UUID] = None
    app_metadata: Optional[Dict[str, Any]] = None
    
    @model_validator(mode='after')
    def validate_update(self):
        """Ensure update maintains the constraint requirements."""
        # For updates, we don't enforce the constraints since we're doing partial updates
        # The service layer should ensure the resulting record still meets constraints
        return self


class UserAppResumeMetadataRead(UserAppResumeMetadataBase):
    """Schema for reading UserAppResumeMetadata."""
    id: uuid.UUID
    org_id: uuid.UUID
    user_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class UserAppResumeMetadataListQuery(CommonListQuery):
    """Query parameters for listing UserAppResumeMetadata."""
    workflow_name: Optional[str] = Field(None, description="Filter by workflow name")
    asset_id: Optional[uuid.UUID] = Field(None, description="Filter by asset ID")
    entity_tag: Optional[str] = Field(None, description="Filter by entity tag")
    frontend_stage: Optional[str] = Field(None, description="Filter by frontend stage")
    run_id: Optional[uuid.UUID] = Field(None, description="Filter by run ID")
    user_id: Optional[uuid.UUID] = Field(None, description="Filter by user ID (superuser only)")
    org_id: Optional[uuid.UUID] = Field(None, description="Filter by organization (superuser only)")


# --- Bulk Delete Schemas --- #

class WorkflowRunBulkDeleteRequest(BaseModel):
    """
    Schema for bulk deleting workflow runs based on time criteria.
    Exactly one time parameter must be provided (seconds, minutes, hours, or days).
    """
    # Time filtering - exactly one must be provided
    last_n_seconds: Optional[int] = Field(None, ge=1, le=86400, description="Delete runs from last N seconds (1-86400)")
    last_n_minutes: Optional[int] = Field(None, ge=1, le=1440, description="Delete runs from last N minutes (1-1440)")  
    last_n_hours: Optional[int] = Field(None, ge=1, le=24, description="Delete runs from last N hours (1-24)")
    last_n_days: Optional[int] = Field(None, ge=1, le=30, description="Delete runs from last N days (1-30)")
    
    # Optional filtering
    workflow_name: Optional[str] = Field(None, description="Filter by workflow name before deletion")
    status: Optional[WorkflowRunStatus] = Field(None, description="Filter by workflow run status before deletion")
    
    # User filtering options
    delete_workflow_runs_for_all_users_in_org: bool = Field(True, description="If false, only delete runs for current user or on_behalf_of_user_id")
    on_behalf_of_user_id: Optional[uuid.UUID] = Field(None, description="Delete runs for specific user (superuser only, requires admin privileges if provided)")
    
    # Control options
    dry_run: bool = Field(False, description="If true, only returns what would be deleted without actually deleting")
    
    @model_validator(mode='after')
    def validate_time_parameters(self):
        """Ensure exactly one time parameter is provided."""
        time_params = [
            self.last_n_seconds,
            self.last_n_minutes, 
            self.last_n_hours,
            self.last_n_days
        ]
        provided_params = [p for p in time_params if p is not None]
        
        if len(provided_params) == 0:
            raise ValueError("Exactly one time parameter must be provided (last_n_seconds, last_n_minutes, last_n_hours, or last_n_days)")
        elif len(provided_params) > 1:
            raise ValueError("Only one time parameter can be provided at a time")
            
        return self

    model_config = ConfigDict(from_attributes=True)


class WorkflowRunDeleteInfo(BaseModel):
    """Information about a deleted workflow run."""
    run_id: uuid.UUID = Field(..., description="The ID of the deleted workflow run")
    workflow_name: Optional[str] = Field(None, description="Name of the workflow this run belonged to")
    status: WorkflowRunStatus = Field(..., description="Status of the deleted workflow run")
    parent_deleted: bool = Field(..., description="True if parent_run_id was also deleted or was null/not found")
    created_at: datetime = Field(..., description="When this run was originally created")
    parent_run_id: Optional[uuid.UUID] = Field(None, description="Parent run ID if it existed")
    triggered_by_user_id: Optional[uuid.UUID] = Field(None, description="User ID who triggered this run")
    inputs: Optional[Dict[str, Any]] = Field(None, description="Inputs that were provided to this workflow run")
    
    model_config = ConfigDict(from_attributes=True)


class WorkflowRunBulkDeleteResponse(BaseModel):
    """Response schema for bulk workflow run deletion."""
    deleted_count: int = Field(..., description="Number of workflow runs deleted")
    dry_run: bool = Field(..., description="Whether this was a dry run (no actual deletion)")
    time_filter_applied: str = Field(..., description="Description of the time filter that was applied")
    additional_filters: Dict[str, Any] = Field(default_factory=dict, description="Additional filters that were applied")
    deleted_runs: List[WorkflowRunDeleteInfo] = Field(default_factory=list, description="Details of deleted runs")
    
    model_config = ConfigDict(from_attributes=True)


# Legacy aliases for backward compatibility
DocumentMoveOperation = DocumentOperation
DocumentCopyOperation = DocumentOperation
DocumentMoveResult = DocumentOperationResult
DocumentCopyResult = DocumentOperationResult
