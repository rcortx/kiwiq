import asyncio
from collections import defaultdict
import json
import uuid
import traceback
from typing import List, Optional, Dict, Any, Annotated, Union, AsyncGenerator

from fastapi import (
    APIRouter, Depends, HTTPException, status, Query, WebSocket,
    WebSocketDisconnect, Body, Path, Response # Added Path, Response
)
import jsonschema
from pydantic import ValidationError as PydanticValidationError
from jsonschema import ValidationError
from jsonschema.validators import Draft202012Validator
from sqlalchemy.ext.asyncio import AsyncSession
import jwt # For encoding/decoding tokens (requires python-jose)
from datetime import datetime, timedelta

# --- Core Dependencies ---
from db.session import get_async_session, get_async_db_dependency
from global_utils import datetime_now_utc
from kiwi_app.auth import crud as auth_crud, dependencies as auth_deps
from kiwi_app.settings import settings

# --- Auth Dependencies ---
from kiwi_app.auth.models import User
from kiwi_app.auth.dependencies import (
    get_current_active_verified_user,
    get_active_org_id,
    get_current_active_superuser,
)
from kiwi_app.workflow_app import crud as wf_crud
# Import function to fetch user from auth crud (used in websocket auth)
# from kiwi_app.auth.crud import user as user_crud

# --- Workflow App Dependencies ---
# Import local schemas, services, models, constants, exceptions
from kiwi_app.workflow_app import schemas, services, models, dependencies as wf_deps, exceptions, constants

# --- Event Schemas ---
# Import event schemas for stream response type
from workflow_service.registry import registry
from workflow_service.services import events as event_schemas
# TODO: FIX: circular imports!
# from workflow_service.graph.graph import GraphSchema

# from kiwi_app.workflow_app.websockets import websocket_router
# from kiwi_app.workflow_app.utils import workflow_logger
from kiwi_app.utils import get_kiwi_logger

# Import the updated validation function
from kiwi_app.workflow_app.workflow_config_override import _validate_updated_graph_schema

workflow_logger = get_kiwi_logger(name="kiwi_app.workflow")

# --- Permission Checker Dependencies (Imported from wf_deps correctly now) ---
# Example: wf_deps.RequireWorkflowRead, wf_deps.RequireWorkflowCreate, etc.

# --- Setup Routers ---
# Group routes logically using tags
workflow_router = APIRouter(prefix="/workflows", tags=["Workflows"])
run_router = APIRouter(prefix="/runs", tags=["Workflow Runs"])
template_router = APIRouter(prefix="/templates", tags=["Templates"])
notification_router = APIRouter(prefix="/notifications", tags=["User Notifications"])
hitl_router = APIRouter(prefix="/hitl", tags=["HITL Jobs"])
workflow_config_override_router = APIRouter(prefix="/workflow-config-overrides", tags=["Workflow Config Overrides"])
chat_thread_router = APIRouter(
    prefix="/chat-threads",
    tags=["Chat Threads"],
)
asset_router = APIRouter(prefix="/assets", tags=["Assets"])
user_app_resume_router = APIRouter(prefix="/user-app-resume", tags=["User App Resume Metadata"])

# === Template Endpoints ===

# -- Node Templates --
@template_router.get(
    "/nodes",
    response_model=List[schemas.NodeTemplateRead],
    summary="List Node Templates",
    # dependencies=[Depends(wf_deps.RequireTemplateRead)] # Requires TEMPLATE_READ on active org
)
async def list_node_templates(
    query_params: Annotated[schemas.NodeTemplateListQuery, Query()],
    user: User = Depends(get_current_active_verified_user),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Retrieves a list of available Node Templates.

    - By default, returns **STAGING** and **PRODUCTION** templates.
    - Use the `launch_status` query parameter (e.g., `?launch_status=production&launch_status=staging`)
      to filter by specific statuses.
    - Supports pagination via `skip` and `limit`.
    - Requires `template:read` permission on the active organization.
    """
    # NOTE: workflow builder is not available publically so only allow superusers to list node templates!
    if not user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Node templates are not accessible to regular users before workflow builder is released.")
    try:
        if any(status in query_params.launch_status for status in constants.PRIVATE_STATUS) and not user.is_superuser:
            workflow_logger.warning(f"User {user.id} attempted to access experimental templates without superuser privileges")
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Experimental templates are not accessible to regular users.")

        templates = await workflow_service.list_node_templates(
            db=db,
            launch_statuses=query_params.launch_status,
            skip=query_params.skip,
            limit=query_params.limit
        )
        workflow_logger.info(f"User {user.id} listed {len(templates)} node templates with filters: {query_params}")
        return templates
    except HTTPException as e:
        # Re-raise HTTP exceptions as they're already properly formatted
        raise
    except Exception as e:
        # Log unexpected errors with traceback
        workflow_logger.error(f"Error listing node templates: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while listing node templates")

@template_router.get(
    "/nodes/{name}/{version}",
    response_model=schemas.NodeTemplateRead,
    summary="Get Node Template by Name and Version; set version to 'latest' for latest production version.",
    # dependencies=[Depends(wf_deps.RequireTemplateRead)]
)
async def get_node_template(
    user: User = Depends(get_current_active_verified_user),
    name: str = Path(..., description="Name of the node template"),
    version: str = Path(..., description="Version of the node template"),
    db_registry: registry.DBRegistry = Depends(wf_deps.get_node_template_registry),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Retrieves a specific Node Template by its unique name and version.

    - Requires `template:read` permission.

    NOTE: if version == "latest", the latest version will be returned.
    """
    # NOTE: if we disallow this, non superusers will not be able to modify workflow configs since they won't know what to validate!
    # # NOTE: workflow builder is not available publically so only allow superusers to list node templates!
    if not user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Node templates are not accessible to regular users before workflow builder is released.")
    try:
        # NOTE: this can only be implemented via the registry!
        if version == "latest":
            try:
                node = db_registry.get_node(node_name=name, version=None)
                version = node.node_version
            except ValueError as e:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        template = await workflow_service.get_node_template(db=db, name=name, version=version)
        if not template:
            workflow_logger.warning(f"User {user.id} attempted to access non-existent node template: {name} v{version}")
            raise exceptions.TemplateNotFoundException(f"Node template '{name}' version '{version}' not found.")
        
        if template.launch_status in constants.PRIVATE_STATUS and not user.is_superuser:
            workflow_logger.warning(f"User {user.id} attempted to access experimental template {name} v{version} without superuser privileges")
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Experimental templates are not accessible to regular users.")
        
        workflow_logger.info(f"User {user.id} retrieved node template: {name} v{version}")
        return template
    except exceptions.TemplateNotFoundException as e:
        # Re-raise template not found exception
        raise
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log unexpected errors with traceback
        workflow_logger.error(f"Error retrieving node template {name} v{version}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while retrieving the node template")

# -- Prompt Templates --
@template_router.post(
    "/prompts",
    response_model=schemas.PromptTemplateRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create Prompt Template",
    # dependencies=[Depends(wf_deps.RequireTemplateCreateActiveOrg)] # Requires TEMPLATE_CREATE on active org
)
async def create_prompt_template(
    template_in: schemas.PromptTemplateCreate,
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(wf_deps.RequireTemplateCreateActiveOrg),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Creates a new organization-specific prompt template.

    - Requires `template:create_org` permission on the active organization.

    NOTE: Only system admins can create system templates.
    """
    if template_in.is_system_entity and not current_user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only system admins can create system templates.")
    try:
        # Service layer handles potential name/version conflicts within the org
        template = await workflow_service.create_prompt_template(
            db=db, template_in=template_in, owner_org_id=active_org_id, user=current_user
        )
        workflow_logger.info(f"User {current_user.id} created prompt template '{template.name}' (id: {template.id}) for org {active_org_id}")
        return template
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log unexpected errors with traceback
        workflow_logger.error(f"Error creating prompt template '{template_in.name}' for org {active_org_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while creating the prompt template")

@template_router.get(
    "/prompts",
    response_model=List[schemas.PromptTemplateRead],
    summary="List Prompt Templates. System templates are accessible to all users and can configure if requested or not. Pagination applies separately to org and system templates.",
    dependencies=[Depends(wf_deps.RequireTemplateReadActiveOrg)] # Requires TEMPLATE_READ on active org
)
async def list_prompt_templates(
    query_params: Annotated[schemas.PromptTemplateListQuery, Query()],
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(get_current_active_verified_user),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Lists accessible prompt templates.

    - Returns templates owned by the active organization.
    - Optionally includes system-wide templates if `include_system=true` (default).
    - Supports pagination via `skip` and `limit`.
    - Superusers can potentially filter by a different `owner_org_id` (requires service/checker support).
    - Requires `template:read` permission on the active organization.
    """
    # NOTE: this is workflow builder prelaunch behaviour, since we don't want ordinary users to be able to list all system templates!
    if not current_user.is_superuser:
        if query_params.include_system:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only system admins can list system templates.")
    try:
        # Superuser filtering by owner_org_id needs to be handled in the service layer
        # based on the current_user's is_superuser flag if needed.
        templates = await workflow_service.list_prompt_templates(
            db=db,
            owner_org_id=active_org_id, # Pass active org as context
            include_system=query_params.include_system,
            skip=query_params.skip,
            limit=query_params.limit
        )
        workflow_logger.info(f"User {current_user.id} listed {len(templates)} prompt templates for org {active_org_id} (include_system: {query_params.include_system})")
        return templates
    except Exception as e:
        workflow_logger.error(f"Error listing prompt templates for org {active_org_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while listing prompt templates")

@template_router.get(
    "/prompts/{template_id}",
    response_model=schemas.PromptTemplateRead,
    summary="Get Prompt Template by ID. System templates are accessible to all users.",
    dependencies=[Depends(wf_deps.RequireTemplateReadActiveOrg)] # Check basic read permission first
)
async def get_prompt_template(
    template_id: uuid.UUID = Path(..., description="The ID of the prompt template"),
    active_org_id: uuid.UUID = Depends(get_active_org_id), # For context if accessing org template
    current_user: User = Depends(get_current_active_verified_user),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Retrieves a specific prompt template by its ID.

    - Can fetch both organization-specific templates (checking ownership against active org)
      and system-wide templates.
    - Requires `template:read` permission.
    """
    try:
        template = await workflow_service.get_prompt_template(
            db=db, template_id=template_id, owner_org_id=active_org_id # Pass org context for check
        )
        workflow_logger.info(f"User {current_user.id} retrieved prompt template {template_id}")
        return template
    except exceptions.TemplateNotFoundException as e:
        workflow_logger.warning(f"User {current_user.id} attempted to access non-existent prompt template: {template_id}")
        raise e # Re-raise the specific 404 exception from the service layer
    except Exception as e:
        workflow_logger.error(f"Error retrieving prompt template {template_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while retrieving the prompt template")


@template_router.put(
    "/prompts/{template_id}",
    response_model=schemas.PromptTemplateRead,
    summary="Update Prompt Template",
    # dependencies=[Depends(wf_deps.RequireTemplateUpdateActiveOrg)] # Requires TEMPLATE_UPDATE on active org
)
async def update_prompt_template(
    template_update: schemas.PromptTemplateUpdate,
    template_id: uuid.UUID = Path(..., description="The ID of the prompt template"),
    active_org_id: uuid.UUID = Depends(get_active_org_id), # For context if accessing org template
    # Fetch the template ensuring it belongs to the active org using the specific dependency
    # template: models.PromptTemplate = Depends(wf_deps.get_prompt_template_for_active_org),
    current_user: User = Depends(wf_deps.RequireTemplateUpdateActiveOrg),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Updates an organization-specific prompt template.

    - Only templates owned by the active organization can be updated.
    - Requires `template:update_org` permission on the active organization.
    """
    try:
        template = await workflow_service.get_prompt_template(
            db=db, template_id=template_id, owner_org_id=active_org_id
        )
        updated_template = await workflow_service.update_prompt_template(
            db=db, template=template, template_update=template_update, user=current_user
        )
        workflow_logger.info(f"User {current_user.id} updated prompt template {template_id} for org {active_org_id}")
        return updated_template
    except exceptions.TemplateNotFoundException as e:
        workflow_logger.warning(f"User {current_user.id} attempted to update non-existent prompt template: {template_id}")
        raise e
    except Exception as e:
        workflow_logger.error(f"Error updating prompt template {template_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while updating the prompt template")

@template_router.delete(
    "/prompts/{template_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Prompt Template",
    # dependencies=[Depends(wf_deps.RequireTemplateDeleteActiveOrg)] # Requires TEMPLATE_DELETE on active org
)
async def delete_prompt_template(
    # template: models.PromptTemplate = Depends(wf_deps.get_prompt_template_for_active_org),
    template_id: uuid.UUID = Path(..., description="The ID of the prompt template"),
    active_org_id: uuid.UUID = Depends(get_active_org_id), # For context if accessing org template
    current_user: User = Depends(wf_deps.RequireTemplateDeleteActiveOrg), # Pass user for audit/logging
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Deletes an organization-specific prompt template.

    - Only templates owned by the active organization can be deleted.
    - Requires `template:delete_org` permission on the active organization.
    """
    try:
        template = await workflow_service.get_prompt_template(
            db=db, template_id=template_id, owner_org_id=active_org_id
        )
        await workflow_service.delete_prompt_template(db=db, template=template, user=current_user)
        workflow_logger.info(f"User {current_user.id} deleted prompt template {template_id} for org {active_org_id}")
        return Response(status_code=status.HTTP_204_NO_CONTENT) # Explicitly return 204
    except exceptions.TemplateNotFoundException as e:
        workflow_logger.warning(f"User {current_user.id} attempted to delete non-existent prompt template: {template_id}")
        raise e
    except Exception as e:
        workflow_logger.error(f"Error deleting prompt template {template_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while deleting the prompt template")

# TODO: FIXME: add custom validators to ensure every prompt variable exists in the prompt template
#   and every placeholder variable `{...}` in template exists as variable
#   maybe use a templating system like Jinja etc??

# -- Schema Templates (Mirrors Prompt Templates) --
@template_router.post(
    "/schemas",
    response_model=schemas.SchemaTemplateRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create Schema Template",
    # dependencies=[Depends(wf_deps.RequireTemplateCreate)]
)
async def create_schema_template(
    template_in: schemas.SchemaTemplateCreate,
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(wf_deps.RequireTemplateCreateActiveOrg),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Creates a new organization-specific schema template.

    - Requires `template:create_org` permission on the active organization.

    NOTE: Only system admins can create system templates.
    """
    if not current_user.is_superuser:
        if template_in.is_system_entity:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only system admins can create system templates.")
    try:
        template = await workflow_service.create_schema_template(
            db=db, template_in=template_in, owner_org_id=active_org_id, user=current_user
        )
        workflow_logger.info(f"User {current_user.id} created schema template '{template.name}' (id: {template.id}) for org {active_org_id}")
        return template
    except HTTPException as e:
        raise
    except Exception as e:
        workflow_logger.error(f"Error creating schema template '{template_in.name}' for org {active_org_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while creating the schema template")

@template_router.get(
    "/schemas",
    response_model=List[schemas.SchemaTemplateRead],
    summary="List Schema Templates",
    # dependencies=[Depends(wf_deps.RequireTemplateReadActiveOrg)]
)
async def list_schema_templates(
    query_params: Annotated[schemas.SchemaTemplateListQuery, Query()],
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(get_current_active_verified_user),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Lists accessible schema templates (organization-specific and optionally system).

     - Requires `template:read` permission on the active organization.
    """
    # NOTE: this is workflow builder prelaunch behaviour, since we don't want ordinary users to be able to list all system templates!
    if not current_user.is_superuser:
        if query_params.include_system:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only system admins can list system templates.")
    try:
        templates = await workflow_service.list_schema_templates(
            db=db,
            owner_org_id=(query_params.owner_org_id if current_user.is_superuser else None) or active_org_id,
            include_system=query_params.include_system,
            skip=query_params.skip,
            limit=query_params.limit
        )
        workflow_logger.info(f"User {current_user.id} listed {len(templates)} schema templates for org {active_org_id} (include_system: {query_params.include_system})")
        return templates
    except Exception as e:
        workflow_logger.error(f"Error listing schema templates for org {active_org_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while listing schema templates")

@template_router.get(
    "/schemas/{template_id}",
    response_model=schemas.SchemaTemplateRead,
    summary="Get Schema Template by ID. System templates are accessible to all users, otherwise only org-specific templates are accessible, org indicated via org header.",
    dependencies=[Depends(wf_deps.RequireTemplateReadActiveOrg)]
)
async def get_schema_template(
    template_id: uuid.UUID = Path(..., description="The ID of the schema template"),
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(get_current_active_verified_user),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Retrieves a specific schema template by its ID.

    - Can fetch both organization-specific and system templates.
    - Requires `template:read` permission.
    """
    try:
        template = await workflow_service.get_schema_template(
            db=db, template_id=template_id, owner_org_id=active_org_id
        )
        workflow_logger.info(f"User {current_user.id} retrieved schema template {template_id}")
        return template
    except exceptions.TemplateNotFoundException as e:
        workflow_logger.warning(f"User {current_user.id} attempted to access non-existent schema template: {template_id}")
        raise e
    except Exception as e:
        workflow_logger.error(f"Error retrieving schema template {template_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while retrieving the schema template")

@template_router.put(
    "/schemas/{template_id}",
    response_model=schemas.SchemaTemplateRead,
    summary="Update Schema Template",
    # dependencies=[Depends(wf_deps.RequireTemplateUpdateActiveOrg)]
)
async def update_schema_template(
    template_update: schemas.SchemaTemplateUpdate,
    template_id: uuid.UUID = Path(..., description="The ID of the schema template"),
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    # template: models.SchemaTemplate = Depends(wf_deps.get_schema_template_for_active_org),
    current_user: User = Depends(wf_deps.RequireTemplateUpdateActiveOrg),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Updates an organization-specific schema template.

     - Requires `template:update_org` permission on the active organization.
    """
    try:
        template = await workflow_service.get_schema_template(
            db=db, template_id=template_id, owner_org_id=active_org_id
        )
        updated_template = await workflow_service.update_schema_template(
            db=db, template=template, template_update=template_update, user=current_user
        )
        workflow_logger.info(f"User {current_user.id} updated schema template {template_id} for org {active_org_id}")
        return updated_template
    except exceptions.TemplateNotFoundException as e:
        workflow_logger.warning(f"User {current_user.id} attempted to update non-existent schema template: {template_id}")
        raise e
    except Exception as e:
        workflow_logger.error(f"Error updating schema template {template_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while updating the schema template")

@template_router.delete(
    "/schemas/{template_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Schema Template",
    # dependencies=[Depends(wf_deps.RequireTemplateDeleteActiveOrg)]
)
async def delete_schema_template(
    # template: models.SchemaTemplate = Depends(wf_deps.get_schema_template_for_active_org),
    template_id: uuid.UUID = Path(..., description="The ID of the schema template"),
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(wf_deps.RequireTemplateDeleteActiveOrg),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Deletes an organization-specific schema template.

     - Requires `template:delete_org` permission on the active organization.
    """
    try:
        template = await workflow_service.get_schema_template(
            db=db, template_id=template_id, owner_org_id=active_org_id
        )
        await workflow_service.delete_schema_template(db=db, template=template, user=current_user)
        workflow_logger.info(f"User {current_user.id} deleted schema template {template_id} for org {active_org_id}")
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except exceptions.TemplateNotFoundException as e:
        workflow_logger.warning(f"User {current_user.id} attempted to delete non-existent schema template: {template_id}")
        raise e
    except Exception as e:
        workflow_logger.error(f"Error deleting schema template {template_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while deleting the schema template")

@template_router.post(
    "/prompts/search",
    response_model=List[schemas.PromptTemplateRead],
    summary="Search for prompt templates by name and optional version (uses POST to support search params, including system entities); there are unique constaints to name/version within a specific org so you may receive unique results within an org.",
    # dependencies=[Depends(wf_deps.RequireTemplateReadActiveOrg)]
)
async def search_prompt_templates(
    search_params: schemas.PromptTemplateSearchQuery,
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(wf_deps.RequireTemplateReadActiveOrg),  # Depends(get_current_active_verified_user),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Search for prompt templates by name and optional version.
    
    Returns prompt templates from:
    - The active organization matching name/version
    - Public prompt templates matching name/version (if include_public=True)
    - System prompt templates matching name/version (if include_system_entities=True and user is superuser)
    
    Requires `template:read` permission on the active organization.
    """
    try:
        templates = await workflow_service.search_prompt_templates(
            db=db,
            name=search_params.name,
            version=search_params.version,
            owner_org_id=active_org_id,
            include_public=search_params.include_public,
            include_system_entities=search_params.include_system_entities,
            include_public_system_entities=search_params.include_public_system_entities,
            sort_by=search_params.sort_by,
            sort_order=search_params.sort_order,
            user=current_user
        )
        workflow_logger.info(f"User {current_user.id} searched for prompt templates with name '{search_params.name}' in org {active_org_id}")
        return templates
    except Exception as e:
        workflow_logger.error(f"Error searching prompt templates for org {active_org_id}: {str(e)}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while searching prompt templates")

@template_router.post(
    "/schemas/search",
    response_model=List[schemas.SchemaTemplateRead],
    summary="Search for schema templates by name and optional version (uses POST to support search params, including system entities); there are unique constaints to name/version within a specific org so you may receive unique results within an org.",
    # dependencies=[Depends(wf_deps.RequireTemplateReadActiveOrg)]
)
async def search_schema_templates(
    search_params: schemas.SchemaTemplateSearchQuery,
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(wf_deps.RequireTemplateReadActiveOrg),  # Depends(get_current_active_verified_user),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Search for schema templates by name and optional version.
    
    Returns schema templates from:
    - The active organization matching name/version
    - Public schema templates matching name/version (if include_public=True)
    - System schema templates matching name/version (if include_system_entities=True and user is superuser)
    
    Requires `template:read` permission on the active organization.
    """
    try:
        templates = await workflow_service.search_schema_templates(
            db=db,
            name=search_params.name,
            version=search_params.version,
            owner_org_id=active_org_id,
            include_public=search_params.include_public,
            include_system_entities=search_params.include_system_entities,
            include_public_system_entities=search_params.include_public_system_entities,
            sort_by=search_params.sort_by,
            sort_order=search_params.sort_order,
            user=current_user
        )
        workflow_logger.info(f"User {current_user.id} searched for schema templates with name '{search_params.name}' in org {active_org_id}")
        return templates
    except Exception as e:
        workflow_logger.error(f"Error searching schema templates for org {active_org_id}: {str(e)}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while searching schema templates")

# === Workflow Endpoints ===

@workflow_router.post(
    "/validate",
    response_model=schemas.WorkflowGraphValidationResult,
    summary="Validate a workflow graph configuration for structural integrity and node configuration validity",
)
async def validate_graph(
    graph_config: Dict[str, Any] = Body(..., description="The graph configuration to validate"),
    validate_nodes_and_structure: bool = Query(True, description="Whether to validate node configurations and overall graph structure using GraphBuilder"),
    user: User = Depends(get_current_active_verified_user),
    db_registry: registry.DBRegistry = Depends(wf_deps.get_node_template_registry),
):
    """
    Validates a workflow graph configuration for correctness and consistency.
    
    This endpoint performs validation:
    1. Base GraphSchema Pydantic validation.
    2. If `validate_nodes_and_structure` is true:
        a. Node configuration validation (ensures each node's configuration is valid against its template).
        b. GraphBuilder validation (ensures overall structural integrity).
    
    Returns detailed validation results with specific errors if validation fails.
    """
    workflow_logger.info(f"User {user.id} requested graph validation. Validate nodes and structure: {validate_nodes_and_structure}")

    overall_is_valid, pydantic_schema_is_valid, detailed_validation_is_valid, errors_dict, validation_errors_dict = \
        await _validate_updated_graph_schema(
            graph_config_dict=graph_config,
            db_registry=db_registry,
            perform_detailed_validation=validate_nodes_and_structure
        )

    # For the response model, node_configs_valid means the detailed validation part.
    # If validate_nodes_and_structure was false, then this part was "skipped", so it's considered valid for the purpose of the response field.
    # If it was true, then detailed_validation_is_valid reflects its actual success/failure.
    # The overall_is_valid from the function correctly reflects whether all *performed* steps passed.

    response_node_configs_valid = detailed_validation_is_valid 
    # If detailed validation was not performed, overall_is_valid only depends on pydantic_schema_is_valid
    # If detailed validation was performed, overall_is_valid considers everything.
    
    validation_result = schemas.WorkflowGraphValidationResult(
        is_valid=overall_is_valid,
        graph_schema_valid=pydantic_schema_is_valid,
        node_configs_valid=response_node_configs_valid, # This field indicates if node/builder validation passed or was skipped
        errors=errors_dict,
        validation_errors=validation_errors_dict if validation_errors_dict else {}
    )
    
    if overall_is_valid:
        workflow_logger.info(f"Graph validation request for user {user.id} completed. Overall result: VALID.")
    else:
        error_count = sum(len(el) for el in errors_dict.values())
        workflow_logger.warning(f"Graph validation request for user {user.id} completed. Overall result: INVALID with {error_count} errors. Details: {errors_dict}")
            
    return validation_result

@workflow_router.post(
    "",
    response_model=schemas.WorkflowRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create Workflow. active_org_id is optional for superuser to create workflows without org context (potentially for testing purposes or global workflows which can be used across all orgs).",
    # dependencies=[Depends(wf_deps.RequireWorkflowCreateActiveOrg)] # Check permission first
)
async def create_workflow(
    workflow_in: schemas.WorkflowCreate,
    active_org_id: Optional[uuid.UUID] = Depends(get_active_org_id),
    current_user: User = Depends(wf_deps.RequireWorkflowCreateActiveOrg),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Creates a new workflow configuration within the active organization.

    - The `graph_config` defines the structure and nodes of the workflow.
    - Requires `workflow:create` permission on the active organization.

    NOTE: Only system admins can create system workflows.
    """
    if workflow_in.is_system_entity and not current_user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only system admins can create system workflows.")
    # TODO: instead add a node blacklist, regular users are not able to use those nodes in graph schema, eg python code submissions!
    # # Current users are system admins only!
    if not current_user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only system admins can create workflows.")
    try:
        # Service layer handles potential name conflicts within the org
        workflow = await workflow_service.create_workflow(
            db=db, workflow_in=workflow_in, owner_org_id=active_org_id, user=current_user
        )
        workflow_logger.info(f"User {current_user.id} created workflow '{workflow.name}' (id: {workflow.id}) for org {active_org_id}")
        return workflow
    except HTTPException as e:
        raise
    except Exception as e:
        workflow_logger.error(f"Error creating workflow '{workflow_in.name}' for org {active_org_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while creating the workflow")

@workflow_router.get(
    "",
    response_model=List[schemas.WorkflowRead],
    summary="List Workflows for the active org. Superusers can list workflows for any org by providing the `owner_org_id` query parameter or using the active org context from header (the former overrides the latter).",
    # dependencies=[Depends(wf_deps.RequireWorkflowRead)] # Requires workflow:read on active org
)
async def admin_list_workflows(
    query_params: Annotated[schemas.WorkflowListQuery, Query()],
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(wf_deps.RequireWorkflowReadActiveOrg),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Lists workflows accessible only to admins (superusers).

    - By default, lists workflows belonging to the active organization.
    - **Superusers** can list workflows for any organization by providing the `owner_org_id` query parameter.
    - Supports pagination via `skip` and `limit`.
    - Requires `workflow:read` permission on the active organization (or superuser status for cross-org listing).

    NOTE: when a user modifies configuration of a system workflow to execute it, it needs to be saved as a new workflow here!
    """
    if not current_user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Workflows are not accessible to regular users before workflow builder is released.")
    try:
        list_org_id = active_org_id
        # Allow superuser to list workflows for other orgs
        if current_user.is_superuser and query_params.owner_org_id:
            list_org_id = query_params.owner_org_id
        elif not current_user.is_superuser and query_params.owner_org_id and query_params.owner_org_id != active_org_id:
             # If not superuser, cannot list for other orgs
             raise HTTPException(
                 status_code=status.HTTP_403_FORBIDDEN,
                 detail="Insufficient permissions to list workflows for other organizations."
             )

        # Pass relevant filters (excluding pagination) to the service layer
        # launch_status filter is removed as it's not on the workflow model according to models.py
        workflows = await workflow_service.list_workflows(
            db=db,
            owner_org_id=list_org_id, # Use the determined org_id
            # launch_status=query_params.launch_status,
            only_public=query_params.only_public,
            only_system_entities=query_params.only_system_entities,
            workflow_name_search=query_params.workflow_name_search,
            skip=query_params.skip,
            limit=query_params.limit
        )
        workflow_logger.info(f"User {current_user.id} listed workflows for org {list_org_id}")

        # If not returning graphs schemas, replace them with empty dicts
        if not query_params.return_graphs_schemas:
            new_workflows = []
            for workflow in workflows:
                _workflow = schemas.WorkflowRead(**workflow.model_dump())
                _workflow.graph_config = {}
                new_workflows.append(_workflow)
            workflows = new_workflows
        return workflows
    except HTTPException as e:
        workflow_logger.warning(f"Permission error for user {current_user.id} listing workflows: {str(e)}")
        raise
    except Exception as e:
        workflow_logger.error(f"Error listing workflows for org {active_org_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while listing workflows")

@workflow_router.post(
    "/search",
    response_model=List[schemas.WorkflowRead],
    summary="Search for workflows by name and optional version_tag (uses POST to support search params, including system entities); there are unique constaints to name/version within a specific org so you may receive unique results within an org.",
    # dependencies=[Depends(wf_deps.RequireWorkflowRead)]
)
async def search_workflows(
    search_params: schemas.WorkflowSearchQuery,
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(wf_deps.RequireWorkflowReadActiveOrg),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Search for workflows by name and optional version tag.
    
    Returns workflows from:
    - The active organization matching name/version_tag
    - Public workflows matching name/version_tag (if include_public=True)
    - System workflows matching name/version_tag (if include_system_entities=True and user is superuser)
    
    Requires `workflow:read` permission on the active organization.
    """
    try:
        workflows = await workflow_service.search_workflows(
            db=db,
            name=search_params.name,
            version_tag=search_params.version_tag,
            owner_org_id=active_org_id,
            include_public=search_params.include_public,
            include_system_entities=search_params.include_system_entities,
            include_public_system_entities=search_params.include_public_system_entities,
            sort_by=search_params.sort_by,
            sort_order=search_params.sort_order,
            user=current_user
        )
        workflow_logger.info(f"User {current_user.id} searched for workflows with name '{search_params.name}' in org {active_org_id}")
        schema_workflows = []
        for workflow in workflows:
            workflow_schema_obj = schemas.WorkflowRead(**workflow.model_dump())
            if not current_user.is_superuser:
                workflow_schema_obj.graph_config = {}
            schema_workflows.append(workflow_schema_obj)
        return schema_workflows
    except Exception as e:
        workflow_logger.error(f"Error searching workflows for org {active_org_id}: {str(e)}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while searching workflows")

@workflow_router.get(
    "/{workflow_id}",
    response_model=schemas.WorkflowRead,
    summary="Get Workflow by ID",
    # dependencies=[Depends(wf_deps.RequireWorkflowReadActiveOrg)] # Basic check on active org context
)
async def get_workflow(
    workflow_id: uuid.UUID = Path(..., description="The ID of the workflow"),
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    # Use dependency to fetch workflow and ensure it belongs to active org
    # Dependency raises 404 if not found in active org
    current_user: User = Depends(wf_deps.RequireWorkflowReadActiveOrg),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Retrieves the details of a specific workflow configuration.

    - Ensures the workflow belongs to the user's active organization.
    - Requires `workflow:read` permission on the active organization.
    """
    # The user won't be able to modify workflow configs if we disable fetching system workflows!
    # include_system_entities = False
    # if current_user.is_superuser:
    #     include_system_entities = True
    
    include_system_entities = True
    if not current_user.is_superuser:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Workflows are not accessible to regular users before workflow builder is released.")
    try:
        workflow = await workflow_service.get_workflow(
            db=db,
            user=current_user,
            workflow_id=workflow_id,
            owner_org_id=active_org_id,
            include_system_entities=include_system_entities
        )
        workflow_logger.info(f"User {current_user.id} retrieved workflow {workflow_id} for org {active_org_id}")
        return workflow
    except exceptions.WorkflowNotFoundException as e:
        workflow_logger.warning(f"User {current_user.id} attempted to retrieve non-existent workflow: {workflow_id}")
        raise e
    except Exception as e:
        workflow_logger.error(f"Error retrieving workflow {workflow_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while retrieving the workflow")

@workflow_router.put(
    "/{workflow_id}",
    response_model=schemas.WorkflowRead,
    summary="Update Workflow",
    # dependencies=[Depends(wf_deps.RequireWorkflowUpdateActiveOrg)] # Check update perm on active org
)
async def update_workflow(
    workflow_update: schemas.WorkflowUpdate,
    workflow_id: uuid.UUID = Path(..., description="The ID of the workflow"),
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    # Fetch workflow, ensuring it belongs to active org for modification
    # workflow: models.Workflow = Depends(wf_deps.get_workflow_for_active_org),
    workflow_dao: wf_crud.WorkflowDAO = Depends(wf_deps.get_workflow_dao),
    current_user: User = Depends(wf_deps.RequireWorkflowUpdateActiveOrg),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Updates an existing workflow configuration.

    - Only workflows belonging to the active organization can be updated.
    - Requires `workflow:update` permission on the active organization.
    """
    try:
        workflow = await workflow_dao.get_by_id_and_org(db, workflow_id=workflow_id, org_id=active_org_id)
        # Dependency handles fetch and ensures it's in the active org
        updated_workflow = await workflow_service.update_workflow(
            db=db, workflow=workflow, workflow_update=workflow_update, user=current_user
        )
        workflow_logger.info(f"User {current_user.id} updated workflow {workflow_id} for org {active_org_id}")
        return updated_workflow
    except exceptions.WorkflowNotFoundException as e:
        workflow_logger.warning(f"User {current_user.id} attempted to update non-existent workflow: {workflow_id}")
        raise e
    # except exceptions.WorkflowNameConflictException as e:
    #     workflow_logger.warning(f"User {current_user.id} attempted to update workflow {workflow_id} with conflicting name: {str(e)}")
    #     raise e
    except Exception as e:
        workflow_logger.error(f"Error updating workflow {workflow_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while updating the workflow")

@workflow_router.delete(
    "/{workflow_id}",
    response_model=schemas.WorkflowRead,
    summary="Delete Workflow",
    # dependencies=[Depends(wf_deps.RequireWorkflowDeleteActiveOrg)] # Check delete perm on active org
)
async def delete_workflow(
    workflow_id: uuid.UUID = Path(..., description="The ID of the workflow"),
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    workflow_dao: wf_crud.WorkflowDAO = Depends(wf_deps.get_workflow_dao),
    current_user: User = Depends(wf_deps.RequireWorkflowDeleteActiveOrg),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Deletes a workflow configuration.

    - Only workflows belonging to the active organization can be deleted.
    - Note: Associated workflow runs are typically *not* deleted by default (check FK constraints).
    - Requires `workflow:delete` permission on the active organization.
    """
    try:
        workflow = await workflow_dao.get_by_id_and_org(db, workflow_id=workflow_id, org_id=active_org_id)
        # Dependency handles fetch and ensures it's in the active org
        await workflow_service.delete_workflow(db=db, workflow=workflow, user=current_user)
        workflow_logger.info(f"User {current_user.id} deleted workflow {workflow_id} for org {active_org_id}")
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except exceptions.WorkflowNotFoundException as e:
        workflow_logger.warning(f"User {current_user.id} attempted to delete non-existent workflow: {workflow_id}")
        raise e
    # TODO: handle this case when workflow is in use!
    # except exceptions.WorkflowInUseException as e:
    #     workflow_logger.warning(f"User {current_user.id} attempted to delete workflow {workflow_id} that is in use: {str(e)}")
    #     raise e
    except Exception as e:
        workflow_logger.error(f"Error deleting workflow {workflow_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while deleting the workflow")

@workflow_router.get(
    "/{workflow_id}/effective-config",
    response_model=schemas.WorkflowEffectiveConfigResponse,
    summary="Get Effective Workflow Configuration with Overrides",
    dependencies=[Depends(wf_deps.RequireWorkflowReadActiveOrg)] # Ensures user has read access to the base workflow
)
async def get_workflow_effective_config(
    # query_params: Annotated[schemas.WorkflowEffectiveConfigQuery, Query(...)],
    include_active: Optional[bool] = Query(True, description="Whether to include active overrides when calculating the effective config."),
    include_tags: Optional[List[str]] = Query(None, description="Optional list of override tags to specifically include."),
    workflow_id: uuid.UUID = Path(..., description="The ID of the workflow"),
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(get_current_active_verified_user), # get_current_active_verified_user instead of RequireWorkflowReadActiveOrg for current_user
    db: AsyncSession = Depends(get_async_db_dependency),
    user_id: Optional[uuid.UUID] = Query(None, description="The ID of the user whose effective config to retrieve (superuser only)"), # Optional, defaults to current_user
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
    user_dao: auth_crud.UserDAO = Depends(auth_deps.get_user_dao),
):
    """
    Retrieves the effective graph configuration for a specific workflow, 
    after applying all relevant and active configuration overrides.

    - Requires `workflow:read` permission on the active organization for the base workflow.
    """
    if not current_user.is_superuser and user_id and user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You must be a superuser to get effective config for other users.")
    effective_user = current_user
    if user_id and user_id != current_user.id:
        effective_user = await user_dao.get(db, id=user_id)
        if not effective_user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    try:
        applied_overrides, effective_graph_schema = await workflow_service.get_workflow_config_with_overrides(
            db=db,
            workflow_id=workflow_id,
            active_org_id=active_org_id,
            requesting_user=effective_user,
            include_active=include_active,
            include_tags=include_tags
        )
        effective_user_msg = f" for user {effective_user.id}" if effective_user.id != current_user.id else ""
        workflow_logger.info(f"User {current_user.id} retrieved effective config{effective_user_msg} for workflow {workflow_id} in org {active_org_id} with {len(applied_overrides)} overrides.")
        return schemas.WorkflowEffectiveConfigResponse(
            applied_overrides=applied_overrides,
            effective_graph_schema=effective_graph_schema
        )
    except exceptions.WorkflowNotFoundException as e:
        workflow_logger.warning(f"User {current_user.id} attempted to get effective config{effective_user_msg} for non-existent workflow: {workflow_id}")
        raise e
    except HTTPException as e:
        # Re-raise known HTTP exceptions (e.g., from override application failure)
        workflow_logger.warning(f"HTTP error for user {current_user.id} getting effective config{effective_user_msg} for workflow {workflow_id}: {str(e)}")
        raise
    except Exception as e:
        workflow_logger.error(f"Error retrieving effective config{effective_user_msg} for workflow {workflow_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while retrieving the effective workflow configuration")


# === Workflow Run Endpoints ===

@run_router.post(
    "",
    response_model=schemas.WorkflowRunRead,
    status_code=status.HTTP_202_ACCEPTED, # Indicate async processing started
    summary="Submit Workflow Run. Supports: new run, HITL resume (provide `run_id` + `resume_after_hitl=true`), or manual retry (provide `run_id` of FAILED/CANCELLED run with `is_manual_retry=true`).",
    dependencies=[Depends(wf_deps.RequireWorkflowExecuteActiveOrg)] # Check execute perm on active org
)
async def submit_workflow_run(
    run_submit: schemas.WorkflowRunCreate,
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(wf_deps.RequireWorkflowExecuteActiveOrg),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
    user_dao: auth_crud.UserDAO = Depends(auth_deps.get_user_dao),
):
    """
    Submits a new workflow run for execution, resumes after HITL, or retries a failed/cancelled run.

    **New Workflow Run:**
    - Provide `workflow_id` to run an existing saved workflow.
    - **OR** provide `graph_schema` to define and run an ad-hoc workflow
      (an associated workflow record will be created automatically).
    - Provide `inputs` required by the workflow.

    **HITL Resume:**
    - Provide `run_id` of the run to resume
    - Set `resume_after_hitl=true`
    - Provide `inputs` with HITL response data

    **Manual Retry (FAILED/CANCELLED runs):**
    - Provide `run_id` of the failed/cancelled run
    - Set `is_manual_retry=true`
    - Original inputs and configuration are automatically reused
    - Retry count is automatically incremented
    - If `manual_retry_search_by_workflow_name=true`, the workflow will be searched by name too if ID is not found to find the latest version of the workflow to retry. Useful for retrying a newly ingested workflow even though workflow ID changed.

    - Returns the `WorkflowRun` record (usually in `SCHEDULED` state).
    - Requires `workflow:execute` permission on the active organization.
    """
    try:
        if not active_org_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Active organization not found")
        
        if not current_user.is_superuser and run_submit.graph_schema:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You must be a superuser to submit ad-hoc workflow runs with custom graph schemas.")
        
        user = current_user
        if run_submit.on_behalf_of_user_id:
            if not current_user.is_superuser:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You must be a superuser to act on behalf of another user")
            if run_submit.on_behalf_of_user_id != current_user.id:
                user = await user_dao.get(db, id=run_submit.on_behalf_of_user_id)
                if not user:
                    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
            
        # Service layer handles creating ad-hoc workflow if needed and triggering execution
        run = await workflow_service.submit_workflow_run(
            db=db, run_submit=run_submit, owner_org_id=active_org_id, user=user
        )
        on_behalf_of_user_id_msg = f" on behalf of user {user.id}" if user.id != current_user.id else ""
        workflow_logger.info(f"User {current_user.id} submitted workflow run{on_behalf_of_user_id_msg} for org {active_org_id}, run_id: {run.id}")
        return run
    except exceptions.WorkflowNotFoundException as e:
        workflow_logger.warning(f"User {current_user.id} attempted to run non-existent workflow: {str(e)}")
        raise e
    # except exceptions.InvalidWorkflowInputsException as e:
    #     workflow_logger.warning(f"User {current_user.id} submitted invalid workflow inputs: {str(e)}")
    #     raise e
    except Exception as e:
        workflow_logger.error(f"Error submitting workflow run for org {active_org_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while submitting the workflow run")


@run_router.get(
    "",
    response_model=List[schemas.WorkflowRunRead],
    summary="List Workflow Runs",
    # dependencies=[Depends(wf_deps.RequireRunReadActiveOrg)] # Check read perm on active org
)
async def list_runs(
    query_params: Annotated[schemas.WorkflowRunListQuery, Query()],
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(wf_deps.RequireRunReadActiveOrg), # Needed for superuser check
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Lists workflow runs accessible to the current user.

    - By default, lists runs belonging to the active organization.
    - **Superusers** can list runs for any organization by providing `owner_org_id`.
    - Supports filtering by:
        - `workflow_id`
        - `status`
        - `triggered_by_user_id` (only own runs for non-superusers)
    - Supports pagination via `skip` and `limit`.
    - Requires `run:read` permission on the active organization (or superuser status).
    """
    try:
        list_org_id = active_org_id
        list_user_id = query_params.triggered_by_user_id

        # Apply superuser logic for cross-org and cross-user filtering
        if current_user.is_superuser:
            if query_params.owner_org_id:
                list_org_id = query_params.owner_org_id or active_org_id
            # Superuser can filter by any user ID or None
            list_user_id = query_params.triggered_by_user_id or current_user.id
        else:
            # Non-superuser checks
            if query_params.owner_org_id and query_params.owner_org_id != active_org_id:
                 raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot list runs for other organizations.")
            if query_params.triggered_by_user_id and query_params.triggered_by_user_id != current_user.id:
                 raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot filter runs triggered by other users.")
            # If filter is None or current_user.id, it's allowed
            list_user_id = query_params.triggered_by_user_id

        # Prepare filters for service layer, converting schema to dict/kwargs if needed
        query_params.owner_org_id = list_org_id
        query_params.triggered_by_user_id = list_user_id
        # service_filters = schemas.WorkflowRunListQuery(
        #     workflow_id=query_params.workflow_id,
        #     workflow_name=query_params.workflow_name,
        #     status=query_params.status,
        #     triggered_by_user_id=list_user_id,
        #     owner_org_id=list_org_id, # Pass the determined org_id to the service
        #     parent_run_id=query_params.parent_run_id,  # Include parent_run_id filter
        #     tag=query_params.tag,  # Include tag filter
        #     applied_workflow_config_overrides=query_params.applied_workflow_config_overrides  # Include overrides filter
        #     # Pagination handled separately
        # )

        runs = await workflow_service.list_runs(
            db=db,
            owner_org_id=list_org_id, # Pass final org id for main query scope
            filters=query_params, # Pass filters object
            skip=query_params.skip,
            limit=query_params.limit
        )
        workflow_logger.info(f"User {current_user.id} listed workflow runs for org {list_org_id}")

        if current_user.is_superuser:
            runs_dump = []
            for run in runs:
                run_dump = run.model_dump(mode="json")
                run_ids = []
                for prefect_run_id in run.prefect_run_ids.split(","):
                    if prefect_run_id:
                        try:
                            run_name = await services.WorkflowService.get_run_name(uuid.UUID(prefect_run_id))
                            run_ids.append(f"{prefect_run_id}: {run_name}")
                        except ValueError as e:
                            run_ids.append(f"{prefect_run_id}: (invalid UUID): {str(e)}")

                run_dump["run_ids"] = "; ".join(run_ids)

                runs_dump.append(run_dump)

            return runs_dump

        return runs
    except HTTPException as e:
        workflow_logger.warning(f"Permission error for user {current_user.id} listing runs: {str(e)}")
        raise
    except Exception as e:
        workflow_logger.error(f"Error listing workflow runs for org {active_org_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while listing workflow runs")

@run_router.get(
    "/{run_id}/logs",
    response_model=schemas.WorkflowRunLogs,
    summary="Get Workflow Run Logs",
    # dependencies=[Depends(wf_deps.RequireRunReadActiveOrg)] # Basic check on active org context
)
async def get_run_logs(
    # Fetch run and ensure it belongs to active org
    # run: models.WorkflowRun = Depends(wf_deps.get_workflow_run_for_org),
    run_id: uuid.UUID = Path(..., description="The ID of the workflow run"),
    run_dao: wf_crud.WorkflowRunDAO = Depends(wf_deps.get_workflow_run_dao),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
    current_superuser: User = Depends(auth_deps.get_current_active_superuser),
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(10000, ge=1, le=10000, description="Maximum number of items to return"),
):
    """
    Gets the logs of a specific workflow run.

    - Fetches data from Prefect for all flow runs associated with this workflow run
      and its direct subworkflow runs (children with this run as parent). This is
      non-recursive and does not include grandchildren.
    - Returns logs with level (as string), message, and timestamp
    - Supports pagination via skip and limit
    - Requires `run:read` permission on the active organization.
    """
    try:
        run = await run_dao.get(db, id=run_id)
        if not run:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workflow run not found")
        logs = await workflow_service.get_run_logs(
            db=db,
            run=run,
            skip=skip,
            limit=limit,
        )
        workflow_logger.info(f"Retrieved {len(logs.logs)} workflow run logs for run {run.id}")
        return logs
    except Exception as e:
        workflow_logger.error(f"Error retrieving workflow run logs for run {run.id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while retrieving the workflow run logs")


@run_router.get(
    "/{run_id}",
    response_model=schemas.WorkflowRunRead,
    summary="Get Workflow Run Status Summary",
    # dependencies=[Depends(wf_deps.RequireRunReadActiveOrg)] # Basic check on active org context
)
async def get_run_status(
    # Use dependency to fetch run and ensure it belongs to active org
    # run: models.WorkflowRun = Depends(wf_deps.get_workflow_run_for_org),
    # Re-inject active_org_id for service call if needed, though run object has it
    current_user: User = Depends(wf_deps.RequireRunReadActiveOrg),
    ignore_org_check_for_run_fetch: bool = Query(False, description="Ignore organization check and fetch run regardless of org (superuser only)"),
    # db_manager: AsyncGenerator[AsyncSession, None] = Depends(get_async_db_as_manager),
    run_id: uuid.UUID = Path(..., description="The ID of the workflow run"),
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    db: AsyncSession = Depends(get_async_db_dependency),
    run_dao: wf_crud.WorkflowRunDAO = Depends(wf_deps.get_workflow_run_dao),
    # workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Gets the status summary of a specific workflow run.

    - Fetches data primarily from the SQL database.
    - May attempt to augment the status with the latest update from the event stream (MongoDB) if available.
    - Requires `run:read` permission on the active organization.
    """
    try:
        if ignore_org_check_for_run_fetch and not current_user.is_superuser:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot fetch run without org check for non-superusers")
        if current_user.is_superuser and ignore_org_check_for_run_fetch:
            run = await run_dao.get(db, id=run_id)
        else:
            run = await run_dao.get_run_by_id_and_org(db, run_id=run_id, org_id=active_org_id)
        # if not run:
        #     raise WorkflowRunNotFoundException()
        if not run:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workflow run not found")
        # Dependency handles fetch and access check against active org
        # Call service method that might augment with Mongo status
        # async with db_manager as db:
        #     return await workflow_service.get_run_summary_with_mongo_status(
        #         db=db, run_id=run.id, owner_org_id=run.owner_org_id, user=current_user # Use org_id from fetched run
        #     )
        workflow_logger.info(f"Retrieved workflow run status for run {run.id}")
        
        run_dump = run.model_dump(mode="json")
        if current_user.is_superuser:
            run_ids = []
            for prefect_run_id in run.prefect_run_ids.split(","):
                if prefect_run_id:
                    try:
                        run_name = await services.WorkflowService.get_run_name(uuid.UUID(prefect_run_id))
                        run_ids.append(f"{prefect_run_id}: {run_name}")
                    except ValueError as e:
                        run_ids.append(f"{prefect_run_id}: (invalid UUID): {str(e)}")

            run_dump["run_ids"] = "; ".join(run_ids)

            return run_dump
        
        run_dump["prefect_run_ids"] = None
        return run_dump

    except HTTPException as e:
        raise e
    except Exception as e:
        workflow_logger.error(f"Error retrieving workflow run status for run {run.id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while retrieving the workflow run status")


# NOTE: state exposes intermediate prompts, system prompts, extra metadata we may not wanna show the user!
@run_router.get(
    "/{run_id}/state",
    response_model=schemas.WorkflowRunState,
    summary="Get Workflow Run State",
    dependencies=[Depends(wf_deps.RequireRunReadActiveOrg)] # Basic check on active org context
)
async def get_run_state(
    # Use dependency to fetch run and ensure it belongs to active org
    # run: models.WorkflowRun = Depends(wf_deps.get_workflow_run_for_org),
    run_id: uuid.UUID = Path(..., description="The ID of the workflow run"),
    run_dao: wf_crud.WorkflowRunDAO = Depends(wf_deps.get_workflow_run_dao),
    # Re-inject active_org_id for service call if needed, though run object has it
    current_superuser: User = Depends(auth_deps.get_current_active_superuser),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Gets the state of a specific workflow run. (ONLY FOR DEBUGGING for SUPERUSERS!)

    - Fetches data primarily from the SQL database.
    - May attempt to augment the state with the latest update from the event stream (MongoDB) if available.
    - Requires `run:read` permission on the active organization.
    """
    run = None
    try:
        run = await run_dao.get(db, id=run_id)
        if not run:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workflow run not found")
        state = await workflow_service.get_run_state(db, run=run)
        workflow_logger.info(f"Retrieved workflow run state for run {run.id}")
        return state
    except Exception as e:
        if not run:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Workflow run not found")
        workflow_logger.error(f"Error retrieving workflow run state for run {run.id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while retrieving the workflow run state")


@run_router.get(
    "/{run_id}/details",
    response_model=schemas.WorkflowRunDetailRead,
    summary="Get Workflow Run Details",
    # dependencies=[Depends(wf_deps.RequireRunReadActiveOrg)] # Basic check on active org context
)
async def get_run_details(
    run: models.WorkflowRun = Depends(wf_deps.get_workflow_run_for_org),
    db: AsyncSession = Depends(get_async_db_dependency),
    current_user: User = Depends(wf_deps.RequireRunReadActiveOrg),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
    skip: int = Query(0, ge=0, description="Number of events to skip"),
    limit: int = Query(100, ge=1, le=5000, description="Maximum number of events to return"),
):
    """
    Gets detailed results for a specific workflow run.

    Default: reverse chronological order (descending), returns last 100 events.

    - Combines the summary data from the SQL database with the detailed event stream
      fetched from the configured NoSQL store (e.g., MongoDB).
    - Includes node outputs, status changes, message chunks, etc.
    - Requires `run:read` permission on the active organization.
    """
    try:
        # Dependency handles fetch and access check
        details = await workflow_service.get_run_details(db=db, run=run, skip=skip, limit=limit, user=current_user)
        workflow_logger.info(f"User {current_user.id} retrieved details for workflow run {run.id}")
        return details
    # except exceptions.RunDetailsNotFoundException as e:
    #     workflow_logger.warning(f"User {current_user.id} attempted to retrieve details for run with no events: {run.id}")
    #     raise e
    except Exception as e:
        workflow_logger.error(f"Error retrieving workflow run details for run {run.id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while retrieving the workflow run details")

@run_router.get(
    "/{run_id}/stream",
    response_model=List[schemas.WorkflowRunEventDetail],
    summary="Get Workflow Run Event Stream",
    # dependencies=[Depends(wf_deps.RequireRunReadActiveOrg)] # Basic check on active org context
)
async def get_run_stream(
    run: models.WorkflowRun = Depends(wf_deps.get_workflow_run_for_org),
    db: AsyncSession = Depends(get_async_db_dependency),
    current_user: User = Depends(wf_deps.RequireRunReadActiveOrg),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
    skip: int = Query(0, ge=0, description="Number of events to skip"),
    limit: int = Query(100, ge=1, le=5000, description="Maximum number of events to return"),
):
    """
    Retrieves the sequence of events for a specific workflow run from the event store (e.g., MongoDB).

    Default: reverse chronological order (descending), returns last 100 events.

    - Useful for displaying progress or debugging.
    - Supports pagination using `skip` and `limit`.
    - Requires `run:read` permission on the active organization.
    """
    if not current_user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Workflow run event stream is not accessible to regular users before workflow builder is released.")
    try:
        # Dependency handles fetch and access check
        events = await workflow_service.get_run_stream(db=db, run=run,
                                                    skip=skip, limit=limit,
                                                    user=current_user)
        workflow_logger.info(f"User {current_user.id} retrieved event stream for workflow run {run.id}")
        return events
    # NOTE: internally, 404 is raised if object not found!
    # except exceptions.RunEventsNotFoundException as e:
    #     workflow_logger.warning(f"User {current_user.id} attempted to retrieve events for run with no events: {run.id}")
    #     raise e
    except Exception as e:
        workflow_logger.error(f"Error retrieving workflow run events for run {run.id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while retrieving the workflow run events")


@run_router.post(
    "/{run_id}/cancel",
    response_model=schemas.WorkflowRunRead,
    summary="Cancel Workflow Run",
    # dependencies=[Depends(wf_deps.RequireRunManageActiveOrg)] # Requires RUN_MANAGE permission
)
async def cancel_run(
    run_id: uuid.UUID = Path(..., description="The ID of the workflow run"),
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(get_current_active_verified_user),
    db: AsyncSession = Depends(get_async_db_dependency),
    run_dao: wf_crud.WorkflowRunDAO = Depends(wf_deps.get_workflow_run_dao),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Attempts to cancel a workflow run that is currently `SCHEDULED` or `RUNNING`.

    - Sends a cancellation signal to the Prefect execution engine (best-effort).
    - Updates the run status to `CANCELLED` in the database.
    - Only works for runs in SCHEDULED or RUNNING state.
    - Requires the run to be in the user's active organization.
    
    **Note:** Cancellation is best-effort. The run may still complete if it's already
    finishing or if it doesn't check for cancellation signals.
    """
    try:
        # Fetch the run ensuring it belongs to the active org
        run = await run_dao.get_run_by_id_and_org(db, run_id=run_id, org_id=active_org_id)
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow run not found"
            )
        
        # Validate run state
        if run.status not in [constants.WorkflowRunStatus.SCHEDULED, constants.WorkflowRunStatus.RUNNING]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel run in '{run.status}' state. Only SCHEDULED or RUNNING runs can be cancelled."
            )
        
        # Call service to cancel the run
        cancelled_run = await workflow_service.cancel_run(
            db=db, 
            run=run, 
            user=current_user
        )
        
        workflow_logger.info(
            f"User {current_user.id} cancelled workflow run {run_id} "
            f"(was in '{run.status}' state) for org {active_org_id}"
        )
        
        return cancelled_run
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        workflow_logger.error(
            f"Error cancelling workflow run {run_id} for user {current_user.id}: {str(e)}", 
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while cancelling the workflow run"
        )

# TODO: Add endpoints for Pause / Resume Run if needed

@run_router.delete(
    "/bulk-delete",
    response_model=schemas.WorkflowRunBulkDeleteResponse,
    summary="Bulk Delete Workflow Runs by Time Criteria (Superuser Only)",
    # No permission dependency needed since we check superuser directly
)
async def bulk_delete_workflow_runs(
    delete_request: schemas.WorkflowRunBulkDeleteRequest,
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(get_current_active_superuser),  # Superuser only
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Bulk delete workflow runs based on time criteria with optional filtering.
    
    **Time Filtering (exactly one required):**
    - `last_n_seconds`: Delete runs from last N seconds (1-86400)
    - `last_n_minutes`: Delete runs from last N minutes (1-1440) 
    - `last_n_hours`: Delete runs from last N hours (1-24)
    - `last_n_days`: Delete runs from last N days (1-30)
    
    **Optional Filtering:**
    - `workflow_name`: Filter by workflow name before deletion
    - `status`: Filter by workflow run status before deletion
    
    **User Filtering Options:**
    - `delete_workflow_runs_for_all_users_in_org`: If false, only delete runs for current user or on_behalf_of_user_id
    - `on_behalf_of_user_id`: Delete runs for specific user (requires admin privileges if provided)
    
    **Control Options:**
    - `dry_run`: If true, only returns what would be deleted without actually deleting
    
    **Response includes:**
    - Count of deleted runs
    - Details of each deleted run including parent deletion status, workflow inputs, and triggered user
    - Applied filters for audit purposes
    
    **Access:** Superuser only. If `on_behalf_of_user_id` is provided, user must be admin.
    """
    # Validate on_behalf_of_user_id requires admin privileges
    if delete_request.on_behalf_of_user_id is not None:
        # For future-proofing: if endpoint opens to non-superusers, check admin status
        if not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required to delete runs on behalf of another user"
            )
    
    try:
        result = await workflow_service.bulk_delete_workflow_runs(
            db=db,
            delete_request=delete_request,
            owner_org_id=active_org_id,
            user=current_user
        )
        
        action = "would delete" if delete_request.dry_run else "deleted"
        workflow_logger.info(
            f"User {current_user.id} {action} {result.deleted_count} workflow runs in org {active_org_id} "
            f"with time filter: {result.time_filter_applied}"
        )
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions without logging
        raise
    except Exception as e:
        workflow_logger.error(
            f"Error bulk deleting workflow runs for user {current_user.id} in org {active_org_id}: {str(e)}", 
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail="An error occurred while bulk deleting workflow runs"
        )


# === User Notification Endpoints ===

@notification_router.get(
    "",
    response_model=List[schemas.UserNotificationRead],
    summary="List User Notifications"
    # Permissions: Requires authenticated user. Active org context is used implicitly.
)
async def list_user_notifications(
    query_params: schemas.NotificationListQuery = Depends(),
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(get_current_active_verified_user),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Lists notifications for the current authenticated user within their active organization. Set `get_notifications_for_all_user_orgs=true` to get notifications for all user organizations.

    - Supports filtering by read status (`is_read=true` or `is_read=false`).
    - Supports sorting by `created_at` (default: descending).
    - Supports pagination via `skip` and `limit`.
    """
    try:
        notifications = await workflow_service.list_user_notifications(
            db=db,
            user_id=current_user.id,
            org_id=None if query_params.get_notifications_for_all_user_orgs else active_org_id,
            filters=query_params,
            skip=query_params.skip,
            limit=query_params.limit
        )
        workflow_logger.info(f"User {current_user.id} listed notifications for org {active_org_id}")
        return notifications
    except Exception as e:
        workflow_logger.error(f"Error listing notifications for user {current_user.id} in org {active_org_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while listing notifications")

@notification_router.post(
    "/{notification_id}/read",
    response_model=schemas.UserNotificationRead,
    status_code=status.HTTP_200_OK,
    summary="Mark Notification as Read"
    # Permissions: User must be authenticated. Dependency checks ownership.
)
async def mark_notification_read(
    # Fetch notification ensuring it belongs to the current user using the dependency
    notification: models.UserNotification = Depends(wf_deps.get_notification_for_user),
    current_user: User = Depends(get_current_active_verified_user), # For service call context consistency
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Marks a specific notification as read for the current user.

    - Raises 404 if the notification does not exist or does not belong to the user.
    """
    try:
        # Dependency fetched and verified ownership
        marked_notification = await workflow_service.mark_notification_read(
            db=db,
            notification_id=notification.id,
            user_id=current_user.id
        )
        workflow_logger.info(f"User {current_user.id} marked notification {notification.id} as read")
        return marked_notification
    # NOTE: internally, 404 is raised if object not found!
    # except exceptions.NotificationNotFoundException as e:
    #     workflow_logger.warning(f"User {current_user.id} attempted to mark non-existent notification as read: {notification.id}")
    #     raise e
    except Exception as e:
        workflow_logger.error(f"Error marking notification {notification.id} as read: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while marking the notification as read")

@notification_router.post(
    "/read-all",
    status_code=status.HTTP_200_OK,
    summary="Mark All Notifications as Read"
    # Permissions: User must be authenticated.
)
async def mark_all_notifications_read(
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(get_current_active_verified_user),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Marks all currently unread notifications as read for the current user
    within their active organization.
    """
    try:
        count = await workflow_service.mark_all_notifications_read(
            db=db,
            user_id=current_user.id,
            org_id=active_org_id
        )
        workflow_logger.info(f"User {current_user.id} marked {count} notifications as read in org {active_org_id}")
        return {"message": f"{count} notifications marked as read"}
    except Exception as e:
        workflow_logger.error(f"Error marking all notifications as read for user {current_user.id} in org {active_org_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while marking all notifications as read")

@notification_router.get(
    "/unread-count",
    response_model=int,
    summary="Get Unread Notification Count"
    # Permissions: User must be authenticated.
)
async def get_unread_notification_count(
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(get_current_active_verified_user),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Gets the count of unread notifications for the current user
    in their active organization.
    """
    try:
        count = await workflow_service.count_unread_notifications(
            db=db, user_id=current_user.id, org_id=active_org_id
        )
        workflow_logger.info(f"User {current_user.id} retrieved unread notification count for org {active_org_id}")
        return count
    except Exception as e:
        workflow_logger.error(f"Error getting unread notification count for user {current_user.id} in org {active_org_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while getting the unread notification count")

# === HITL Job Endpoints ===

@hitl_router.get(
    "",
    response_model=List[schemas.HITLJobRead],
    summary="List HITL Jobs"
    # Permissions: Requires authenticated user. Service layer filters based on access rules.
)
async def list_hitl_jobs(
    query_params: schemas.HITLJobListQuery = Depends(),
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(get_current_active_verified_user),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Lists Human-in-the-Loop (HITL) jobs accessible by the current user.

    - By default, lists jobs within the active organization that are either
      assigned to the current user or are unassigned.
    - **Superusers** can list jobs for any organization by providing `owner_org_id`.
    - Supports filtering by:
        - `run_id`: Show jobs for a specific workflow run.
        - `assigned_user_id`: Filter by assignee ('me' for current user, or a specific user UUID).
        - `status`: Filter by job status (e.g., 'pending', 'responded').
        - `pending_only=true`: Shortcut to only show 'pending' jobs.
        - `exclude_cancelled=false`: Set to false to include 'cancelled' jobs (default is true).
    - Supports pagination via `skip` and `limit`.
    """
    try:
        # Service layer handles complex filtering logic, including 'me' translation and superuser checks
        jobs = await workflow_service.list_hitl_jobs(
            db=db,
            owner_org_id=active_org_id, # Pass active org as context
            user=current_user,
            filters=query_params,
            skip=query_params.skip,
            limit=query_params.limit
        )
        workflow_logger.info(f"User {current_user.id} listed HITL jobs for org {active_org_id} with filters: {query_params}")
        return jobs
    # except PermissionDeniedException as e:
    #     workflow_logger.warning(f"Permission denied for user {current_user.id} listing HITL jobs: {str(e)}")
    #     raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        workflow_logger.error(f"Error listing HITL jobs for user {current_user.id} in org {active_org_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while listing HITL jobs")

@hitl_router.get(
    "/{job_id}",
    response_model=schemas.HITLJobRead,
    summary="Get HITL Job Details"
    # Permissions: User must be authenticated. Dependency checks access.
)
async def get_hitl_job(
    # Fetch job ensuring it belongs to active org and user can access it
    job: models.HITLJob = Depends(wf_deps.get_hitl_job_for_org),
    # Re-inject dependencies needed by the service method (if any beyond db)
    current_user: User = Depends(get_current_active_verified_user), # Already injected by get_hitl_job_for_user
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Gets the details of a specific HITL job.

    - Requires the job to be in the active organization and accessible by the user
      (assigned to them or unassigned).
    """
    try:
        # Dependency handles fetch and access checks
        workflow_logger.info(f"User {current_user.id} retrieved HITL job {job.id}")
        return job
    except Exception as e:
        workflow_logger.error(f"Error retrieving HITL job {job.id} for user {current_user.id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while retrieving the HITL job")

# @hitl_router.post(
#     "/{job_id}/respond",
#     response_model=schemas.HITLJobRead,
#     summary="Respond to HITL Job"
#     # Permissions: User must be authenticated. Dependency checks access.
# )
# async def respond_to_hitl_job(
#     response_in: schemas.HITLJobRespond,
#     job: models.HITLJob = Depends(wf_deps.get_hitl_job_for_user), # Fetch job ensuring access
#     current_user: User = Depends(get_current_active_verified_user),
#     db_manager: AsyncGenerator[AsyncSession, None] = Depends(get_async_db_as_manager),
#     workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
# ):
#     """
#     Submits a response to a pending HITL job.

#     - Requires the job to be in `PENDING` status and accessible by the user.
#     - Validates the `response_data` against the job's `response_schema` (if defined).
#     - Updates the job status to `RESPONDED`.
#     - Triggers the resumption of the associated workflow run.
#     """
#     # Dependency handles fetch and access checks
#     # Service handles validation, DB update, and triggering resumption
#     async with db_manager as db:
#         return await workflow_service.respond_to_hitl_job(
#             db=db,
#             job=job,
#             response_data=response_in.response_data,
#             user=current_user
#         )

@hitl_router.post(
    "/{job_id}/cancel",
    response_model=schemas.HITLJobRead,
    summary="Cancel HITL Job"
    # Permissions: Requires access to the job (checked by dependency)
    # TODO: Define specific permission? run:manage? hitl:manage? Assume job access is sufficient for now.
)
async def cancel_hitl_job(
    job: models.HITLJob = Depends(wf_deps.get_hitl_job_for_org), # Fetch job ensuring access
    current_user: User = Depends(get_current_active_verified_user),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency),
):
    """
    Cancels a pending HITL job.

    - Requires the job to be in `PENDING` status and accessible by the user.
    - Updates the job status to `CANCELLED`.
    - Note: This may or may not automatically fail the associated workflow run,
      depending on the workflow's design.
    """
    try:
        updated_job = await workflow_service.cancel_hitl_job(db=db, job=job, user=current_user)
        workflow_logger.info(f"User {current_user.id} cancelled HITL job {job.id}")
        return updated_job
    except ValueError as e:
        # For expected validation errors like job already cancelled
        workflow_logger.warning(f"Invalid attempt to cancel HITL job {job.id} by user {current_user.id}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    # except PermissionDeniedException as e:
    #     workflow_logger.warning(f"Permission denied for user {current_user.id} cancelling HITL job {job.id}: {str(e)}")
    #     raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except Exception as e:
        workflow_logger.error(f"Error cancelling HITL job {job.id} by user {current_user.id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An error occurred while cancelling the HITL job")



# --- Workflow Config Override Routes --- #

@workflow_config_override_router.post(
    "",
    response_model=schemas.WorkflowConfigOverrideRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create workflow configuration override"
)
async def create_config_override(
    override_in: schemas.WorkflowConfigOverrideCreate,
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(get_current_active_verified_user),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency)
):
    """Create a new workflow configuration override."""
    # Only superusers can create system-wide overrides
    if override_in.is_system_entity and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only superusers can create system-wide overrides"
        )
    
    # Users can only create overrides for themselves or their active org (unless superuser)
    if override_in.user_id and override_in.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot create overrides for other users"
        )
    
    if override_in.org_id and override_in.org_id != active_org_id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot create overrides for other organizations"
        )
    
    # Create the override
    return await workflow_service.create_workflow_config_override(
        db=db,
        override_in=override_in,
        user=current_user,
        active_org_id=active_org_id
    )

# Add specialized endpoints for more intuitive API access

@workflow_config_override_router.get(
    "/user",
    response_model=List[schemas.WorkflowConfigOverrideRead],
    summary="List workflow configuration overrides for a specific user"
)
async def list_user_overrides(
    query: Annotated[schemas.UserOverrideListQuery, Query()],
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(get_current_active_verified_user),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency)
):
    """List workflow configuration overrides for a specific user."""
    return await workflow_service.list_user_overrides(
        db=db,
        user_id=query.user_id or current_user.id,
        active_org_id=active_org_id,
        is_active=query.is_active,
        skip=query.skip,
        limit=query.limit,
        requesting_user=current_user
    )

@workflow_config_override_router.get(
    "/organization",
    response_model=List[schemas.WorkflowConfigOverrideRead],
    summary="List workflow configuration overrides for a specific organization"
)
async def list_org_overrides(
    query: Annotated[schemas.OrgOverrideListQuery, Query()],
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(get_current_active_verified_user),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency)
):
    """List workflow configuration overrides for a specific organization."""
    return await workflow_service.list_org_overrides(
        db=db,
        active_org_id=query.org_id or active_org_id,
        is_active=query.is_active,
        skip=query.skip,
        limit=query.limit,
        requesting_user=current_user
    )

@workflow_config_override_router.get(
    "/system",
    response_model=List[schemas.WorkflowConfigOverrideRead],
    summary="List system-wide workflow configuration overrides (Superuser only)"
)
async def list_system_overrides(
    query: Annotated[schemas.SystemOverrideListQuery, Query()],
    current_user: User = Depends(get_current_active_superuser),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency)
):
    """List system-wide workflow configuration overrides. Only accessible to superusers."""
    return await workflow_service.list_system_overrides(
        db=db,
        is_active=query.is_active,
        skip=query.skip,
        limit=query.limit,
        requesting_user=current_user
    )

@workflow_config_override_router.post(
    "/workflow/search", # Changed from GET /workflow and added /search convention
    response_model=List[schemas.WorkflowConfigOverrideRead],
    summary="List workflow configuration overrides for a specific workflow (supports tag list in body)"
)
async def list_workflow_specific_overrides(
    payload: schemas.WorkflowSpecificOverrideListPayload, # Added payload for request body
    workflow_id: Optional[uuid.UUID] = Query(None, description="The ID of the workflow"),
    workflow_name: Optional[str] = Query(None, description="The name of the workflow"),
    workflow_version: Optional[str] = Query(None, description="The version of the workflow"),
    include_active: bool = Query(True, description="Whether to include active overrides"),
    # tag: Optional[str] = Query(None, description="Filter by tag"), # Removed query param tag
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=200, description="Maximum number of items to return"),
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(get_current_active_verified_user),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency)
):
    """List workflow configuration overrides for a specific workflow."""
    if not workflow_id and not workflow_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Either workflow_id or workflow_name must be provided"
        )
        
    # include_tags = [tag] if tag else None # Old logic for single tag query param
    
    overrides, _ = await workflow_service.list_workflow_specific_overrides_and_optional_apply(
        db=db,
        workflow_id=workflow_id,
        workflow_name=workflow_name,
        workflow_version=workflow_version,
        include_active=include_active,
        include_tags=payload.include_tags, # Use include_tags from payload
        active_org_id=active_org_id,
        skip=skip,
        limit=limit,
        requesting_user=current_user
    )

    return overrides

@workflow_config_override_router.get(
    "/tag/{tag}",
    response_model=List[schemas.WorkflowConfigOverrideRead],
    summary="List workflow configuration overrides with a specific tag"
)
async def list_tag_overrides(
    query: Annotated[schemas.TagOverrideListQuery, Query()],
    tag: str = Path(..., description="The tag to filter by"),
    active_org_id: uuid.UUID = Depends(get_active_org_id),
    current_user: User = Depends(get_current_active_verified_user),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency)
):
    """List workflow configuration overrides with a specific tag."""
    return await workflow_service.list_tag_overrides(
        db=db,
        tag=tag,
        is_active=query.is_active,
        active_org_id=active_org_id,
        skip=query.skip,
        limit=query.limit,
        requesting_user=current_user
    )

@workflow_config_override_router.get(
    "/override/{override_id}",
    response_model=schemas.WorkflowConfigOverrideRead,
    summary="Get workflow configuration override by ID"
)
async def get_config_override(
    override_id: uuid.UUID = Path(..., description="The ID of the config override"),
    current_user: User = Depends(get_current_active_verified_user),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency)
):
    """Get a specific workflow configuration override by ID."""
    return await workflow_service.get_workflow_config_override(
        db=db,
        override_id=override_id,
        user=current_user
    )

@workflow_config_override_router.put(
    "/override/{override_id}",
    response_model=schemas.WorkflowConfigOverrideRead,
    summary="Update workflow configuration override"
)
async def update_config_override(
    override_update: schemas.WorkflowConfigOverrideUpdate,
    override_id: uuid.UUID = Path(..., description="The ID of the config override"),
    active_org_id: uuid.UUID = Depends(get_active_org_id), # Added active_org_id
    current_user: User = Depends(get_current_active_verified_user),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency)
):
    """Update a specific workflow configuration override."""
    # First get the override to check permissions
    override = await workflow_service.get_workflow_config_override(
        db=db,
        override_id=override_id,
        user=current_user
    )
    
    # Perform the update
    return await workflow_service.update_workflow_config_override(
        db=db,
        override_id=override_id,
        override_update=override_update,
        user=current_user,
        active_org_id=active_org_id # Pass active_org_id
    )

@workflow_config_override_router.delete(
    "/override/{override_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete workflow configuration override"
)
async def delete_config_override(
    override_id: uuid.UUID = Path(..., description="The ID of the config override"),
    active_org_id: uuid.UUID = Depends(get_active_org_id), # Added active_org_id
    current_user: User = Depends(get_current_active_verified_user),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency)
):
    """Delete a specific workflow configuration override."""
    # First get the override to check permissions
    override = await workflow_service.get_workflow_config_override(
        db=db,
        override_id=override_id,
        user=current_user
    )
    
    # Delete the override
    await workflow_service.delete_workflow_config_override(
        db=db,
        override_id=override_id,
        user=current_user,
        active_org_id=active_org_id # Pass active_org_id
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@workflow_config_override_router.get(
    "/all",
    response_model=List[schemas.WorkflowConfigOverrideRead],
    summary="List all workflow configuration overrides (superuser only)"
)
async def get_all_workflow_config_overrides(
    skip: int = Query(0, ge=0, description="Number of items to skip"),
    limit: int = Query(100, ge=1, le=200, description="Maximum number of items to return"),
    current_user: User = Depends(get_current_active_superuser),
    db: AsyncSession = Depends(get_async_db_dependency),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency)
):
    """
    List all workflow configuration overrides across the system.
    Only accessible to superusers.
    """
    overrides = await workflow_service.get_all_workflow_config_overrides(
        db=db,
        skip=skip,
        limit=limit,
        requesting_user=current_user
    )
    return overrides

# --- Chat Thread Router --- #



@chat_thread_router.post(
    "",
    response_model=schemas.ChatThreadRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new chat thread"
)
async def create_chat_thread(
    thread_data: schemas.ChatThreadCreate,
    db: AsyncSession = Depends(get_async_db_dependency),
    current_user: User = Depends(get_current_active_verified_user),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency)
):
    """
    Create a new chat thread.
    
    The chat thread can be associated with a workflow by providing the workflow_name
    and optionally the workflow_version.
    
    Only superusers can create threads owned by other users by providing user_id.
    Non-superusers will always create threads with themselves as the owner.
    """
    try:
        # Check permission for non-superusers trying to create threads for others
        if not current_user.is_superuser and thread_data.user_id is not None and thread_data.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only superusers can create threads for other users"
            )
        
        # Create the thread
        return await workflow_service.create_chat_thread(
            db=db, 
            thread_data=thread_data,
            user=current_user
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating chat thread: {str(e)}"
        )

@chat_thread_router.get(
    "",
    response_model=List[schemas.ChatThreadRead],
    summary="List chat threads"
)
async def list_chat_threads(
    query_params: Annotated[schemas.ChatThreadListQuery, Query()],
    db: AsyncSession = Depends(get_async_db_dependency),
    current_user: User = Depends(get_current_active_verified_user),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency)
):
    """
    List chat threads with optional filtering by workflow name, version, and tag.
    
    Regular users can only see their own threads.
    Superusers can see all threads or filter by user_id.
    
    Results are paginated using skip and limit parameters.
    """
    try:
        # Non-superusers can only see their own threads
        if not current_user.is_superuser and query_params.user_id is not None and query_params.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only view your own chat threads"
            )
        
        # Get threads
        return await workflow_service.list_chat_threads(
            db=db,
            user=current_user,
            workflow_name=query_params.workflow_name,
            workflow_version=query_params.workflow_version,
            user_id=query_params.user_id,
            tag=query_params.tag,
            skip=query_params.skip,
            limit=query_params.limit
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing chat threads: {str(e)}"
        )

@chat_thread_router.get(
    "/{thread_id}",
    response_model=schemas.ChatThreadRead,
    summary="Get a chat thread by ID"
)
async def get_chat_thread(
    thread_id: uuid.UUID = Path(..., description="The ID of the chat thread"),
    db: AsyncSession = Depends(get_async_db_dependency),
    current_user: User = Depends(get_current_active_verified_user),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency)
):
    """
    Get a chat thread by its ID.
    
    Users can only access their own threads, unless they are superusers.
    Superusers can access any thread.
    
    Returns a 404 error if the thread doesn't exist or the user doesn't have access.
    """
    try:
        # Get the thread first to check access
        thread = await workflow_service.get_chat_thread(
            db=db, 
            thread_id=thread_id,
            user=current_user
        )
        
        if not thread:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat thread not found"
            )
        
        # Check permission for non-superusers trying to access others' threads
        if not current_user.is_superuser and thread.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this chat thread"
            )
        
        return thread
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving chat thread: {str(e)}"
        )

@chat_thread_router.put(
    "/{thread_id}",
    response_model=schemas.ChatThreadRead,
    summary="Update a chat thread"
)
async def update_chat_thread(
    thread_data: schemas.ChatThreadUpdate,
    thread_id: uuid.UUID = Path(..., description="The ID of the chat thread"),
    db: AsyncSession = Depends(get_async_db_dependency),
    current_user: User = Depends(get_current_active_verified_user),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency)
):
    """
    Update a chat thread.
    
    Users can only update their own threads, unless they are superusers.
    Superusers can update any thread.
    
    Returns a 404 error if the thread doesn't exist or the user doesn't have access.
    """
    try:
        # Get the thread first to check permissions
        thread = await workflow_service.get_chat_thread(
            db=db, 
            thread_id=thread_id,
            user=current_user
        )
        
        if not thread:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat thread not found"
            )
        
        # Check permission for non-superusers trying to update others' threads
        if not current_user.is_superuser and thread.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to update this chat thread"
            )
        
        # Update the thread
        updated_thread = await workflow_service.update_chat_thread(
            db=db, 
            thread_id=thread_id, 
            thread_data=thread_data,
            user=current_user
        )
        
        return updated_thread
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating chat thread: {str(e)}"
        )

@chat_thread_router.delete(
    "/{thread_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a chat thread"
)
async def delete_chat_thread(
    thread_id: uuid.UUID = Path(..., description="The ID of the chat thread"),
    db: AsyncSession = Depends(get_async_db_dependency),
    current_user: User = Depends(get_current_active_verified_user),
    workflow_service: services.WorkflowService = Depends(wf_deps.get_workflow_service_dependency)
):
    """
    Delete a chat thread.
    
    Users can only delete their own threads, unless they are superusers.
    Superusers can delete any thread.
    
    Returns a 404 error if the thread doesn't exist or the user doesn't have access.
    """
    try:
        # Get the thread first to check permissions
        thread = await workflow_service.get_chat_thread(
            db=db, 
            thread_id=thread_id,
            user=current_user
        )
        
        if not thread:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat thread not found"
            )
        
        # Check permission for non-superusers trying to delete others' threads
        if not current_user.is_superuser and thread.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to delete this chat thread"
            )
        
        # Delete the thread
        success = await workflow_service.delete_chat_thread(
            db=db, 
            thread_id=thread_id,
            user=current_user
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat thread not found"
            )
        
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting chat thread: {str(e)}"
        )

# === Asset Endpoints ===

@asset_router.get("/types", response_model=List[schemas.AssetTypeInfo])
async def list_asset_types(
    current_user: User = Depends(get_current_active_verified_user),
    asset_service: services.AssetService = Depends(wf_deps.get_asset_service_dependency),
):
    """List all available asset types and their schemas."""
    return asset_service.list_asset_types()


@asset_router.get("/types/{asset_type}", response_model=schemas.AssetTypeInfo)
async def get_asset_type_info(
    asset_type: str = Path(..., description="Asset type identifier"),
    current_user: User = Depends(get_current_active_verified_user),
    asset_service: services.AssetService = Depends(wf_deps.get_asset_service_dependency),
):
    """Get information about a specific asset type including its schema."""
    return asset_service.get_asset_type_info(asset_type)


@asset_router.post("", response_model=schemas.AssetRead, dependencies=[Depends(wf_deps.RequireAssetCreateActiveOrg)])
async def create_asset(
    asset_in: schemas.AssetCreate,
    db: AsyncSession = Depends(get_async_db_dependency),
    current_user: User = Depends(get_current_active_verified_user),
    org_id: uuid.UUID = Depends(get_active_org_id),
    asset_service: services.AssetService = Depends(wf_deps.get_asset_service_dependency),
):
    """Create a new asset."""
    return await asset_service.create_asset(
        db,
        user=current_user,
        asset_in=asset_in,
        org_id=org_id,
        is_superuser=current_user.is_superuser
    )


@asset_router.get("/{asset_id}", response_model=schemas.AssetRead, dependencies=[Depends(wf_deps.RequireAssetReadActiveOrg)])
async def get_asset(
    asset_id: uuid.UUID = Path(..., description="Asset ID"),
    query: schemas.AssetReadQuery = Depends(),
    db: AsyncSession = Depends(get_async_db_dependency),
    current_user: User = Depends(get_current_active_verified_user),
    org_id: uuid.UUID = Depends(get_active_org_id),
    asset_service: services.AssetService = Depends(wf_deps.get_asset_service_dependency),
):
    """Get an asset by ID."""
    return await asset_service.get_asset(
        db,
        user=current_user,
        asset_id=asset_id,
        org_id=org_id,
        is_superuser=current_user.is_superuser,
        app_data_fields=query.app_data_fields
    )


@asset_router.patch("/{asset_id}", response_model=schemas.AssetRead, dependencies=[Depends(wf_deps.RequireAssetUpdateActiveOrg)])
async def update_asset(
    asset_id: uuid.UUID = Path(..., description="Asset ID"),
    asset_update: schemas.AssetUpdate = Body(...),
    db: AsyncSession = Depends(get_async_db_dependency),
    current_user: User = Depends(get_current_active_verified_user),
    org_id: uuid.UUID = Depends(get_active_org_id),
    asset_service: services.AssetService = Depends(wf_deps.get_asset_service_dependency),
):
    """Update an asset."""
    return await asset_service.update_asset(
        db,
        user=current_user,
        asset_id=asset_id,
        asset_update=asset_update,
        org_id=org_id,
        is_superuser=current_user.is_superuser
    )


@asset_router.patch("/{asset_id}/app-data", response_model=schemas.AssetRead, dependencies=[Depends(wf_deps.RequireAssetUpdateActiveOrg)])
async def update_asset_app_data(
    asset_id: uuid.UUID = Path(..., description="Asset ID"),
    app_data_update: schemas.AssetAppDataUpdate = Body(...),
    db: AsyncSession = Depends(get_async_db_dependency),
    current_user: User = Depends(get_current_active_verified_user),
    org_id: uuid.UUID = Depends(get_active_org_id),
    asset_service: services.AssetService = Depends(wf_deps.get_asset_service_dependency),
):
    """Update asset app_data using JSONB operations. 
    
    If the asset is shared, any user can update the asset app_data. eg. if asset app data includes onboarding, any org user can complete the onboarding and update the asset app data.
    If the asset is not shared, only the managing user can update the asset app_data.
    """
    return await asset_service.update_asset_app_data(
        db,
        user=current_user,
        asset_id=asset_id,
        app_data_update=app_data_update,
        org_id=org_id,
        is_superuser=current_user.is_superuser
    )


@asset_router.patch("/{asset_id}/app-data/increment", response_model=schemas.AssetRead, dependencies=[Depends(wf_deps.RequireAssetUpdateActiveOrg)])
async def increment_asset_app_data_field(
    asset_id: uuid.UUID = Path(..., description="Asset ID"),
    increment_data: schemas.AssetAppDataIncrement = Body(...),
    db: AsyncSession = Depends(get_async_db_dependency),
    current_user: User = Depends(get_current_active_verified_user),
    org_id: uuid.UUID = Depends(get_active_org_id),
    asset_service: services.AssetService = Depends(wf_deps.get_asset_service_dependency),
):
    """Atomically increment a numeric field in asset app_data. NOTE: increment can be any float/int, including negative values.
    
    If the asset is shared, any user can update the asset app_data. eg. if asset app data includes onboarding, any org user can complete the onboarding and update the asset app data.
    If the asset is not shared, only the managing user can update the asset app_data.
    """
    return await asset_service.increment_asset_app_data_field(
        db,
        user=current_user,
        asset_id=asset_id,
        path=increment_data.path,
        increment=increment_data.increment,
        org_id=org_id
    )


@asset_router.post("/{asset_id}/deactivate", response_model=schemas.AssetRead, dependencies=[Depends(wf_deps.RequireAssetDeleteActiveOrg)])
async def deactivate_asset(
    asset_id: uuid.UUID = Path(..., description="Asset ID"),
    db: AsyncSession = Depends(get_async_db_dependency),
    current_user: User = Depends(get_current_active_verified_user),
    org_id: uuid.UUID = Depends(get_active_org_id),
    asset_service: services.AssetService = Depends(wf_deps.get_asset_service_dependency),
):
    """Deactivate an asset (soft delete).
    
    Can be deactivated by Org admins, superusers, or the asset managing user.
    NOTE: only managing user can activate the asset again!
    """
    return await asset_service.deactivate_asset(
        db,
        user=current_user,
        asset_id=asset_id,
        org_id=org_id,
        is_superuser=current_user.is_superuser
    )


@asset_router.get("/managed", response_model=List[schemas.AssetRead], dependencies=[Depends(wf_deps.RequireAssetReadActiveOrg)])
async def list_managed_assets(
    query: schemas.AssetListQuery = Depends(),
    db: AsyncSession = Depends(get_async_db_dependency),
    current_user: User = Depends(get_current_active_verified_user),
    org_id: uuid.UUID = Depends(get_active_org_id),
    asset_service: services.AssetService = Depends(wf_deps.get_asset_service_dependency),
):
    """
    List all assets managed by the current user.
    
    If include_all_orgs is True, returns assets from all organizations.
    Otherwise, returns assets only from the current organization.
    """
    # Handle 'me' for managing_user_id
    on_behalf_of_user_id = None
    if query.managing_user_id and str(query.managing_user_id) != "me":
        on_behalf_of_user_id = query.managing_user_id
    
    return await asset_service.list_managed_assets(
        db,
        user=current_user,
        org_id=org_id if not query.include_all_orgs else None,
        asset_type=query.asset_type,
        is_active=query.is_active,
        include_all_orgs=query.include_all_orgs,
        skip=query.skip,
        limit=query.limit,
        is_superuser=current_user.is_superuser,
        on_behalf_of_user_id=on_behalf_of_user_id,
        app_data_fields=query.app_data_fields,
        sort_by=query.sort_by,
        sort_order=query.sort_order
    )


@asset_router.get("/org/all", response_model=List[schemas.AssetRead], dependencies=[Depends(wf_deps.RequireAssetUpdateActiveOrg)])
async def list_all_org_assets(
    query: schemas.AssetListQuery = Depends(),
    db: AsyncSession = Depends(get_async_db_dependency),
    current_user: User = Depends(get_current_active_verified_user),
    org_id: uuid.UUID = Depends(get_active_org_id),
    asset_service: services.AssetService = Depends(wf_deps.get_asset_service_dependency),
):
    """
    List all assets in an organization.
    
    Requires ORG_DATA_WRITE permission.
    """
    # Handle superuser override for org_id
    effective_org_id = query.org_id if query.org_id and current_user.is_superuser else org_id
    
    return await asset_service.list_all_org_assets(
        db,
        user=current_user,
        org_id=effective_org_id,
        asset_type=query.asset_type,
        is_active=query.is_active,
        is_shared=query.is_shared,
        skip=query.skip,
        limit=query.limit,
        is_superuser=current_user.is_superuser,
        app_data_fields=query.app_data_fields,
        sort_by=query.sort_by,
        sort_order=query.sort_order
    )


@asset_router.get("", response_model=List[schemas.AssetRead], dependencies=[Depends(wf_deps.RequireAssetReadActiveOrg)])
async def list_assets(
    query: schemas.AssetListQuery = Depends(),
    db: AsyncSession = Depends(get_async_db_dependency),
    current_user: User = Depends(get_current_active_verified_user),
    org_id: uuid.UUID = Depends(get_active_org_id),
    asset_service: services.AssetService = Depends(wf_deps.get_asset_service_dependency),
):
    """
    List assets accessible to the current user within an organization.
    
    An asset is accessible if:
    - The user is the managing user, OR
    - The asset is shared within the organization
    """
    # Handle superuser override for org_id
    effective_org_id = query.org_id if query.org_id and current_user.is_superuser else org_id
    
    return await asset_service.list_accessible_assets(
        db,
        user=current_user,
        org_id=effective_org_id,
        asset_type=query.asset_type,
        is_active=query.is_active,
        is_shared=query.is_shared,
        skip=query.skip,
        limit=query.limit,
        is_superuser=current_user.is_superuser,
        app_data_fields=query.app_data_fields,
        sort_by=query.sort_by,
        sort_order=query.sort_order
    )


# === UserAppResumeMetadata Endpoints ===

@user_app_resume_router.post("", response_model=schemas.UserAppResumeMetadataRead, dependencies=[Depends(wf_deps.RequireOrgDataWriteActiveOrg)])
async def create_user_app_resume_metadata(
    metadata_in: schemas.UserAppResumeMetadataCreate,
    db: AsyncSession = Depends(get_async_db_dependency),
    current_user: User = Depends(get_current_active_verified_user),
    org_id: uuid.UUID = Depends(get_active_org_id),
    asset_service: services.AssetService = Depends(wf_deps.get_asset_service_dependency),
):
    """Create a new user app resume metadata record."""
    return await asset_service.create_user_app_resume_metadata(
        db,
        user=current_user,
        metadata_in=metadata_in,
        org_id=org_id,
        is_superuser=current_user.is_superuser
    )


@user_app_resume_router.get("/{metadata_id}", response_model=schemas.UserAppResumeMetadataRead, dependencies=[Depends(wf_deps.RequireOrgDataReadActiveOrg)])
async def get_user_app_resume_metadata(
    metadata_id: uuid.UUID = Path(..., description="Metadata ID"),
    db: AsyncSession = Depends(get_async_db_dependency),
    current_user: User = Depends(get_current_active_verified_user),
    asset_service: services.AssetService = Depends(wf_deps.get_asset_service_dependency),
):
    """Get a user app resume metadata record by ID."""
    return await asset_service.get_user_app_resume_metadata(
        db,
        user=current_user,
        metadata_id=metadata_id,
        is_superuser=current_user.is_superuser
    )


@user_app_resume_router.patch("/{metadata_id}", response_model=schemas.UserAppResumeMetadataRead, dependencies=[Depends(wf_deps.RequireOrgDataWriteActiveOrg)])
async def update_user_app_resume_metadata(
    metadata_id: uuid.UUID = Path(..., description="Metadata ID"),
    metadata_update: schemas.UserAppResumeMetadataUpdate = Body(...),
    db: AsyncSession = Depends(get_async_db_dependency),
    current_user: User = Depends(get_current_active_verified_user),
    asset_service: services.AssetService = Depends(wf_deps.get_asset_service_dependency),
):
    """Update a user app resume metadata record."""
    return await asset_service.update_user_app_resume_metadata(
        db,
        user=current_user,
        metadata_id=metadata_id,
        metadata_update=metadata_update,
        is_superuser=current_user.is_superuser
    )


@user_app_resume_router.delete("/{metadata_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(wf_deps.RequireOrgDataWriteActiveOrg)])
async def delete_user_app_resume_metadata(
    metadata_id: uuid.UUID = Path(..., description="Metadata ID"),
    db: AsyncSession = Depends(get_async_db_dependency),
    current_user: User = Depends(get_current_active_verified_user),
    asset_service: services.AssetService = Depends(wf_deps.get_asset_service_dependency),
):
    """Delete a user app resume metadata record."""
    success = await asset_service.delete_user_app_resume_metadata(
        db,
        user=current_user,
        metadata_id=metadata_id,
        is_superuser=current_user.is_superuser
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User app resume metadata not found"
        )
    return None


@user_app_resume_router.get("", response_model=List[schemas.UserAppResumeMetadataRead], dependencies=[Depends(wf_deps.RequireOrgDataReadActiveOrg)])
async def list_user_app_resume_metadata(
    query: schemas.UserAppResumeMetadataListQuery = Depends(),
    db: AsyncSession = Depends(get_async_db_dependency),
    current_user: User = Depends(get_current_active_verified_user),
    org_id: uuid.UUID = Depends(get_active_org_id),
    asset_service: services.AssetService = Depends(wf_deps.get_asset_service_dependency),
):
    """List user app resume metadata records with filters."""
    # Handle superuser overrides
    effective_org_id = query.org_id if query.org_id and current_user.is_superuser else org_id
    effective_user_id = query.user_id if query.user_id and current_user.is_superuser else None
    
    return await asset_service.list_user_app_resume_metadata(
        db,
        user=current_user,
        org_id=effective_org_id,
        workflow_name=query.workflow_name,
        asset_id=query.asset_id,
        entity_tag=query.entity_tag,
        frontend_stage=query.frontend_stage,
        run_id=query.run_id,
        user_id=effective_user_id,
        skip=query.skip,
        limit=query.limit,
        is_superuser=current_user.is_superuser
    )


# Don't forget to include the router in the main application
# In services/kiwi_app/main.py or wherever you include your routers:
# app.include_router(workflow_app.routes.chat_thread_router, prefix="/api/v1/workflow")
# app.include_router(workflow_app.routes.asset_router, prefix="/api/v1/workflow")
# app.include_router(workflow_app.routes.user_app_resume_router, prefix="/api/v1/workflow")
