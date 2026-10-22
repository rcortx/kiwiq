"""
Generic Weaviate Base Client

This module provides a generic async client for Weaviate operations that can be
extended for specific use cases. It handles:
- Connection management (async and sync clients)
- Generic schema creation and management
- Generic batch operations with callbacks
- Generic search capabilities (vector, keyword, hybrid)
- Filter utilities

This base client is designed to be extended by specific implementations
(e.g., DocChunk client) which define their own schemas and business logic.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import weaviate
from weaviate import WeaviateAsyncClient, WeaviateClient
from weaviate.classes.config import Configure, Property
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.util import generate_uuid5

from global_config.settings import global_settings

# Configure logging
logger = logging.getLogger(__name__)

INTEGRATION_HEADER = "X-Weaviate-Client-Integration"


def with_integration_header(headers: Optional[Dict[str, str]]) -> Dict[str, str]:
    """
    Return a copy of ``headers`` that also identifies KiwiQ to Weaviate.

    Weaviate (>= 1.38) reads ``X-Weaviate-Client-Integration: {name}/{version}`` to
    attribute traffic to the integration that sent it; older servers ignore it. Headers
    passed at client creation reach both the REST and the gRPC transport. A header the
    caller already set (any casing) wins, and failing to build the value never breaks
    the connection.
    """
    merged = dict(headers or {})
    try:
        if not any(key.lower() == INTEGRATION_HEADER.lower() for key in merged):
            version = str(global_settings.VERSION or "").strip() or "unknown"
            merged[INTEGRATION_HEADER] = f"kiwiq/{version}"
    except Exception:  # telemetry must never break the connection
        logger.debug("Could not add the Weaviate integration header", exc_info=True)
    return merged


class WeaviateBaseClient:
    """
    Generic async client for Weaviate operations.

    This base class provides:
    - Automatic connection management (async and sync)
    - Generic schema setup with configurable properties
    - Generic batch ingestion with UUID callbacks
    - Efficient deletion operations
    - Vector, keyword, and hybrid search capabilities
    - Generic filter building utilities

    Note: Per Weaviate recommendations, batch operations use the synchronous client
    as it's already optimized for concurrent requests.

    This class should be extended by specific implementations that define:
    - Collection-specific schemas
    - Business logic for data transformation
    - Collection-specific filter builders
    """

    def __init__(
        self,
        url: Optional[str] = None,
        host: Optional[str] = None,
        api_key: Optional[str] = None,
        additional_headers: Optional[Dict[str, str]] = None,
        collection_name: str = "GenericCollection",
        batch_size: int = 200,
        delete_batch_size: int = 500,
    ):
        """
        Initialize the Weaviate base client.

        Args:
            url: Weaviate instance URL
            host: Weaviate host (for local connections)
            api_key: Optional API key for authentication
            additional_headers: Optional additional headers
            collection_name: Name of the collection
            batch_size: Size of each batch for ingestion (default: 200)
            delete_batch_size: Size of each batch for deletion (default: 500)
        """
        # Use provided values or fall back to global settings
        self.url = url or global_settings.WEAVIATE_URL
        self.host = host or global_settings.WEAVIATE_HOST
        self.api_key = api_key or global_settings.WEAVIATE_API_KEY

        # Setup headers
        self.additional_headers = additional_headers or {}

        self.collection_name = collection_name
        self.batch_size = batch_size
        self.delete_batch_size = delete_batch_size

        self.client: Optional[WeaviateAsyncClient] = None
        self.sync_client: Optional[WeaviateClient] = None  # For batch operations

        logger.info(f"Initialized WeaviateBaseClient for collection '{self.collection_name}' at {self.url or self.host}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def connect(self) -> None:
        """
        Establish connection to Weaviate.

        Creates both async and sync clients. The sync client is used for
        batch operations as recommended by Weaviate documentation.

        Raises:
            Exception: If connection fails
        """
        try:
            headers = with_integration_header(self.additional_headers)

            # Create async client for most operations
            if self.api_key and self.url:
                self.client = weaviate.use_async_with_weaviate_cloud(
                    cluster_url=self.url,
                    auth_credentials=weaviate.auth.Auth.api_key(self.api_key),
                    headers=headers,
                )
            else:
                self.client = weaviate.use_async_with_local(
                    host=self.host,
                    headers=headers,
                )

            await self.client.connect()

            # Create sync client for batch operations
            if self.api_key and self.url:
                self.sync_client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=self.url,
                    auth_credentials=weaviate.auth.Auth.api_key(self.api_key),
                    headers=headers,
                )
            else:
                self.sync_client = weaviate.connect_to_local(
                    host=self.host,
                    headers=headers,
                )

            logger.info("Successfully connected to Weaviate (async and sync clients)")

        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise Exception(f"Connection failed: {e}")

    async def close(self) -> None:
        """Close the Weaviate connections."""
        if self.client:
            await self.client.close()
        if self.sync_client:
            self.sync_client.close()
        logger.info("Closed Weaviate connections")

    async def collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of collection to check (defaults to self.collection_name)

        Returns:
            bool: True if collection exists
        """
        if not self.client:
            raise RuntimeError("Client not connected. Call connect() first.")

        name = collection_name or self.collection_name
        collections = self.client.collections
        return await collections.exists(name)

    async def delete_collection(self, collection_name: Optional[str] = None) -> None:
        """
        Delete a collection.

        Args:
            collection_name: Name of collection to delete (defaults to self.collection_name)
        """
        if not self.client:
            raise RuntimeError("Client not connected. Call connect() first.")

        name = collection_name or self.collection_name
        collections = self.client.collections

        if await collections.exists(name):
            await collections.delete(name)
            logger.info(f"Deleted collection: {name}")
        else:
            logger.warning(f"Collection does not exist: {name}")

    async def setup_schema(
        self,
        properties: List[Property],
        vectorizer_config: Any,
        description: str = "",
        vector_index_config: Optional[Any] = None,
        inverted_index_config: Optional[Any] = None,
        recreate: bool = False,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Setup or validate a collection schema.

        This is a generic method that can be called by collection-specific clients
        with their own property definitions and configurations.

        Args:
            properties: List of Property definitions for the schema
            vectorizer_config: Vectorizer configuration (e.g., Configure.Vectorizer.text2vec_openai(...))
            description: Collection description
            vector_index_config: Optional vector index configuration
            inverted_index_config: Optional inverted index configuration
            recreate: If True, delete and recreate the collection
            collection_name: Collection name (defaults to self.collection_name)
        """
        if not self.client:
            raise RuntimeError("Client not connected. Call connect() first.")

        name = collection_name or self.collection_name
        collections = self.client.collections

        # Check if collection exists
        if await collections.exists(name):
            if recreate:
                logger.warning(f"Recreating collection {name}")
                await collections.delete(name)
            else:
                logger.info(f"Collection {name} already exists")
                return

        # Create collection with schema
        logger.info(f"Creating collection {name}")

        create_kwargs = {
            "name": name,
            "description": description or f"Collection: {name}",
            "vectorizer_config": vectorizer_config,
            "properties": properties,
        }

        if vector_index_config:
            create_kwargs["vector_index_config"] = vector_index_config

        if inverted_index_config:
            create_kwargs["inverted_index_config"] = inverted_index_config

        await collections.create(**create_kwargs)

        logger.info(f"Successfully created collection {name}")

    async def ingest_objects(
        self,
        objects: List[Dict[str, Any]],
        uuid_generator: Callable[[Dict[str, Any]], str],
        property_transformer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        generate_vectors: bool = True,
        collection_name: Optional[str] = None,
    ) -> List[str]:
        """
        Ingest objects in batches using the sync client for optimal performance.

        This is a generic ingestion method that uses callbacks for UUID generation
        and property transformation, making it reusable across different collection types.

        Args:
            objects: List of object dictionaries
            uuid_generator: Callback function that generates UUID from an object
            property_transformer: Optional callback to transform properties before ingestion
            generate_vectors: Whether to generate vectors (default: True)
            collection_name: Collection name (defaults to self.collection_name)

        Returns:
            List[str]: List of generated UUIDs for the objects

        Note:
        - Uses sync client for batch operations as recommended by Weaviate
        - Handles timezone conversion for dates automatically
        """
        return await asyncio.to_thread(
            self._ingest_objects_sync,
            objects,
            uuid_generator,
            property_transformer,
            generate_vectors,
            collection_name,
        )

    def _ingest_objects_sync(
        self,
        objects: List[Dict[str, Any]],
        uuid_generator: Callable[[Dict[str, Any]], str],
        property_transformer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]],
        generate_vectors: bool,
        collection_name: Optional[str],
    ) -> List[str]:
        """
        Synchronous implementation of object ingestion.

        This is called from ingest_objects via asyncio.to_thread.
        """
        if not self.sync_client:
            raise RuntimeError("Sync client not connected. Call connect() first.")

        name = collection_name or self.collection_name
        collection = self.sync_client.collections.get(name)
        uuids = []

        # Process objects in batches using sync client's batch functionality
        with collection.batch.fixed_size(batch_size=self.batch_size) as batch:
            for obj in objects:
                # Generate UUID
                obj_uuid = uuid_generator(obj)
                uuids.append(obj_uuid)

                # Transform properties if transformer provided
                if property_transformer:
                    properties = property_transformer(obj)
                else:
                    properties = obj.copy()

                # Determine if a custom vector should be used
                vector = obj.get("vector") if not generate_vectors else None

                # Add object to the batch
                batch.add_object(
                    properties=properties,
                    uuid=obj_uuid,
                    vector=vector,
                )

        # Check for failed objects
        if hasattr(collection.batch, 'failed_objects') and collection.batch.failed_objects:
            logger.error(f"Batch insertion errors: {len(collection.batch.failed_objects)} objects failed")
            for error in collection.batch.failed_objects[:5]:  # Log first 5 errors
                logger.error(f"Failed object: {error}")

        logger.info(f"Successfully ingested {len(objects)} objects into {name}")
        return uuids

    async def delete_many(
        self,
        where_filter: Filter,
        collection_name: Optional[str] = None,
    ) -> Tuple[int, int, int]:
        """
        Delete multiple objects matching a filter.

        Args:
            where_filter: Filter criteria for deletion
            collection_name: Collection name (defaults to self.collection_name)

        Returns:
            Tuple[int, int, int]: (failed, matched, successful)
        """
        if not self.client:
            raise RuntimeError("Client not connected. Call connect() first.")

        name = collection_name or self.collection_name
        collection = self.client.collections.get(name)

        try:
            result = await collection.data.delete_many(where=where_filter)

            failed = result.failed if hasattr(result, 'failed') else 0
            matched = result.matches if hasattr(result, 'matches') else 0
            successful = result.successful if hasattr(result, 'successful') else 0

            logger.debug(f"Delete result - failed: {failed}, matched: {matched}, successful: {successful}")

            if matched > 0:
                logger.info(f"Deleted {matched} objects from {name}")
            else:
                logger.info(f"No objects found to delete in {name}")

            return failed, matched, successful

        except Exception as e:
            logger.error(f"Failed to delete objects: {e}")
            raise

    # ============================================================================
    # Generic Search Methods
    # ============================================================================

    async def vector_search(
        self,
        query_vector: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
        where_filter: Optional[Filter] = None,
        return_properties: Optional[List[str]] = None,
        include_vector: bool = False,
        target_vector: Optional[Union[List[str], str]] = None,
        return_metadata: bool = True,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.

        Args:
            query_vector: Query vector (if None, query_text must be provided)
            query_text: Query text for vectorization
            limit: Maximum results to return
            offset: Number of results to skip before applying the limit
            where_filter: Optional prefilter
            return_properties: Properties to return (None = all)
            include_vector: Whether to include vectors in results
            target_vector: Optional target vector field(s) - only use with named vectors
            return_metadata: Whether to return metadata
            collection_name: Collection to search (defaults to self.collection_name)

        Returns:
            List[Dict[str, Any]]: Search results

        Note: target_vector should only be used when the collection is configured
        with named vectors. For default single vector space, leave as None.
        """
        if not self.client:
            raise RuntimeError("Client not connected. Call connect() first.")

        if query_vector is None and query_text is None:
            raise ValueError("Either query_vector or query_text must be provided")

        name = collection_name or self.collection_name
        collection = self.client.collections.get(name)

        # Build query kwargs
        base_kwargs = {
            "limit": limit,
            "offset": offset,
        }
        if where_filter:
            base_kwargs["filters"] = where_filter
        if return_properties:
            base_kwargs["return_properties"] = return_properties
        if include_vector:
            base_kwargs["include_vector"] = include_vector
        if target_vector is not None:
            base_kwargs["target_vector"] = target_vector

        # Execute query
        try:
            if query_vector:
                result = await collection.query.near_vector(
                    near_vector=query_vector,
                    return_metadata=MetadataQuery.full() if return_metadata else None,
                    **base_kwargs
                )
            else:
                logger.debug(f"Performing vector search with text: '{query_text}', limit: {limit}, offset: {offset}")
                result = await collection.query.near_text(
                    query=query_text,
                    return_metadata=MetadataQuery.full() if return_metadata else None,
                    **base_kwargs
                )

        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            raise

        # Convert to dictionaries
        formatted_results = self._format_results(result.objects)
        if not formatted_results and query_text:
            logger.warning(f"Vector search returned no results for query: {query_text}")
        return formatted_results

    async def keyword_search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        where_filter: Optional[Filter] = None,
        return_properties: Optional[List[str]] = None,
        bm25_properties: Optional[List[str]] = None,
        return_metadata: bool = True,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query
            limit: Maximum results to return
            offset: Number of results to skip before applying the limit
            where_filter: Optional prefilter
            return_properties: Properties to return
            bm25_properties: Properties to search (implementation-specific)
            return_metadata: Whether to return metadata
            collection_name: Collection to search (defaults to self.collection_name)

        Returns:
            List[Dict[str, Any]]: Search results
        """
        if not self.client:
            raise RuntimeError("Client not connected. Call connect() first.")

        name = collection_name or self.collection_name
        collection = self.client.collections.get(name)

        kwargs = {
            "query": query,
            "limit": limit,
            "offset": offset,
        }
        if where_filter:
            kwargs["filters"] = where_filter
        if return_properties:
            kwargs["return_properties"] = return_properties
        if bm25_properties:
            kwargs["query_properties"] = bm25_properties

        try:
            logger.debug(f"Performing BM25 search with query: '{query}', properties: {bm25_properties}")
            result = await collection.query.bm25(
                return_metadata=MetadataQuery.full() if return_metadata else None,
                **kwargs
            )

        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            raise

        return self._format_results(result.objects)

    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        alpha: Optional[float] = None,
        where_filter: Optional[Filter] = None,
        return_properties: Optional[List[str]] = None,
        vector: Optional[List[float]] = None,
        return_metadata: bool = True,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and keyword search.

        Args:
            query: Search query
            limit: Maximum results to return
            offset: Number of results to skip before applying the limit
            alpha: Balance between vector (1.0) and keyword (0.0) search
            where_filter: Optional prefilter
            return_properties: Properties to return
            vector: Optional pre-computed query vector
            return_metadata: Whether to return metadata
            collection_name: Collection to search (defaults to self.collection_name)

        Returns:
            List[Dict[str, Any]]: Search results

        Alpha controls the balance:
        - alpha = 1.0: Pure vector search
        - alpha = 0.5: Equal weight to both
        - alpha = 0.0: Pure keyword search
        """
        if not self.client:
            raise RuntimeError("Client not connected. Call connect() first.")

        name = collection_name or self.collection_name
        collection = self.client.collections.get(name)

        kwargs = {
            "query": query,
            "limit": limit,
            "offset": offset,
        }
        if alpha is not None:
            kwargs["alpha"] = alpha
        if where_filter:
            kwargs["filters"] = where_filter
        if return_properties:
            kwargs["return_properties"] = return_properties
        if vector:
            kwargs["vector"] = vector

        try:
            result = await collection.query.hybrid(
                return_metadata=MetadataQuery.full() if return_metadata else None,
                **kwargs
            )

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            raise

        return self._format_results(result.objects)

    async def fetch_objects(
        self,
        where_filter: Optional[Filter] = None,
        limit: int = 100,
        offset: int = 0,
        return_properties: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
        include_vector: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Fetch objects by filter criteria.

        Args:
            where_filter: Filter criteria
            limit: Maximum results to return
            offset: Number of results to skip before applying the limit
            return_properties: Properties to return
            collection_name: Collection to query (defaults to self.collection_name)
            include_vector: Whether to include vectors in response (default: False)

        Returns:
            List[Dict[str, Any]]: Fetched objects
        """
        if not self.client:
            raise RuntimeError("Client not connected. Call connect() first.")

        name = collection_name or self.collection_name
        collection = self.client.collections.get(name)

        kwargs = {
            "limit": limit,
            "offset": offset,
            "include_vector": include_vector,
        }
        if where_filter:
            kwargs["filters"] = where_filter
        if return_properties:
            kwargs["return_properties"] = return_properties

        try:
            result = await collection.query.fetch_objects(**kwargs)

        except Exception as e:
            logger.error(f"Error in fetch objects: {e}")
            raise

        return self._format_results(result.objects)

    def _format_results(self, objects: List[Any]) -> List[Dict[str, Any]]:
        """
        Format Weaviate objects into dictionaries.

        Converts Weaviate response objects into clean dictionaries
        with metadata and properties.
        """
        results = []

        for obj in objects:
            metadata = getattr(obj, "metadata", None)
            result = {
                "uuid": str(obj.uuid),
                "properties": obj.properties,
                "metadata": {
                    "distance": getattr(metadata, "distance", None),
                    "certainty": getattr(metadata, "certainty", None),
                    "score": getattr(metadata, "score", None),
                    "explain_score": getattr(metadata, "explain_score", None),
                },
            }

            # Include vector if present
            if hasattr(obj, "vector") and obj.vector:
                vector = obj.vector
                # Handle named vectors vs single vector space
                if isinstance(vector, dict) and len(vector) == 1 and "default" in vector:
                    # Single vector space returned as {'default': [values]}
                    # Extract just the vector values for backwards compatibility
                    result["vector"] = vector["default"]
                else:
                    # Either a list directly or multiple named vectors
                    result["vector"] = vector

            results.append(result)

        return results
