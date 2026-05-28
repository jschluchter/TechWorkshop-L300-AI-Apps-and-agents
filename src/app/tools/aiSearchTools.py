import os
import sys
import json
from pathlib import Path
import requests
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

# Cosmos DB configuration
COSMOS_ENDPOINT = os.environ.get("COSMOS_ENDPOINT")
DATABASE_NAME = os.environ.get("DATABASE_NAME")
CONTAINER_NAME = os.environ.get("CONTAINER_NAME")

# Embedding service configuration (used to encode the query)
EMBEDDING_ENDPOINT = os.environ.get("embedding_endpoint")
EMBEDDING_DEPLOYMENT = os.environ.get("embedding_deployment")
EMBEDDING_API_VERSION = os.environ.get("embedding_api_version")

credential = DefaultAzureCredential()

# Validate required Cosmos env vars
if not COSMOS_ENDPOINT:
    raise ValueError("COSMOS_ENDPOINT environment variable is not set")
if not DATABASE_NAME:
    raise ValueError("DATABASE_NAME environment variable is not set")
if not CONTAINER_NAME:
    raise ValueError("CONTAINER_NAME environment variable is not set")


def get_cosmos_client(endpoint: str | None):
    if not endpoint:
        raise ValueError("COSMOS_ENDPOINT must be provided in environment variables")

    client = CosmosClient(endpoint, credential=credential)
    _ = list(client.list_databases())
    return client


def get_request_embedding(text: str) -> list[float] | None:
    """Call embedding endpoint and return the embedding vector or None on failure."""
    if not EMBEDDING_ENDPOINT or not EMBEDDING_DEPLOYMENT or not EMBEDDING_API_VERSION:
        raise ValueError("Embedding endpoint configuration missing. Set EMBEDDING_ENDPOINT, EMBEDDING_DEPLOYMENT, EMBEDDING_API_VERSION")

    url = EMBEDDING_ENDPOINT.rstrip("/") + f"/openai/deployments/{EMBEDDING_DEPLOYMENT}/embeddings?api-version={EMBEDDING_API_VERSION}"
    token = credential.get_token("https://cognitiveservices.azure.com/.default")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token.token}",
    }
    payload = {"input": text}

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    embedding = data.get("data", [{}])[0].get("embedding")
    return embedding


# Initialize Cosmos client and container
_cosmos_client = get_cosmos_client(COSMOS_ENDPOINT)
_database = _cosmos_client.get_database_client(DATABASE_NAME)
_container = _database.get_container_client(CONTAINER_NAME)


def _load_local_catalog() -> list[dict]:
    """Load local catalog as a deterministic fallback source."""
    catalog_path = Path(__file__).resolve().parents[2] / "data" / "product_catalog.json"
    if not catalog_path.exists():
        return []
    with catalog_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_paint_shade_request(question: str) -> bool:
    q = question.lower()
    paint_terms = {"paint", "shade", "color", "colour", "wall color", "wall paint"}
    accessory_terms = {
        "sprayer", "roller", "brush", "tray", "tape", "drop cloth", "accessory", "tool"
    }
    has_paint_intent = any(term in q for term in paint_terms)
    has_accessory_intent = any(term in q for term in accessory_terms)
    return has_paint_intent and not has_accessory_intent


def _is_white_paint_request(question: str) -> bool:
    q = question.lower()
    white_terms = {"white", "off-white", "off white", "ivory", "cream", "vanilla", "pearl"}
    return any(term in q for term in white_terms)


def _is_paint_shade_item(item: dict) -> bool:
    category = str(item.get("ProductCategory", "")).lower()
    return "paint shade" in category


def _is_white_like_item(item: dict) -> bool:
    text = f"{item.get('ProductName', '')} {item.get('ProductDescription', '')}".lower()
    white_like_terms = {"white", "off-white", "off white", "ivory", "cream", "vanilla", "pearl"}
    return any(term in text for term in white_like_terms)


def _fallback_white_paint_items(limit: int) -> list[dict]:
    """Fetch white/off-white paint shades directly from catalog fields.

    This avoids vector-search misses where accessory products dominate top-k.
    """
    fallback_query = (
        "SELECT c.id, c.ProductID, c.ProductName, c.ProductCategory, c.ProductDescription, "
        "c.ImageURL, c.ProductPunchLine, c.Price "
        "FROM c "
        "WHERE CONTAINS(LOWER(c.ProductCategory), 'paint shade') "
        "AND ("
        "CONTAINS(LOWER(c.ProductName), 'white') OR "
        "CONTAINS(LOWER(c.ProductName), 'ivory') OR "
        "CONTAINS(LOWER(c.ProductName), 'cream') OR "
        "CONTAINS(LOWER(c.ProductName), 'vanilla') OR "
        "CONTAINS(LOWER(c.ProductName), 'pearl') OR "
        "CONTAINS(LOWER(c.ProductDescription), 'white') OR "
        "CONTAINS(LOWER(c.ProductDescription), 'off-white') OR "
        "CONTAINS(LOWER(c.ProductDescription), 'off white')"
        ")"
    )

    items = list(
        _container.query_items(
            query=fallback_query,
            enable_cross_partition_query=True,
            max_item_count=limit,
        )
    )

    # If Cosmos does not return white shades, fallback to local catalog file.
    if items:
        return items

    local_items = _load_local_catalog()
    white_local = [
        item
        for item in local_items
        if _is_paint_shade_item(item) and _is_white_like_item(item)
    ]
    return white_local[:limit]


def product_recommendations(question: str, top_k: int = 8):
    """
    Input:
        question (str): Natural language user query
        top_k (int): number of nearest neighbors to return
    Output:
        list of product dicts with product information
    """

    # Generate embedding for the query
    query_vector = get_request_embedding(question)
    if query_vector is None:
        raise RuntimeError("Failed to generate query embedding")

    # Cosmos DB vector search SQL. Requires Cosmos account with vector search enabled
    query = (
        "SELECT c.id, c.ProductID, c.ProductName, c.ProductCategory, c.ProductDescription, "
        "c.ImageURL, c.ProductPunchLine, c.Price "
        "FROM c "
        "ORDER BY VECTORDISTANCE(c.request_vector, @vector) "
        "OFFSET 0 LIMIT @top"
    )

    effective_top_k = top_k
    if _is_paint_shade_request(question):
        # Pull a wider vector set, then filter down to relevant paint shades.
        effective_top_k = max(top_k, 20)

    parameters = [
        {"name": "@vector", "value": query_vector},
        {"name": "@top", "value": effective_top_k},
    ]

    items = list(_container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True,
        max_item_count=effective_top_k
    ))

    # When user intent is paint shades, suppress tool/accessory products.
    if _is_paint_shade_request(question):
        shade_items = [item for item in items if _is_paint_shade_item(item)]
        if shade_items:
            items = shade_items

    # For white paint asks, prefer white-like shades first.
    if _is_white_paint_request(question):
        white_like = [item for item in items if _is_white_like_item(item)]
        non_white_like = [item for item in items if not _is_white_like_item(item)]
        if not white_like:
            fallback_items = _fallback_white_paint_items(limit=max(top_k, 12))
            if fallback_items:
                white_like = [item for item in fallback_items if _is_white_like_item(item)]
                non_white_like = [item for item in fallback_items if not _is_white_like_item(item)]

        if white_like:
            items = white_like + non_white_like

    # Keep response size consistent for downstream prompts.
    items = items[:top_k]

    get = dict.get
    response = [
        {
            "id": get(item, "ProductID", None),
            "name": get(item, "ProductName", None),
            "type": get(item, "ProductCategory", None),
            "description": get(item, "ProductDescription", None),
            "imageURL": get(item, "ImageURL", None),
            "punchLine": get(item, "ProductPunchLine", None),
            "price": get(item, "Price", None)
        }
        for item in items
    ]

    return response
