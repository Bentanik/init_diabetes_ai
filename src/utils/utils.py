"""Utility functions with improved error handling and type safety."""

import re
from typing import Optional, List, Dict, Any, Union

from core.exceptions import ServiceError
from core.logging_config import get_logger

logger = get_logger(__name__)


"""Utility functions with improved error handling and type safety."""


def extract_json(text: str) -> str:
    """Extract JSON array from text response.

    Args:
        text: Input text containing JSON array

    Returns:
        Extracted JSON array as string

    Raises:
        ParsingException: If no valid JSON array found
    """
    if not text or not isinstance(text, str):
        logger.error(f"Invalid input for JSON extraction: {type(text)}")
        raise ServiceError("Input text is empty or not a string")

    # Try to find JSON array pattern
    patterns = [
        r"\[\s*{.*}\s*\]",  # Standard array pattern
        r"\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]",  # More specific pattern
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            result = match.group(0)
            logger.debug(f"Successfully extracted JSON array of length: {len(result)}")
            return result

    logger.error(f"No JSON array found in text of length: {len(text)}")
    raise ServiceError("Không tìm thấy JSON array trong phản hồi")


def validate_patient_id(patient_id: Optional[str]) -> bool:
    """Validate patient ID format.

    Args:
        patient_id: Patient identifier

    Returns:
        True if valid, False otherwise
    """
    if not patient_id or not isinstance(patient_id, str):
        return False

    # Validation - adjust according to your ID format requirements
    return len(patient_id.strip()) >= 3 and patient_id.strip().isalnum()


def sanitize_text(text: Optional[str], max_length: Optional[int] = None) -> str:
    """Sanitize and clean text input.

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Sanitization
    sanitized = text.strip()

    # Remove excessive whitespace
    sanitized = re.sub(r"\s+", " ", sanitized)

    # Truncate if necessary
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
        logger.warning(f"Text truncated to {max_length} characters")

    return sanitized


def safe_getattr(obj, attr_name: str, default=None):
    """
    Safely get attribute from object with proper error handling.

    Args:
        obj: Object to get attribute from
        attr_name: Name of attribute to get
        default: Default value if attribute doesn't exist

    Returns:
        Attribute value or default
    """
    try:
        return getattr(obj, attr_name, default)
    except (AttributeError, TypeError):
        return default


def check_retriever_type(retriever):
    """
    Check the type of retriever and return its characteristics.

    Args:
        retriever: Retriever instance to check

    Returns:
        Dict with retriever type information
    """
    from features.rag.retrieval import (
        HybridRetriever,
        MultiCollectionHybridRetriever,
    )

    info = {
        "type": type(retriever).__name__,
        "has_vector_store": False,
        "has_vector_stores": False,
        "is_hybrid": False,
        "is_multi_collection": False,
    }

    if isinstance(retriever, HybridRetriever):
        info["is_hybrid"] = True
        info["has_vector_store"] = (
            hasattr(retriever, "vector_store") and retriever.vector_store is not None
        )

    elif isinstance(retriever, MultiCollectionHybridRetriever):
        info["is_multi_collection"] = True
        info["has_vector_stores"] = (
            hasattr(retriever, "_vector_stores")
            and retriever._vector_stores is not None
            and len(retriever._vector_stores) > 0
        )

    return info


def safe_access_vector_store(retriever, collection_name: Optional[str] = None):
    """
    Safely access vector store(s) from retriever.

    Args:
        retriever: Retriever instance
        collection_name: Optional collection name for multi-collection retrievers

    Returns:
        Vector store instance or None if not found
    """
    from features.rag.retrieval import (
        HybridRetriever,
        MultiCollectionHybridRetriever,
    )

    try:
        if isinstance(retriever, HybridRetriever):
            return safe_getattr(retriever, "vector_store")

        elif isinstance(retriever, MultiCollectionHybridRetriever):
            vector_stores = safe_getattr(retriever, "_vector_stores", {})
            if collection_name:
                return vector_stores.get(collection_name)
            else:
                # Return first available store if no specific collection requested
                return next(iter(vector_stores.values())) if vector_stores else None

    except Exception as e:
        logger.error(f"Error accessing vector store: {e}")
        return None

    return None


def validate_retriever_initialization(retriever):
    """
    Validate that a retriever is properly initialized.

    Args:
        retriever: Retriever instance to validate

    Returns:
        Tuple of (is_valid: bool, error_message: str)
    """
    from features.rag.retrieval import (
        HybridRetriever,
        MultiCollectionHybridRetriever,
    )

    try:
        if isinstance(retriever, HybridRetriever):
            if not hasattr(retriever, "vector_store") or retriever.vector_store is None:
                return False, "HybridRetriever missing vector_store attribute"
            return True, ""

        elif isinstance(retriever, MultiCollectionHybridRetriever):
            if not hasattr(retriever, "_vector_stores") or not retriever._vector_stores:
                return (
                    False,
                    "MultiCollectionHybridRetriever missing _vector_stores attribute",
                )
            if len(retriever._vector_stores) == 0:
                return False, "MultiCollectionHybridRetriever has empty vector_stores"
            return True, ""

        else:
            return False, f"Unknown retriever type: {type(retriever)}"

    except Exception as e:
        return False, f"Error validating retriever: {e}"


def extract_variables(template: str) -> List[str]:
    """Extract variables from a template string.

    Args:
        template: The template string containing variables in {{variable_name}} format

    Returns:
        List of variable names found in the template
    """
    pattern = r"\{\{([^}]+)\}\}"
    matches = re.findall(pattern, template)
    # Remove any whitespace and duplicates
    variables = list(set(var.strip() for var in matches))
    return variables
