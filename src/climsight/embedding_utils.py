import os
import logging
from typing import Optional
from langchain_openai.embeddings import OpenAIEmbeddings

logger = logging.getLogger(__name__)

def create_embeddings(
    model_type: str,
    embedding_model: str,
    openai_api_key: Optional[str] = None,
    aitta_api_key: Optional[str] = None,
    aitta_url: Optional[str] = None,
    model_name: Optional[str] = None
) -> OpenAIEmbeddings:
    """
    Creates an embedding model instance based on the configuration.
    
    Args:
        model_type (str): Which backend to use ('openai', 'aitta', ...)
        embedding_model (str): The embedding model name or type
        openai_api_key (str, optional): OpenAI API key for OpenAI models
        aitta_api_key (str, optional): AITTA API key for open models
        aitta_url (str, optional): AITTA API URL for open models
        model_name (str, optional): Specific model name for open models
        
    Returns:
        OpenAIEmbeddings: Configured embedding model instance
        
    Raises:
        ValueError: If required parameters are missing or configuration is invalid
    """
    
    if model_type == 'openai':
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI models")
        return OpenAIEmbeddings(
            api_key=openai_api_key,  # type: ignore
            model=embedding_model
        )
    elif model_type == 'aitta':
        if not aitta_api_key:
            raise ValueError("AITTA_API_KEY is required for aitta models")
        if not aitta_url:
            raise ValueError("AITTA URL is required for aitta models")
        if not model_name:
            model_name = embedding_model
            
        try:
            # Import aitta_client only when needed
            from aitta_client import Model, Client
            from aitta_client.authentication import APIKeyAccessTokenSource
            
            client = Client(aitta_url, APIKeyAccessTokenSource(aitta_api_key, aitta_url))
            model = Model.load(model_name, client)
            
            return OpenAIEmbeddings(
                model=model.id,
                api_key=client.access_token_source.get_access_token(),  # type: ignore
                base_url=model.openai_api_url,
                tiktoken_enabled=False
            )
        except ImportError:
            raise ImportError("aitta_client is required for aitta models. Install with: pip install aitta-client")
        except Exception as e:
            logger.error(f"Failed to create aitta model embeddings: {e}")
            raise ValueError(f"Failed to create aitta model embeddings: {e}")
    # elif model_type == 'mistral':
    #     # Add logic for mistral here
    #     pass
    else:
        raise ValueError(f"Unknown model_type: {model_type}") 