import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

import chromadb
from chromadb.utils.embedding_functions import LocalEmbeddingFunction
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger("ai-service")

class RAGPipeline:
    def __init__(self, db_path: str = "./chroma_db", groq_api_key: str = None):
        self.groq_api_key = (groq_api_key or "").strip() or None
        self.db_path = db_path

        os.makedirs(db_path, exist_ok=True)

        # Cloud PDF Store
        self.connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.container_name = "documents"
        self.blob_service_client = None
        
        if self.connection_string:
            try:
                # Ensure it looks like a connection string to prevent startup crashes
                if "DefaultEndpointsProtocol=" in self.connection_string:
                    self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
                    # Ensure container exists
                    container_client = self.blob_service_client.get_container_client(self.container_name)
                    if not container_client.exists():
                        container_client.create_container()
                    logger.info("Azure Blob Storage connected successfully.")
                else:
                    logger.warning("AZURE_STORAGE_CONNECTION_STRING is not a valid connection string. Cloud storage disabled.")
            except Exception as e:
                logger.error(f"Failed to connect to Azure Blob Storage: {str(e)}")

        self.embedding_fn = LocalEmbeddingFunction()
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_documents",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """Adds documents to the vector store"""
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def query(self, query_text: str, n_results: int = 3) -> Dict[str, Any]:
        """Queries the vector store for relevant documents"""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results

    async def get_latest_documents(self):
        """Fetches latest documents from Azure Blob Storage if configured"""
        if not self.blob_service_client:
            return []
            
        container_client = self.blob_service_client.get_container_client(self.container_name)
        blobs = container_client.list_blobs()
        return [blob.name for blob in blobs]
