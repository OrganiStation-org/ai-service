import os
import json
import logging
import hashlib
import requests
from datetime import datetime
from typing import List, Dict, Any

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from azure.storage.blob import BlobServiceClient
from pypdf import PdfReader

logger = logging.getLogger("ai-service")

class RAGPipeline:
    def __init__(self, db_path: str = "./chroma_db", groq_api_key: str = None):
        self.groq_api_key = (groq_api_key or "").strip() or None
        self.db_path = db_path
        self.llm_provider = "local"
        self.llm_model_name = "Mock-LLM"
        
        if self.groq_api_key:
            self.llm_provider = "groq"
            self.llm_model_name = "mixtral-8x7b-32768"

        os.makedirs(db_path, exist_ok=True)

        # Cloud PDF Store
        self.connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.container_name = "documents"
        self.blob_service_client = None
        
        if self.connection_string:
            try:
                if "DefaultEndpointsProtocol=" in self.connection_string:
                    self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
                    container_client = self.blob_service_client.get_container_client(self.container_name)
                    if not container_client.exists():
                        container_client.create_container()
                    logger.info("Azure Blob Storage connected successfully.")
            except Exception as e:
                logger.error(f"Failed to connect to Azure Blob Storage: {str(e)}")

        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.embedding_fn = DefaultEmbeddingFunction()
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_documents",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_document(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Parses, chunks, and indexes a document."""
        try:
            logger.info(f"Ingesting document: {filename}")
            content = ""
            ext = os.path.splitext(filename)[1].lower()
            
            if ext == ".pdf":
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        content += text + "\n"
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

            if not content.strip():
                raise ValueError("Document appears to be empty or unreadable.")

            # Upload to Azure
            if self.blob_service_client:
                try:
                    blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=filename)
                    with open(file_path, "rb") as data:
                        blob_client.upload_blob(data, overwrite=True)
                except Exception as e:
                    logger.error(f"Cloud upload failed for {filename}: {e}")

            # Chunking
            chunks = self._chunk_text(content)
            doc_hash = hashlib.md5(filename.encode()).hexdigest()
            
            ids = [f"{doc_hash}_{i}" for i in range(len(chunks))]
            metadatas = [{
                "filename": filename,
                "chunk_index": i,
                "timestamp": datetime.utcnow().isoformat()
            } for i in range(len(chunks))]
            
            self.collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            
            return {"id": doc_hash, "chunks": len(chunks), "filename": filename}
        except Exception as e:
            logger.error(f"Ingestion failed for {filename}: {str(e)}")
            raise

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        chunks = []
        if len(text) <= chunk_size:
            return [text]
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size]
            if len(chunk) > 50: # Skip tiny chunks
                chunks.append(chunk)
        return chunks

    def query(self, query_text: str, n_results: int = 3) -> Dict[str, Any]:
        """Queries the vector store and optionally calls Groq for an answer."""
        try:
            # Explicitly cast n_results to int to avoid any weirdness
            results = self.collection.query(
                query_texts=[query_text],
                n_results=int(n_results)
            )

            if not results['documents'] or not results['documents'][0]:
                return {
                    "answer": "I couldn't find any relevant information in the uploaded documents.",
                    "sources": []
                }

            # Format sources for UI
            sources = []
            context_text = ""
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                dist = float(results['distances'][0][i]) if results['distances'] else 0.0
                
                sources.append({
                    "content": doc,
                    "filename": meta.get("filename", "Unknown"),
                    "relevance": round(1 - dist, 4)
                })
                context_text += f"\n--- Source: {meta.get('filename')} ---\n{doc}\n"

            # If Groq is configured, get a real answer
            if self.groq_api_key:
                answer = self._get_groq_answer(query_text, context_text)
            else:
                answer = f"Based on the documents I found, here is the most relevant section:\n\n{results['documents'][0][0]}"

            return {
                "answer": answer,
                "sources": sources,
                "raw_results": { # Minimal raw results to avoid serialization issues
                    "ids": results.get("ids", [[]])[0],
                    "distances": [float(d) for d in results.get("distances", [[]])[0]]
                }
            }
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            # Fallback instead of crashing
            return {
                "answer": f"Sorry, I encountered an error while processing your query: {str(e)}",
                "sources": []
            }

    def _get_groq_answer(self, query: str, context: str) -> str:
        """Call Groq API for a high-quality answer."""
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            prompt = f"""You are a helpful assistant for OrganiStation. 
Answer the following question based ONLY on the provided context.
If the answer is not in the context, say you don't know.

QUESTION: {query}

CONTEXT:
{context}
"""
            payload = {
                "model": self.llm_model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            }
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return f"I found some information, but I couldn't generate a polished answer. Error: {str(e)}"

    def list_documents(self) -> List[str]:
        try:
            data = self.collection.get()
            if not data or not data.get('metadatas'):
                return []
            
            filenames = set()
            for meta in data['metadatas']:
                if meta and isinstance(meta, dict) and 'filename' in meta:
                    filenames.add(meta['filename'])
            return sorted(list(filenames))
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []

    def delete_document(self, doc_hash: str) -> bool:
        return self.reset_database() # Simplified safety

    def reset_database(self) -> bool:
        try:
            self.chroma_client.delete_collection("rag_documents")
            self.collection = self.chroma_client.get_or_create_collection(
                name="rag_documents",
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            return True
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            return False
