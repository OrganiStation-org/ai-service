import hashlib
import os
import logging
from datetime import datetime
from typing import Any, Dict, List

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from pypdf import PdfReader
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger("ai-service")
logging.basicConfig(level=logging.INFO)


class LocalEmbeddingFunction(EmbeddingFunction):
    """On-device semantic embeddings via Chroma's default MiniLM model (no API key)."""

    def __init__(self):
        self._ef = DefaultEmbeddingFunction()
        self.model_name = "all-MiniLM-L6-v2"

    def __call__(self, input: Documents) -> Embeddings:
        return self._ef(input)


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
                    container_client.create_container()
                logger.info("Azure Blob Storage connected successfully.")
            except Exception as e:
                logger.error(f"Failed to connect to Azure Blob Storage: {str(e)}")

        self.embedding_fn = LocalEmbeddingFunction()
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_documents",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        self.llm_provider = "none"
        self.llm_model_name = "None (Local Fallback)"
        if self.groq_api_key:
            self.llm_provider = "groq"
            self.llm_model_name = "llama-3.3-70b-versatile"
            print("Groq LLM active for RAG answers.")
        else:
            print("No GROQ_API_KEY — using local excerpt fallback for answers.")

        print(f"Embeddings: {self.embedding_fn.model_name} (local, no API key).")

    def extract_text_from_file(self, file_path: str, original_filename: str) -> str:
        ext = os.path.splitext(original_filename)[1].lower()

        if ext == ".pdf":
            try:
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
                return text.strip()
            except Exception as e:
                raise ValueError(f"Failed to read PDF document: {str(e)}")

        if ext in [".txt", ".md"]:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read().strip()
            except Exception as e:
                raise ValueError(f"Failed to read text/markdown file: {str(e)}")

        raise ValueError(f"Unsupported file type '{ext}'. Please upload PDF, TXT, or MD files.")

    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        if not text:
            return []

        if len(text) <= chunk_size:
            return [text]

        separators = ["\n\n", "\n", ". ", " ", ""]
        chunks = []
        current_idx = 0
        text_length = len(text)

        while current_idx < text_length:
            end_idx = min(current_idx + chunk_size, text_length)

            if end_idx == text_length:
                chunks.append(text[current_idx:])
                break

            chunk_slice = text[current_idx:end_idx]
            split_idx = -1

            for sep in separators[:-1]:
                last_occurrence = chunk_slice.rfind(sep)
                if last_occurrence != -1 and last_occurrence > chunk_size // 2:
                    split_idx = last_occurrence + len(sep)
                    break

            if split_idx == -1:
                split_idx = chunk_size

            chunks.append(text[current_idx : current_idx + split_idx].strip())
            current_idx += (split_idx - chunk_overlap) if (split_idx > chunk_overlap) else split_idx

        return [c for c in chunks if len(c) > 10]

    def ingest_document(self, temp_file_path: str, filename: str) -> Dict[str, Any]:
        raw_text = self.extract_text_from_file(temp_file_path, filename)
        if not raw_text:
            raise ValueError("No extractable text found in this document.")

        chunks = self.chunk_text(raw_text)
        if not chunks:
            raise ValueError("Document was empty or too small to chunk.")

        # Generate unique hash for the document
        doc_hash = hashlib.md5(raw_text.encode()).hexdigest()
        
        # Upload to Azure Blob Storage if available
        blob_url = ""
        if self.blob_service_client:
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=f"{doc_hash}_{filename}")
            with open(temp_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            blob_url = blob_client.url

        self.collection.add(
            ids=[f"{doc_hash}_{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{
                "filename": filename,
                "doc_hash": doc_hash,
                "chunk_id": i,
                "blob_url": blob_url,
                "ingested_at": datetime.now().isoformat()
            } for i in range(len(chunks))]
        )

        return {
            "id": doc_hash,
            "filename": filename,
            "chunks_count": len(chunks),
            "total_characters": len(raw_text),
        }

    def list_documents(self) -> List[Dict[str, Any]]:
        results = self.collection.get()
        if not results or not results["ids"]:
            return []

        docs_map = {}
        for meta, text in zip(results["metadatas"], results["documents"]):
            doc_hash = meta["doc_hash"]
            if doc_hash not in docs_map:
                docs_map[doc_hash] = {
                    "id": doc_hash,
                    "filename": meta["filename"],
                    "chunks_count": 0,
                    "characters_count": 0,
                }
            docs_map[doc_hash]["chunks_count"] += 1
            docs_map[doc_hash]["characters_count"] += len(text)

        return list(docs_map.values())

    def delete_document(self, doc_hash: str) -> bool:
        docs = self.collection.get(where={"doc_hash": doc_hash})
        if not docs or not docs["ids"]:
            return False
        self.collection.delete(where={"doc_hash": doc_hash})
        return True

    def reset_database(self):
        self.chroma_client.delete_collection("rag_documents")
        self.collection = self.chroma_client.create_collection(
            name="rag_documents",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def _keyword_score(self, query: str, text: str) -> int:
        keywords = [w.strip("?,.!-").lower() for w in query.split() if len(w) > 2]
        lower = text.lower()
        return sum(1 for kw in keywords if kw in lower)

    def _retrieve_context(self, user_query: str, max_results: int = 4) -> tuple[List[str], List[Dict[str, Any]]]:
        """Vector search with keyword reranking for better recall on policy documents."""
        pool_size = max(max_results * 3, 12)
        results = self.collection.query(query_texts=[user_query], n_results=pool_size)

        if not results or not results["documents"] or not results["documents"][0]:
            return [], []

        ranked = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            vector_score = 1 - dist if dist is not None else 1.0
            keyword_score = self._keyword_score(user_query, text)
            combined = vector_score + (keyword_score * 0.15)
            ranked.append((combined, text, meta, vector_score))

        ranked.sort(key=lambda x: x[0], reverse=True)
        top = ranked[:max_results]

        context_chunks = [item[1] for item in top]
        sources = [
            {
                "filename": item[2]["filename"],
                "chunk": item[2]["chunk_index"],
                "relevance": round(item[3], 4),
            }
            for item in top
        ]
        return context_chunks, sources

    def query(self, user_query: str, max_results: int = 4) -> Dict[str, Any]:
        context_chunks, sources = self._retrieve_context(user_query, max_results)

        context_str = "\n---\n".join(
            [f"Source: {src['filename']}\nContent: {txt}" for src, txt in zip(sources, context_chunks)]
        )

        if self.llm_provider == "groq" and self.groq_api_key:
            try:
                answer = self._query_groq_llm(user_query, context_str)
                is_fallback = False
            except Exception as e:
                print(f"Error querying Groq LLM API: {e}. Falling back to local excerpt mode.")
                answer = self._generate_local_answer(user_query, context_chunks, sources)
                is_fallback = True
        else:
            answer = self._generate_local_answer(user_query, context_chunks, sources)
            is_fallback = True

        return {
            "query": user_query,
            "answer": answer,
            "sources": sources,
            "local_fallback": is_fallback,
        }

    def _query_groq_llm(self, user_query: str, context_str: str) -> str:
        import requests

        system_prompt = (
            "You are OrganiStation's AI assistant. Answer using the Context documents below.\n\n"
            "RULES:\n"
            "1. Use professional markdown (headers, bullets, tables when helpful).\n"
            "2. Answer primarily from the Context. Cite source filenames inline, e.g. [leave-policy.txt].\n"
            "3. If the Context contains the answer, do NOT say it is missing — quote or summarize it directly.\n"
            "4. Only if the Context is completely empty or irrelevant, say no matching policy was found.\n"
        )

        user_content = f"Context:\n{context_str or '(no documents retrieved)'}\n\nQuestion: {user_query}\n\nAnswer:"

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.1,
            },
            timeout=30,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Groq API returned error {response.status_code}: {response.text}")

        return response.json()["choices"][0]["message"]["content"]

    def _generate_local_answer(self, query: str, chunks: List[str], sources: List[Dict[str, Any]]) -> str:
        if not chunks:
            return (
                "### No matching documents\n\n"
                "Upload PDF, TXT, or MD files first. "
                "Set `GROQ_API_KEY` for full AI-generated answers."
            )

        keywords = [w.strip("?,.!-").lower() for w in query.split() if len(w) > 3]
        matched_results = []
        for text, src in zip(chunks, sources):
            score = sum(1 for kw in keywords if kw in text.lower())
            matched_results.append((score, text, src))
        matched_results.sort(key=lambda x: x[0], reverse=True)

        best_match = matched_results[0]
        lines = [
            "### Local excerpt mode",
            "Set `GROQ_API_KEY` for full AI summaries. Best matching passage:",
            "",
        ]

        text_body = best_match[1]
        first_kw = next((kw for kw in keywords if kw in text_body.lower()), None)
        if first_kw:
            pos = text_body.lower().find(first_kw)
            start = max(0, pos - 150)
            end = min(len(text_body), pos + 350)
            excerpt = ("..." if start > 0 else "") + text_body[start:end] + ("..." if end < len(text_body) else "")
        else:
            excerpt = text_body[:400] + ("..." if len(text_body) > 400 else "")

        lines.append(f"**[{best_match[2]['filename']}]**")
        lines.append(f"> {excerpt.strip()}")
        return "\n".join(lines)
