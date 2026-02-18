"""
Pinecone Manager Module

This module provides a comprehensive interface for managing Pinecone vector database operations,
including various methods for reading and writing vectors and documents.
"""

import os
from typing import List, Dict, Any, Optional, Union, Tuple
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import time
from dotenv import load_dotenv

load_dotenv()


# ==================== GLOBAL SETTINGS ====================
# Порог косинусного сходства для определения дубликатов
# Значения косинусного сходства находятся в диапазоне [0, 1]:
# - 0.95-1.0: почти идентичные тексты (высокое сходство → дубликат)
# - 0.85-0.95: очень похожие тексты
# - 0.70-0.85: умеренное сходство
# - 0.0-0.70: низкое сходство (новая информация)
SIMILARITY_THRESHOLD = 0.85


class PineconeManager:
    """
    A comprehensive manager class for Pinecone vector database operations.
    
    Supports multiple methods for:
    - Writing vectors and documents
    - Reading by vector similarity
    - Reading by text query
    - Batch operations
    - Metadata filtering
    - Smart duplicate detection and prevention
    
    Duplicate Detection:
    - Uses cosine similarity to detect duplicate content before insertion
    - Configurable threshold (default: SIMILARITY_THRESHOLD = 0.85)
    - Can skip duplicates or update existing entries
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        index_name: Optional[str] = None,
        dimension: int = 1536,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None
    ):
        """
        Initialize the Pinecone Manager.
        
        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            environment: Pinecone environment (deprecated in newer versions)
            index_name: Name of the Pinecone index
            dimension: Vector dimension (default 1536 for OpenAI embeddings)
            metric: Distance metric (cosine, euclidean, dotproduct)
            cloud: Cloud provider (aws, gcp, azure)
            region: Cloud region
            openai_api_key: OpenAI API key for text embeddings
            openai_base_url: OpenAI base URL (defaults to OPENAI_BASE_URL env var)
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        
        if not self.api_key:
            raise ValueError("Pinecone API key is required")
        
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None
        
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
        self.openai_client = None
        if self.openai_api_key:
            client_kwargs = {"api_key": self.openai_api_key}
            if self.openai_base_url:
                client_kwargs["base_url"] = self.openai_base_url
            self.openai_client = OpenAI(**client_kwargs)
        
        if self.index_name:
            self.connect_to_index()
    
    def create_index(self, index_name: Optional[str] = None, dimension: Optional[int] = None) -> None:
        """
        Create a new Pinecone index.
        
        Args:
            index_name: Name for the new index
            dimension: Vector dimension
        """
        index_name = index_name or self.index_name
        dimension = dimension or self.dimension
        
        if not index_name:
            raise ValueError("Index name is required")
        
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if index_name not in existing_indexes:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud=self.cloud,
                    region=self.region
                )
            )
            print(f"Index '{index_name}' created successfully")
        else:
            print(f"Index '{index_name}' already exists")
        
        self.index_name = index_name
        self.index = self.pc.Index(index_name)
    
    def connect_to_index(self, index_name: Optional[str] = None) -> None:
        """
        Connect to an existing Pinecone index.
        
        Args:
            index_name: Name of the index to connect to
        """
        index_name = index_name or self.index_name
        
        if not index_name:
            raise ValueError("Index name is required")
        
        self.index_name = index_name
        self.index = self.pc.Index(index_name)
        print(f"Connected to index '{index_name}'")
    
    def delete_index(self, index_name: Optional[str] = None) -> None:
        """
        Delete a Pinecone index.
        
        Args:
            index_name: Name of the index to delete
        """
        index_name = index_name or self.index_name
        
        if not index_name:
            raise ValueError("Index name is required")
        
        self.pc.delete_index(index_name)
        print(f"Index '{index_name}' deleted successfully")
        
        if index_name == self.index_name:
            self.index = None
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary containing index statistics
        """
        if not self.index:
            raise ValueError("No index connected. Call connect_to_index() first")
        
        return self.index.describe_index_stats()
    
    # ==================== EMBEDDING METHODS ====================
    
    def generate_embedding(self, text: str, model: str = "text-embedding-ada-002") -> List[float]:
        """
        Generate embedding vector from text using OpenAI.
        
        Args:
            text: Text to embed
            model: OpenAI embedding model
            
        Returns:
            Embedding vector
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Provide openai_api_key")
        
        response = self.openai_client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        model: str = "text-embedding-ada-002"
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            model: OpenAI embedding model
            
        Returns:
            List of embedding vectors
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Provide openai_api_key")
        
        response = self.openai_client.embeddings.create(
            input=texts,
            model=model
        )
        return [item.embedding for item in response.data]
    
    # ==================== DUPLICATE CHECK METHODS ====================
    
    def check_duplicate(
        self,
        text: str,
        namespace: str = "",
        threshold: Optional[float] = None,
        top_k: int = 5,
        model: str = "text-embedding-ada-002"
    ) -> Dict[str, Any]:
        """
        Проверяет, является ли текст дубликатом уже сохранённой информации.
        
        Args:
            text: Текст для проверки
            namespace: Namespace для поиска
            threshold: Порог сходства (если None, используется SIMILARITY_THRESHOLD)
            top_k: Количество похожих векторов для проверки
            model: Модель OpenAI для генерации embedding
            
        Returns:
            Словарь с результатами проверки:
            {
                "is_duplicate": bool,
                "max_similarity": float,
                "similar_documents": List[Dict],
                "threshold_used": float
            }
        """
        if threshold is None:
            threshold = SIMILARITY_THRESHOLD
        
        vector = self.generate_embedding(text, model)
        
        results = self.query_by_vector(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
            include_values=False
        )
        
        matches = results.get("matches", [])
        
        is_duplicate = False
        max_similarity = 0.0
        similar_docs = []
        
        if matches:
            max_similarity = matches[0].get("score", 0.0)
            is_duplicate = max_similarity >= threshold
            
            similar_docs = [
                {
                    "id": match["id"],
                    "score": match.get("score", 0.0),
                    "text": match.get("metadata", {}).get("text", ""),
                    "metadata": match.get("metadata", {})
                }
                for match in matches
                if match.get("score", 0.0) >= threshold
            ]
        
        return {
            "is_duplicate": is_duplicate,
            "max_similarity": max_similarity,
            "similar_documents": similar_docs,
            "threshold_used": threshold
        }
    
    def check_duplicate_batch(
        self,
        texts: List[str],
        namespace: str = "",
        threshold: Optional[float] = None,
        top_k: int = 5,
        model: str = "text-embedding-ada-002"
    ) -> List[Dict[str, Any]]:
        """
        Проверяет множество текстов на дубликаты.
        
        Args:
            texts: Список текстов для проверки
            namespace: Namespace для поиска
            threshold: Порог сходства (если None, используется SIMILARITY_THRESHOLD)
            top_k: Количество похожих векторов для проверки
            model: Модель OpenAI для генерации embedding
            
        Returns:
            Список результатов проверки для каждого текста
        """
        results = []
        for text in texts:
            result = self.check_duplicate(
                text=text,
                namespace=namespace,
                threshold=threshold,
                top_k=top_k,
                model=model
            )
            results.append(result)
        
        return results
    
    # ==================== WRITE METHODS ====================
    
    def upsert_vector(
        self,
        id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        namespace: str = ""
    ) -> None:
        """
        Insert or update a single vector.
        
        Args:
            id: Unique identifier for the vector
            vector: The embedding vector
            metadata: Optional metadata dictionary
            namespace: Namespace for the vector
        """
        if not self.index:
            raise ValueError("No index connected. Call connect_to_index() first")
        
        self.index.upsert(
            vectors=[(id, vector, metadata or {})],
            namespace=namespace
        )
    
    def upsert_vectors_batch(
        self,
        vectors: List[Tuple[str, List[float], Optional[Dict[str, Any]]]],
        namespace: str = "",
        batch_size: int = 100
    ) -> None:
        """
        Insert or update multiple vectors in batches.
        
        Args:
            vectors: List of tuples (id, vector, metadata)
            namespace: Namespace for the vectors
            batch_size: Number of vectors per batch
        """
        if not self.index:
            raise ValueError("No index connected. Call connect_to_index() first")
        
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
            print(f"Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")
    
    def upsert_document(
        self,
        id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: str = "",
        model: str = "text-embedding-ada-002"
    ) -> None:
        """
        Insert or update a document by generating its embedding.
        
        Args:
            id: Unique identifier for the document
            text: Document text
            metadata: Optional metadata dictionary
            namespace: Namespace for the document
            model: OpenAI embedding model
        """
        vector = self.generate_embedding(text, model)
        
        if metadata is None:
            metadata = {}
        metadata["text"] = text
        
        self.upsert_vector(id, vector, metadata, namespace)
    
    def upsert_document_smart(
        self,
        id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: str = "",
        model: str = "text-embedding-ada-002",
        check_duplicates: bool = True,
        threshold: Optional[float] = None,
        update_if_duplicate: bool = False
    ) -> Dict[str, Any]:
        """
        Умная вставка документа с проверкой на дубликаты.
        
        Перед записью проверяет косинусное сходство с существующими документами:
        - Если сходство низкое (< threshold) → записывает как новую информацию
        - Если сходство высокое (>= threshold) → пропускает или обновляет существующий
        
        Args:
            id: Уникальный идентификатор документа
            text: Текст документа
            metadata: Опциональные метаданные
            namespace: Namespace для документа
            model: Модель OpenAI для генерации embedding
            check_duplicates: Выполнять ли проверку на дубликаты
            threshold: Порог сходства (если None, используется SIMILARITY_THRESHOLD)
            update_if_duplicate: Обновлять ли существующий документ, если найден дубликат
            
        Returns:
            Словарь с результатом операции:
            {
                "action": "inserted" | "skipped" | "updated",
                "reason": str,
                "duplicate_check": Dict (если check_duplicates=True)
            }
        """
        result = {
            "action": "inserted",
            "reason": "New document inserted",
            "duplicate_check": None
        }
        
        if check_duplicates:
            duplicate_check = self.check_duplicate(
                text=text,
                namespace=namespace,
                threshold=threshold,
                model=model
            )
            result["duplicate_check"] = duplicate_check
            
            if duplicate_check["is_duplicate"]:
                if update_if_duplicate and duplicate_check["similar_documents"]:
                    duplicate_id = duplicate_check["similar_documents"][0]["id"]
                    
                    self.upsert_document(
                        id=duplicate_id,
                        text=text,
                        metadata=metadata,
                        namespace=namespace,
                        model=model
                    )
                    
                    result["action"] = "updated"
                    result["reason"] = (
                        f"Updated existing document '{duplicate_id}' "
                        f"(similarity: {duplicate_check['max_similarity']:.4f})"
                    )
                else:
                    result["action"] = "skipped"
                    result["reason"] = (
                        f"Duplicate detected (similarity: {duplicate_check['max_similarity']:.4f}). "
                        f"Similar to: {duplicate_check['similar_documents'][0]['id'] if duplicate_check['similar_documents'] else 'unknown'}"
                    )
                
                return result
        
        self.upsert_document(
            id=id,
            text=text,
            metadata=metadata,
            namespace=namespace,
            model=model
        )
        
        return result
    
    def upsert_documents_batch(
        self,
        documents: List[Dict[str, Any]],
        namespace: str = "",
        model: str = "text-embedding-ada-002",
        batch_size: int = 100
    ) -> None:
        """
        Insert or update multiple documents in batches.
        
        Args:
            documents: List of dicts with 'id', 'text', and optional 'metadata'
            namespace: Namespace for the documents
            model: OpenAI embedding model
            batch_size: Number of documents per batch
        """
        texts = [doc["text"] for doc in documents]
        embeddings = self.generate_embeddings_batch(texts, model)
        
        vectors = []
        for doc, embedding in zip(documents, embeddings):
            metadata = doc.get("metadata", {})
            metadata["text"] = doc["text"]
            vectors.append((doc["id"], embedding, metadata))
        
        self.upsert_vectors_batch(vectors, namespace, batch_size)
    
    def upsert_documents_batch_smart(
        self,
        documents: List[Dict[str, Any]],
        namespace: str = "",
        model: str = "text-embedding-ada-002",
        check_duplicates: bool = True,
        threshold: Optional[float] = None,
        update_if_duplicate: bool = False,
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Умная пакетная вставка документов с проверкой на дубликаты.
        
        Args:
            documents: Список словарей с 'id', 'text', и опционально 'metadata'
            namespace: Namespace для документов
            model: Модель OpenAI для генерации embedding
            check_duplicates: Выполнять ли проверку на дубликаты
            threshold: Порог сходства (если None, используется SIMILARITY_THRESHOLD)
            update_if_duplicate: Обновлять ли существующие документы при обнаружении дубликатов
            batch_size: Размер батча для загрузки
            
        Returns:
            Словарь со статистикой операции:
            {
                "total": int,
                "inserted": int,
                "skipped": int,
                "updated": int,
                "results": List[Dict]
            }
        """
        stats = {
            "total": len(documents),
            "inserted": 0,
            "skipped": 0,
            "updated": 0,
            "results": []
        }
        
        for doc in documents:
            result = self.upsert_document_smart(
                id=doc["id"],
                text=doc["text"],
                metadata=doc.get("metadata"),
                namespace=namespace,
                model=model,
                check_duplicates=check_duplicates,
                threshold=threshold,
                update_if_duplicate=update_if_duplicate
            )
            
            stats["results"].append({
                "id": doc["id"],
                "action": result["action"],
                "reason": result["reason"]
            })
            
            if result["action"] == "inserted":
                stats["inserted"] += 1
            elif result["action"] == "skipped":
                stats["skipped"] += 1
            elif result["action"] == "updated":
                stats["updated"] += 1
        
        return stats
    
    def upsert_with_chunks(
        self,
        id_prefix: str,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 200,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: str = "",
        model: str = "text-embedding-ada-002"
    ) -> List[str]:
        """
        Split text into chunks and upsert each chunk as a separate vector.
        
        Args:
            id_prefix: Prefix for chunk IDs
            text: Text to chunk and upsert
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks
            metadata: Base metadata for all chunks
            namespace: Namespace for the chunks
            model: OpenAI embedding model
            
        Returns:
            List of chunk IDs
        """
        chunks = []
        chunk_ids = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            chunk_id = f"{id_prefix}_chunk_{i // (chunk_size - overlap)}"
            chunk_ids.append(chunk_id)
            
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "chunk_index": i // (chunk_size - overlap),
                "chunk_text": chunk
            })
            
            chunks.append({
                "id": chunk_id,
                "text": chunk,
                "metadata": chunk_metadata
            })
        
        self.upsert_documents_batch(chunks, namespace, model)
        return chunk_ids
    
    # ==================== READ/QUERY METHODS ====================
    
    def query_by_vector(
        self,
        vector: List[float],
        top_k: int = 10,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
        include_values: bool = False,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Query the index using a vector.
        
        Args:
            vector: Query vector
            top_k: Number of results to return
            namespace: Namespace to query
            filter: Metadata filter
            include_values: Include vector values in results
            include_metadata: Include metadata in results
            
        Returns:
            Query results
        """
        if not self.index:
            raise ValueError("No index connected. Call connect_to_index() first")
        
        return self.index.query(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_values=include_values,
            include_metadata=include_metadata
        )
    
    def query_by_text(
        self,
        text: str,
        top_k: int = 10,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
        include_values: bool = False,
        include_metadata: bool = True,
        model: str = "text-embedding-ada-002"
    ) -> Dict[str, Any]:
        """
        Query the index using text (generates embedding automatically).
        
        Args:
            text: Query text
            top_k: Number of results to return
            namespace: Namespace to query
            filter: Metadata filter
            include_values: Include vector values in results
            include_metadata: Include metadata in results
            model: OpenAI embedding model
            
        Returns:
            Query results
        """
        vector = self.generate_embedding(text, model)
        return self.query_by_vector(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_values=include_values,
            include_metadata=include_metadata
        )
    
    def query_by_id(
        self,
        id: str,
        top_k: int = 10,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
        include_values: bool = False,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Query the index using an existing vector ID.
        
        Args:
            id: ID of the vector to use as query
            top_k: Number of results to return
            namespace: Namespace to query
            filter: Metadata filter
            include_values: Include vector values in results
            include_metadata: Include metadata in results
            
        Returns:
            Query results
        """
        if not self.index:
            raise ValueError("No index connected. Call connect_to_index() first")
        
        return self.index.query(
            id=id,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_values=include_values,
            include_metadata=include_metadata
        )
    
    def fetch_vectors(
        self,
        ids: List[str],
        namespace: str = ""
    ) -> Dict[str, Any]:
        """
        Fetch specific vectors by their IDs.
        
        Args:
            ids: List of vector IDs to fetch
            namespace: Namespace to fetch from
            
        Returns:
            Dictionary of fetched vectors
        """
        if not self.index:
            raise ValueError("No index connected. Call connect_to_index() first")
        
        return self.index.fetch(ids=ids, namespace=namespace)
    
    def search_by_metadata(
        self,
        filter: Dict[str, Any],
        top_k: int = 10,
        namespace: str = "",
        include_values: bool = False,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Search vectors using metadata filters only.
        Note: Requires a dummy vector for the query.
        
        Args:
            filter: Metadata filter dictionary
            top_k: Number of results to return
            namespace: Namespace to query
            include_values: Include vector values in results
            include_metadata: Include metadata in results
            
        Returns:
            Query results
        """
        dummy_vector = [0.0] * self.dimension
        
        return self.query_by_vector(
            vector=dummy_vector,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_values=include_values,
            include_metadata=include_metadata
        )
    
    # ==================== DELETE METHODS ====================
    
    def delete_by_ids(
        self,
        ids: List[str],
        namespace: str = ""
    ) -> None:
        """
        Delete vectors by their IDs.
        
        Args:
            ids: List of vector IDs to delete
            namespace: Namespace to delete from
        """
        if not self.index:
            raise ValueError("No index connected. Call connect_to_index() first")
        
        self.index.delete(ids=ids, namespace=namespace)
        print(f"Deleted {len(ids)} vectors")
    
    def delete_by_filter(
        self,
        filter: Dict[str, Any],
        namespace: str = ""
    ) -> None:
        """
        Delete vectors matching a metadata filter.
        
        Args:
            filter: Metadata filter dictionary
            namespace: Namespace to delete from
        """
        if not self.index:
            raise ValueError("No index connected. Call connect_to_index() first")
        
        self.index.delete(filter=filter, namespace=namespace)
        print(f"Deleted vectors matching filter: {filter}")
    
    def delete_all(self, namespace: str = "") -> None:
        """
        Delete all vectors in a namespace.
        
        Args:
            namespace: Namespace to clear (empty string for default)
        """
        if not self.index:
            raise ValueError("No index connected. Call connect_to_index() first")
        
        self.index.delete(delete_all=True, namespace=namespace)
        print(f"Deleted all vectors in namespace: '{namespace}'")
    
    # ==================== UTILITY METHODS ====================
    
    def list_indexes(self) -> List[str]:
        """
        List all available indexes.
        
        Returns:
            List of index names
        """
        return [index.name for index in self.pc.list_indexes()]
    
    def format_results(
        self,
        results: Dict[str, Any],
        include_scores: bool = True,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Format query results into a more readable structure.
        
        Args:
            results: Raw query results from Pinecone
            include_scores: Include similarity scores
            include_metadata: Include metadata
            
        Returns:
            List of formatted result dictionaries
        """
        formatted = []
        
        for match in results.get("matches", []):
            result = {"id": match["id"]}
            
            if include_scores:
                result["score"] = match.get("score")
            
            if include_metadata and "metadata" in match:
                result["metadata"] = match["metadata"]
                if "text" in match["metadata"]:
                    result["text"] = match["metadata"]["text"]
            
            formatted.append(result)
        
        return formatted
    
    def __repr__(self) -> str:
        """String representation of the PineconeManager."""
        return (
            f"PineconeManager(index='{self.index_name}', "
            f"dimension={self.dimension}, metric='{self.metric}')"
        )



if __name__ == "__main__":
    pinecone_manager = PineconeManager()
    print(f"Connected to index: {pinecone_manager.index_name}\n")
    
    query_text = "Я люблю мамины блины"
    top_k = 10
    print(f"Query: '{query_text}' (searching across all namespaces)\n")
    
    stats = pinecone_manager.get_index_stats()
    namespaces_dict = getattr(stats, "namespaces", None) or stats.get("namespaces", {})
    namespace_list = list(namespaces_dict.keys()) if namespaces_dict else [""]
    
    if not namespace_list or (len(namespace_list) == 1 and namespace_list[0] == ""):
        namespace_list = [""]
    
    all_matches = []
    for ns in namespace_list:
        result = pinecone_manager.query_by_text(
            text=query_text,
            top_k=top_k,
            namespace=ns,
            include_values=True,
            include_metadata=True
        )
        for m in result.matches:
            match_with_ns = (m, ns)
            all_matches.append(match_with_ns)
    
    all_matches.sort(key=lambda x: x[0].score or 0, reverse=True)
    all_matches = all_matches[:top_k]
    
    print("=" * 80)
    print("QUERY RESULTS")
    print("=" * 80)
    
    if all_matches:
        for i, (match, ns) in enumerate(all_matches, 1):
            print(f"\n--- Match {i} (namespace: {ns or '(default)'}) ---")
            print(f"ID: {match.id}")
            print(f"Score: {match.score:.10f}")
            
            if match.metadata:
                print(f"\nMetadata:")
                for key, value in match.metadata.items():
                    if key == 'text':
                        print(f"  {key}: {value}")
                    else:
                        print(f"  {key}: {value}")
            
            if match.values:
                print(f"\nVector (first 10 dimensions): {match.values[:10]}")
                print(f"Vector dimension: {len(match.values)}")
            
            print("-" * 80)
    else:
        print("No matches found")
    
    print(f"\nNamespaces in index: {namespace_list}")
    print(f"Total matches: {len(all_matches)}") 