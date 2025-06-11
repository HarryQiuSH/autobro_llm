"""
RAG Engine Module

This module handles the Retrieval-Augmented Generation functionality,
including vector storage, similarity search, and context retrieval.

Author: Harry Qiu
"""

from typing import Any

import streamlit as st
from langchain.schema import Document as LangchainDocument

# LangChain imports
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Sentence transformers for local embeddings
from sentence_transformers import SentenceTransformer


class RAGEngine:
    """Handles RAG functionality including vector storage and retrieval."""

    def __init__(self, embedding_type: str = "openai", openai_api_key: int | None = None):
        """
        Initialize RAG engine with specified embedding type.

        Args:
            embedding_type: Either "openai" or "local"
            openai_api_key: Required if using OpenAI embeddings
        """
        self.embedding_type = embedding_type
        self.embeddings = None
        self.vector_store = None
        self.retriever = None

        self._initialize_embeddings(openai_api_key)
        self._initialize_vector_store()

    def _initialize_embeddings(self, openai_api_key: str | None = None):
        """Initialize embedding model based on type."""
        try:
            if self.embedding_type == "openai" and openai_api_key:
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=openai_api_key, model="text-embedding-3-small"
                )
                st.success("✅ OpenAI embeddings initialized")
            else:
                # Use local sentence transformers model
                self.embeddings = LocalEmbeddings()
                st.success("✅ Local embeddings initialized")
        except Exception as e:
            st.error(f"Failed to initialize embeddings: {str(e)}")
            # Fallback to local embeddings
            self.embeddings = LocalEmbeddings()

    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store."""
        try:
            if self.embeddings:
                self.vector_store = Chroma(
                    embedding_function=self.embeddings, persist_directory="./chroma_db"
                )
                st.success("✅ Vector store initialized")
        except Exception as e:
            st.error(f"Failed to initialize vector store: {str(e)}")

    def add_documents(self, documents: list[LangchainDocument]) -> bool:
        """Add documents to the vector store."""
        if not self.vector_store or not documents:
            return False

        try:
            # Add documents to vector store
            self.vector_store.add_documents(documents)

            # Update retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            )

            st.success(f"✅ Added {len(documents)} document chunks to vector store")
            return True

        except Exception as e:
            st.error(f"Failed to add documents to vector store: {str(e)}")
            return False

    def retrieve_context(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        """Retrieve relevant context for a query."""
        if not self.retriever:
            return []

        try:
            # Retrieve relevant documents
            docs = self.retriever.get_relevant_documents(query)

            # Format results
            context_docs = []
            for doc in docs[:k]:
                context_docs.append(
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "source": doc.metadata.get("filename", "Unknown"),
                    }
                )

            return context_docs

        except Exception as e:
            st.error(f"Failed to retrieve context: {str(e)}")
            return []

    def create_rag_prompt(self, query: str, context_docs: list[dict[str, Any]]) -> str:
        """Create an augmented prompt with retrieved context."""
        if not context_docs:
            return query

        # Build context section
        context_text = "\n\n".join(
            [f"**Source: {doc['source']}**\n{doc['content']}" for doc in context_docs]
        )

        return f"""
        Based on the following context information, please answer the question.
        If the answer cannot be found in the context, please say so.

**Context:**
{context_text}

**Question:** {query}

**Answer:**"""

    def get_vector_store_info(self) -> dict[str, Any]:
        """Get information about the current vector store."""
        if not self.vector_store:
            return {"status": "not_initialized", "count": 0}

        try:
            # Get collection info
            collection = self.vector_store._collection
            count = collection.count()

            return {
                "status": "active",
                "count": count,
                "embedding_type": self.embedding_type,
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "count": 0}

    def clear_vector_store(self) -> bool:
        """Clear all documents from the vector store."""
        try:
            if self.vector_store:
                # Delete the collection and recreate
                self.vector_store.delete_collection()
                self._initialize_vector_store()
                self.retriever = None
                st.success("✅ Vector store cleared")
                return True
        except Exception as e:
            st.error(f"Failed to clear vector store: {str(e)}")
            return False


class LocalEmbeddings:
    """Local embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize local embedding model."""
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            st.error(f"Failed to load local embedding model: {str(e)}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        if not self.model:
            return []

        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            st.error(f"Failed to embed documents: {str(e)}")
            return []

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        if not self.model:
            return []

        try:
            embedding = self.model.encode([text], convert_to_tensor=False)
            return embedding[0].tolist()
        except Exception as e:
            st.error(f"Failed to embed query: {str(e)}")
            return []


class RAGChatManager:
    """Manages RAG-enhanced chat functionality."""

    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine

    def process_rag_query(self, query: str, chat_history: list[dict[str, str]]) -> dict[str, Any]:
        """Process a query with RAG enhancement."""
        # Retrieve relevant context
        context_docs = self.rag_engine.retrieve_context(query)

        # Create augmented prompt
        augmented_prompt = self.rag_engine.create_rag_prompt(query, context_docs) if context_docs else query

        return {
            "augmented_prompt": augmented_prompt,
            "context_docs": context_docs,
            "original_query": query,
        }

    def format_rag_response_with_sources(
        self, response: str, context_docs: list[dict[str, Any]]
    ) -> str:
        """Format response with source citations."""
        if not context_docs:
            return response

        # Add sources section
        sources = set()
        for doc in context_docs:
            source = doc.get("source", "Unknown")
            sources.add(source)

        if sources:
            sources_list = [f"• {source}" for source in sorted(sources)]
            sources_text = "\n\n**Sources:**\n" + "\n".join(sources_list)
            return response + sources_text

        return response
