"""
Auto Bro Chatbot Application

This Streamlit application serves as a chatbot interface for users to check information about their cars.
It maintains a chat history and displays previous messages upon app rerun.

Author: Harry Qiu

Modules:
    - streamlit: Provides the Streamlit library for creating the web application.

Usage:
    Run the script using Streamlit to start the chatbot application.
"""

import hashlib
import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

import llm_tools as llm

# RAG imports
from document_processor import DocumentProcessor
from rag_engine import RAGChatManager, RAGEngine

# Load environment variables
load_dotenv()
# Retrieve the API key and password hash from .env
api_key = os.getenv("API_KEY")
stored_password_hash = os.getenv("PASSWORD")
DRIVE_API_KEY = os.getenv("GOOGLE")
DRIVE_FOLDER_ID = "1Mm8vIu0Wflpm6oILFIyM_YsZ02l4d-yn"


st.set_page_config(
    page_title="Universal LLM hub",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.html("""<h4 style="text-align: center;">🔍 <i> API access gets you fair answer among LLM providers! </i> </h4>""")

st.title("HQ Agentic Helper")
st.subheader("Api access is the real access! ")

# Sidebar for API settings
st.sidebar.title("Model API Setup")

user_password = st.sidebar.text_input("Enter your password 请输入密码", type="password")
user_password_hash = hashlib.sha256(user_password.encode()).hexdigest()

if user_password_hash == stored_password_hash:
    # Filter selection for API provider
    api_provider = st.sidebar.selectbox("Select API Provider", ["", "OpenAI", "Anthropic", "DeepSeek"])

    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_KEY")
    MODELS = {
        "GPT4o": "openai/gpt-4o",
        "GPT4.1": "openai/gpt-4.1",
        "GPT4omini": "openai/gpt-4o-mini",
        "o1": "openai/o1",
        "o3": "openai/o3",
        "Claude4 Opus": "anthropic/claude-opus-4-20250514",
        "Claude4 Sonnet": "anthropic/claude-4-sonnet-20250514",
        "Claude3.5 Sonnet": "anthropic/claude-3-5-sonnet-20241022",
        "Claude3.7 Sonnet": "anthropic/claude-3-7-sonnet-20250219",
        "Claude3.5 Haiku": "anthropic/claude-3-5-haiku-20241022",
        "DeepSeek R1": "deepseek/deepseek-reasoner",
        "DeepSeek V3": "deepseek/deepseek-chat",
    }

    # --- Initial Setup ---
    if (
        "session_id" not in st.session_state
        or "rag_sources" not in st.session_state
        or "messages" not in st.session_state
        or "rag_engine" not in st.session_state
        or "rag_chat_manager" not in st.session_state
        or "document_processor" not in st.session_state
        or "uploaded_documents" not in st.session_state
        or "rag_mode" not in st.session_state
    ):
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        if "rag_sources" not in st.session_state:
            st.session_state.rag_sources = []
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "user", "content": "Let's go!"}]
        # RAG Setup
        if "rag_engine" not in st.session_state:
            embedding_type = "openai" if openai_key and "sk-" in openai_key else "local"
            st.session_state.rag_engine = RAGEngine(
                embedding_type=embedding_type,
                openai_api_key=openai_key if embedding_type == "openai" else None,
            )
        if "rag_chat_manager" not in st.session_state:
            st.session_state.rag_chat_manager = RAGChatManager(st.session_state.rag_engine)
        if "document_processor" not in st.session_state:
            st.session_state.document_processor = DocumentProcessor()
        if "uploaded_documents" not in st.session_state:
            st.session_state.uploaded_documents = []
        if "rag_mode" not in st.session_state:
            st.session_state.rag_mode = False
    # --- Main Content ---
    # Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
    missing_openai = openai_key == "" or openai_key is None or "sk-" not in openai_key
    missing_anthropic = anthropic_key == "" or anthropic_key is None
    missing_deepseek = deepseek_key == "" or deepseek_key is None
    if missing_openai and missing_anthropic and missing_deepseek:
        st.write("#")
        st.warning("⬅️ Please introduce an API Key to continue...")
    if api_provider:
        with st.sidebar:
            st.divider()
            models = []
            for model_name, address in MODELS.items():
                if (
                    ("openai" in address and api_provider == "OpenAI")
                    or ("anthropic" in address and api_provider == "Anthropic")
                    or ("deepseek" in address and api_provider == "DeepSeek")
                ):
                    models.append(model_name)
            st.selectbox(
                "🤖 Select a Model",
                options=models,
                key="model",
            )
            cols0 = st.columns(2)
            with cols0[1]:
                st.button(
                    "Clear Chat",
                    on_click=lambda: st.session_state.messages.clear(),
                    type="primary",
                )

            # RAG Section
            st.divider()
            st.subheader("📚 RAG Documents")

            # RAG Mode Toggle
            st.session_state.rag_mode = st.checkbox(
                "Enable RAG Mode",
                value=st.session_state.rag_mode,
                help="Use uploaded documents to enhance responses",
            )

            # File Upload
            uploaded_files = st.file_uploader(
                "Upload Documents",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                help="Upload PDF, DOCX, or TXT files for RAG",
            )

            # Process uploaded files
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in [doc["filename"] for doc in st.session_state.uploaded_documents]:
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            processed_doc = st.session_state.document_processor.process_uploaded_file(uploaded_file)
                            if processed_doc:
                                st.session_state.uploaded_documents.append(processed_doc)

                                # Add to vector store
                                chunked_docs = st.session_state.document_processor.chunk_documents([processed_doc])
                                st.session_state.rag_engine.add_documents(chunked_docs)

                                st.success(f"✅ {uploaded_file.name} processed successfully!")

            # Display uploaded documents
            if st.session_state.uploaded_documents:
                st.write("**Uploaded Documents:**")
                for doc in st.session_state.uploaded_documents:
                    st.write(f"• {doc['filename']} ({doc['file_type'].upper()})")

                # Clear documents button
                if st.button("Clear All Documents", type="secondary"):
                    st.session_state.uploaded_documents = []
                    st.session_state.rag_engine.clear_vector_store()
                    st.success("All documents cleared!")
                    st.rerun()

            # Vector store info
            vector_info = st.session_state.rag_engine.get_vector_store_info()
            if vector_info["count"] > 0:
                st.info(f"📊 Vector Store: {vector_info['count']} chunks indexed")

        # Main chat app
        model_provider = MODELS[st.session_state.model].split("/")[0]
        if model_provider == "openai":
            model_name = MODELS[st.session_state.model].split("/")[-1]
            temperature_value = 1 if model_name.startswith("o") else 0.5
            llm_stream = ChatOpenAI(
                api_key=openai_key,
                model_name=model_name,
                temperature=temperature_value,
                streaming=True,
            )
        elif model_provider == "anthropic":
            llm_stream = ChatAnthropic(
                api_key=anthropic_key,
                model=MODELS[st.session_state.model].split("/")[-1],
                temperature=0.5,
                streaming=True,
            )
        elif model_provider == "deepseek":
            llm_stream = ChatOpenAI(
                api_key=deepseek_key,
                model=MODELS[st.session_state.model].split("/")[-1],
                temperature=0.5,
                streaming=True,
                base_url="https://api.deepseek.com/v1",
            )

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Anything you like to check?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # Check if RAG mode is enabled and we have documents
                if st.session_state.rag_mode and st.session_state.uploaded_documents:
                    # Process with RAG
                    rag_result = st.session_state.rag_chat_manager.process_rag_query(prompt, st.session_state.messages)

                    # Use augmented prompt for LLM
                    final_prompt = rag_result["augmented_prompt"]
                    context_docs = rag_result["context_docs"]

                    # Show retrieved context in an expander
                    if context_docs:
                        with st.expander("📄 Retrieved Context", expanded=False):
                            for i, doc in enumerate(context_docs):
                                st.write(f"**Source {i + 1}: {doc['source']}**")
                                st.write(doc["content"][:300] + "..." if len(doc["content"]) > 300 else doc["content"])
                                st.divider()

                    # Create messages with RAG context
                    messages = [
                        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                        for m in st.session_state.messages[:-1]  # Exclude the last user message
                    ]
                    # Add the augmented prompt as the final message
                    messages.append(HumanMessage(content=final_prompt))

                else:
                    # Regular chat without RAG
                    messages = [
                        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                        for m in st.session_state.messages
                    ]
                    context_docs = []

                # Stream the response
                response_generator = llm.stream_llm_response(llm_stream, messages)
                response = st.write_stream(response_generator)

                # Add sources to response if RAG was used
                if st.session_state.rag_mode and context_docs:
                    final_response = st.session_state.rag_chat_manager.format_rag_response_with_sources(
                        response,
                        context_docs,
                    )
                    # Update the last message with sources
                    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                        st.session_state.messages[-1]["content"] = final_response
else:
    st.warning("⬅️ Please input correct password to continue...")
