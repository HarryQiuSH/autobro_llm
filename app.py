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

import base64
import hashlib
import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from openai import OpenAI

import llm_tools as llm

# RAG imports
from document_processor import DocumentProcessor
from rag_engine import RAGChatManager, RAGEngine

# Load environment variables
load_dotenv()
# Retrieve the API key and password hash from .env


def generate_image_bytes(openai_api_key: str, prompt: str, size: str = "1024x1024", n: int = 1):
    """
    Generate images via OpenAI Images API (gpt-image-1).
    Returns a list of bytes objects (PNG).

    This function is robust to different SDK response shapes: dicts or objects
    where fields may be attributes (e.g., b64_json, b64, url) or dict keys.
    """
    if not openai_api_key or "sk-" not in openai_api_key:
        raise ValueError("OPENAI_API_KEY missing or invalid")

    client = OpenAI(api_key=openai_api_key)
    resp = client.images.generate(model="gpt-image-1", prompt=prompt, size=size, n=n)

    # Normalize data list whether resp is dict-like or object-like
    if isinstance(resp, dict):
        data_list = resp.get("data", []) or []
    else:
        data_list = getattr(resp, "data", []) or []

    images = []
    for item in data_list:
        # helper to safely get value from dict or object
        def _get(obj, key):
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        b64 = _get(item, "b64_json") or _get(item, "b64")
        url = _get(item, "url")

        if b64:
            # sometimes b64 may already be bytes (less common), handle both
            if isinstance(b64, (bytes, bytearray)):
                images.append(bytes(b64))
            else:
                images.append(base64.b64decode(b64))
        elif url:
            r = requests.get(url)
            r.raise_for_status()
            images.append(r.content)
        else:
            # As a final fallback, try to interpret the item itself as base64-encodable string
            try:
                raw = str(item)
                images.append(base64.b64decode(raw))
            except Exception:
                raise RuntimeError("No image data returned from OpenAI (unexpected response shape).")
    return images


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
        "GPT5": "openai/gpt-5",
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
            # Temperature slider for model responses (set per-session in sidebar)
            # Default heuristic: models starting with "o" use higher default temperature
            default_temp = 1.0 if (st.session_state.get("model") and str(st.session_state.get("model")).startswith("o")) else 0.5
            st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=default_temp,
                step=0.01,
                key="temperature",
                help="Controls randomness of model responses (0.0 - deterministic, 1.0 - creative)",
            )
            # Sidebar toggle to show/hide image generation UI in the main area
            st.checkbox(
                "Enable Image Generation",
                value=st.session_state.get("show_image_ui", False),
                key="show_image_ui",
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
            # Use temperature from sidebar slider if present; otherwise fall back to model heuristic
            temperature_value = st.session_state.get("temperature", 1.0 if model_name.startswith("o") else 0.5)
            llm_stream = ChatOpenAI(
                api_key=openai_key,
                model_name=model_name,
                temperature=temperature_value,
                streaming=True,
            )

        # Image generation UI for OpenAI (separate from chat)
        if api_provider == "OpenAI" and openai_key and "sk-" in openai_key:
            # Only show image UI when the sidebar toggle is enabled.
            if st.session_state.get("show_image_ui", False):
                st.divider()
                st.subheader("🖼️ Image generation (OpenAI)")
                img_prompt = st.text_input(
                    "Image prompt",
                    value="",
                    placeholder="Describe the image you want",
                    key="img_prompt",
                )
                size = st.selectbox(
                    "Size",
                    ["1024*1536", "1536*1024", "1024x1024", "auto"],
                    index=2,
                    key="img_size",
                )
                n_images = st.slider("Number of images", 1, 4, 1, key="img_n")
                if st.button("Generate image", key="generate_image_btn"):
                    if not img_prompt.strip():
                        st.warning("Enter an image prompt first.")
                    else:
                        with st.spinner("Generating image..."):
                            try:
                                imgs = generate_image_bytes(openai_key, img_prompt, size=size, n=n_images)
                                # Save generated images in session_state so they can be cleared when toggled off
                                st.session_state.generated_images = imgs
                            except Exception as e:
                                st.error(f"Image generation failed: {e}")
                # Display generated images when the UI is enabled
                if st.session_state.get("generated_images"):
                    for i, img_bytes in enumerate(st.session_state.generated_images):
                        st.image(
                            img_bytes,
                            caption=f"Generated image #{i + 1}",
                            use_column_width=True,
                        )
                        st.download_button(
                            f"Download image #{i + 1}",
                            data=img_bytes,
                            file_name=f"image_{i + 1}.png",
                            mime="image/png",
                        )
            else:
                # When the image UI is disabled, clear any previously generated images and rerun to refresh the UI
                if st.session_state.get("generated_images"):
                    del st.session_state["generated_images"]
                    st.experimental_rerun()
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
