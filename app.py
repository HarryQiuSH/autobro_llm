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

import os
import getpass
import hashlib
from dotenv import load_dotenv
import uuid

import streamlit as st
import llm_tools as llm

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage


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
    initial_sidebar_state="expanded"
)

st.html("""<h4 style="text-align: center;">🔍 <i> Api is the real access! </i> </h4>""")

st.title("Universal LLM hub")
st.subheader(" Api access is the real access! ")

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
        "GPT4omini": "openai/gpt-4o-mini",
        "o1-preview": "openai/o1-preview",
        "o1mini": "openai/o1-mini",
        "Claude3.5 Sonnet": "anthropic/claude-3-5-sonnet-20241022",
        "Claude3.7 Sonnet": "anthropic/claude-3-7-sonnet-20250219",
        "Claude3.5 Haiku": "anthropic/claude-3-5-haiku-20241022",
        "DeepSeek R1": "deepseek/deepseek-reasoner",
        "DeepSeek V3": "deepseek/deepseek-chat"
    }

    # --- Initial Setup ---
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "rag_sources" not in st.session_state:
        st.session_state.rag_sources = []

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "user", "content": "Let's go!"}]
    # --- Main Content ---
    # Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
    missing_openai = openai_key == "" or openai_key is None or "sk-" not in openai_key
    missing_anthropic = anthropic_key == "" or anthropic_key is None
    missing_deepseek = deepseek_key == "" or deepseek_key is None
    if missing_openai and missing_anthropic and missing_deepseek:
        st.write("#")
        st.warning("⬅️ Please introduce an API Key to continue...")
    elif api_provider:
        with st.sidebar:
            st.divider()
            models = []
            for model_name, address in MODELS.items():
                if "openai" in address and api_provider == "OpenAI":
                    models.append(model_name)
                elif "anthropic" in address and api_provider == "Anthropic":
                    models.append(model_name)
                elif "deepseek" in address and api_provider == "DeepSeek":
                    models.append(model_name)
            st.selectbox(
                "🤖 Select a Model",
                options=models,
                key="model",
            )
            cols0 = st.columns(2)
            with cols0[1]:
                st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

        # Main chat app
        model_provider = MODELS[st.session_state.model].split("/")[0]
        if model_provider == "openai":
            temperature_value = 1 if model_name.startswith("o1") else 0.5
            model_name = MODELS[st.session_state.model].split("/")[-1]
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
                base_url="https://api.deepseek.com/v1"
            )

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Anything you like to check?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                messages = [
                    HumanMessage(content=m["content"])
                    if m["role"] == "user"
                    else AIMessage(content=m["content"])
                    for m in st.session_state.messages
                ]

                st.write_stream(llm.stream_llm_response(llm_stream, messages))
else:
    st.warning("⬅️ Please input correct password to continue...")
