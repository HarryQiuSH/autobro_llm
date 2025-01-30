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
import uuid


import streamlit as st
import llm_tools as llm

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage


st.set_page_config(
    page_title="AutoBro LLM app", 
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

st.html("""<h4 style="text-align: center;">🔍 <i> Get stuck looking for info in maintenance manual? </i> </h4>""")

st.title("AutoBro Chatbot")
st.subheader("Check information about your car with AutoBro Chatbot")

# Sidebar for API settings
st.sidebar.title("API Settings")

# Filter selection for API provider
api_provider = st.sidebar.selectbox("Select API Provider", ["OpenAI", "Anthropic", "DeepSeek"])


openai_key = None
anthropic_key = None
deepseek_key = None
MODELS = [
    # "openai/o1-mini",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/o1-2024-12-17",
    "openai/o1-mini",
    "anthropic/claude-3-5-sonnet-20240620",
    "deepseek/deepseek-reasoner",
    "deepseek/deepseek-chat"
]

# Input field for API key based on selection
if api_provider == "OpenAI":
    openai_key = st.sidebar.text_input("Enter your OpenAI Secret Key", type="password")
elif api_provider == "Anthropic":
    anthropic_key = st.sidebar.text_input("Enter your Anthropic Key", type="password")
elif api_provider == "DeepSeek":
    deepseek_key = st.sidebar.text_input("Enter your DeepSeek Key", type="password")

# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?\n 你好, 有什么可以帮您？"}
    ]

# --- Main Content ---
# Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
missing_openai = openai_key == "" or openai_key is None or "sk-" not in openai_key
missing_anthropic = anthropic_key == "" or anthropic_key is None
missing_deepseek = deepseek_key == "" or deepseek_key is None
if missing_openai and missing_anthropic and missing_deepseek:
    st.write("#")
    st.warning("⬅️ Please introduce an API Key to continue...")
else:
    with st.sidebar:
        st.divider()
        models = []
        for model in MODELS:
            if "openai" in model and not missing_openai:
                models.append(model)
            elif "anthropic" in model and not missing_anthropic:
                models.append(model)
            elif "deepseek" in model and not missing_deepseek:
                models.append(model)
        st.selectbox(
            "🤖 Select a Model", 
            options=models,
            key="model",
        )
        cols0 = st.columns(2)
        with cols0[1]:
            st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")



    # Main chat app
    model_provider = st.session_state.model.split("/")[0]
    if model_provider == "openai":
        llm_stream = ChatOpenAI(
            api_key=openai_key,
            model_name=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
        )
    elif model_provider == "anthropic":
        llm_stream = ChatAnthropic(
            api_key=anthropic_key,
            model=st.session_state.model.split("/")[-1],
            temperature=0.3,
            streaming=True,
        )
    elif model_provider == "deepseek":
        llm_stream = ChatOpenAI(
            api_key=deepseek_key,
            model=st.session_state.model.split("/")[-1],
            temperature=0.5,
            streaming=True,
            base_url="https://api.deepseek.com/v1"
        )


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])



    if prompt := st.chat_input("Anything you like to check about your car?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]

            st.write_stream(llm.stream_llm_response(llm_stream, messages))
