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
import dotenv
import uuid
if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import streamlit as st


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
api_provider = st.sidebar.selectbox("Select API Provider", ["OpenAI", "Anthropic"])

# Input field for API key based on selection
if api_provider == "OpenAI":
    openai_key = st.sidebar.text_input("Enter your OpenAI Secret Key", type="password")
elif api_provider == "Anthropic":
    claude_key = st.sidebar.text_input("Enter your Claude Key", type="password")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Anything you like to check about your car?")
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")
