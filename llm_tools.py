
import os
import dotenv
from time import time
import streamlit as st
import io
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build

dotenv.load_dotenv()

def stream_llm_response(llm_stream, messages):
    response_message = ""

    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})

import io
from googleapiclient.http import MediaIoBaseDownload

def list_drive_files(folder_id, drive_api_key):
    service = build('drive', 'v3', developerKey=drive_api_key)
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    return results.get('files', [])

def download_file_content(file_id, drive_api_key):
    service = build('drive', 'v3', developerKey=drive_api_key)
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    # Adjust the decoding/processing based on file type (assuming text here)
    return fh.read().decode('utf-8')