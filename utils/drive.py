import streamlit as st
import requests
import json
import os
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# SCOPES
SCOPES = ['https://www.googleapis.com/auth/drive'] # Full access required to see files created by others
FOLDER_NAME = "TopoBox_Submissions"
KNOWN_FOLDER_ID = "19vjHtlChlrH56p8tbnZjgr9WkFK2qegD" # Shared Folder ID

def get_creds():
    """Gets valid credentials from secrets."""
    if "gcp_service_account" not in st.secrets:
        return None
    try:
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=SCOPES
        )
        # Refresh if needed
        if not creds.valid:
            creds.refresh(Request())
        return creds
    except Exception as e:
        print(f"Auth Error: {e}")
        return None

def get_headers():
    """Returns headers with Bearer token."""
    creds = get_creds()
    if not creds: return None
    return {"Authorization": f"Bearer {creds.token}"}

# --- API METHODS ---

def get_folder_id(service_unused=None, folder_name=FOLDER_NAME):
    """Returns the hardcoded folder ID."""
    return KNOWN_FOLDER_ID

def list_drive_files():
    """Lists JSON files using Requests."""
    headers = get_headers()
    if not headers: return []
    
    folder_id = KNOWN_FOLDER_ID
    query = f"'{folder_id}' in parents and mimeType='application/json' and trashed=false"
    
    url = "https://www.googleapis.com/drive/v3/files"
    params = {
        'q': query,
        'fields': 'files(id, name)',
        'pageSize': 100
    }
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10) # Fast timeout
        if resp.status_code == 200:
            return resp.json().get('files', [])
        else:
            print(f"List Error: {resp.text}")
            return []
    except Exception as e:
        print(f"List Exception: {e}")
        return []

def upload_file(filepath, filename):
    """Uploads using Requests (Multipart)."""
    headers = get_headers()
    if not headers: return None

    folder_id = KNOWN_FOLDER_ID
    
    # Metadata
    metadata = {
        'name': filename,
        'parents': [folder_id]
    }
    
    files = {
        'data': ('metadata', json.dumps(metadata), 'application/json'),
        'file': (filename, open(filepath, 'rb'), 'application/json')
    }
    
    url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
    
    try:
        resp = requests.post(url, headers=headers, files=files, timeout=60)
        if resp.status_code == 200:
            return resp.json().get('id')
        else:
            raise Exception(f"Drive API Error {resp.status_code}: {resp.text}")
    except Exception as e:
        raise e  # Bubble up to UI

def download_file(file_id, dest_path=None):
    """Downloads content using Requests. Returns bytes if dest_path is None."""
    headers = get_headers()
    if not headers: return False
    
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    
    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=60)
        if resp.status_code == 200:
            if dest_path:
                with open(dest_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            else:
                return resp.content # Return bytes
        else:
            print(f"Download Fail: {resp.text}")
            return False
    except Exception as e:
        print(f"Download Exception: {e}")
        return False

def share_folder_with_user(user_email, role="writer"):
    """Shares folder using Requests."""
    headers = get_headers()
    if not headers: return "Auth Failed"
    
    url = f"https://www.googleapis.com/drive/v3/files/{KNOWN_FOLDER_ID}/permissions"
    
    body = {
        'role': role,
        'type': 'user',
        'emailAddress': user_email
    }
    
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=10)
        if resp.status_code == 200:
            return None # Success
        else:
            return f"Share Error: {resp.text}"
    except Exception as e:
        return f"Exception: {str(e)}"

# --- NEW DB METHODS ---

def get_file_id_by_name(filename):
    """Finds a file ID by exact name in the shared folder."""
    headers = get_headers()
    if not headers: return None
    
    query = f"'{KNOWN_FOLDER_ID}' in parents and name='{filename}' and trashed=false"
    url = "https://www.googleapis.com/drive/v3/files"
    params = {'q': query, 'fields': 'files(id, name)'}
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        files = resp.json().get('files', [])
        if files:
            return files[0]['id']
        return None
    except Exception as e:
        print(f"Find Error: {e}")
        return None

def update_file_content(file_id, new_content_str):
    """Updates the content of an existing file (Bypasses Quota)."""
    headers = get_headers()
    if not headers: return "Auth Failed"
    
    # uploadType=media is used for updating content via PATCH
    url = f"https://www.googleapis.com/upload/drive/v3/files/{file_id}?uploadType=media"
    
    try:
        # Request body is the raw content
        resp = requests.patch(url, headers=headers, data=new_content_str, timeout=30)
        if resp.status_code == 200:
            return None # Success
        else:
            return f"Update Error {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"Update Exception: {str(e)}"
