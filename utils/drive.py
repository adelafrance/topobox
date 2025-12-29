import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import os

# SCOPES
SCOPES = ['https://www.googleapis.com/auth/drive.file']
FOLDER_NAME = "TopoBox_Submissions"

def get_drive_service():
    """Authenticates and returns the Drive API service."""
    try:
        if "gcp_service_account" not in st.secrets:
            return None
            
        # Create credentials from the secrets dict
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=SCOPES
        )
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(f"Drive Auth Error: {e}")
        return None

def get_folder_id(service, folder_name):
    """Finds or creates the 'TopoBox_Submissions' folder."""
    try:
        # Search for existing folder
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        files = results.get('files', [])
        
        if files:
            return files[0]['id']
        
        # Create if not exists
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        file = service.files().create(body=file_metadata, fields='id').execute()
        return file.get('id')
    except Exception as e:
        print(f"Folder Error: {e}")
        return None

def upload_file(filepath, filename):
    """Uploads a local file to the Drive folder."""
    service = get_drive_service()
    if not service:
        return None

    folder_id = get_folder_id(service, FOLDER_NAME)
    if not folder_id:
        return None

    file_metadata = {
        'name': filename,
        'parents': [folder_id]
    }
    media = MediaFileUpload(filepath, mimetype='application/json')
    
    try:
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return file.get('id')
    except Exception as e:
        print(f"Upload Error: {e}")
        return None

def list_drive_files():
    """Lists JSON files in the submissions folder."""
    service = get_drive_service()
    if not service:
        return []
        
    folder_id = get_folder_id(service, FOLDER_NAME)
    if not folder_id:
        return []
        
    query = f"'{folder_id}' in parents and mimeType='application/json' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    return results.get('files', [])

def download_file(file_id, dest_path):
    """Downloads a file from Drive."""
    service = get_drive_service()
    if not service:
        return False
        
    try:
        request = service.files().get_media(fileId=file_id)
        from io import BytesIO
        fh = BytesIO()
        downloader = request.execute() # .execute() returns the content directly for get_media? 
        # Actually for googleapiclient get_media, strictly we might need MediaIoBaseDownload
        # But let's check simple 'execute' return type or use standard bytes.
        # Actually, simpler method:
        content = request.execute()
        with open(dest_path, 'wb') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Download Error: {e}")
        return False
