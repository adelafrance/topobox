import requests
import tomllib
import json
import os
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# Load Secrets
try:
    with open(".streamlit/secrets.toml", "rb") as f:
        secrets = tomllib.load(f)
    print("Secrets loaded.")
except Exception as e:
    print(f"Secrets Error: {e}")
    exit(1)

SCOPES = ['https://www.googleapis.com/auth/drive']
FOLDER_ID = "19vjHtlChlrH56p8tbnZjgr9WkFK2qegD" # Shared Folder

def get_headers():
    if "gcp_service_account" not in secrets: return None
    try:
        creds = service_account.Credentials.from_service_account_info(
            secrets["gcp_service_account"], scopes=SCOPES
        )
        if not creds.valid: creds.refresh(Request())
        return {"Authorization": f"Bearer {creds.token}"}
    except Exception as e:
        print(f"Auth Error: {e}")
        return None

headers = get_headers()
if not headers:
    print("Failed to authenticate.")
    exit(1)

# 1. Find 'test_edit.txt'
print("Listing files in Shared Folder...")
query = f"'{FOLDER_ID}' in parents and trashed=false"
url_list = "https://www.googleapis.com/drive/v3/files"
params = {'q': query, 'fields': 'files(id, name)'}

try:
    resp = requests.get(url_list, headers=headers, params=params)
    files = resp.json().get('files', [])
except Exception as e:
    print(f"List Error: {e}")
    exit(1)

print(f"Found {len(files)} files:")
target_file_id = None
for f in files:
    print(f" - {f['name']} ({f['id']})")
    if f['name'] == 'test_edit.txt':
        target_file_id = f['id']

if not target_file_id:
    print("\nFile 'test_edit.txt' NOT FOUND.")
    exit(1)

file_id = target_file_id
print(f"\nFound target file! ID: {file_id}")

# 2. Update Content
print("Attempting to UPDATE content...")
url_update = f"https://www.googleapis.com/upload/drive/v3/files/{file_id}?uploadType=media"

new_content = "The Robot was here! 🤖✅"

try:
    # We use PATCH to update metadata, but 'upload/drive/v3/files/ID' with method=PATCH allows content update?
    # Actually for content update, we typically use PATCH on the upload endpoint.
    resp = requests.patch(
        url_update, 
        headers=headers, 
        data=new_content,
        timeout=30
    )
    
    if resp.status_code == 200:
        print("SUCCESS! File content updated.")
        print("Result:", resp.json())
        print("CONCLUSION: The Robot CAN edit your files.")
    else:
        print("FAILURE.")
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text}")
        print("CONCLUSION: The Robot CANNOT edit your files (Quota Limit applies to edits too?).")

except Exception as e:
    print(f"Update Exception: {e}")
