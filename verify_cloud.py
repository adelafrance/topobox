import streamlit as st
import os
import json
import tomllib  # Standard library in 3.11

# Load secrets manually
try:
    with open(".streamlit/secrets.toml", "rb") as f:
        secrets = tomllib.load(f)
    # Monkey patch st.secrets to be a dict
    st.secrets = secrets
    print("Secrets loaded from .streamlit/secrets.toml")
except Exception as e:
    print(f"Secrets load failed: {e}")
    exit(1)

# Now we can import drive, which uses st.secrets
from utils import drive

# 1. Test Listing
print("\n--- TEST 1: LISTING ---")
files = drive.list_drive_files()
print(f"Found {len(files)} files.")
for f in files:
    print(f" - {f['name']} ({f['id']})")

# 2. Test Upload
print("\n--- TEST 2: UPLOAD ---")
real_file = "submissions/wuppertal_0_test3_SUBMISSION.json"
if os.path.exists(real_file):
    print(f"Uploading real file: {real_file}")
    target_file = real_file
else:
    print("Real file not found, creating dummy.")
    dummy_file = "test_upload_verify.json"
    with open(dummy_file, "w") as f:
        json.dump({"test": "data", "source": "verify_script"}, f)
    target_file = dummy_file

file_id = drive.upload_file(target_file, os.path.basename(target_file))

if file_id:
    print(f"Upload Result ID: {file_id}")
else:
    print("Upload returned None (Failed).")

# 3. Verify
print("\n--- TEST 3: VERIFY ---")
files_after = drive.list_drive_files()
found = False
for f in files_after:
    if f['id'] == file_id:
        print(f"SUCCESS: Found uploaded file {f['name']} with ID {f['id']}")
        found = True
        break

if not found:
    print("FAILURE: Uploaded file not found in list.")

# Cleanup
if os.path.exists(dummy_file):
    os.remove(dummy_file)
