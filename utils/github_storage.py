import streamlit as st
import requests
import json
import base64

def get_config():
    """Retrieves GitHub config from st.secrets."""
    if "github" not in st.secrets:
        return None
    return st.secrets["github"]

def push_json(filename, data, message="New Submission"):
    """
    Pushes a JSON file to the 'submissions/' directory in the repo.
    Handles both Create (new file) and Update (overwrite existing).
    """
    config = get_config()
    if not config:
        return "❌ GitHub Secrets ('[github]' block) missing in secrets.toml."
        
    token = config.get("token")
    owner = config.get("owner")
    repo = config.get("repo")
    
    if not token or not owner or not repo:
         return "❌ GitHub Config incomplete (token, owner, repo required)."
    
    # Target Path
    path = f"submissions/{filename}"
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # 1. Check if file exists (We need the SHA to update it)
    sha = None
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            sha = resp.json().get("sha")
    except Exception as e:
        return f"❌ Connection Error during check: {e}"
        
    # 2. Prepare Payload
    try:
        content_str = json.dumps(data, indent=4)
        content_b64 = base64.b64encode(content_str.encode("utf-8")).decode("utf-8")
        
        payload = {
            "message": f"{message} ({filename})",
            "content": content_b64,
            "branch": "main"
        }
        if sha:
            payload["sha"] = sha
            
        # 3. PUT Request
        resp = requests.put(url, headers=headers, json=payload, timeout=20)
        
        if resp.status_code in [200, 201]:
            return None # Success (None = no error)
        else:
            return f"❌ GitHub Error {resp.status_code}: {resp.json().get('message', resp.text)}"
            
    except Exception as e:
        return f"❌ Connection Error during upload: {e}"
