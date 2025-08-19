import os
import sys
import json
import time
import webbrowser
import argparse
import urllib.parse as up
from typing import Dict, List, Optional, Tuple

import requests
from flask import Flask, request
from tqdm import tqdm

# ====== Config (from env) ======
# CLIENT_ID = os.getenv("SKETCHFAB_CLIENT_ID")
# CLIENT_SECRET = os.getenv("SKETCHFAB_CLIENT_SECRET")
# REDIRECT_URI = os.getenv("SKETCHFAB_REDIRECT_URI")

CLIENT_ID = "L2NG1hQYb0vXhXNdzE4HAEF5sCBbScnBNt0Em1ck"
CLIENT_SECRET = "4tbvRrZ5rmdkZv8CHgDW70WTSxckN5YQEcK0mL5TQZMjI01Ug1hqa1JM2ZrJpXCtvdCCS0QuxxEGC9Tz01LpbeZllRrllwyo8Hv2n4KPLVniDMlysTmc8mgNsnQqg60k"
REDIRECT_URI = "http://127.0.0.1:5000/callback"
OAUTH_AUTHORIZE_URL = "https://sketchfab.com/oauth2/authorize/"
OAUTH_TOKEN_URL = "https://sketchfab.com/oauth2/token/"
API_BASE = "https://api.sketchfab.com/v3"

# ====== Simple local receiver for the authorization code ======
app = Flask(__name__)
auth_code_holder = {"code": None, "error": None}

@app.route("/callback")
def oauth_callback():
    error = request.args.get("error")
    code = request.args.get("code")
    if error:
        auth_code_holder["error"] = error
        return f"OAuth error: {error}. You can close this tab."
    if code:
        auth_code_holder["code"] = code
        return "Success! Authorization code received. You can close this tab and return to the app."
    return "Missing code. You can close this tab."

def get_access_token_interactive(scope: str = "read write") -> Dict:
    """
    Standard OAuth Authorization Code flow:
    1) Open browser to authorize
    2) Receive ?code=... on REDIRECT_URI
    3) Exchange code for tokens
    """
    # Start local Flask server in background thread
    from threading import Thread
    t = Thread(target=lambda: app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False))
    t.daemon = True
    t.start()

    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": scope,
    }
    url = f"{OAUTH_AUTHORIZE_URL}?{up.urlencode(params)}"
    print("Opening browser for Sketchfab authorization...")
    webbrowser.open(url)

    # Wait for the callback to fill auth_code_holder
    print("Waiting for authorization...")
    for _ in range(600):  # up to 60s
        if auth_code_holder["error"]:
            raise RuntimeError(f"OAuth error: {auth_code_holder['error']}")
        if auth_code_holder["code"]:
            break
        time.sleep(0.1)

    if not auth_code_holder["code"]:
        raise TimeoutError("Timed out waiting for OAuth authorization code.")

    code = auth_code_holder["code"]
    # print("code:", code)
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
    }
    r = requests.post(OAUTH_TOKEN_URL, data=data)

    if r.status_code >= 400:
        raise RuntimeError(f"Token exchange failed: {r.status_code} {r.text}")

    tokens = r.json()
    # tokens: {access_token, token_type, expires_in, refresh_token, scope}
    return tokens

def refresh_access_token(refresh_token: str) -> Dict:
    data = {
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": refresh_token,
    }
    r = requests.post(OAUTH_TOKEN_URL, data=data)
    if r.status_code >= 400:
        raise RuntimeError(f"Refresh failed: {r.status_code} {r.text}")
    return r.json()

def api_get(path: str, access_token: str, params: Optional[Dict] = None) -> Dict:
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"{API_BASE}{path}"
    r = requests.get(url, headers=headers, params=params or {}, timeout=30)
    if r.status_code == 401:
        raise PermissionError("Unauthorized (401). Token may be invalid/expired.")
    r.raise_for_status()
    return r.json()

def search_models(q: str, access_token: str, limit: int) -> List[Dict]:
    """
    Search free & downloadable models.
    The v3 search endpoint supports filtering; we request downloadable models.
    """
    results: List[Dict] = []
    page = 1
    page_size = 24  # Sketchfab paginates; typical size is 24
    while len(results) < limit:
        params = {
            "type": "models",
            "q": q,
            "downloadable": "true",  # filter for downloadable models
            "sort_by": "relevance",  # "relevance", "likeCount", etc.
            "page": page,
            "per_page": page_size,
        }
        data = api_get("/search", access_token, params)
        items = data.get("results", [])
        if not items:
            break
        # Keep only items that are actually downloadable and free (non-store)
        for it in items:
            if it.get("isDownloadable") and not it.get("isStoreModel", False):
                results.append(it)
                if len(results) >= limit:
                    break
        page += 1
    return results[:limit]

def choose_archive(download_info: Dict) -> Tuple[str, str]:
    """
    Pick GLB if available, otherwise GLTF (zip), otherwise USDZ (Apple).
    download_info looks like:
      { 'gltf': {'url': ...}, 'glb': {'url': ...}, 'usdz': {'url': ...}, ... }
    """
    for key in ("glb", "gltf", "usdz"):
        entry = download_info.get(key)
        if entry and entry.get("url"):
            url = entry["url"]
            # extension hint
            ext = ".glb" if key == "glb" else (".zip" if key == "gltf" else ".usdz")
            return url, ext

    return None, None

def request_download_url(uid: str, access_token: str) -> Dict:
    """
    Per Download API: first call /v3/models/{uid}/download with Bearer,
    which returns signed URLs for archives. Then download the file.
    """
    headers = {"Authorization": f"Bearer {access_token}"}
    r = requests.get(f"{API_BASE}/models/{uid}/download", headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def stream_download(url: str, out_path: str):
    try:
        while True:
            r = requests.get(url, stream=True, timeout=60)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 30))
                print(f"Rate limit exceeded. Waiting for {wait} seconds...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            break

        total = int(r.headers.get("Content-Length", 0))
        with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(out_path)) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except:
        print(f"Failed to download {url} to {out_path}")
        return False

def write_credit_row(csv_path: str, row: List[str], header: Optional[List[str]] = None):
    new_file = not os.path.exists(csv_path)
    import csv
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file and header:
            w.writerow(header)
        w.writerow(row)

def download_by_query(query: str, n: int, out_dir: str, tokens: Dict):
    access_token = tokens["access_token"]
    refresh_token = tokens.get("refresh_token")

    os.makedirs(out_dir, exist_ok=True)
    credits_csv = os.path.join(out_dir, "CREDITS.csv")
    write_credit_row(
        credits_csv,
        [],
        header=["model_uid", "title", "author", "author_profile", "license", "model_url", "saved_file"]
    )

    found = search_models(query, access_token, n * 2)
    if not found:
        print("No downloadable results found.")
        return

    download_cnt = 0
    for idx, item in enumerate(found, 1):
        uid = item["uid"]
        title = item.get("name") or f"model_{uid}"
        user = item.get("user", {})
        author = user.get("displayName") or user.get("username") or "unknown"
        author_profile = user.get("profileUrl") or user.get("profileUrl", "")
        model_url = item.get("viewerUrl") or item.get("uri") or f"https://sketchfab.com/3d-models/{uid}"
        license_label = (item.get("license") or {}).get("label", "")
        safe_title = "".join(ch for ch in title if ch.isalnum() or ch in "._- ").strip().replace(" ", "_")

        print(f"[{idx}/{len(found)}] Requesting download URL for: {title} ({uid})")
        try:
            info = request_download_url(uid, access_token)
        except requests.HTTPError as e:
            # Token might have expired; try refresh once
            if e.response is not None and e.response.status_code == 401 and refresh_token:
                print("Access token expired. Refreshing...")
                tokens_new = refresh_access_token(refresh_token)
                access_token = tokens_new["access_token"]
                info = request_download_url(uid, access_token)
            else:
                print(f"Failed to get download info for {title} ({uid}): {e}")
                continue

        url, ext = choose_archive(info)
        if url is None:
            print(f"No downloadable URL found for: {title} ({uid})")
            continue
        out_path = os.path.join(out_dir, f"{safe_title}_{uid}{ext}")
        if stream_download(url, out_path):
            download_cnt += 1
        else:
            print(f"Failed to download {title} ({uid}) to {out_path}")
            continue
        write_credit_row(
            credits_csv,
            [uid, title, author, author_profile, license_label, model_url, os.path.basename(out_path)]
        )
        if download_cnt >= n:
            break

def download_from_sketchfab(query: str, n: int = 10, out_dir: str = "./downloads"):
    tokens = get_access_token_interactive(scope="read write")
    print("Authorized. Starting search and download...")
    download_by_query(query, n, out_dir, tokens)
    print("Done. Remember to credit authors — see CREDITS.csv")

def main():
    parser = argparse.ArgumentParser(description="Download free Sketchfab models by keyword.")
    parser.add_argument("--q", required=True, help="Keyword/category (e.g., 'mug', 'chair', 'cat')")
    parser.add_argument("--n", type=int, default=10, help="How many models to download")
    parser.add_argument("--out", default="./downloads", help="Output directory")
    args = parser.parse_args()

    download_from_sketchfab(args.q, args.n, args.out)

if __name__ == "__main__":
    main()
