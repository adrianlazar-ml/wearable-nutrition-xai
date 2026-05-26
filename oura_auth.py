"""
OAuth2 flow pentru Oura Ring API v2.
Deschide browserul, prinde callback-ul pe localhost:8080,
obtine access token si il salveaza in .env.

Rulare: py oura_auth.py
"""

import os
import re
import webbrowser
import urllib.parse
import http.server
import threading
import requests
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("OURA_CLIENT_ID")
CLIENT_SECRET = os.getenv("OURA_CLIENT_SECRET")
REDIRECT_URI = os.getenv("OURA_REDIRECT_URI")

AUTH_URL = "https://cloud.ouraring.com/oauth/authorize"
TOKEN_URL = "https://api.ouraring.com/oauth/token"

access_token = None


def save_token_to_env(token):
    """
    Salveaza token-ul in .env, INLOCUIND valoarea existenta.
    FIX: Nu mai face append duplicat la fiecare rulare.
    """
    env_path = ".env"

    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            content = f.read()

        if "OURA_ACCESS_TOKEN" in content:
            content = re.sub(
                r"OURA_ACCESS_TOKEN=.*",
                f"OURA_ACCESS_TOKEN={token}",
                content
            )
        else:
            content = content.rstrip("\n") + f"\nOURA_ACCESS_TOKEN={token}\n"

        with open(env_path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(f"OURA_ACCESS_TOKEN={token}\n")


class CallbackHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        global access_token
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if "code" in params:
            code = params["code"][0]
            response = requests.post(TOKEN_URL, data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": REDIRECT_URI,
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
            })

            if response.status_code != 200:
                print(f"\nEroare la obtinere token: {response.status_code}")
                print(response.text)
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"Eroare la obtinere token.")
                return

            token_data = response.json()
            access_token = token_data.get("access_token")
            save_token_to_env(access_token)

            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Autorizare reusita! Poti inchide aceasta fereastra.")
            print(f"\nAccess token obtinut si salvat in .env")

    def log_message(self, format, *args):
        pass


def get_token():
    if not CLIENT_ID or not CLIENT_SECRET:
        print("EROARE: OURA_CLIENT_ID si OURA_CLIENT_SECRET trebuie setate in .env")
        return

    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": "email personal daily heartrate workout session spo2 stress",
    }
    url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

    server = http.server.HTTPServer(("localhost", 8080), CallbackHandler)
    thread = threading.Thread(target=server.handle_request)
    thread.start()

    print("Deschid browserul pentru autorizare Oura...")
    webbrowser.open(url)
    thread.join()
    server.server_close()


if __name__ == "__main__":
    get_token()
