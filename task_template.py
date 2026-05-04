import sys
import argparse
from pathlib import Path

import requests


BASE = Path(__file__).parent
OUTPUT_CSV = BASE / "submission.csv"

BASE_URL = "http://34.63.153.158"   # DO NOT CHANGE
API_KEY = "0f6ca9a2cb7b67808c0b7b619fbcf665"       
TASK_ID = "01-mia"                  # DO NOT CHANGE


def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


parser = argparse.ArgumentParser(description="Submit a CSV file to the server.")
args = parser.parse_args()

submit_path = OUTPUT_CSV

if API_KEY == "YOUR_API_KEY_HERE":
    die("You forgot to replace API_KEY with your real API key.")

if not submit_path.exists():
    die(f"File not found: {submit_path}")

print("Submitting file:", submit_path)

try:
    with open(submit_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": (submit_path.name, f, "application/csv")},
            timeout=(10, 600),
        )

    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    if resp.status_code == 413:
        die("Upload rejected: file too large (HTTP 413).")

    resp.raise_for_status()

    print("Successfully submitted.")
    print("Server response:", body)

    submission_id = body.get("submission_id")
    if submission_id:
        print(f"Submission ID: {submission_id}")

except requests.exceptions.RequestException as e:
    detail = getattr(e, "response", None)
    print(f"Submission error: {e}")

    if detail is not None:
        try:
            print("Server response:", detail.json())
        except Exception:
            print("Server response (text):", detail.text)

    sys.exit(1)