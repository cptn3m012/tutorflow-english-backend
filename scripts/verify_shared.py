"""Run the frontend's opt-in browser smoke test against a local sharing server."""
import argparse
import os
from pathlib import Path
import subprocess

from dotenv import dotenv_values

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--url", default="http://127.0.0.1:8012")
parser.add_argument("--frontend")
args = parser.parse_args()
root = Path(__file__).resolve().parent.parent
values = dotenv_values(root / ".env")
frontend = args.frontend or os.getenv("FRONTEND_PROJECT_DIR") or values.get("FRONTEND_PROJECT_DIR")
if not frontend:
    parser.error("Provide --frontend or configure FRONTEND_PROJECT_DIR in local .env.")
username = os.getenv("SHARE_USERNAME") or values.get("SHARE_USERNAME")
password = os.getenv("SHARE_PASSWORD") or values.get("SHARE_PASSWORD")
if not username or not password:
    raise SystemExit("Configure sharing credentials in backend .env first.")
environment = dict(os.environ)
environment.update({
    "TUTORFLOW_SHARED_URL": args.url,
    "TUTORFLOW_SHARED_USERNAME": username,
    "TUTORFLOW_SHARED_PASSWORD": password,
})
# Credentials pass only in the child environment, never command arguments/logs.
result = subprocess.run(
    ["npm.cmd" if os.name == "nt" else "npm", "--prefix", frontend, "run", "test:e2e"],
    env=environment, check=False,
)
raise SystemExit(result.returncode)
