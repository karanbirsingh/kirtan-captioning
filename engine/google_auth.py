"""Google Cloud auth credentials management for end-customer distribution.

End customers can't be expected to install `gcloud` CLI and run
`application-default login`. The supported production flow is:

  1. Customer creates a service account in their own GCP project
     with role "Cloud Speech Client".
  2. Downloads the JSON key file.
  3. Uploads it to the app via Settings → Google Chirp 2.

This module handles persistence of that uploaded key and the (single)
function the ASR backend uses to obtain a fresh OAuth access token.

Auth resolution priority (first one that resolves wins):
  1. Uploaded service account JSON at ``<app data>/google_credentials.json``
     — the customer-facing path.
  2. Application Default Credentials (ADC) — the dev path, requires
     ``gcloud auth application-default login`` once. Kept as a fallback
     so the dev workflow continues to work without an uploaded key.

Storage location follows the same OS-native convention the sidecar uses
for its log file (see ``desktop/sidecar/server.py::_setup_file_logging``):

  Windows: ``%LOCALAPPDATA%/bani-mic/google_credentials.json``
  macOS:   ``~/Library/Application Support/bani-mic/google_credentials.json``
  Linux:   ``$XDG_DATA_HOME/bani-mic/google_credentials.json``

File permissions are set to 0o600 on POSIX (owner read/write only). On
Windows we don't tighten the ACL — the file lives under the user's
LOCALAPPDATA which is already per-user.

Why not the OS keychain? The ``keyring`` library doesn't ergonomically
store multi-KB blobs (service account JSON ~2KB), and many Linux
distros ship without a configured Secret Service. A 0o600 file under
the user's app-data dir is the same security posture as `gcloud`
itself, which is good enough for v1. Upgrade path is open: swap out
``save_credentials`` / ``load_credentials`` for keyring without
touching any other module.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("live_detection")


# ─── Storage location ─────────────────────────────────────────────────────

def _creds_dir() -> Path:
    """OS-native app-data directory for the sidecar (same as sidecar.log)."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return base / "bani-mic"


def creds_path() -> Path:
    """Absolute path to the saved service account key file."""
    return _creds_dir() / "google_credentials.json"


# ─── Validation + save / load / clear ─────────────────────────────────────

class InvalidServiceAccountKey(ValueError):
    """The uploaded JSON isn't a service account key we can use."""


def validate_service_account_json(data: Any) -> dict:
    """Return the parsed key dict if valid; raise ``InvalidServiceAccountKey``
    otherwise. Accepts either a parsed dict or a JSON string.

    Validation is intentionally narrow: we check the fields we'll actually
    use plus the ``type`` discriminator. We don't try to verify the key
    cryptographically here — that happens on first token exchange when
    google-auth actually tries to sign a JWT.
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            raise InvalidServiceAccountKey(f"Not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise InvalidServiceAccountKey("Top-level value must be a JSON object")
    if data.get("type") != "service_account":
        raise InvalidServiceAccountKey(
            f"Expected type=service_account, got type={data.get('type')!r}. "
            "Make sure you downloaded a service account key, not an OAuth "
            "client ID or user credentials."
        )
    for required in ("client_email", "private_key", "token_uri", "project_id"):
        if not data.get(required):
            raise InvalidServiceAccountKey(f"Missing required field: {required}")
    return data


def save_credentials(data: dict) -> dict:
    """Validate and persist a service account key. Returns a small dict
    describing what was saved (email, project_id) — never echoes the
    private key back to the caller.

    Atomicity: writes to a temp file in the same dir then ``os.replace``
    so a crash mid-write doesn't leave a half-truncated key file.
    """
    parsed = validate_service_account_json(data)
    path = creds_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(parsed), encoding="utf-8")
    if os.name == "posix":
        # Tighten before the rename so the published file is born 0o600
        # rather than going through a window of default perms.
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
    os.replace(tmp, path)
    logger.info("Saved Google service account credentials for %s (project=%s)",
                parsed["client_email"], parsed["project_id"])
    return {
        "email": parsed["client_email"],
        "project_id": parsed["project_id"],
    }


def load_credentials() -> Optional[dict]:
    """Read the saved key dict, or None if not present / corrupt."""
    path = creds_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Stored Google credentials are unreadable: %s", e)
        return None


def clear_credentials() -> bool:
    """Delete the saved key file. Returns True if a file was removed."""
    path = creds_path()
    if not path.exists():
        return False
    try:
        path.unlink()
        logger.info("Cleared Google service account credentials")
        return True
    except OSError as e:
        logger.warning("Could not delete %s: %s", path, e)
        return False


def credentials_status() -> dict:
    """Public-safe summary of the stored creds. Never includes the key.

    Returns ``{"connected": False}`` if nothing is saved, otherwise
    ``{"connected": True, "email": ..., "project_id": ...}``.
    """
    data = load_credentials()
    if data is None:
        return {"connected": False}
    return {
        "connected": True,
        "email": data.get("client_email", ""),
        "project_id": data.get("project_id", ""),
    }


# ─── Credential resolution for the ASR backend ────────────────────────────

# Speech-to-Text V2 (Chirp 2) requires this scope. cloud-platform is broad;
# we'd narrow if Google ever publishes a Speech-specific scope for V2.
SPEECH_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def get_credentials() -> tuple[Any, str]:
    """Return (google.auth.credentials.Credentials, project_id).

    Used by ``asr.GoogleCloudASR._get_access_token``. The returned
    Credentials object refreshes itself on demand via
    ``creds.refresh(google.auth.transport.requests.Request())``.

    Resolution order:
      1. Saved service account JSON (production customer flow)
      2. ADC via ``google.auth.default()`` (dev / gcloud fallback)

    Raises ``RuntimeError`` with an actionable message if neither path
    resolves — the caller surfaces it to the client so the user sees
    "upload a service account key" rather than a stack trace.
    """
    # Imported lazily so anything that imports this module (e.g. for the
    # validate / save helpers from a unit test) doesn't pay the cost of
    # google-auth's import-time setup.
    from google.oauth2 import service_account
    import google.auth
    import google.auth.exceptions

    data = load_credentials()
    if data is not None:
        try:
            creds = service_account.Credentials.from_service_account_info(
                data, scopes=[SPEECH_SCOPE]
            )
            return creds, data["project_id"]
        except (ValueError, KeyError) as e:
            logger.warning(
                "Stored Google credentials failed to load (%s); falling back to ADC.",
                e,
            )

    try:
        creds, project_id = google.auth.default(scopes=[SPEECH_SCOPE])
    except google.auth.exceptions.DefaultCredentialsError as e:
        raise RuntimeError(
            "No Google Cloud credentials available. Upload a service "
            "account key via Settings → Google Chirp 2, or run "
            "`gcloud auth application-default login` (dev path)."
        ) from e
    if not project_id:
        raise RuntimeError(
            "Google credentials loaded but no project_id is associated. "
            "Set `gcloud config set project <PROJECT>` (ADC path) or "
            "upload a service account key (which embeds the project_id)."
        )
    return creds, project_id


def get_access_token() -> tuple[str, str]:
    """Return (access_token, project_id) for an immediate API call.

    Convenience wrapper around ``get_credentials()`` that performs the
    refresh dance up front so ``GoogleCloudASR.transcribe_async`` can
    stay simple — it gets a string token, not a Credentials object.
    """
    from google.auth.transport.requests import Request as GAuthRequest

    creds, project_id = get_credentials()
    if not creds.valid:
        creds.refresh(GAuthRequest())
    return creds.token, project_id
