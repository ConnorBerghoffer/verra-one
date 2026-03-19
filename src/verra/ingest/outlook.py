"""Microsoft Outlook / Office 365 email ingestion via Microsoft Graph API.

Authentication uses MSAL device-code flow — no client secret needed.
The user opens a URL in their browser, enters a code, and grants access.

Token cache is stored at ~/.verra/oauth/<account>_outlook_cache.json.

Usage:
    ingestor = OutlookIngestor("you@company.com", client_id="...")
    if ingestor.authenticate():
        stats = ingest_outlook(ingestor, metadata_store, vector_store)

Requires: msal (pip install msal)
"""


from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time
import urllib.parse
import urllib.request
from base64 import b64decode
from pathlib import Path
from typing import Any

from verra.config import VERRA_HOME
from verra.ingest.pipeline import IngestStats

logger = logging.getLogger(__name__)

_SCOPES = ["Mail.Read"]
_TOKEN_DIR = VERRA_HOME / "oauth"
_GRAPH_BASE = "https://graph.microsoft.com/v1.0"
_PAGE_SIZE = 50


class OutlookIngestor:
    """Microsoft Outlook/365 email ingestor via Graph API."""

    def __init__(self, account: str, client_id: str | None = None) -> None:
        self.account = account
        self.client_id = client_id
        self._access_token: str | None = None
        self._cache_path = _TOKEN_DIR / f"{account}_outlook_cache.json"

    def authenticate(self) -> bool:
        """Authenticate via MSAL device-code flow. Returns True on success."""
        try:
            import msal
        except ImportError:
            logger.error("msal is required for Outlook. Install with: pip install msal")
            return False

        if not self.client_id:
            self._print_setup_instructions()
            return False

        _TOKEN_DIR.mkdir(parents=True, exist_ok=True)

        cache = msal.SerializableTokenCache()
        if self._cache_path.exists():
            cache.deserialize(self._cache_path.read_text())

        app = msal.PublicClientApplication(
            self.client_id,
            authority="https://login.microsoftonline.com/common",
            token_cache=cache,
        )

        accounts = app.get_accounts()
        result = None
        if accounts:
            result = app.acquire_token_silent(_SCOPES, account=accounts[0])

        if not result or "access_token" not in result:
            flow = app.initiate_device_flow(scopes=_SCOPES)
            if "user_code" not in flow:
                logger.error("Could not initiate device code flow.")
                return False
            print(f"\n  To sign in, open: {flow['verification_uri']}")
            print(f"  Enter code: {flow['user_code']}\n")
            result = app.acquire_token_by_device_flow(flow)

        if "access_token" not in result:
            logger.error("Auth failed: %s", result.get("error_description", "unknown"))
            return False

        self._access_token = result["access_token"]
        if cache.has_state_changed:
            self._cache_path.write_text(cache.serialize())
            # Restrict token cache permissions — contains OAuth credentials
            try:
                self._cache_path.chmod(0o600)
            except OSError:
                pass
        return True

    def _print_setup_instructions(self) -> None:
        print(
            "\n  Outlook requires an Azure AD app registration. One-time setup:\n"
            "\n"
            "  1. Go to portal.azure.com\n"
            "  2. Navigate to Azure Active Directory -> App registrations -> New registration\n"
            "  3. Name your app (e.g. 'Verra'), set account type to\n"
            "     'Accounts in any organizational directory and personal Microsoft accounts'\n"
            "  4. Under 'Authentication', add a Mobile/desktop redirect URI:\n"
            "     https://login.microsoftonline.com/common/oauth2/nativeclient\n"
            "  5. Under 'API permissions', add Microsoft Graph -> Delegated -> Mail.Read\n"
            "  6. Copy the Application (client) ID from the Overview page\n"
            "\n"
            f"  Then run: verra outlook {self.account} --client-id <your-client-id>\n"
            "\n"
            "  The client ID will be saved to config so you only need to pass it once.\n"
        )

    def _graph_get(self, url: str) -> dict[str, Any]:
        """Authenticated GET to Microsoft Graph."""
        import ssl

        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
        })
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            return json.loads(resp.read())

    def fetch_messages(
        self,
        since: str | None = None,
        folder: str = "inbox",
        max_results: int = 500,
    ) -> list[dict[str, Any]]:
        """Fetch messages from Outlook via Graph API."""
        messages: list[dict[str, Any]] = []
        params: dict[str, str] = {
            "$top": str(min(max_results, _PAGE_SIZE)),
            "$select": "id,subject,from,toRecipients,ccRecipients,receivedDateTime,body,conversationId,hasAttachments",
            "$orderby": "receivedDateTime desc",
        }
        if since:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", since):
                raise ValueError(f"Invalid date format for 'since': {since!r} — expected YYYY-MM-DD")
            params["$filter"] = f"receivedDateTime ge {since}T00:00:00Z"

        url: str | None = f"{_GRAPH_BASE}/me/mailFolders/{folder}/messages?{urllib.parse.urlencode(params)}"

        while url and len(messages) < max_results:
            try:
                data = self._graph_get(url)
            except urllib.error.HTTPError as exc:
                if exc.code == 401:
                    raise RuntimeError("Outlook token expired. Re-authenticate.") from exc
                logger.warning("Graph API error %d", exc.code)
                break
            messages.extend(data.get("value", []))
            url = data.get("@odata.nextLink")

        return messages[:max_results]

    def fetch_attachments(self, message_id: str) -> list[dict[str, Any]]:
        """Fetch file attachments for a message."""
        url = f"{_GRAPH_BASE}/me/messages/{message_id}/attachments"
        try:
            data = self._graph_get(url)
        except Exception as exc:
            logger.warning("Failed to fetch attachments for message %s: %s", message_id, exc)
            return []
        return [
            a for a in data.get("value", [])
            if a.get("@odata.type") == "#microsoft.graph.fileAttachment"
        ]


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_NL = re.compile(r"\n{3,}")


def _strip_html(html: str) -> str:
    text = html.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    text = text.replace("</p>", "\n\n").replace("</div>", "\n")
    text = _TAG_RE.sub("", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return _MULTI_NL.sub("\n\n", text).strip()


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_outlook(
    ingestor: OutlookIngestor,
    metadata_store: Any,
    vector_store: Any,
    entity_store: Any | None = None,
    since: str | None = None,
    folder: str = "inbox",
    max_results: int = 500,
) -> IngestStats:
    """Ingest emails from Outlook/365 into the Verra knowledge base."""
    from verra.ingest.chunking import chunk_document
    from verra.ingest.email_cleaner import clean_email_body
    from verra.ingest.extractors import ContentExtractor

    stats = IngestStats()
    start = time.time()

    messages = ingestor.fetch_messages(since=since, folder=folder, max_results=max_results)
    stats.files_found = len(messages)

    for msg in messages:
        msg_id = msg.get("id", "")
        conv_id = msg.get("conversationId", "")
        subject = msg.get("subject", "(no subject)")
        received = msg.get("receivedDateTime", "")

        from_data = msg.get("from", {}).get("emailAddress", {})
        from_addr = from_data.get("address", "")
        to_addrs = ", ".join(
            r.get("emailAddress", {}).get("address", "")
            for r in msg.get("toRecipients", [])
        )
        cc_addrs = ", ".join(
            r.get("emailAddress", {}).get("address", "")
            for r in msg.get("ccRecipients", [])
        )

        body = msg.get("body", {})
        body_content = body.get("content", "")
        if body.get("contentType", "").lower() == "html":
            text = _strip_html(body_content)
        else:
            text = body_content

        text = clean_email_body(text)
        if not text.strip():
            stats.files_skipped += 1
            continue

        content_hash = hashlib.sha256(text.encode()).hexdigest()
        existing = metadata_store.get_document_by_hash(content_hash)
        if existing:
            stats.files_skipped += 1
            continue

        try:
            doc_id = metadata_store.add_document(
                file_path=f"outlook://{from_addr}/{msg_id}",
                file_name=subject,
                source_type="email",
                format="email",
                content_hash=content_hash,
                page_count=1,
                extra_metadata=json.dumps({
                    "from": from_addr, "to": to_addrs,
                    "subject": subject, "date": received,
                    "conversation_id": conv_id, "provider": "outlook",
                }),
                document_type="email",
                authority_weight=50,
            )
        except Exception as exc:
            stats.errors.append(f"Store error for {subject[:40]}: {exc}")
            continue

        full_text = f"From: {from_addr}\nTo: {to_addrs}\nSubject: {subject}\nDate: {received}\n\n{text}"
        chunks = chunk_document(
            full_text,
            metadata={
                "document_id": doc_id,
                "file_name": subject,
                "source_type": "email",
                "from": from_addr,
            },
        )

        try:
            chunk_ids = metadata_store.add_chunks(doc_id, chunks)
            vector_store.add_chunks(chunk_ids, chunks)
            stats.chunks_created += len(chunks)
        except Exception as exc:
            stats.errors.append(f"Chunk store error in {subject[:30]}: {exc}")

        try:
            metadata_store.add_email(
                thread_id=conv_id, message_id=msg_id,
                from_addr=from_addr, to_addr=to_addrs, cc_addr=cc_addrs,
                subject=subject, date=received, labels="[]", chunk_id=None,
            )
        except Exception as exc:
            stats.errors.append(f"Email metadata for {msg_id}: {exc}")

        # Attachments
        if msg.get("hasAttachments"):
            try:
                for att in ingestor.fetch_attachments(msg_id):
                    att_name = att.get("name", "attachment")
                    content_b64 = att.get("contentBytes", "")
                    if not content_b64:
                        continue
                    ext = Path(att_name).suffix.lower()
                    if ext not in (".pdf", ".docx", ".txt", ".csv", ".xlsx"):
                        continue
                    raw = b64decode(content_b64)
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode="wb") as tmp:
                        os.chmod(tmp.name, 0o600)
                        tmp.write(raw)
                        tmp_path = tmp.name
                    try:
                        att_text = ContentExtractor.extract(Path(tmp_path))
                        if att_text.strip():
                            att_hash = hashlib.sha256(att_text.encode()).hexdigest()
                            if not metadata_store.get_document_by_hash(att_hash):
                                att_doc_id = metadata_store.add_document(
                                    file_path=f"outlook-att://{msg_id}/{att_name}",
                                    file_name=att_name, source_type="email",
                                    format=ext.lstrip("."), content_hash=att_hash,
                                    page_count=1,
                                    extra_metadata=json.dumps({"parent_email": subject, "provider": "outlook"}),
                                )
                                att_chunks = chunk_document(
                                    att_text,
                                    metadata={
                                        "document_id": att_doc_id,
                                        "file_name": att_name,
                                        "source_type": "email",
                                    },
                                )
                                att_chunk_ids = metadata_store.add_chunks(att_doc_id, att_chunks)
                                vector_store.add_chunks(att_chunk_ids, att_chunks)
                                stats.chunks_created += len(att_chunks)
                    except Exception as exc:
                        stats.errors.append(f"Attachment {att_name}: {exc}")
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)
            except Exception as exc:
                stats.errors.append(f"Attachments for {subject[:30]}: {exc}")

        stats.files_processed += 1

    stats.elapsed_seconds = time.time() - start
    return stats
