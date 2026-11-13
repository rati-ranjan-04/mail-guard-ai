"""Safe e-mail parsing (pasted text, raw RFC 822 or .eml bytes).

Attachments are only *described* (name, type, size). They are never written
to disk, opened or executed. HTML is converted to text with a tolerant parser
that discards <script>/<style>; nothing is rendered.
"""

from __future__ import annotations

import email
import re
from dataclasses import dataclass, field
from email import policy
from html.parser import HTMLParser

from .security import MAX_TEXT_CHARS

_HEADER_HINTS = (
    "from", "to", "date", "received", "message-id", "return-path", "reply-to",
    "mime-version", "authentication-results", "received-spf", "dkim-signature",
    "content-type",
)
_STRONG_HINTS = ("received", "authentication-results", "return-path", "message-id",
                 "received-spf", "dkim-signature", "from", "mime-version")


@dataclass
class ParsedEmail:
    subject: str = ""
    body_text: str = ""
    from_header: str = ""
    reply_to: str = ""
    return_path: str = ""
    to_header: str = ""
    auth_results: list = field(default_factory=list)
    received_spf: list = field(default_factory=list)
    dkim_signatures: list = field(default_factory=list)
    html_links: list = field(default_factory=list)      # [(href, anchor_text)]
    attachments: list = field(default_factory=list)     # [{name, content_type, size}]
    has_headers: bool = False
    has_html: bool = False
    truncated: bool = False

    @property
    def ml_text(self) -> str:
        """Text given to the classifier (subject + body, as in the training data)."""
        return f"{self.subject}\n{self.body_text}".strip()


class _HTMLToText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._skip = 0
        self._href: str | None = None
        self._anchor: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self._skip += 1
        elif tag == "a":
            self._href = dict(attrs).get("href") or ""
            self._anchor = []
        elif tag in ("br", "p", "div", "tr", "li"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self._skip = max(0, self._skip - 1)
        elif tag == "a" and self._href is not None:
            self.links.append((self._href.strip(), " ".join("".join(self._anchor).split())))
            self._href = None

    def handle_data(self, data):
        if self._skip:
            return
        self.parts.append(data)
        if self._href is not None:
            self._anchor.append(data)


def html_to_text(markup: str):
    parser = _HTMLToText()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:
        pass
    text = re.sub(r"[ \t]+", " ", "".join(parser.parts))
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    return text, parser.links


def _looks_like_headers(raw: str) -> bool:
    head = raw.lstrip()[:8000]
    block = head.split("\n\n", 1)[0] if "\n\n" in head else head.split("\r\n\r\n", 1)[0]
    names = re.findall(r"(?mi)^([A-Za-z][A-Za-z0-9-]{1,40}):[ \t]", block)
    lowered = {n.lower() for n in names}
    if lowered & set(_STRONG_HINTS) and len(lowered & set(_HEADER_HINTS)) >= 2:
        return True
    return len(lowered & set(_HEADER_HINTS)) >= 3


def _hdr(msg, name: str) -> str:
    try:
        value = msg.get(name)
        return " ".join(str(value).split()) if value is not None else ""
    except Exception:
        return ""


def parse_email(raw) -> ParsedEmail:
    """Parse raw text/bytes into a ParsedEmail. Never raises on malformed mail."""
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = bytes(raw).decode("utf-8")
        except UnicodeDecodeError:
            raw = bytes(raw).decode("latin-1")
    raw = raw or ""
    truncated = len(raw) > MAX_TEXT_CHARS
    raw = raw[:MAX_TEXT_CHARS]

    parsed = ParsedEmail(truncated=truncated)
    if not _looks_like_headers(raw):
        # Plain text, optionally with a leading "Subject:" line (dataset style).
        m = re.match(r"(?is)^\s*subject:[ \t]*([^\n]*)\n(.*)$", raw)
        if m and "\n" in raw.strip():
            parsed.subject, parsed.body_text = m.group(1).strip(), m.group(2).strip()
        else:
            parsed.body_text = raw.strip()
        return parsed

    try:
        msg = email.message_from_string(raw, policy=policy.default)
    except Exception:
        parsed.body_text = raw.strip()
        return parsed

    parsed.has_headers = True
    parsed.subject = _hdr(msg, "Subject")
    parsed.from_header = _hdr(msg, "From")
    parsed.reply_to = _hdr(msg, "Reply-To")
    parsed.return_path = _hdr(msg, "Return-Path")
    parsed.to_header = _hdr(msg, "To")
    parsed.auth_results = [" ".join(str(v).split()) for v in (msg.get_all("Authentication-Results") or [])]
    parsed.received_spf = [" ".join(str(v).split()) for v in (msg.get_all("Received-SPF") or [])]
    parsed.dkim_signatures = [" ".join(str(v).split()) for v in (msg.get_all("DKIM-Signature") or [])]

    texts, htmls = [], []
    try:
        for part in msg.walk():
            if part.is_multipart():
                continue
            ctype = part.get_content_type()
            fname = part.get_filename()
            disp = (part.get_content_disposition() or "").lower()
            if fname or disp == "attachment":
                try:
                    size = len(part.get_payload(decode=True) or b"")
                except Exception:
                    size = 0
                parsed.attachments.append({"name": str(fname or "(unnamed)")[:255],
                                           "content_type": ctype, "size": size})
                continue
            try:
                content = part.get_content()
            except Exception:
                payload = part.get_payload(decode=True) or b""
                content = payload.decode("utf-8", "replace")
            if ctype == "text/plain":
                texts.append(str(content))
            elif ctype == "text/html":
                htmls.append(str(content))
    except Exception:
        pass

    body = "\n".join(texts).strip()
    if htmls:
        parsed.has_html = True
        for markup in htmls:
            html_text, links = html_to_text(markup)
            parsed.html_links.extend(links)
            if not body:
                body = html_text
    parsed.body_text = body
    return parsed
