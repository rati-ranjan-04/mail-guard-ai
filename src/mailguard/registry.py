"""Model versioning, integrity manifest and loading.

``models/manifest.json`` records the SHA-256 of the active pipeline file. joblib files
are pickles (loading one can run arbitrary code), so the engine verifies the hash before
loading and the README tells users to load only models they trained themselves.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def write_manifest(model_path: Path, manifest_path: Path, extra: dict | None = None) -> dict:
    import sklearn
    manifest = {
        "model_file": Path(model_path).name,
        "sha256": sha256_file(model_path),
        "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version_at_registration": sklearn.__version__,
        **(extra or {}),
    }
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_manifest(manifest_path: Path) -> dict | None:
    try:
        return json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def verify_model(model_path: Path, manifest_path: Path) -> tuple[str, str]:
    """Return (status, message); status in {'verified','unverified','mismatch','missing'}."""
    if not Path(model_path).exists():
        return "missing", "Model file not found."
    manifest = read_manifest(manifest_path)
    if not manifest or "sha256" not in manifest:
        return "unverified", "No integrity manifest found; model hash could not be verified."
    if sha256_file(model_path) != manifest["sha256"]:
        return "mismatch", "Model file hash does not match models/manifest.json."
    return "verified", "Model file hash matches the manifest."


def make_version_id(data_sha: str, params: dict, seed: int, sklearn_version: str) -> str:
    """Deterministic id: same data + params + seed + sklearn => same version."""
    blob = json.dumps({"d": data_sha, "p": params, "s": seed, "v": sklearn_version}, sort_keys=True)
    return "nb-" + sha256_text(blob)[:10]
