"""Target inspection, slug/path helpers, manifest writer, LLM-facing formatter."""
from __future__ import annotations

import hashlib
import json
import mimetypes
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

from agent_harness import __version__ as _HARNESS_VERSION

_UNSAFE_SLUG = re.compile(r"[^A-Za-z0-9._-]")

_USER_AGENT = f"agent-harness/{_HARNESS_VERSION}"

_PDF_SUFFIXES = frozenset({".pdf"})
_IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".jp2", ".webp", ".gif", ".bmp"}
)

_MIME_TO_SUFFIX: dict[str, str] = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}

Kind = Literal["pdf", "image", "unknown"]


@dataclass(frozen=True)
class TargetInspection:
    is_local: bool
    size_bytes: int | None
    size_mb: float | None
    pages: int | None
    name: str
    mime: str | None
    kind: Kind


def is_local_path(target: str) -> bool:
    return not target.startswith(("http://", "https://"))


async def inspect_target(target: str) -> TargetInspection:
    if is_local_path(target):
        return _inspect_local(target)
    return await _inspect_remote(target)


def _kind_from_mime(mime: str | None) -> Kind:
    if mime is None:
        return "unknown"
    m = mime.lower().split(";")[0].strip()
    if m == "application/pdf":
        return "pdf"
    if m.startswith("image/"):
        return "image"
    return "unknown"


def _kind_from_suffix(suffix: str) -> Kind:
    s = suffix.lower()
    if s in _PDF_SUFFIXES:
        return "pdf"
    if s in _IMAGE_SUFFIXES:
        return "image"
    return "unknown"


def _inspect_local(target: str) -> TargetInspection:
    p = Path(target).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(target)
    size = p.stat().st_size
    kind = _kind_from_suffix(p.suffix)
    mime, _ = mimetypes.guess_type(str(p))

    pages: int | None
    if kind == "pdf":
        pages = _pdf_pages(p)
    elif kind == "image":
        pages = 1
    else:
        pages = None

    return TargetInspection(
        is_local=True,
        size_bytes=size,
        size_mb=size / 1024 / 1024,
        pages=pages,
        name=p.name,
        mime=mime,
        kind=kind,
    )


def _pdf_pages(p: Path) -> int | None:
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(p)).pages)
    except Exception:
        return None


async def _inspect_remote(target: str) -> TargetInspection:
    from agent_harness.utils.http_retry import http_head_with_retry
    name = Path(unquote(urlparse(target).path)).name or "remote"
    size: int | None = None
    mime: str | None = None
    try:
        status, headers = await http_head_with_retry(
            target, timeout=15,
            headers={"User-Agent": _USER_AGENT},
        )
        if status == 200:
            mime, size = _read_head_metadata(headers)
    except Exception:
        pass

    kind = _kind_from_mime(mime)
    if kind == "unknown":
        kind = _kind_from_suffix(Path(name).suffix)
    pages = 1 if kind == "image" else None
    return TargetInspection(
        is_local=False,
        size_bytes=size,
        size_mb=(size / 1024 / 1024) if size is not None else None,
        pages=pages,
        name=name,
        mime=mime,
        kind=kind,
    )


def _read_head_metadata(
    headers: Mapping[str, str],
) -> tuple[str | None, int | None]:
    mime: str | None = None
    size: int | None = None
    for k, v in headers.items():
        lk = k.lower()
        if lk == "content-length" and v.isdigit():
            size = int(v)
        elif lk == "content-type":
            mime = v.split(";")[0].strip().lower() or None
    return mime, size


def session_documents_root(session_id: str | None) -> Path:
    base = Path.home() / ".agent-harness"
    if session_id:
        return base / "sessions" / session_id / "documents"
    return base / "anonymous" / "documents"


def make_slug(
    *, source: str, content_hash: str | None, suggested: str | None = None,
) -> str:
    if suggested:
        base = _UNSAFE_SLUG.sub("_", suggested)[:64].strip("_") or "doc"
    else:
        stem = Path(unquote(urlparse(source).path)).stem
        base = _UNSAFE_SLUG.sub("_", stem)[:64].strip("_") or "doc"
    suffix = (content_hash or hashlib.sha256(source.encode()).hexdigest())[:8]
    return f"{base}_{suffix}"


def hash_source(target: str, is_local: bool) -> str:
    if is_local:
        p = Path(target).expanduser().resolve()
        if p.exists():
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(64 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
    return hashlib.sha256(target.encode("utf-8")).hexdigest()


def already_parsed(dir_: Path) -> bool:
    content_md = dir_ / "content.md"
    manifest = dir_ / "manifest.json"
    if not (content_md.exists() and manifest.exists()):
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict)


def write_manifest(
    dir_: Path, *,
    slug: str,
    source: dict[str, Any],
    backend: dict[str, str],
    size_bytes: int | None,
    mime: str | None,
    kind: Kind,
    page_count: int | None,
    image_count: int,
    content_md_tokens: int,
    content_md_lines: int,
    successful_tier_elapsed_ms: int,
    fallback_chain: list[dict[str, Any]],
    skipped_tiers: list[dict[str, str]],
) -> Path:
    manifest: dict[str, Any] = {
        "slug": slug,
        "source": source,
        "size_bytes": size_bytes,
        "mime": mime,
        "kind": kind,
        "backend": backend,
        "parsed_at": datetime.now().isoformat(),
        "artifacts": detect_artifacts(dir_),
        "stats": {
            "page_count": page_count,
            "image_count": image_count,
            "content_md_tokens": content_md_tokens,
            "content_md_lines": content_md_lines,
            "successful_tier_elapsed_ms": successful_tier_elapsed_ms,
        },
        "fallback_chain": fallback_chain,
        "skipped_tiers": skipped_tiers,
    }
    path = dir_ / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def detect_artifacts(dir_: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    if (dir_ / "content.md").exists():
        artifacts["content_md"] = "content.md"
    images = dir_ / "images"
    if images.is_dir() and any(images.iterdir()):
        artifacts["images_dir"] = "images/"
    if (dir_ / "layout.json").exists():
        artifacts["layout_json"] = "layout.json"
    return artifacts


def format_success(
    *,
    slug_dir: Path,
    source: str,
    name: str,
    kind: Kind,
    page_count: int | None,
    size_mb: float | None,
    backend_name: str,
    backend_model: str,
    content_md_tokens: int,
    content_md_lines: int,
    image_count: int,
) -> str:
    fmt_label = kind if kind != "unknown" else (
        Path(name).suffix.lower().lstrip(".") or "doc"
    )
    fmt_parts: list[str] = [fmt_label]
    if page_count is not None and size_mb is not None:
        fmt_parts.append(f"({page_count} pages, {size_mb:.1f} MB)")
    elif page_count is not None:
        fmt_parts.append(f"({page_count} pages)")
    elif size_mb is not None:
        fmt_parts.append(f"({size_mb:.1f} MB)")

    lines = [
        "Document parsed and saved.",
        f"  source : {source}",
        f"  format : {' '.join(fmt_parts)}",
        f"  backend: {backend_name} ({backend_model})",
        "",
        f"  content   {_homeify(slug_dir)}/content.md "
        f"({_fmt_tokens(content_md_tokens)} tokens, {content_md_lines} lines)",
    ]
    if image_count > 0:
        lines.append(
            f"  images    {_homeify(slug_dir)}/images/ ({image_count} figures)"
        )
    lines.append(f"  layout    {_homeify(slug_dir)}/layout.json")
    lines.append(f"  manifest  {_homeify(slug_dir)}/manifest.json")
    return "\n".join(lines)


def format_no_viable(
    skipped: list[dict[str, str]],
    chain: list[dict[str, Any]],
    unattempted: list[str] | None = None,
) -> str:
    unattempted = unattempted or []
    tier_widths: list[int] = [len(a["tier"]) for a in chain]
    tier_widths.extend(len(s["tier"]) for s in skipped)
    tier_widths.extend(len(n) for n in unattempted)
    tier_w = max(tier_widths) if tier_widths else 0
    mode_w = max((len(a["mode"]) for a in chain), default=0)

    lines: list[str] = ["Error: document parsing failed.", ""]
    if chain:
        lines.append("Tried:")
        grouped: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        for a in chain:
            if a["tier"] not in grouped:
                order.append(a["tier"])
                grouped[a["tier"]] = []
            grouped[a["tier"]].append(a)
        for i, tier in enumerate(order, 1):
            attempts = grouped[tier]
            for j, a in enumerate(attempts):
                head = f"  {i}. {tier:<{tier_w}}" if j == 0 else f"     {'':<{tier_w}}"
                lines.append(
                    f"{head} {a['mode']:<{mode_w}} "
                    f"{a['error_class']} ({a['error_message']})"
                )
    if unattempted:
        lines.append("")
        lines.append("Skipped (aborted):")
        for name in unattempted:
            lines.append(f"  - {name}")
    if skipped:
        lines.append("")
        lines.append("Skipped (preflight):")
        for s in skipped:
            lines.append(f"  - {s['tier']:<{tier_w}} {s['reason']}")
    return "\n".join(lines)


def format_cached(dest_dir: Path, target: str) -> str:
    raw = json.loads((dest_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest: dict[str, Any] = raw if isinstance(raw, dict) else {}

    stats = _opt_dict(manifest, "stats")
    backend = _opt_dict(manifest, "backend")
    source = _opt_dict(manifest, "source")

    size_bytes = _opt_int(manifest, "size_bytes")

    return format_success(
        slug_dir=dest_dir,
        source=target,
        name=_opt_str(source, "name") or Path(target).name,
        kind=_opt_kind(manifest),
        page_count=_opt_int(stats, "page_count"),
        size_mb=(size_bytes / 1024 / 1024) if size_bytes else None,
        backend_name=_opt_str(backend, "name") or "cached",
        backend_model=_opt_str(backend, "model") or "cached",
        content_md_tokens=_opt_int(stats, "content_md_tokens") or 0,
        content_md_lines=_opt_int(stats, "content_md_lines") or 0,
        image_count=_opt_int(stats, "image_count") or 0,
    )


def _homeify(p: Path) -> str:
    try:
        return "~/" + str(p.relative_to(Path.home()))
    except ValueError:
        return str(p)


def _fmt_tokens(n: int) -> str:
    return f"~{n / 1000:.1f}k" if n >= 1000 else f"~{n}"


def _opt_int(d: dict[str, Any], key: str) -> int | None:
    v = d.get(key)
    return v if isinstance(v, int) else None


def _opt_str(d: dict[str, Any], key: str) -> str | None:
    v = d.get(key)
    return v if isinstance(v, str) and v else None


def _opt_dict(d: dict[str, Any], key: str) -> dict[str, Any]:
    v = d.get(key)
    return v if isinstance(v, dict) else {}


def _opt_kind(d: dict[str, Any]) -> Kind:
    v = d.get("kind")
    if v == "pdf":
        return "pdf"
    if v == "image":
        return "image"
    return "unknown"
