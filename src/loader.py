from __future__ import annotations

from pathlib import Path

from schemas import ReferenceDocument


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf_file(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "PDF support requires pypdf. Install project dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"[Page {page_number}]\n{page_text.strip()}")

    return "\n\n".join(pages)


def _read_reference_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf_file(path)
    return _read_text_file(path)


def load_reference_documents(folder: str | Path) -> list[ReferenceDocument]:
    reference_folder = Path(folder).expanduser().resolve()
    if not reference_folder.exists():
        raise FileNotFoundError(f"Reference folder does not exist: {reference_folder}")
    if not reference_folder.is_dir():
        raise NotADirectoryError(f"Reference path is not a folder: {reference_folder}")

    documents: list[ReferenceDocument] = []
    for path in sorted(reference_folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            content = _read_reference_file(path)
        except OSError as exc:
            raise OSError(f"Could not read reference file {path}") from exc

        if not content.strip():
            continue

        documents.append(
            ReferenceDocument(
                filename=str(path.relative_to(reference_folder)),
                content=content,
            )
        )

    return documents
