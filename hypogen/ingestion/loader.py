"""Paper and corpus loading utilities.

Responsibilities:
- Read individual paper files as plain text.
- Walk a directory and load all ``.txt`` files as a corpus.
- Return consistent ``(filename, content)`` tuples so downstream components
  always know which paper a claim came from.
"""

from __future__ import annotations

import logging
from pathlib import Path

from hypogen.exceptions import CorpusLoadError

logger = logging.getLogger("hypogen.ingestion")


def load_paper(path: Path) -> str:
    """Read a single research paper text file and return its content.

    Parameters
    ----------
    path:
        Absolute or relative path to a ``.txt`` file.

    Returns
    -------
    str
        The full text content of the paper.

    Raises
    ------
    CorpusLoadError
        If the file does not exist, is not readable, or is empty.
    """
    path = Path(path)
    if not path.exists():
        raise CorpusLoadError(f"File not found: {path}", path=str(path))
    if not path.is_file():
        raise CorpusLoadError(f"Path is not a file: {path}", path=str(path))

    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise CorpusLoadError(f"Cannot read file: {exc}", path=str(path)) from exc

    if not content:
        raise CorpusLoadError(f"File is empty: {path}", path=str(path))

    logger.debug("Loaded paper: %s (%d chars)", path.name, len(content))
    return content


def load_corpus(corpus_dir: Path) -> list[tuple[str, str]]:
    """Load all ``.txt`` files from a directory as a corpus.

    Each file is treated as one paper. Files are sorted alphabetically for
    deterministic ordering.

    Parameters
    ----------
    corpus_dir:
        Directory containing ``.txt`` paper files. Non-``.txt`` files are
        silently skipped.

    Returns
    -------
    list[tuple[str, str]]
        List of ``(filename, content)`` tuples, one per paper.

    Raises
    ------
    CorpusLoadError
        If the directory does not exist or contains no ``.txt`` files.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise CorpusLoadError(
            f"Corpus directory not found: {corpus_dir}", path=str(corpus_dir)
        )
    if not corpus_dir.is_dir():
        raise CorpusLoadError(
            f"Path is not a directory: {corpus_dir}", path=str(corpus_dir)
        )

    txt_files = sorted(corpus_dir.glob("*.txt"))
    if not txt_files:
        raise CorpusLoadError(
            f"No .txt files found in {corpus_dir}", path=str(corpus_dir)
        )

    corpus: list[tuple[str, str]] = []
    errors: list[str] = []

    for fp in txt_files:
        try:
            content = load_paper(fp)
            corpus.append((fp.name, content))
        except CorpusLoadError as exc:
            logger.warning("Skipping %s: %s", fp.name, exc)
            errors.append(str(exc))

    if not corpus:
        raise CorpusLoadError(
            f"All files in {corpus_dir} failed to load. Errors: {errors}",
            path=str(corpus_dir),
        )

    logger.info(
        "Loaded corpus: %d papers from %s", len(corpus), corpus_dir
    )
    return corpus


def load_corpus_from_file(path: Path) -> list[tuple[str, str]]:
    """Load a single file and treat it as a one-paper corpus.

    Useful for quick single-document analysis without setting up a directory.

    Parameters
    ----------
    path:
        Path to a single ``.txt`` file.

    Returns
    -------
    list[tuple[str, str]]
        A single-element list: ``[(filename, content)]``.

    Raises
    ------
    CorpusLoadError
        If the file cannot be loaded.
    """
    path = Path(path)
    content = load_paper(path)
    logger.info("Loaded single-file corpus: %s", path.name)
    return [(path.name, content)]
