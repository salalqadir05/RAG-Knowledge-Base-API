"""
Document Parser - Extracts plain text from various file formats.
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def parse_document(path: Path, ext: str) -> str:
    """Parse a document file and return extracted text."""
    ext = ext.lower()

    if ext == ".txt" or ext == ".md":
        return path.read_text(encoding="utf-8", errors="ignore")

    elif ext == ".pdf":
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            return "\n\n".join(text_parts)
        except ImportError:
            raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")

    elif ext == ".docx":
        try:
            from docx import Document
            doc = Document(path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except ImportError:
            raise RuntimeError("python-docx not installed. Run: pip install python-docx")

    elif ext == ".csv":
        try:
            import pandas as pd
            df = pd.read_csv(path)
            # Convert dataframe to readable text
            return df.to_string(index=False)
        except ImportError:
            raise RuntimeError("pandas not installed. Run: pip install pandas")

    else:
        raise ValueError(f"Unsupported file extension: {ext}")
