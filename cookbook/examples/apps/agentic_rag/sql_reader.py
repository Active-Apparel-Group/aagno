import io
import hashlib
from pathlib import Path
from typing import List, Union

from agno.document.base import Document
from agno.document.reader.base import Reader
from agno.utils.log import log_info, logger


def generate_unique_id(content: str, name: str) -> str:
    """Generate a stable, unique ID using the filename and content hash"""
    short_hash = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
    return f"{name}_{short_hash}"


class SQLScriptReader(Reader):
    """Reader for SQL files (.sql, stored procedures, view defs, etc)"""

    def read(self, file: Union[Path, io.BytesIO]) -> List[Document]:
        try:
            # Detect name and read raw bytes
            if isinstance(file, Path):
                log_info(f"Reading: {file}")
                raw_bytes = file.read_bytes()
                base_name = file.stem
            else:
                log_info(f"Reading uploaded file: {file.name}")
                file.seek(0)
                raw_bytes = file.read()
                base_name = file.name.rsplit(".", 1)[0]

            # Try decoding with fallbacks
            for encoding in ["utf-8-sig", "utf-16", "latin-1"]:
                try:
                    text = raw_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise UnicodeDecodeError("Could not decode SQL file using fallback encodings.")

            doc_id = generate_unique_id(text, base_name)

            document = Document(
                id=doc_id,
                name=base_name,
                content=text.strip(),
                meta_data={"source": file.name if hasattr(file, "name") else str(file)}
            )

            if self.chunk:
                return self.chunk_document(document)
            return [document]

        except Exception as e:
            logger.error(f"Error reading SQL file: {e}")
            return []
