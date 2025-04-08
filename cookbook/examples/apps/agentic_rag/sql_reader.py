import io
from pathlib import Path
from typing import List, Union

from agno.document.base import Document
from agno.document.reader.base import Reader
from agno.utils.log import log_info, logger


class SQLScriptReader(Reader):
    """Reader for SQL files (.sql, stored procedures, view defs, etc)"""

    def read(self, file: Union[Path, io.BytesIO]) -> List[Document]:
        try:
            # Detect name
            if isinstance(file, Path):
                log_info(f"Reading: {file}")
                raw_bytes = file.read_bytes()
                sql_name = file.stem
            else:
                log_info(f"Reading uploaded file: {file.name}")
                file.seek(0)
                raw_bytes = file.read()
                sql_name = file.name.split(".")[0]

            # Try decoding with fallback
            for encoding in ["utf-8-sig", "utf-16", "latin-1"]:
                try:
                    text = raw_bytes.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise UnicodeDecodeError("Could not decode SQL file using fallback encodings.")

            # Build document
            documents = [
                Document(
                    name=sql_name,
                    id=sql_name,
                    content=text.strip()
                )
            ]
            if self.chunk:
                chunked_documents = []
                for document in documents:
                    chunked_documents.extend(self.chunk_document(document))
                return chunked_documents

            return documents

        except Exception as e:
            logger.error(f"Error reading SQL file: {e}")
            return []
