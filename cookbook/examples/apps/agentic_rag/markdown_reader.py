import io
from pathlib import Path
from typing import List, Union

from agno.document.base import Document
from agno.document.reader.base import Reader
from agno.utils.log import log_info, logger


class MarkdownReader(Reader):
    """Reader for Markdown (.md) files"""

    def read(self, file: Union[Path, io.BytesIO]) -> List[Document]:
        try:
            if isinstance(file, Path):
                log_info(f"Reading: {file}")
                content = file.read_text(encoding="utf-8")
                doc_name = file.stem
            else:
                log_info(f"Reading uploaded file: {file.name}")
                file.seek(0)
                content = file.read().decode("utf-8")
                doc_name = file.name.split(".")[0]

            documents = [Document(name=doc_name, id=doc_name, content=content)]

            if self.chunk:
                chunked_documents = []
                for document in documents:
                    chunked_documents.extend(self.chunk_document(document))
                return chunked_documents

            return documents

        except Exception as e:
            logger.error(f"Error reading Markdown file: {e}")
            return []