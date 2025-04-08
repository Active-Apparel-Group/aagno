import io
from pathlib import Path
from typing import List, Union

from agno.document.base import Document
from agno.document.reader.base import Reader
from agno.utils.log import log_info, logger


class SQLScriptReader(Reader):
    """Reader for SQL scripts (.sql files)"""

    def read(self, file: Union[Path, io.BytesIO]) -> List[Document]:
        try:
            if isinstance(file, Path):
                log_info(f"Reading: {file}")
                sql_name = file.stem
                sql_content = file.read_text("utf-8")
            else:
                log_info(f"Reading uploaded file: {file.name}")
                sql_name = file.name.split(".")[0]
                file.seek(0)
                sql_content = file.read().decode("utf-8")

            document = Document(
                name=sql_name,
                id=sql_name,
                content=sql_content,
            )

            if self.chunk:
                return self.chunk_document(document)
            return [document]
        except Exception as e:
            logger.error(f"Error reading SQL file: {e}")
            return []