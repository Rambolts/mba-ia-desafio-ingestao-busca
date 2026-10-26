import logging
import math
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from openai import APIError, OpenAI
from pgvector.psycopg import Vector, register_vector
from psycopg import sql
from pypdf import PdfReader
from pypdf.errors import PdfReadError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_DIMENSIONS = 1536
EMBEDDING_BATCH_SIZE = 64

logger = logging.getLogger(__name__)


def read_pdf(pdf_path: Path) -> str:
    """Extraia o texto das páginas na ordem, preservando suas quebras de linha."""
    if not pdf_path.is_file():
        raise ValueError(f"PDF não encontrado: {pdf_path}")

    reader = PdfReader(pdf_path)
    if reader.is_encrypted:
        raise ValueError("O PDF está protegido por senha. Use uma cópia desbloqueada.")

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\x00", "")
        if not text.strip():
            logger.warning("Página %d sem texto extraível; pode exigir OCR.", page_number)
        pages.append(text)

    text = "\n".join(pages).strip()
    if not text:
        raise ValueError("O PDF não contém texto extraível. Verifique se ele exige OCR.")

    logger.info("PDF lido: %d páginas, %d caracteres.", len(pages), len(text))
    return text


def split_text(text: str) -> list[str]:
    """Divida o texto em até 1.000 caracteres, repetindo os últimos 150."""
    chunks = []
    for start in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = text[start : start + CHUNK_SIZE]
        if not chunk.strip():
            raise ValueError("Foi encontrado um chunk sem conteúdo. Revise o texto do PDF.")
        chunks.append(chunk)
        if start + CHUNK_SIZE >= len(text):
            break
    return chunks


def generate_embeddings(chunks: list[str], model: str, api_key: str) -> list[list[float]]:
    """Gere e valide um embedding por chunk, mantendo a ordem original."""
    embeddings = []
    with OpenAI(api_key=api_key, timeout=60.0) as client:
        for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
            response = client.embeddings.create(
                model=model,
                input=batch,
                encoding_format="float",
            )
            results = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in results] != list(range(len(batch))):
                raise ValueError("A API não retornou exatamente um embedding por chunk.")

            for item in results:
                vector = item.embedding
                if len(vector) != EMBEDDING_DIMENSIONS:
                    raise ValueError(
                        f"O modelo retornou {len(vector)} dimensões; "
                        f"a tabela exige {EMBEDDING_DIMENSIONS}. "
                        "Use text-embedding-3-small ou ajuste o código e o DDL."
                    )
                if not all(math.isfinite(value) for value in vector) or not any(vector):
                    raise ValueError("A API retornou um embedding inválido.")
                embeddings.append(vector)

            logger.info("Embeddings gerados: %d/%d.", len(embeddings), len(chunks))

    return embeddings


def ingest_pdf() -> None:
    """Leia o PDF, gere os embeddings e substitua sua ingestão atomicamente."""
    load_dotenv(PROJECT_ROOT / ".env")
    settings = {
        name: os.getenv(name, "").strip()
        for name in (
            "PDF_PATH",
            "DATABASE_URL",
            "PG_VECTOR_COLLECTION_NAME",
            "OPENAI_API_KEY",
            "OPENAI_EMBEDDING_MODEL",
        )
    }
    missing = [name for name, value in settings.items() if not value]
    if missing:
        raise ValueError(f"Variáveis obrigatórias ausentes: {', '.join(missing)}.")

    table_name = settings["PG_VECTOR_COLLECTION_NAME"]
    if len(table_name.encode("utf-8")) > 63:
        raise ValueError("PG_VECTOR_COLLECTION_NAME deve ter no máximo 63 bytes.")
    table = sql.Identifier("public", table_name)

    pdf_path = Path(settings["PDF_PATH"]).expanduser()
    if not pdf_path.is_absolute():
        pdf_path = PROJECT_ROOT / pdf_path
    source = str(pdf_path.resolve())
    chunks = split_text(read_pdf(pdf_path))

    database_url = settings["DATABASE_URL"]
    if database_url == "localhost":
        # Usa as credenciais já definidas no docker-compose.yml.
        database_url = "postgresql://postgres:postgres@localhost:5432/rag"

    with psycopg.connect(database_url, connect_timeout=5) as connection:
        with connection.transaction():
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(connection)
            connection.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                        source TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
                        content TEXT NOT NULL
                            CHECK (char_length(content) BETWEEN 1 AND 1000),
                        embedding VECTOR(1536) NOT NULL,
                        UNIQUE (source, chunk_index)
                    )
                    """
                ).format(table)
            )

        embeddings = generate_embeddings(
            chunks,
            settings["OPENAI_EMBEDDING_MODEL"],
            settings["OPENAI_API_KEY"],
        )

        with connection.transaction():
            connection.execute(
                sql.SQL("DELETE FROM {} WHERE source = %s").format(table),
                (source,),
            )
            with connection.cursor() as cursor:
                cursor.executemany(
                    sql.SQL(
                        """
                        INSERT INTO {} (source, chunk_index, content, embedding)
                        VALUES (%s, %s, %s, %s)
                        """
                    ).format(table),
                    [
                        (source, index, chunk, Vector(embedding))
                        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
                    ],
                )

    logger.info("Ingestão concluída: %d chunks gravados em %s.", len(chunks), table_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        ingest_pdf()
    except (ValueError, OSError, PdfReadError, psycopg.Error, APIError) as error:
        logger.error("Falha na ingestão: %s", error)
        raise SystemExit(1) from None
