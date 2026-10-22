"""Descarga los archivos mensuales de viajes de MiBici."""

import csv
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import typer
from loguru import logger

from MiBici.config import PROJ_ROOT, RAW_DATA_DIR

app = typer.Typer(help="Descarga los datos mensuales originales de MiBici.")

DEFAULT_URLS_FILE = PROJ_ROOT / "references" / "mibici_urls_2014_2026.txt"
DEFAULT_OUTPUT_DIR = RAW_DATA_DIR / "mibici"
CHUNK_SIZE = 1024 * 1024
MAX_ATTEMPTS = 3
TIMEOUT_SECONDS = 60


def utc_now() -> str:
    """Devuelve la fecha y hora actual en UTC."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_urls(urls_file: Path) -> list[str]:
    """Lee y valida URLs únicas de archivos CSV."""
    if not urls_file.is_file():
        raise typer.BadParameter(f"No se encontró el archivo de URLs: {urls_file}")

    urls = []
    seen = set()
    for line_number, line in enumerate(
        urls_file.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        url = line.strip()
        if not url or url.startswith("#"):
            continue

        parsed = urlparse(url)
        filename = Path(parsed.path).name
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise typer.BadParameter(f"URL no válida en línea {line_number}: {url}")
        if not filename.lower().endswith(".csv"):
            raise typer.BadParameter(
                f"La URL no apunta a un CSV en la línea {line_number}: {url}"
            )
        if url in seen:
            logger.warning("URL duplicada omitida en línea {}: {}", line_number, url)
            continue
        seen.add(url)
        urls.append(url)

    if not urls:
        raise typer.BadParameter(f"El archivo no contiene URLs CSV: {urls_file}")
    return urls


def sha256_file(path: Path) -> str:
    """Calcula el SHA-256 leyendo el archivo por partes."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(manifest_path: Path) -> dict[str, dict[str, str]]:
    """Lee el manifiesto existente para conservar datos de descargas previas."""
    if not manifest_path.exists():
        return {}
    with manifest_path.open(encoding="utf-8", newline="") as file:
        return {
            row["url"]: row
            for row in csv.DictReader(file)
            if row.get("url")
        }


def write_manifest(
    manifest_path: Path, entries: list[dict[str, str]]
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_name(f"{manifest_path.name}.tmp")
    fields = [
        "filename",
        "url",
        "bytes",
        "sha256",
        "downloaded_at",
        "status",
        "error",
    ]
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(entries)
    # Reemplaza el manifiesto completo para evitar dejarlo escrito a medias.
    temporary_path.replace(manifest_path)


def download_one(url: str, destination: Path) -> tuple[int, str]:
    """Descarga un archivo y devuelve su tamaño y SHA-256."""
    temporary_path = destination.with_name(f"{destination.name}.part")
    request = Request(
        url,
        headers={"User-Agent": "MiBici-Equipo11-DataDownloader/1.0"},
    )
    last_error = None

    # Se reintenta la descarga si hay un error de red o de escritura.
    for attempt in range(1, MAX_ATTEMPTS + 1):
        digest = hashlib.sha256()
        byte_count = 0
        try:
            with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                with temporary_path.open("wb") as output:
                    while chunk := response.read(CHUNK_SIZE):
                        output.write(chunk)
                        digest.update(chunk)
                        byte_count += len(chunk)

            if byte_count == 0:
                raise OSError("El servidor devolvió un archivo vacío")

            # El archivo queda disponible con su nombre final solo al completarse.
            temporary_path.replace(destination)
            return byte_count, digest.hexdigest()

        except (OSError, URLError, TimeoutError) as error:
            last_error = error
            temporary_path.unlink(missing_ok=True)
            if attempt < MAX_ATTEMPTS:
                time.sleep(attempt * 2)

    raise OSError(
        f"No se pudo descargar {url}: {last_error}"
    ) from last_error


@app.command()
def download(
    urls_file: Path = typer.Option(
        DEFAULT_URLS_FILE,
        help="Archivo de texto con una URL CSV por línea.",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        help="Carpeta donde guardar los CSV originales.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Volver a descargar archivos que ya existen.",
    ),
) -> None:
    """Descarga los CSV mensuales y omite los que ya existen, salvo con --force."""
    urls = read_urls(urls_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    previous_entries = read_manifest(manifest_path)
    entries = []
    downloaded = skipped = failed = 0

    logger.info("{} URLs encontradas. Destino: {}", len(urls), output_dir)
    logger.info("Archivo de URLs: {}", urls_file)
    logger.info(
        "Inicio de descarga: {} archivos por descargar; forzar descarga: {}",
        len(urls),
        force,
    )

    for index, url in enumerate(urls, 1):
        filename = Path(urlparse(url).path).name
        destination = output_dir / filename
        previous = previous_entries.get(url, {})

        entry = {
            "filename": filename,
            "url": url,
            "bytes": "",
            "sha256": "",
            "downloaded_at": "",
            "status": "",
            "error": "",
        }

        if destination.is_file() and not force:
            entry.update(
                bytes=str(destination.stat().st_size),
                sha256=sha256_file(destination),
                downloaded_at=previous.get("downloaded_at") or "unknown",
                status="already_present",
            )
            skipped += 1
            logger.info(
                "[{}/{}] Ya existe: {}",
                index,
                len(urls),
                filename,
            )
        else:
            try:
                size, checksum = download_one(url, destination)
                entry.update(
                    bytes=str(size),
                    sha256=checksum,
                    downloaded_at=utc_now(),
                    status="downloaded",
                )
                downloaded += 1
                logger.success(
                    "[{}/{}] Descargado: {} ({} bytes)",
                    index,
                    len(urls),
                    filename,
                    size,
                )
            except OSError as error:
                entry.update(
                    status="error",
                    error=str(error),
                    downloaded_at=utc_now(),
                )
                failed += 1
                logger.error("[{}/{}] {}", index, len(urls), error)

        entries.append(entry)

        # Guarda el avance para poder continuar si la ejecución se interrumpe.
        write_manifest(manifest_path, entries)

    logger.info(
        "Resumen de descarga: {} URLs revisadas; {} descargadas; {} ya existentes; {} errores",
        len(urls),
        downloaded,
        skipped,
        failed,
    )
    logger.info("Archivos originales: {}", output_dir)
    logger.info("Manifiesto de descarga: {}", manifest_path)

    if failed:
        logger.error("Revisa los errores anteriores y vuelve a ejecutar la descarga.")
        raise typer.Exit(code=1)

    logger.success(
        "Descarga completada. Para preparar los datos ejecuta: python -m MiBici.prepare_data"
    )


if __name__ == "__main__":
    app()
