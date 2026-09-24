"""Imágenes de entrada (fase 4, decisión 0007): referencia por contenido, límites y carga verificada.

- En un dataset, ``image_path`` debe ser ``images/<sha256>.<png|jpg|jpeg>``: el nombre es el hash
  del fichero. Así el hash de entrada de un ejemplo (y el de sus filas) identifica la imagen sin
  leerla, y el nombre no puede revelar la clase.
- Límites iniciales de la spec §9: 5 MiB decodificables y 16 megapíxeles; sólo PNG/JPEG sin
  animación. Las dimensiones se comprueban en la cabecera, antes de decodificar los píxeles.
- Al cargar se verifica el sha256 del nombre y se convierte a RGB (el procesador también lo hace).
- La imagen va antes del texto de la fila en el turno de usuario (``IMAGE_PLACEMENT``).
"""

from __future__ import annotations

import hashlib
import io
import re
from pathlib import Path

IMAGE_PLACEMENT = "image_before_text_v1"
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 16_000_000
FORMATS = {"PNG": "png", "JPEG": "jpg"}
CONTENT_ADDRESSED = re.compile(r"^images/(?P<sha>[0-9a-f]{64})\.(?P<ext>png|jpg|jpeg)$")


class ImageError(ValueError):
    pass


class ImageTooLargeError(ImageError):
    """Bytes o píxeles por encima de los límites (HTTP 413 en la API)."""


def content_address(data: bytes, ext: str) -> str:
    return f"images/{hashlib.sha256(data).hexdigest()}.{ext}"


def check_image_file(root: Path, rel: str) -> list[str]:
    """Errores de una imagen del dataset (vacío si es válida)."""
    m = CONTENT_ADDRESSED.fullmatch(rel)
    if m is None:
        return [f"{rel}: image_path debe ser images/<sha256>.png|jpg (dirección por contenido)"]
    base = Path(root).resolve()
    path = (Path(root) / rel).resolve()
    if not path.is_relative_to(base):
        return [f"{rel}: sale de la raíz del dataset"]
    if not path.is_file():
        return [f"{rel}: no existe"]
    try:
        image = load_image(path)
        image.close()
    except ImageError as exc:
        return [f"{rel}: {exc}"]
    return []


def image_content_key(rel: str) -> str:
    """Identidad de bytes independiente de los alias admitidos .jpg/.jpeg.

    No cambia hashes históricos de filas o splits; se usa para detectar fugas.
    """
    match = CONTENT_ADDRESSED.fullmatch(rel)
    return match["sha"] if match else rel


def decode_image_bytes(data: bytes, *, expected_ext: str | None = None, name: str = "imagen"):
    """RGB decodificado de unos bytes ya en memoria, con los límites y controles del proyecto.

    Compartido por el dataset (``load_image``) y la API: tamaño en bytes, formato PNG/JPEG,
    extensión coherente, sin animación, píxeles leídos de la cabecera **antes** de decodificar y
    decodificación completa (``verify()`` no decodifica los píxeles de JPEG). Devuelve (imagen, ext).
    """
    from PIL import Image

    if len(data) > MAX_IMAGE_BYTES:
        raise ImageTooLargeError(f"{name}: supera {MAX_IMAGE_BYTES} bytes")
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format not in FORMATS:
                raise ImageError(f"{name}: formato no admitido ({im.format}); sólo PNG o JPEG")
            ext = FORMATS[im.format]
            if expected_ext is not None and ext != expected_ext.replace("jpeg", "jpg"):
                raise ImageError(f"{name}: formato no admitido o extensión incoherente")
            if getattr(im, "is_animated", False) or getattr(im, "n_frames", 1) > 1:
                raise ImageError(f"{name}: imagen animada")
            if im.size[0] * im.size[1] > MAX_PIXELS:
                raise ImageTooLargeError(f"{name}: supera {MAX_PIXELS} píxeles")
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            return im.convert("RGB"), ext
    except ImageError:
        raise
    except Exception as exc:
        raise ImageError(f"{name}: no decodificable ({type(exc).__name__})") from exc


def load_image(path: Path):
    """RGB decodificado con los mismos controles en validación y extracción.

    Se limita la lectura antes de materializar el fichero completo y se decodifican
    los mismos bytes cuyo hash se comprobó (sin reabrir la ruta).
    """
    path = Path(path)
    m = CONTENT_ADDRESSED.fullmatch(f"images/{path.name}")
    if m is None:
        raise ImageError(f"{path.name}: nombre sin hash de contenido válido")
    try:
        with path.open("rb") as fh:
            data = fh.read(MAX_IMAGE_BYTES + 1)
    except OSError as exc:
        raise ImageError(f"{path.name}: no legible ({type(exc).__name__})") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageTooLargeError(f"{path.name}: supera {MAX_IMAGE_BYTES} bytes")
    if hashlib.sha256(data).hexdigest() != m["sha"]:
        raise ImageError(f"{path.name}: el contenido no coincide con el hash del nombre (sha256)")
    return decode_image_bytes(data, expected_ext=m["ext"], name=path.name)[0]
