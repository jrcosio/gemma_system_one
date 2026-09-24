"""Regresiones de revisión de imágenes: decodificación, formato y fugas por alias."""

import io

import pytest
from PIL import Image

from conftest import make_example
from gemma_system_one.data.split import leakage_checks
from gemma_system_one.images import ImageError, check_image_file, content_address, load_image


def save_image(root, data, ext):
    rel = content_address(data, ext)
    path = root / rel
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(data)
    return rel, path


def test_dataset_validation_rejects_truncated_jpeg(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(buf, format="JPEG")
    rel, _ = save_image(tmp_path, buf.getvalue()[:-2], "jpg")
    assert check_image_file(tmp_path, rel), "verify() no garantiza decodificación de píxeles"


@pytest.mark.parametrize("kind", ["wrong_format", "animated"])
def test_runtime_loader_rejects_invalid_image_format(tmp_path, kind):
    buf = io.BytesIO()
    im = Image.new("RGB", (16, 16), "blue")
    if kind == "wrong_format":
        im.save(buf, format="BMP")
    else:
        im.save(
            buf,
            format="PNG",
            save_all=True,
            append_images=[Image.new("RGB", im.size, "red")],
            duration=10,
            loop=0,
        )
    _, path = save_image(tmp_path, buf.getvalue(), "png")
    with pytest.raises(ImageError):
        load_image(path)


def test_jpeg_extension_alias_cannot_cross_splits(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "blue").save(buf, format="JPEG")
    jpg, _ = save_image(tmp_path, buf.getvalue(), "jpg")
    jpeg, _ = save_image(tmp_path, buf.getvalue(), "jpeg")
    assert check_image_file(tmp_path, jpg) == check_image_file(tmp_path, jpeg) == []
    a = make_example(id="a", group_id="a", image_path=jpg)
    b = make_example(id="b", group_id="b", image_path=jpeg)
    assert any("imágenes compartidas" in e for e in leakage_checks({"train": [a], "test": [b]})["errors"])
