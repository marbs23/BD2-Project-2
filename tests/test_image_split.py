"""Tests del split en patches (issue #7)."""
import numpy as np

from src.indexing.image.split import split_patches, load_gray, GRID


def _imagen(h=400, w=600):
    return np.random.randint(0, 256, (h, w), dtype=np.uint8)


def test_numero_de_patches():
    patches = split_patches(_imagen())
    assert len(patches) == GRID * GRID


def test_patches_no_vacios():
    for p in split_patches(_imagen()):
        assert p.size > 0
        assert p.ndim == 2


def test_solapamiento_agranda_patches():
    # Con solape, cada patch es mayor que una celda exacta (h/grid, w/grid).
    h, w = 400, 600
    patches = split_patches(_imagen(h, w))
    ph, pw = h // GRID, w // GRID
    centrales = [patches[GRID + 1]]  # un patch interior tiene solape en los 4 lados
    for p in centrales:
        assert p.shape[0] > ph and p.shape[1] > pw


def test_load_gray_redimensiona(tmp_path):
    import cv2
    grande = np.random.randint(0, 256, (1000, 2000), dtype=np.uint8)
    ruta = tmp_path / "grande.jpg"
    cv2.imwrite(str(ruta), grande)
    img = load_gray(str(ruta), max_side=512)
    assert max(img.shape) == 512


def test_load_gray_inexistente():
    assert load_gray("no_existe_123.jpg") is None
