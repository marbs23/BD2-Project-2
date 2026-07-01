"""Tests del extractor SIFT (issue #8)."""
import numpy as np

from src.indexing.image.extractor import extract_patch


def _patch_con_textura(size=120):
    # El ruido tiene textura suficiente para que SIFT detecte keypoints.
    return np.random.randint(0, 256, (size, size), dtype=np.uint8)


def test_descriptores_son_128d():
    des = extract_patch(_patch_con_textura())
    assert des is not None
    assert des.shape[1] == 128


def test_patch_plano_sin_keypoints():
    plano = np.full((120, 120), 128, dtype=np.uint8)
    des = extract_patch(plano)
    assert des is None or len(des) == 0


def test_patch_mayor_tiene_mas_keypoints():
    # SIFT escala con el área texturada: un patch más grande detecta más keypoints.
    np.random.seed(0)
    chico = extract_patch(_patch_con_textura(80))
    grande = extract_patch(_patch_con_textura(240))
    assert chico is not None and grande is not None
    assert len(grande) > len(chico)
