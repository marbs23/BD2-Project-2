"""Tests de la lógica pura del índice de imagen (issue #10)."""
import numpy as np

from image_module.index import _histogramas_por_patch


def test_reconstruye_histogramas_por_patch():
    # Dos patches apilados: el primero con labels [0,0], el segundo con [1,2].
    labels = np.array([0, 0, 1, 2])
    counts = np.array([2, 2])
    hist = _histogramas_por_patch(labels, counts, k=4)
    assert len(hist) == 2
    assert np.array_equal(hist[0], np.array([2, 0, 0, 0], dtype=np.float64))
    assert np.array_equal(hist[1], np.array([0, 1, 1, 0], dtype=np.float64))


def test_longitud_fija_k():
    labels = np.array([0, 3])
    counts = np.array([2])
    hist = _histogramas_por_patch(labels, counts, k=8)
    assert hist[0].shape == (8,)
