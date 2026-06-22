"""Tests del codebook visual (issue #9)."""
import numpy as np

from image_module.codebook import train, assign


def test_train_devuelve_k_centroides():
    descriptors = np.random.rand(2000, 128).astype(np.float32)
    centroids = train(descriptors, k=64)
    assert centroids.shape == (64, 128)


def test_assign_etiqueta_en_rango():
    descriptors = np.random.rand(500, 128).astype(np.float32)
    centroids = train(descriptors, k=32)
    labels = assign(descriptors, centroids)
    assert labels.shape == (500,)
    assert labels.min() >= 0 and labels.max() < 32


def test_assign_centroide_se_asigna_a_si_mismo():
    centroids = np.random.rand(16, 128).astype(np.float32)
    # Cada centroide debe cuantizarse a su propio índice.
    labels = assign(centroids, centroids)
    assert np.array_equal(labels, np.arange(16))
