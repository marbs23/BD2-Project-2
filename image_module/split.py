"""
Módulo Split — patches (issue #7)

Divide cada imagen en patches, el equivalente visual al split en párrafos del
texto. Se trabaja en escala de grises (SIFT no usa color) y se redimensiona al
lado mayor fijo para que el número de keypoints no dependa del tamaño original.

El grid es 4x4 (16 patches) con un solapamiento que evita perder keypoints
justo en los bordes entre celdas.
"""
import cv2
import numpy as np

MAX_SIDE = 512      # lado mayor tras redimensionar
GRID = 4            # 4x4 = 16 patches por imagen
OVERLAP = 0.25      # cada patch se extiende 25% sobre las celdas vecinas


def load_gray(path: str, max_side: int = MAX_SIDE) -> np.ndarray | None:
    """Carga una imagen en grises y la redimensiona al lado mayor. None si no abre."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


def split_patches(img: np.ndarray, grid: int = GRID,
                  overlap: float = OVERLAP) -> list[np.ndarray]:
    """Parte la imagen en grid*grid patches solapados. Devuelve los recortes."""
    h, w = img.shape[:2]
    ph, pw = h // grid, w // grid
    oh, ow = int(ph * overlap), int(pw * overlap)

    patches = []
    for i in range(grid):
        for j in range(grid):
            y1, y2 = max(0, i * ph - oh), min(h, (i + 1) * ph + oh)
            x1, x2 = max(0, j * pw - ow), min(w, (j + 1) * pw + ow)
            patches.append(img[y1:y2, x1:x2])
    return patches


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sample = sys.argv[1] if len(sys.argv) > 1 else next(
        (str(p) for p in Path("data/raw/images").rglob("*.jpg")), None
    )
    if not sample:
        print("No hay imágenes en data/raw/images")
        sys.exit(1)

    img = load_gray(sample)
    patches = split_patches(img)
    print(f"Imagen: {sample}")
    print(f"  tamaño tras resize: {img.shape}")
    print(f"  patches: {len(patches)}")
    print(f"  tamaño de un patch: {patches[0].shape}")
