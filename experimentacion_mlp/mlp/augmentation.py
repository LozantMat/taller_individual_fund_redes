"""
Aumento de datos sobre lotes de imágenes aplanadas.

A cada imagen se le aplica una transformación afín aleatoria: una rotación de
hasta ±max_rotation grados alrededor del centro y una traslación entera de
hasta ±max_shift píxeles en vertical y en horizontal. Se implementa con mapeo
inverso (para cada píxel de salida se calcula de dónde viene en la imagen
original) e interpolación bilineal, todo vectorizado sobre el lote. Los
píxeles que caen fuera de la imagen se rellenan con 0 (fondo negro de MNIST).
"""
import numpy as np

from .data import IMAGE_SHAPE


class ImageAugmenter:
    def __init__(self, cfg, rng, image_shape=IMAGE_SHAPE):
        self.max_shift = int(cfg.max_shift)
        self.max_rotation = float(cfg.max_rotation)
        self.probability = float(cfg.probability)
        self.rng = rng
        self.H, self.W = image_shape
        rows, cols = np.mgrid[0:self.H, 0:self.W]
        # Coordenadas de cada píxel de salida relativas al centro de la imagen
        self._cy, self._cx = (self.H - 1) / 2.0, (self.W - 1) / 2.0
        self._r = (rows.ravel() - self._cy).astype(np.float32)[None, :]
        self._c = (cols.ravel() - self._cx).astype(np.float32)[None, :]

    def sample_params(self, n):
        apply = self.rng.random(n) < self.probability
        angles = np.deg2rad(self.rng.uniform(-self.max_rotation, self.max_rotation, n)) * apply
        shift_r = self.rng.integers(-self.max_shift, self.max_shift + 1, n) * apply
        shift_c = self.rng.integers(-self.max_shift, self.max_shift + 1, n) * apply
        return angles.astype(np.float32), shift_r.astype(np.float32), shift_c.astype(np.float32)

    def transform(self, X, angles, shift_r, shift_c):
        n = X.shape[0]
        images = X.reshape(n, self.H, self.W)

        # Mapeo inverso: deshacer la traslación y luego la rotación
        r = self._r - shift_r[:, None]
        c = self._c - shift_c[:, None]
        cos, sin = np.cos(angles)[:, None], np.sin(angles)[:, None]
        src_r = cos * r - sin * c + self._cy
        src_c = sin * r + cos * c + self._cx

        # Interpolación bilineal con relleno de ceros fuera de los bordes
        r0f, c0f = np.floor(src_r), np.floor(src_c)
        dr, dc = src_r - r0f, src_c - c0f
        r0, c0 = r0f.astype(np.int32), c0f.astype(np.int32)
        flat_images = images.reshape(-1)
        offsets = (np.arange(n, dtype=np.int32) * self.H * self.W)[:, None]
        out = np.zeros((n, self.H * self.W), dtype=X.dtype)
        for ri, ci, w in ((r0, c0, (1 - dr) * (1 - dc)), (r0, c0 + 1, (1 - dr) * dc),
                          (r0 + 1, c0, dr * (1 - dc)), (r0 + 1, c0 + 1, dr * dc)):
            valid = (ri >= 0) & (ri < self.H) & (ci >= 0) & (ci < self.W)
            flat_idx = offsets + np.clip(ri, 0, self.H - 1) * self.W + np.clip(ci, 0, self.W - 1)
            out += w * valid * flat_images[flat_idx]
        return out

    def __call__(self, X, chunk_size=4096):
        # Se procesa por bloques para acotar la memoria al transformar el conjunto completo
        return np.concatenate([self.transform(X[i:i + chunk_size], *self.sample_params(len(X[i:i + chunk_size])))
                               for i in range(0, X.shape[0], chunk_size)])
