"""Coordinate + affine convention firewall.

One place owns every convention so the rest of ``pyreg`` never open-codes a transpose, an
axis swap, or an off-by-one.

Conventions used everywhere in ``pyreg``
----------------------------------------
* **Points** are ``(x, y) = (column, row)``, 0-indexed, as float arrays of shape ``(N, 2)``.
  Convert to/from array-indexing ``(row, col)`` only with :func:`xy_to_rc` / :func:`rc_to_xy`.
* **Affines** are ``3x3`` matrices acting on homogeneous *column* vectors
  ``[x, y, 1]^T`` (the scikit-image convention): ``p' = M @ [x, y, 1]``. Build them with the
  helpers here and apply with :func:`apply_affine`.
* The reference (legacy) implementation stored affines in the *row-vector / post-multiply*
  form ``[x y 1] @ T``; :func:`row_major_affine_to_skimage` converts those to our column form,
  and :func:`compose` matches the legacy ``T1 * T2`` composition order.

Nothing here is named for its origin; names describe the convention (row-vector vs column-vector).
"""
from __future__ import annotations

import numpy as np
from skimage.transform import AffineTransform, SimilarityTransform


# --------------------------------------------------------------------------- #
# Affine construction / conversion
# --------------------------------------------------------------------------- #
def row_major_affine_to_skimage(T):
    """Convert a row-vector/post-multiply affine (``[x y 1] @ T``) to the column-vector form.

    The reference implementation's transforms satisfy ``[x' y' 1] = [x y 1] @ T``; scikit-image
    (and everything here) uses ``[x' y' 1]^T = M @ [x y 1]^T``. The two are transposes:
    ``M = T.T``.
    """
    T = np.asarray(T, float)
    if T.shape != (3, 3):
        raise ValueError(f"expected a 3x3 affine, got {T.shape}")
    return T.T.copy()


def translation(tx, ty):
    """Affine that translates by ``(tx, ty)`` in ``(x, y)``."""
    M = np.eye(3)
    M[0, 2] = tx
    M[1, 2] = ty
    return M


def similarity_matrix(scale=1.0, rotation=0.0, tx=0.0, ty=0.0):
    """Similarity affine (isotropic ``scale``, ``rotation`` in radians, translation ``tx,ty``)."""
    return SimilarityTransform(scale=scale, rotation=rotation,
                               translation=(tx, ty)).params.copy()


def compose(*mats):
    """Compose column-vector affines: ``compose(A, B, C)`` applies C, then B, then A.

    (i.e. returns ``A @ B @ C`` — the transform you get by first doing C to a point, then B,
    then A.) Note the reference implementation composed in the *opposite* textual order because
    it used row vectors: its ``T_total = T1 * T2`` (apply T1 then T2) equals, in our column
    convention, ``compose(to_col(T2), to_col(T1))`` = ``T2.T @ T1.T``.
    """
    if not mats:
        return np.eye(3)
    out = np.asarray(mats[0], float)
    for M in mats[1:]:
        out = out @ np.asarray(M, float)
    return out


def compose_after(first, second):
    """Return the affine that applies ``first`` then ``second`` (both column-vector form)."""
    return np.asarray(second, float) @ np.asarray(first, float)


def invert(M):
    """Inverse of a column-vector affine."""
    return np.linalg.inv(np.asarray(M, float))


# --------------------------------------------------------------------------- #
# Point application / axis conventions
# --------------------------------------------------------------------------- #
def apply_affine(M, pts_xy):
    """Apply a column-vector affine ``M`` to points ``pts_xy`` ``(N, 2)`` in ``(x, y)``."""
    pts = np.asarray(pts_xy, float).reshape(-1, 2)
    hom = np.column_stack([pts, np.ones(len(pts))])          # [N,3] rows [x,y,1]
    out = hom @ np.asarray(M, float).T                       # (M @ hom.T).T
    return out[:, :2]


def xy_to_rc(pts_xy):
    """``(x, y)`` -> ``(row, col)`` (swap columns)."""
    pts = np.asarray(pts_xy, float).reshape(-1, 2)
    return pts[:, ::-1].copy()


def rc_to_xy(pts_rc):
    """``(row, col)`` -> ``(x, y)`` (swap columns)."""
    pts = np.asarray(pts_rc, float).reshape(-1, 2)
    return pts[:, ::-1].copy()


def fit_affine(src_xy, dst_xy):
    """Least-squares column-vector affine mapping ``src_xy`` -> ``dst_xy`` (both ``(N,2)`` xy)."""
    t = AffineTransform()
    t.estimate(np.asarray(src_xy, float), np.asarray(dst_xy, float))
    return t.params.copy()


def fit_similarity(src_xy, dst_xy, ransac=False):
    """Least-squares column-vector **similarity** (4 DOF: isotropic scale, rotation, translation)
    mapping ``src_xy`` -> ``dst_xy`` (both ``(N, 2)`` xy).

    Unlike :func:`fit_affine` (6 DOF), this cannot encode shear or anisotropic scale, so it does
    not overfit a small set of cross-modal anchor correspondences. Needs at least 2 points. With
    ``ransac=True`` and >= 4 points, fit robustly (rejects outlier correspondences).
    """
    src = np.asarray(src_xy, float)
    dst = np.asarray(dst_xy, float)
    if ransac and len(src) >= 4:
        from skimage.measure import ransac as _ransac
        model, _inliers = _ransac((src, dst), SimilarityTransform, min_samples=2,
                                  residual_threshold=3.0, max_trials=1000)
        return model.params.copy()
    t = SimilarityTransform()
    t.estimate(src, dst)
    return t.params.copy()


# --------------------------------------------------------------------------- #
# Dihedral group D4 (the 8 flips/rotations) — the orientation between the two
# microscopes is one of these and is recovered empirically (see register.py).
# --------------------------------------------------------------------------- #
DIHEDRAL_IMAGE_OPS = {
    "identity":       lambda a: a,
    "rot90":          lambda a: np.rot90(a, 1),     # counter-clockwise
    "rot180":         lambda a: np.rot90(a, 2),
    "rot270":         lambda a: np.rot90(a, 3),
    "fliplr":         np.fliplr,
    "flipud":         np.flipud,
    "transpose":      lambda a: a.T,
    "anti_transpose": lambda a: np.rot90(a, 2).T,
}
DIHEDRAL_NAMES = tuple(DIHEDRAL_IMAGE_OPS)


def apply_dihedral_image(op, image):
    """Apply a named dihedral op to a 2-D image (numpy)."""
    return DIHEDRAL_IMAGE_OPS[op](np.asarray(image))


def dihedral_point_transform(op, in_shape):
    """Column-vector affine mapping input ``(x, y)`` to the ``op``-transformed image's ``(x, y)``.

    Derived *from the actual numpy op* (by tracking where three corner pixels land), so the point
    transform and the image op can never disagree. ``in_shape`` is ``(H, W)`` of the input image.
    """
    H, W = int(in_shape[0]), int(in_shape[1])
    idx = np.arange(H * W).reshape(H, W)
    out = DIHEDRAL_IMAGE_OPS[op](idx)
    src = np.array([[0, 0], [W - 1, 0], [0, H - 1]], float)      # (x, y) corners
    dst = np.empty((3, 2), float)
    for k, (x, y) in enumerate(src):
        val = int(y) * W + int(x)
        pos = np.argwhere(out == val)[0]                        # (row', col')
        dst[k] = [pos[1], pos[0]]                               # (x', y')
    return fit_affine(src, dst)
