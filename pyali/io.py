"""I/O: read the raw camera movie and read/write ``.mat`` (HDF5 v7.3) files.

Movie convention: ``[T, H, W]`` float64, row-major C-order.
"""
import numpy as np


_RAW_DTYPE = {"uint8": "<u1", "u1": "<u1", "uint16": "<u2", "u2": "<u2"}


def read_bin_mov(path, nrow, ncol, read_dtype="uint16", out_dtype="float64"):
    """Read a raw ``frames1.bin`` movie into ``[T, H, W]``.

    Parameters
    ----------
    path : str
        Path to the binary movie file.
    nrow, ncol : int
        Frame height and width in pixels.
    read_dtype : str
        On-disk sample dtype: ``"uint16"`` (default; 6GP002 16-bit) or ``"uint8"``
        (443screen2 8-bit). Little-endian.
    out_dtype : str
        In-RAM movie dtype. ``"float64"`` (default) reproduces the historical result;
        ``"float32"`` halves peak RAM (~21.5 GB vs ~43 GB for an 800x800x8399 movie),
        which is required to fit a large 8-bit movie in a 62 GB box.

    Notes
    -----
    The raw stream is frame-major and row-major within each frame, so a plain C-order
    reshape ``(T, nrow, ncol)`` recovers the frames directly. Returns the full movie;
    the caller decides how many trailing frames to drop. On limited RAM use
    :func:`open_bin_memmap` instead.
    """
    raw = np.fromfile(path, dtype=_RAW_DTYPE.get(str(read_dtype), read_dtype))
    T = raw.size // (nrow * ncol)
    mov = raw[:T * nrow * ncol].reshape(T, nrow, ncol)   # frame, row, col
    return mov.astype(out_dtype)


def open_bin_memmap(path, nrow, ncol, read_dtype="uint16"):
    """Memory-map ``frames1.bin`` as a read-only ``[T, H, W]`` integer view (no RAM cost).

    Same layout as :func:`read_bin_mov` but without materializing a float array — useful
    for streaming over a movie too large to hold in memory. ``read_dtype`` is the on-disk
    sample dtype (``"uint16"`` default, or ``"uint8"`` for 8-bit acquisitions).
    """
    raw = np.memmap(path, dtype=_RAW_DTYPE.get(str(read_dtype), read_dtype), mode="r")
    T = raw.size // (nrow * ncol)
    return raw[:T * nrow * ncol].reshape(T, nrow, ncol)


def _from_h5(dset):
    a = np.array(dset)
    return a.T if a.ndim >= 2 else a          # .mat/HDF5 stores dimensions in reverse order


def load_v73(path, var=None):
    """Load variable(s) from a ``.mat`` (HDF5 v7.3) file.

    Parameters
    ----------
    path : str
        Path to the ``.mat`` file.
    var : str, optional
        Variable name to load. If omitted, returns a ``{name: ndarray}`` dict.

    Uses ``h5py`` when available, otherwise the bundled pure-python reader
    (:mod:`pyali._h5read`), so no extra dependency is strictly required.
    """
    try:
        import h5py
    except ImportError:
        from . import _h5read
        d = _h5read.read_mat_v73(path)
        return d if var is None else d[var]
    with h5py.File(path, "r") as f:
        if var is None:
            return {k: _from_h5(f[k]) for k in f.keys()}
        return _from_h5(f[var])


def save_mat_v73(path, **arrays):
    """Write arrays to a ``.mat`` (HDF5 v7.3) file.

    Each array is stored with reversed dimension order (the ``.mat`` convention) and tagged
    with the class attribute that ``.mat`` readers expect.

    Parameters
    ----------
    path : str
        Output ``.mat`` path.
    **arrays : ndarray
        Named arrays to store.
    """
    import h5py
    _cls = {"float64": "double", "float32": "single", "uint8": "uint8",
            "uint16": "uint16", "int64": "int64", "int32": "int32", "bool": "logical"}
    with h5py.File(path, "w") as f:
        for name, v in arrays.items():
            v = np.asarray(v)
            stored = v.T if v.ndim >= 2 else v
            dset = f.create_dataset(name, data=stored)
            dset.attrs["MATLAB_class"] = np.bytes_(_cls.get(str(v.dtype), "double"))
