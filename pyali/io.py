"""I/O: read the raw camera movie and read/write ``.mat`` (HDF5 v7.3) files.

Movie convention: ``[T, H, W]`` float64, row-major C-order.
"""
import numpy as np


def read_bin_mov(path, nrow, ncol):
    """Read a raw ``frames1.bin`` movie (uint16, little-endian) into ``[T, H, W]`` float64.

    Parameters
    ----------
    path : str
        Path to the binary movie file.
    nrow, ncol : int
        Frame height and width in pixels.

    Notes
    -----
    The raw stream is frame-major and row-major within each frame, so a plain C-order
    reshape ``(T, nrow, ncol)`` recovers the frames directly. Returns the full movie;
    the caller decides how many trailing frames to drop. The array is large
    (~18 GB float64 for a 6389-frame 312x1200 movie); on limited RAM use
    :func:`open_bin_memmap` instead.
    """
    raw = np.fromfile(path, dtype="<u2")                 # uint16, little-endian
    T = raw.size // (nrow * ncol)
    mov = raw[:T * nrow * ncol].reshape(T, nrow, ncol)   # frame, row, col
    return mov.astype(np.float64)


def open_bin_memmap(path, nrow, ncol):
    """Memory-map ``frames1.bin`` as a read-only ``[T, H, W]`` uint16 view (no RAM cost).

    Same layout as :func:`read_bin_mov` but without materializing a float64 array — useful
    for streaming over a movie too large to hold in memory.
    """
    raw = np.memmap(path, dtype="<u2", mode="r")
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
