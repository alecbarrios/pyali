"""I/O: read the raw camera movie and read/write ``.mat`` (HDF5 v7.3) files.

Movie convention: ``[T, H, W]`` float64, row-major C-order.
"""
import os

import numpy as np


_RAW_DTYPE = {"uint8": "<u1", "u1": "<u1", "uint16": "<u2", "u2": "<u2"}


def read_bin_mov(path, nrow, ncol, read_dtype="uint16", out_dtype="float64", chunk_frames=256):
    """Read a raw ``frames1.bin`` movie into ``[T, H, W]``.

    Parameters
    ----------
    path : str
        Path to the binary movie file.
    nrow, ncol : int
        Frame height and width in pixels.
    read_dtype : str
        On-disk sample dtype: ``"uint16"`` (default) or ``"uint8"``. Little-endian.
    out_dtype : str
        In-RAM movie dtype. ``"float64"`` (default) reproduces the historical result;
        ``"float32"`` halves peak RAM, which can be what lets a large movie fit, and
        represents an 8-bit source losslessly.

    chunk_frames : int
        Frames converted per read. The cast is done chunk-wise straight into the output
        array, so peak memory is one float movie plus one chunk of raw samples, rather than
        the float movie plus a complete raw copy (which is 25% more for uint16 -> float64
        and 25% more for uint8 -> float32, and was what tipped a full-frame movie over the
        RAM of a mid-sized box). Values are identical to a whole-file ``astype``.

    Notes
    -----
    The raw stream is frame-major and row-major within each frame, so a plain C-order
    reshape ``(T, nrow, ncol)`` recovers the frames directly. Returns the full movie;
    the caller decides how many trailing frames to drop (a trailing partial frame is
    ignored). On limited RAM use :func:`open_bin_memmap` instead.
    """
    raw_dtype = np.dtype(_RAW_DTYPE.get(str(read_dtype), read_dtype))
    frame = nrow * ncol
    T = os.path.getsize(path) // (frame * raw_dtype.itemsize)
    mov = np.empty((T, nrow, ncol), dtype=out_dtype)
    with open(path, "rb") as f:
        for t0 in range(0, T, chunk_frames):
            n = min(chunk_frames, T - t0)
            raw = np.fromfile(f, dtype=raw_dtype, count=n * frame)
            mov[t0:t0 + n] = raw.reshape(n, nrow, ncol)          # frame, row, col; cast here
    return mov


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


def save_mat_v73(path, compression="gzip", compression_opts=4, **arrays):
    """Write arrays to a ``.mat`` (HDF5 v7.3) file.

    Each array is stored with reversed dimension order (the ``.mat`` convention) and tagged
    with the class attribute that ``.mat`` readers expect.

    Compression is on by default and matters enormously here: ``footprint`` is a dense
    ``(H, W, N)`` array in which each of the N planes is an 11x11 window in a field of zeros,
    a few hundredths of a percent non-zero, so a large field of view with hundreds of cells is
    gigabytes uncompressed. Chunking one plane per chunk and gzipping shrinks it by roughly
    three orders of magnitude, and the round trip is bit-exact.

    gzip is part of the HDF5 spec, so MATLAB's v7.3 reader and :func:`load_v73` both decompress
    transparently — nothing downstream has to change.

    Parameters
    ----------
    path : str
        Output ``.mat`` path.
    compression : str or None
        HDF5 filter; ``None`` disables (restores the previous uncompressed behaviour).
    compression_opts : int
        gzip level. 4 compresses a footprint ~4x better than level 1 for under 2x the time.
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
            kw = {}
            # Chunk along the leading stored axis so each cell's mostly-zero plane is its own
            # chunk; scalars and tiny arrays are left contiguous (HDF5 rejects zero-size chunks).
            if compression and stored.ndim >= 2 and stored.size and min(stored.shape) > 0:
                kw = dict(compression=compression, chunks=(1,) + stored.shape[1:])
                if compression == "gzip":
                    kw["compression_opts"] = compression_opts
            dset = f.create_dataset(name, data=stored, **kw)
            dset.attrs["MATLAB_class"] = np.bytes_(_cls.get(str(v.dtype), "double"))
