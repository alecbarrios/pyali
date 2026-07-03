"""Zero-dependency reader for HDF5-based ``.mat`` (v7.3) files (numpy + zlib only).

Used as a fallback by :func:`pyali.io.load_v73` when ``h5py`` is not installed. Handles the
subset of the format these files use:

  * 512-byte user block (all HDF5 addresses are relative to it -> +512)
  * superblock v0
  * root-group symbol table (v1 B-tree node-type 0 + local heap + SNOD)
  * object header v1 (with continuation blocks)
  * datatypes: IEEE float64/32, fixed-point ints, enum(logical)->uint8
  * data layout: chunked (v3) with the deflate filter, and contiguous
  * chunk B-tree v1 (node-type 1), single- or multi-level

Dimensions are stored reversed in the file, so arrays are transposed to natural order.
"""
import struct
import zlib

import numpy as np

_NULL = 0xFFFFFFFFFFFFFFFF


class _Reader:
    def __init__(self, path):
        self.d = open(path, "rb").read()
        self.base = next((b for b in (0, 512, 1024, 2048)
                          if self.d[b:b + 8] == b"\x89HDF\r\n\x1a\n"), None)
        if self.base is None:
            raise ValueError("HDF5 superblock not found")
        sb = self.base
        if self.d[sb + 8] != 0:
            raise ValueError("only superblock v0 supported")
        self.off_size = self.d[sb + 13]
        self.len_size = self.d[sb + 14]
        p = sb + 24 + 4 * self.off_size          # skip base/free/eof/driver addrs
        self.root_oh = self._addr(self._uoff(p + self.off_size))  # root symbol table entry

    def _uoff(self, p):
        return int.from_bytes(self.d[p:p + self.off_size], "little")

    def _addr(self, a):
        return None if a == _NULL else self.base + a

    # -- object header v1: gather (type, body) for all messages (follow continuations) --
    def _messages(self, oh):
        d = self.d
        num = struct.unpack_from("<H", d, oh + 2)[0]
        hdr = struct.unpack_from("<I", d, oh + 8)[0]
        blocks = [(oh + 16, hdr)]                 # v1 prefix is 12 + 4 pad = 16 bytes
        out, seen, bi = [], 0, 0
        while bi < len(blocks) and seen < num:
            start, size = blocks[bi]; bi += 1
            p, end = start, start + size
            while p + 8 <= end and seen < num:
                mtype = struct.unpack_from("<H", d, p)[0]
                msize = struct.unpack_from("<H", d, p + 2)[0]
                body = p + 8
                out.append((mtype, d[body:body + msize])); seen += 1
                if mtype == 0x10:                 # continuation -> another message block
                    caddr = self._addr(self._uoff(body))
                    clen = self._uoff(body + self.off_size)
                    if caddr is not None:
                        blocks.append((caddr, clen))
                p = body + msize
        return out

    def _dataspace(self, b):
        ver, rank = b[0], b[1]
        p = 8 if ver == 1 else 4
        return tuple(struct.unpack_from("<Q", b, p + 8 * i)[0] for i in range(rank))

    def _datatype(self, b):
        cls = b[0] & 0x0F
        size = struct.unpack_from("<I", b, 4)[0]
        if cls == 1:                              # floating point
            return np.dtype("<f8") if size == 8 else np.dtype("<f4")
        if cls == 0:                              # fixed point
            signed = bool(b[8] & 0x08)
            key = {1: "i1", 2: "i2", 4: "i4", 8: "i8"} if signed else \
                  {1: "u1", 2: "u2", 4: "u4", 8: "u8"}
            return np.dtype("<" + key[size])
        if cls == 8:                              # enum (logical) -> uint8
            return np.dtype("u1")
        raise ValueError(f"unsupported datatype class {cls} size {size}")

    def _layout(self, b):
        assert b[0] == 3, "only layout v3 supported"
        cls, O, L = b[1], self.off_size, self.len_size
        if cls == 1:                              # contiguous
            addr = self._addr(int.from_bytes(b[2:2 + O], "little"))
            return {"class": "contiguous", "addr": addr}
        if cls == 2:                              # chunked
            dim = b[2]
            addr = self._addr(int.from_bytes(b[3:3 + O], "little"))
            base = 3 + O
            cdims = [struct.unpack_from("<I", b, base + 4 * i)[0] for i in range(dim)]
            return {"class": "chunked", "btree": addr, "chunk_dims": cdims}
        if cls == 0:                              # compact
            size = struct.unpack_from("<H", b, 2)[0]
            return {"class": "compact", "data": b[4:4 + size]}
        raise ValueError(f"unsupported layout class {cls}")

    def _walk_chunk_btree(self, addr, rank, out):
        d = self.d
        level = d[addr + 5]
        entries = struct.unpack_from("<H", d, addr + 6)[0]
        p = addr + 8 + 2 * self.off_size
        keysize = 8 + 8 * (rank + 1)
        for _ in range(entries):
            if level == 0:
                csize, mask = struct.unpack_from("<II", d, p)
                offs = [struct.unpack_from("<Q", d, p + 8 + 8 * k)[0] for k in range(rank + 1)]
                child = self._addr(self._uoff(p + keysize))
                out.append((csize, mask, offs, child))
            else:
                self._walk_chunk_btree(self._addr(self._uoff(p + keysize)), rank, out)
            p += keysize + self.off_size

    def _read_dataset(self, oh):
        dims = dtype = layout = None
        for t, b in self._messages(oh):
            if t == 0x01 and dims is None:
                dims = self._dataspace(b)
            elif t == 0x03 and dtype is None:
                dtype = self._datatype(b)
            elif t == 0x08:
                layout = self._layout(b)
        dims = dims or ()
        n = int(np.prod(dims)) if dims else 1
        if layout["class"] == "contiguous":
            if layout["addr"] is None:
                hdf = np.zeros(dims, dtype=dtype)
            else:
                raw = self.d[layout["addr"]:layout["addr"] + n * dtype.itemsize]
                hdf = np.frombuffer(raw, dtype=dtype).copy().reshape(dims)
        elif layout["class"] == "compact":
            hdf = np.frombuffer(layout["data"][:n * dtype.itemsize], dtype=dtype).reshape(dims)
        else:                                     # chunked
            rank = len(dims)
            cdims = layout["chunk_dims"][:rank]
            full = np.zeros(dims, dtype=dtype)
            chunks = []
            if layout["btree"] is not None:
                self._walk_chunk_btree(layout["btree"], rank, chunks)
            for csize, mask, offs, child in chunks:
                raw = self.d[child:child + csize]
                buf = raw if (mask & 1) else zlib.decompress(raw, 15)   # bit0 => deflate skipped
                cd = np.frombuffer(buf, dtype=dtype)
                sl, csl = [], []
                for i in range(rank):
                    length = min(cdims[i], dims[i] - offs[i])
                    sl.append(slice(offs[i], offs[i] + length)); csl.append(slice(0, length))
                full[tuple(sl)] = cd.reshape(cdims)[tuple(csl)]
            hdf = full
        return np.asarray(hdf).transpose()        # stored reversed -> natural order

    def read_all(self):
        d, O = self.d, self.off_size
        st_btree = st_heap = None
        for t, b in self._messages(self.root_oh):
            if t == 0x11:                         # symbol table message
                st_btree = self._addr(int.from_bytes(b[0:O], "little"))
                st_heap = self._addr(int.from_bytes(b[O:2 * O], "little"))
        heap_data = self._addr(self._uoff(st_heap + 8 + 2 * self.len_size))

        def name_at(off):
            e = d.index(b"\x00", heap_data + off)
            return d[heap_data + off:e].decode("ascii", "replace")

        result = {}
        self._walk_group_btree(st_btree, name_at, result)
        return result

    def _walk_group_btree(self, addr, name_at, result):
        d = self.d
        level = d[addr + 5]
        entries = struct.unpack_from("<H", d, addr + 6)[0]
        p = addr + 8 + 2 * self.off_size
        keysize = self.len_size
        for _ in range(entries):
            child = self._addr(self._uoff(p + keysize))
            if level == 0:
                self._read_snod(child, name_at, result)
            else:
                self._walk_group_btree(child, name_at, result)
            p += keysize + self.off_size

    def _read_snod(self, addr, name_at, result):
        d = self.d
        nsym = struct.unpack_from("<H", d, addr + 6)[0]
        p, ent = addr + 8, 2 * self.off_size + 8 + 16
        for _ in range(nsym):
            name = name_at(self._uoff(p))
            result[name] = self._read_dataset(self._addr(self._uoff(p + self.off_size)))
            p += ent


def read_mat_v73(path):
    """Return ``{varname: ndarray}`` for a v7.3 ``.mat`` file."""
    return _Reader(path).read_all()
