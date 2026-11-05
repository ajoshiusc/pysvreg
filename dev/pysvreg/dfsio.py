"""BrainSuite DFS I/O helpers reused for the Python SVReg port."""

from __future__ import annotations

from types import SimpleNamespace
import os
import struct

import numpy as np


def readdfs(fname: str):
    if not os.path.exists(fname):
        raise IOError(f"File name {fname} does not exist.")

    hdr = SimpleNamespace()
    nfv = SimpleNamespace()

    with open(fname, "rb") as stream:
        hdr.ftype_header = np.array(struct.unpack("c" * 12, stream.read(12)), dtype="S1")
        if b"DFS" not in hdr.ftype_header.tobytes():
            raise ValueError(f"Invalid dfs file {fname}")

        hdr.hdrsize = np.array(struct.unpack("i", stream.read(4)), dtype="int32")[0]
        hdr.mdoffset = np.array(struct.unpack("i", stream.read(4)), dtype="int32")[0]
        hdr.pdoffset = np.array(struct.unpack("i", stream.read(4)), dtype="int32")[0]
        hdr.nTriangles = np.array(struct.unpack("i", stream.read(4)), dtype="int32")
        hdr.nVertices = np.array(struct.unpack("i", stream.read(4)), dtype="int32")
        hdr.nStrips = np.array(struct.unpack("i", stream.read(4)), dtype="int32")[0]
        hdr.stripSize = np.array(struct.unpack("i", stream.read(4)), dtype="int32")[0]
        hdr.normals = np.array(struct.unpack("i", stream.read(4)), dtype="int32")[0]
        hdr.uvStart = np.array(struct.unpack("i", stream.read(4)), dtype="int32")[0]
        hdr.vcoffset = np.array(struct.unpack("i", stream.read(4)), dtype="int32")[0]
        hdr.labelOffset = np.array(struct.unpack("i", stream.read(4)), dtype="int32")[0]
        hdr.vertexAttributes = np.array(struct.unpack("i", stream.read(4)), dtype="int32")[0]
        stream.seek(hdr.hdrsize)

        if hdr.nTriangles[0] > 0:
            nfv.faces = np.array(
                struct.unpack("i" * 3 * hdr.nTriangles[0], stream.read(3 * hdr.nTriangles[0] * 4)),
                dtype="int32",
            ).reshape(hdr.nTriangles[0], 3)

        nfv.vertices = np.array(
            struct.unpack("f" * 3 * hdr.nVertices[0], stream.read(3 * hdr.nVertices[0] * 4)),
            dtype="float32",
        ).reshape(hdr.nVertices[0], 3)

        if hdr.normals > 0:
            stream.seek(hdr.normals)
            nfv.normals = np.array(
                struct.unpack("f" * 3 * hdr.nVertices[0], stream.read(3 * hdr.nVertices[0] * 4)),
                dtype="float32",
            ).reshape(hdr.nVertices[0], 3)

        if hdr.vcoffset > 0:
            stream.seek(hdr.vcoffset)
            nfv.vColor = np.array(
                struct.unpack("f" * 3 * hdr.nVertices[0], stream.read(3 * hdr.nVertices[0] * 4)),
                dtype="float32",
            ).reshape(hdr.nVertices[0], 3)

        if hdr.uvStart > 0:
            stream.seek(hdr.uvStart)
            uv = np.array(
                struct.unpack("f" * 2 * hdr.nVertices[0], stream.read(2 * hdr.nVertices[0] * 4)),
                dtype="float32",
            ).reshape(hdr.nVertices[0], 2)
            nfv.u = uv[:, 0]
            nfv.v = uv[:, 1]

        if hdr.labelOffset > 0:
            stream.seek(hdr.labelOffset)
            nfv.labels = np.array(
                struct.unpack("H" * hdr.nVertices[0], stream.read(hdr.nVertices[0] * 2)),
                dtype="uint16",
            )

        if hdr.vertexAttributes > 0:
            stream.seek(hdr.vertexAttributes)
            nfv.attributes = np.array(
                struct.unpack("f" * hdr.nVertices[0], stream.read(hdr.nVertices[0] * 4)),
                dtype="float32",
            )

    nfv.name = fname
    return nfv


def writedfs(fname: str, nfv) -> None:
    header = np.array(list("DFS_LE v2.0\0"), dtype="S1")
    hdrsize = 184
    ntriangles = len(nfv.faces.flatten()) // 3
    nvertices = len(nfv.vertices.flatten()) // 3
    normals = 0
    uvoffset = 0
    vcoffset = 0
    label_offset = 0
    attributes = 0
    next_array_pos = hdrsize + 12 * (ntriangles + nvertices)

    if hasattr(nfv, "normals"):
        normals = next_array_pos
        next_array_pos += nvertices * 12
    if hasattr(nfv, "vColor"):
        vcoffset = next_array_pos
        next_array_pos += nvertices * 12
    if hasattr(nfv, "u") and hasattr(nfv, "v"):
        uvoffset = next_array_pos
        next_array_pos += nvertices * 8
    if hasattr(nfv, "labels"):
        label_offset = next_array_pos
        next_array_pos += nvertices * 2
    if hasattr(nfv, "attributes"):
        attributes = next_array_pos
        next_array_pos += nvertices * 4

    with open(fname, "wb") as stream:
        stream.write(header.tobytes())
        stream.write(np.array(hdrsize, dtype="int32").tobytes())
        stream.write(np.array(0, dtype="int32").tobytes())
        stream.write(np.array(0, dtype="int32").tobytes())
        stream.write(np.array(ntriangles, dtype="int32").tobytes())
        stream.write(np.array(nvertices, dtype="int32").tobytes())
        stream.write(np.array(0, dtype="int32").tobytes())
        stream.write(np.array(0, dtype="int32").tobytes())
        stream.write(np.array(normals, dtype="int32").tobytes())
        stream.write(np.array(uvoffset, dtype="int32").tobytes())
        stream.write(np.array(vcoffset, dtype="int32").tobytes())
        stream.write(np.array(label_offset, dtype="int32").tobytes())
        stream.write(np.array(attributes, dtype="int32").tobytes())
        stream.write(np.zeros([1, 4 + 15 * 8], dtype="uint8").tobytes())
        stream.write(np.array(nfv.faces, dtype="int32").tobytes())
        stream.write(np.array(nfv.vertices, dtype="float32").tobytes())
        if normals > 0:
            stream.write(np.array(nfv.normals, dtype="float32").tobytes())
        if vcoffset > 0:
            stream.write(np.array(nfv.vColor, dtype="float32").tobytes())
        if uvoffset > 0:
            uv = np.column_stack([np.asarray(nfv.u, dtype="float32").reshape(-1), np.asarray(nfv.v, dtype="float32").reshape(-1)])
            stream.write(uv.astype("float32").tobytes())
        if label_offset > 0:
            stream.write(np.array(nfv.labels, dtype="int16").tobytes())
        if attributes > 0:
            stream.write(np.array(nfv.attributes, dtype="float32").tobytes())
