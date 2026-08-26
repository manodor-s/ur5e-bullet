import sys, struct
import numpy as np


def read_stl(path):
    with open(path, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        verts = np.empty((n * 3, 3), dtype=np.float32)
        for i in range(n):
            d = struct.unpack("<12fH", f.read(50))
            verts[i * 3:(i + 1) * 3] = np.array(d[3:12], dtype=np.float32).reshape(3, 3)
    tris = np.arange(n * 3).reshape(-1, 3)
    return verts, tris


def decimate(verts, tris, cell):
    cell_idx = np.floor(verts / cell).astype(np.int64)
    uniq, inv, first = np.unique(
        cell_idx, axis=0, return_inverse=True, return_index=True
    )
    rep = verts[inv]
    t = first[tris]
    keep = (t[:, 0] != t[:, 1]) & (t[:, 0] != t[:, 2]) & (t[:, 1] != t[:, 2])
    new_tris = t[keep]
    return rep, new_tris


def write_stl(path, verts, tris):
    n = len(tris)
    with open(path, "wb") as f:
        f.write(b"decimated lower.stl".ljust(80, b"\0"))
        f.write(struct.pack("<I", n))
        for i in range(n):
            a, b, c = (verts[tris[i, 0]], verts[tris[i, 1]], verts[tris[i, 2]])
            normal = np.cross(b - a, c - a)
            ln = np.linalg.norm(normal)
            normal = normal / ln if ln > 0 else np.array([0.0, 0.0, 1.0])
            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<9f", *a, *b, *c))
            f.write(struct.pack("<H", 0))


if __name__ == "__main__":
    src = sys.argv[1]
    dst = sys.argv[2]
    cell = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    verts, tris = read_stl(src)
    rep, new_tris = decimate(verts, tris, cell)
    write_stl(dst, rep, new_tris)
    orig_aabb = [verts.min(0), verts.max(0)]
    new_aabb = [rep.min(0), rep.max(0)]
    print(f"orig: {len(tris)} Dreiecke, AABB {orig_aabb}")
    print(f"neu : {len(new_tris)} Dreiecke, AABB {new_aabb}")
    print(f"Reduktion: {len(new_tris) / len(tris):.1%}  ({cell} mm Zelle)")
