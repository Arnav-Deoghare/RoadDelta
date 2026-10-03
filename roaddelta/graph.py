"""Loader for MUNO21 .graph files.

ASSUMED FORMAT (from Bastani's earlier road-extraction tools; verify against a real file
with scripts/inspect_data.py before trusting it):
    x y            <- one line per vertex, pixel coords in the matching JPEG
    ...
    <blank line>
    u v            <- one line per undirected edge, 0-based vertex indices
    ...
Official spec: https://favyen.com/muno21/graph-format.txt
"""
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class RoadGraph:
    vertices: np.ndarray  # (N, 2) float32, (x, y) in pixels
    edges: np.ndarray     # (M, 2) int64, indices into vertices

    @property
    def n_vertices(self):
        return len(self.vertices)

    @property
    def n_edges(self):
        return len(self.edges)


def load_graph(path):
    verts, edges, section = [], [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                section += 1
                continue
            p = line.split()
            if section == 0:
                verts.append((float(p[0]), float(p[1])))
            else:
                edges.append((int(p[0]), int(p[1])))
    v = np.array(verts, dtype=np.float32).reshape(-1, 2)
    e = np.array(edges, dtype=np.int64).reshape(-1, 2)
    if len(e) and (e.min() < 0 or e.max() >= len(v)):
        raise ValueError(
            f"{path}: edge indices out of range for {len(v)} vertices. "
            "The .graph format assumption in graph.py is probably wrong; "
            "compare against graph-format.txt."
        )
    return RoadGraph(v, e)


def rasterize(graph, shape, width=3):
    """Binary road mask (H, W) uint8 from a graph. shape = (H, W)."""
    mask = np.zeros(shape, dtype=np.uint8)
    for a, b in graph.edges:
        p1 = tuple(int(round(c)) for c in graph.vertices[a])
        p2 = tuple(int(round(c)) for c in graph.vertices[b])
        cv2.line(mask, p1, p2, 1, thickness=width)
    return mask


def overlay(image_bgr, graph, color=(0, 0, 255), width=2, vertices=True):
    out = image_bgr.copy()
    for a, b in graph.edges:
        p1 = tuple(int(round(c)) for c in graph.vertices[a])
        p2 = tuple(int(round(c)) for c in graph.vertices[b])
        cv2.line(out, p1, p2, color, width)
    if vertices:
        for x, y in graph.vertices:
            cv2.circle(out, (int(round(x)), int(round(y))), 2, (0, 255, 255), -1)
    return out
