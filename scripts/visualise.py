"""Visualise one MUNO21 scenario: old/new aerial images, road graphs, and ground-truth changes.

Run from the project root:
  python scripts/visualize.py --root data/raw/mapupdate --tag constructed --nth 0
  python scripts/visualize.py --root data/raw/mapupdate --index 0
  python scripts/visualize.py --root data/raw/mapupdate --tag nochange

Output is a 2x2 image:
  top-left     old image + old road map        top-right    new image + new road map
  bottom-left  old image + changes             bottom-right new image + changes
Colours: blue = road map, green = added road, red = removed road, yellow = annotation window.

This script is the alignment check: if the blue lines sit on visible roads and the green or red
segments sit inside the yellow window, the coordinate assumptions hold.
"""
import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from roaddelta import load_graph

OLD_YEAR = 2013
OLD_MAP_DATE = "2013-07-01"
NEW_MAP_DATE = "2020-07-01"

# OpenCV uses BGR
BLUE = (255, 120, 0)
GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)


def available_years(root, region, tile):
    pattern = os.path.join(root, "naip", "jpg", f"{region}_{tile[0]}_{tile[1]}_*.jpg")
    years = []
    for p in glob.glob(pattern):
        m = re.search(r"_(\d{4})\.jpg$", p)
        if m:
            years.append(int(m.group(1)))
    return sorted(years)


def pick_scenario(ann, index, tag, nth):
    if index is not None:
        return index, ann[index]
    matches = [i for i, x in enumerate(ann) if tag in x.get("Tags", [])]
    if not matches:
        raise SystemExit(f"No scenario with tag '{tag}'.")
    if nth >= len(matches):
        raise SystemExit(f"Only {len(matches)} scenarios have tag '{tag}'.")
    i = matches[nth]
    return i, ann[i]


def graph_path(root, region, tile, date):
    p = os.path.join(root, "graphs", "graphs", f"{region}_{tile[0]}_{tile[1]}_{date}.graph")
    if not os.path.exists(p):
        raise SystemExit(f"Graph not found: {p}")
    return p


def crop_box(window, pad, width, height):
    x1, y1, x2, y2 = window
    return (max(0, int(x1 - pad)), max(0, int(y1 - pad)),
            min(width, int(x2 + pad)), min(height, int(y2 + pad)))


def read_crop(path, window, pad):
    img = cv2.imread(path)  # full tile is large (hundreds of MB decoded); freed right after cropping
    if img is None:
        raise SystemExit(f"Could not read image: {path}")
    h, w = img.shape[:2]
    box = crop_box(window, pad, w, h)
    crop = img[box[1]:box[3], box[0]:box[2]].copy()
    del img
    return crop, box, (w, h)


def edges_in_box(graph, box):
    """Edges whose bounding box touches the crop box, shifted into crop coordinates."""
    x1, y1, x2, y2 = box
    a = graph.vertices[graph.edges[:, 0]]
    b = graph.vertices[graph.edges[:, 1]]
    keep = ((np.minimum(a[:, 0], b[:, 0]) <= x2) & (np.maximum(a[:, 0], b[:, 0]) >= x1) &
            (np.minimum(a[:, 1], b[:, 1]) <= y2) & (np.maximum(a[:, 1], b[:, 1]) >= y1))
    shift = np.array([x1, y1], dtype=np.float32)
    return a[keep] - shift, b[keep] - shift


def pt(p):
    return (int(round(p[0])), int(round(p[1])))


def draw_map(img, p1, p2, color, thickness):
    for a, b in zip(p1, p2):
        cv2.line(img, pt(a), pt(b), color, thickness)


def draw_changes(img, changes, origin, thickness):
    ox, oy = origin
    for c in changes or []:
        color = RED if c.get("Deleted") else GREEN
        for s in c["Segments"]:
            a = (s["Start"]["X"] - ox, s["Start"]["Y"] - oy)
            b = (s["End"]["X"] - ox, s["End"]["Y"] - oy)
            cv2.line(img, pt(a), pt(b), color, thickness)


def draw_window(img, window, origin, thickness):
    x1, y1, x2, y2 = window
    ox, oy = origin
    cv2.rectangle(img, (int(x1 - ox), int(y1 - oy)), (int(x2 - ox), int(y2 - oy)), YELLOW, thickness)


def title(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def fit(img, max_side):
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to the extracted mapupdate folder")
    ap.add_argument("--index", type=int, help="position in annotations.json")
    ap.add_argument("--tag", default="constructed", help="pick a scenario by tag (default: constructed)")
    ap.add_argument("--nth", type=int, default=0, help="which scenario among those with the tag")
    ap.add_argument("--new-year", type=int, help="image year for the new side (default: latest available)")
    ap.add_argument("--pad", type=int, default=128, help="pixels of context around the window")
    ap.add_argument("--max-side", type=int, default=900, help="max panel side in the output")
    ap.add_argument("--out", help="output path (default: outputs/viz_<index>.jpg)")
    a = ap.parse_args()

    ann = json.load(open(os.path.join(a.root, "annotations.json")))
    idx, sc = pick_scenario(ann, a.index, a.tag, a.nth)
    cl = sc["Cluster"]
    region, tile, window = cl["Region"], cl["Tile"], cl["Window"]

    years = available_years(a.root, region, tile)
    if OLD_YEAR not in years:
        raise SystemExit(f"No {OLD_YEAR} image for {region} tile {tile}. Available: {years}")
    new_year = a.new_year or max(years)
    if new_year == OLD_YEAR:
        raise SystemExit(f"Only the {OLD_YEAR} image exists for {region} tile {tile}.")

    def img_path(y):
        return os.path.join(a.root, "naip", "jpg", f"{region}_{tile[0]}_{tile[1]}_{y}.jpg")

    old_img, box, (w, h) = read_crop(img_path(OLD_YEAR), window, a.pad)
    new_img, _, _ = read_crop(img_path(new_year), window, a.pad)
    if old_img.shape != new_img.shape:
        print(f"WARNING: crop shapes differ (old {old_img.shape}, new {new_img.shape}); "
              "the years may have different resolutions.")
        new_img = cv2.resize(new_img, (old_img.shape[1], old_img.shape[0]))

    g_old = load_graph(graph_path(a.root, region, tile, OLD_MAP_DATE))
    g_new = load_graph(graph_path(a.root, region, tile, NEW_MAP_DATE))
    o1, o2 = edges_in_box(g_old, box)
    n1, n2 = edges_in_box(g_new, box)

    changes = cl.get("Changes")
    origin = (box[0], box[1])
    t_line = max(2, max(old_img.shape[:2]) // 400)

    tl, tr, bl, br = old_img.copy(), new_img.copy(), old_img.copy(), new_img.copy()
    draw_map(tl, o1, o2, BLUE, t_line)
    draw_map(tr, n1, n2, BLUE, t_line)
    draw_changes(bl, changes, origin, t_line + 1)
    draw_changes(br, changes, origin, t_line + 1)
    for p in (tl, tr, bl, br):
        draw_window(p, window, origin, t_line)

    title(tl, f"OLD {OLD_YEAR} image + {OLD_MAP_DATE[:4]} map")
    title(tr, f"NEW {new_year} image + {NEW_MAP_DATE[:4]} map")
    title(bl, "OLD image + changes")
    title(br, "NEW image + changes")

    grid = np.vstack([np.hstack([fit(tl, a.max_side), fit(tr, a.max_side)]),
                      np.hstack([fit(bl, a.max_side), fit(br, a.max_side)])])
    caption = "blue=road map  green=added  red=removed  yellow=annotation window"
    bar = np.zeros((34, grid.shape[1], 3), np.uint8)
    cv2.putText(bar, f"{region} tile {tile} | idx {idx} | tags {sc.get('Tags')} | {caption}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    grid = np.vstack([bar, grid])

    out = a.out or os.path.join("outputs", f"viz_{idx}.jpg")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cv2.imwrite(out, grid)

    n_add = sum(1 for c in (changes or []) if not c.get("Deleted"))
    n_del = sum(1 for c in (changes or []) if c.get("Deleted"))
    print(f"scenario {idx}: {region} tile {tile}  tags={sc.get('Tags')}  roadtype={sc.get('RoadType')!r}")
    print(f"image years available: {years}  using old={OLD_YEAR} new={new_year}")
    print(f"tile size {w}x{h}  window {window}  crop box {box}")
    print(f"change records: {n_add} added, {n_del} removed")
    print(f"map edges in crop: old {len(o1)}, new {len(n1)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()