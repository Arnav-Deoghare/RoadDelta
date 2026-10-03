"""Sanity-check a downloaded MUNO21 folder.

  python scripts/inspect_data.py --root ~/data/mapupdate
  python scripts/inspect_data.py --root ~/data/mapupdate \
      --image naip/jpg/ny_1_0_2019.jpg --graph graphs/graphs/ny_1_0_2018-07-01.graph --out check.jpg
"""
import argparse, json, os, sys
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from roaddelta import load_graph, overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--image")
    ap.add_argument("--graph")
    ap.add_argument("--out", default="check.jpg")
    a = ap.parse_args()
    r = a.root

    jpg = os.path.join(r, "naip", "jpg")
    gdir = os.path.join(r, "graphs", "graphs")
    imgs = sorted(os.listdir(jpg)) if os.path.isdir(jpg) else []
    graphs = sorted(os.listdir(gdir)) if os.path.isdir(gdir) else []
    print(f"images: {len(imgs)}  e.g. {imgs[:3]}")
    print(f"graphs: {len(graphs)}  e.g. {graphs[:3]}")

    ann_path = os.path.join(r, "annotations.json")
    if os.path.exists(ann_path):
        ann = json.load(open(ann_path))
        print(f"annotations: {type(ann).__name__}, len={len(ann)}")
        first = ann[0] if isinstance(ann, list) else next(iter(ann.values()))
        print("first annotation keys:", list(first.keys()) if isinstance(first, dict) else first)
    else:
        print("annotations.json NOT FOUND")

    if graphs:
        g0 = os.path.join(gdir, graphs[0])
        print(f"\nraw head of {graphs[0]}:")
        with open(g0) as f:
            for _ in range(6):
                print("  ", f.readline().rstrip())
        g = load_graph(g0)
        print(f"parsed: {g.n_vertices} vertices, {g.n_edges} edges")

    if a.image and a.graph:
        img = cv2.imread(os.path.join(r, a.image))
        g = load_graph(os.path.join(r, a.graph))
        cv2.imwrite(a.out, overlay(img, g))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
