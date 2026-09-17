from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Iterable
import numpy as np

# ---------------- Params ----------------
@dataclass
class Params:
    y_align_tol: float = 1.5
    char_width_factor: float = 0.6
    x_space_mult: float = 2.5
    para_vgap_mult: float = 2.0
    # indent_tol: float = 6.0
    # col_max_vgap_mult: float = 5.0
    neighbor_radius_mult: float = 1.2


# ---------------- Types ----------------
BBox = Tuple[float, float, float, float]

@dataclass
class Block:
    page: int
    text: str
    bbox: BBox
    kind: str                  # "word"|"line"|"paragraph"|"table_json"
    children: List["Block"] = field(default_factory=list)
    font_size: float = 0.0

    @property
    def x0(self): return self.bbox[0]
    @property
    def y0(self): return self.bbox[1]
    @property
    def x1(self): return self.bbox[2]
    @property
    def y1(self): return self.bbox[3]
    @property
    def w(self):  return self.x1 - self.x0
    @property
    def h(self):  return self.y1 - self.y0
    @property
    def cx(self): return 0.5*(self.x0 + self.x1)
    @property
    def cy(self): return 0.5*(self.y0 + self.y1)


# ---------------- Utils ----------------
def median(nums: List[float], default: float = 12.0) -> float:
    if not nums: return default
    s = sorted(nums); return s[len(s)//2]

def cluster_coords(sorted_coords, tol=2.0):
    ticks = []
    for v in sorted(sorted_coords):
        if not ticks or abs(v - ticks[-1]) > tol: ticks.append(float(v))
    return ticks

def infer_grid_from_cells(cluster, tol=2.0):
        xs, ys = [], []
        for c in cluster:
            x0,y0,x1,y1 = c["bbox"]
            xs.extend([x0,x1]); ys.extend([y0,y1])
        return cluster_coords(xs,tol), cluster_coords(ys,tol)

def find_tick_index(ticks, v, tol=2.0):
    for i, t in enumerate(ticks):
        if abs(v - t) <= tol: return i
    return int(np.argmin([abs(v - t) for t in ticks]))

def build_span_matrix(cluster, x_ticks, y_ticks, tol=2.0):
    R, C = max(0,len(y_ticks)-1), max(0,len(x_ticks)-1)
    M = [[None for _ in range(C)] for __ in range(R)]
    taken = [[False for _ in range(C)] for __ in range(R)]
    ordered = sorted(cluster, key=lambda c: (c["bbox"][2]-c["bbox"][0])*(c["bbox"][3]-c["bbox"][1]), reverse=True)
    for c in ordered:
        x0,y0,x1,y1 = c["bbox"]; t = (c.get("text") or "").strip()
        c0 = find_tick_index(x_ticks, x0, tol); c1 = find_tick_index(x_ticks, x1, tol)
        r0 = find_tick_index(y_ticks, y0, tol); r1 = find_tick_index(y_ticks, y1, tol)
        c0,c1 = min(c0,c1), max(c0,c1); r0,r1 = min(r0,r1), max(r0,r1)
        colspan = max(1, c1-c0); rowspan = max(1, r1-r0)
        if 0 <= r0 < R and 0 <= c0 < C and not taken[r0][c0]:
            M[r0][c0] = {"text": t, "rowspan": rowspan, "colspan": colspan, "bbox": (x0,y0,x1,y1)}
            for rr in range(r0, min(r0+rowspan,R)):
                for cc in range(c0, min(c0+colspan,C)):
                    taken[rr][cc] = True
    return M, taken

def greedy_natural_order(indices: List[int], blocks: List[Block]) -> List[int]:
    if not indices: return []
    remaining = set(indices)
    cur = min(remaining, key=lambda k: (blocks[k].page, blocks[k].y0, blocks[k].x0))
    order=[cur]; remaining.remove(cur)
    def dist(a,b):
        ax,ay = blocks[a].cx, blocks[a].cy; bx,by = blocks[b].cx, blocks[b].cy
        return math.hypot(ax-bx, ay-by)
    while remaining:
        nearest = min(remaining, key=lambda k: (round(dist(cur,k),6), blocks[k].cy, blocks[k].cx))
        order.append(nearest); remaining.remove(nearest); cur=nearest
    return order

def merge_bboxes(bboxes: Iterable[BBox]) -> BBox:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for x0,y0,x1,y1 in bboxes:
        xs0.append(x0); ys0.append(y0); xs1.append(x1); ys1.append(y1)
    return (min(xs0), min(ys0), max(xs1), max(ys1)) if xs0 else (0,0,0,0)

def v_gap(a: Block, b: Block) -> float: return b.y0 - a.y1

def count_tokens(blocks: List[Block]) -> int:
    return sum(len(b.text.split()) for b in blocks)

def clean(s: str) -> str: return " ".join(s.split()).strip()

def strip_bbox(obj):
    if isinstance(obj, dict):
        return {k: strip_bbox(v) for k, v in obj.items() if k != "bbox"}
    if isinstance(obj, list):
        return [strip_bbox(x) for x in obj]
    return obj
