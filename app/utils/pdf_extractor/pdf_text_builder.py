from __future__ import annotations

import io, json, math
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import numpy as np
import fitz, pdfplumber
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from functools import lru_cache
from math import hypot

from app.utils.pdf_extractor.utils import Params, Block
from app.utils.pdf_extractor.utils import median, merge_bboxes, count_tokens, v_gap, clean, infer_grid_from_cells, build_span_matrix, greedy_natural_order, strip_bbox

# ======================================================================
#                              PIPELINE
# ======================================================================
class PDFTextBuilder:
    params = Params()

    # ---------- Step 1: extraction ----------
    @staticmethod
    def extract_text_items(
        pdf_bytes: bytes,
        granularity: str = "word",
        *,
        min_length: float = 5.0
    ) -> Tuple[List[Dict[str, Any]], Dict[int, Tuple[float, float]], List[Dict[str, Any]]]:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        out, page_sizes = [], {}
        for page_index, page in enumerate(doc):
            page_sizes[page_index] = (page.rect.width, page.rect.height)
            if granularity == "word":
                for x0,y0,x1,y1,w,*_ in page.get_text("words"):
                    w = clean(w)
                    if w:
                        out.append({"page_index": page_index, "bbox": (x0,y0,x1,y1), "text": w})
            else:
                layout = page.get_text("dict")
                for blk in layout.get("blocks", []):
                    if blk.get("type", 0) != 0: continue
                    for line in blk.get("lines", []):
                        spans = line.get("spans", [])
                        if not spans: continue
                        text = " ".join(clean(s.get("text","")) for s in spans if s.get("text"))
                        if not text: continue
                        x0 = min(s["bbox"][0] for s in spans)
                        y0 = min(s["bbox"][1] for s in spans)
                        x1 = max(s["bbox"][2] for s in spans)
                        y1 = max(s["bbox"][3] for s in spans)
                        out.append({"page_index": page_index, "bbox": (x0,y0,x1,y1), "text": text})
        doc.close()

        merged_lines = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_index, page in enumerate(pdf.pages):
                W, H = page.width, page.height
                norm_y = lambda y: H - y
                raw_segments = []
                for ln in page.lines:
                    raw_segments.append((ln["x0"], ln["y0"], ln["x1"], ln["y1"]))
                for rect in page.rects:
                    fill = rect.get("fill", False)
                    x0,y0,x1,y1 = rect["x0"],rect["y0"],rect["x1"],rect["y1"]
                    rw, rh = abs(x1-x0), abs(y1-y0)
                    if fill and min(rw, rh) < 3:
                        raw_segments.extend([
                            (x0,y0,x1,y0),(x1,y0,x1,y1),(x0,y1,x1,y1),(x0,y0,x0,y1)
                        ])
                for (x0,y0,x1,y1) in raw_segments:
                    y0,y1 = norm_y(y0), norm_y(y1)
                    dx, dy = x1-x0, y1-y0
                    length = (dx*dx + dy*dy) ** 0.5
                    if length < min_length: continue
                    if abs(dy) < 0.5:
                        kind = "h"; seg = (min(x0,x1), y0, max(x0,x1), y0)
                    elif abs(dx) < 0.5:
                        kind = "v"; seg = (x0, min(y0,y1), x0, max(y0,y1))
                    else:
                        kind = "diag"; seg = (x0,y0,x1,y1)
                    merged_lines.append({"page_index": page_index, "kind": kind, "points": seg, "length": length})
        return out, page_sizes, merged_lines

    # ---------- Step 2: words -> lines ----------
    @classmethod
    def __estimate_space_width(cls, words: List[Block]) -> float:
        widths = []
        for w in words:
            n = max(len(w.text), 1)
            widths.append(w.w / n)
        if not widths: return 4.0
        med = sorted(widths)[len(widths)//2]
        return cls.params.char_width_factor * med

    @classmethod
    def __group_words_into_lines(
        cls, words: List[Block], *, page_sizes: Optional[Dict[int, Tuple[float, float]]] = None
    ) -> List[Block]:
        if not words: return []
        words_sorted = sorted(words, key=lambda b: (b.page, b.y0, b.x0))
        space_w = cls.__estimate_space_width(words_sorted)
        max_gap = cls.params.x_space_mult * space_w

        def page_width(p: int) -> float:
            if page_sizes and p in page_sizes: return float(page_sizes[p][0])
            xs = [w.x1 for w in words_sorted if w.page == p]; return (max(xs) if xs else 1000.0)

        out, used = [], set()
        page_to_idxs: Dict[int, List[int]] = defaultdict(list)
        for idx, w in enumerate(words_sorted): page_to_idxs[w.page].append(idx)

        for page, idxs in page_to_idxs.items():
            pw = page_width(page)
            gutter = min(8.0*space_w, 0.18*pw)
            split_gap = max(3.0*space_w, 0.08*pw)
            i = 0
            while i < len(idxs):
                if idxs[i] in used: i += 1; continue
                base_i = idxs[i]
                base = words_sorted[base_i]
                group = [base_i]; used.add(base_i)
                j = i+1
                while j < len(idxs):
                    k = idxs[j]
                    if k in used: j += 1; continue
                    nxt = words_sorted[k]
                    same_baseline = abs((nxt.y0+nxt.y1)/2 - (base.y0+base.y1)/2) <= cls.params.y_align_tol
                    if not same_baseline: break
                    prev = words_sorted[group[-1]]
                    gap = nxt.x0 - prev.x1
                    if gap > split_gap: break
                    if gap > -0.5*space_w and gap <= max_gap and gap <= gutter:
                        group.append(k); used.add(k); j += 1; continue
                    break
                group_words = [words_sorted[g] for g in group]
                text = " ".join(w.text for w in group_words)
                bbox = merge_bboxes(w.bbox for w in group_words)
                line_font = median([w.font_size for w in group_words], default=12.0)
                out.append(Block(page=page, text=text, bbox=bbox, kind="line", children=group_words, font_size=line_font))
                i = j
            for k in idxs:
                if k not in used:
                    w = words_sorted[k]
                    out.append(Block(page=w.page, text=w.text, bbox=w.bbox, kind="line", children=[], font_size=w.font_size))
        out.sort(key=lambda b: (b.page, b.y0, b.x0))
        return out

    # ---------- Step 3: lines -> paragraphs ----------
    @classmethod
    def __group_lines_into_paragraphs(
        cls, lines: List[Block], *, table_lines: Optional[List[Dict[str, Any]]] = None
    ) -> List[Block]:
        if not lines: return []
        lines_sorted = sorted(lines, key=lambda b: (b.page, b.y0, b.x0))
        out: List[Block] = []

        def line_between(a: Block, b: Block, tol: float = 1.0) -> bool:
            if not table_lines or a.page != b.page: return False
            top, bottom = min(a.y1,b.y1), max(a.y0,b.y0)
            left, right = min(a.x0,b.x0), max(a.x1,b.x1)
            for l in table_lines:
                if l["page_index"] != a.page or l["kind"] != "h": continue
                x0,y0,x1,y1 = l["points"]
                if (top - tol) < y0 < (bottom + tol):
                    if not (x1 < left - tol or x0 > right + tol): return True
            return False

        def compatible(a: Block, b: Block) -> bool:
            if a.page != b.page: return False
            ref = max(a.font_size, b.font_size)
            if v_gap(a, b) > cls.params.para_vgap_mult * ref: return False
            inside = (a.x0 <= b.cx <= a.x1) or (b.x0 <= a.cx <= b.x1)
            if not inside: return False
            if line_between(a, b):
                return False
            return True

        remaining = set(range(len(lines_sorted)))
        while remaining:
            seed = min(remaining, key=lambda i: (lines_sorted[i].y0, lines_sorted[i].x0))
            group = [seed]; remaining.remove(seed)
            while True:
                last = lines_sorted[group[-1]]
                candidates = []
                for j in list(remaining):
                    cand = lines_sorted[j]
                    if compatible(last, cand):
                        candidates.append((max(0.0, cand.y0 - last.y1), j))
                if not candidates: break
                candidates.sort(key=lambda t: t[0])
                _, best_idx = candidates[0]
                group.append(best_idx); remaining.remove(best_idx)

            segs = [lines_sorted[g] for g in group]
            text = " ".join(s.text for s in segs)
            bbox = merge_bboxes([s.bbox for s in segs])
            para_font = median([s.font_size for s in segs], default=12.0)
            out.append(Block(page=segs[0].page, text=text, bbox=bbox, kind="paragraph", children=segs, font_size=para_font))
        out.sort(key=lambda b: (b.page, b.y0, b.x0))
        return out

    # ---------- Step 3': tables first (from WORDS) ----------
    @classmethod
    def __tables_from_words(
        cls,
        words: List[Block],
        *,
        table_lines: Optional[List[Dict[str, Any]]] = None,
        page_sizes: Optional[Dict[int, Tuple[float, float]]] = None,
        include_bbox: bool = True,
        overlap_threshold: float = 0.70,
    ) -> Tuple[List[Block], Dict[int, List[Tuple[float,float,float,float]]]]:
        """
        Build tables directly from geometry and FILL each cell with text derived by:
        words -> lines -> paragraphs (scoped to words within that cell).
        Returns:
          - table Blocks (kind="table_json")
          - dict page->list of cell bboxes (for leftover-word filtering)
        """
        if not words or not table_lines:
            return [], defaultdict(list)

        # index words by page
        by_page_words: Dict[int, List[Block]] = defaultdict(list)
        for w in words:
            by_page_words[w.page].append(w)

        # ---- collect geometry lines per page ----
        page_to_h, page_to_v = defaultdict(list), defaultdict(list)
        for l in table_lines:
            if l["kind"] == "h" and l["length"] > 10: page_to_h[l["page_index"]].append(l)
            elif l["kind"] == "v" and l["length"] > 10: page_to_v[l["page_index"]].append(l)

        def merge_lines(lines, kind, tol=1):
            merged = []
            if not lines: return merged
            if kind == "h":
                lines = sorted(lines, key=lambda l: l["points"][1])
                groups, cur = [], [lines[0]]
                for l in lines[1:]:
                    if abs(l["points"][1] - cur[-1]["points"][1]) <= tol: cur.append(l)
                    else: groups.append(cur); cur = [l]
                groups.append(cur)
                for g in groups:
                    y_avg = float(np.mean([l["points"][1] for l in g]))
                    spans = np.array([(l["points"][0], l["points"][2]) for l in g], dtype=float)
                    spans = spans[np.argsort(spans[:,0])]
                    s,e = spans[0]
                    for s2,e2 in spans[1:]:
                        if s2 <= e + tol: e = max(e, e2)
                        else: merged.append([s,y_avg,e,y_avg]); s,e = s2,e2
                    merged.append([s,y_avg,e,y_avg])
            else:
                lines = sorted(lines, key=lambda l: l["points"][0])
                groups, cur = [], [lines[0]]
                for l in lines[1:]:
                    if abs(l["points"][0] - cur[-1]["points"][0]) <= tol: cur.append(l)
                    else: groups.append(cur); cur = [l]
                groups.append(cur)
                for g in groups:
                    x_avg = float(np.mean([l["points"][0] for l in g]))
                    spans = np.array([(l["points"][1], l["points"][3]) for l in g], dtype=float)
                    spans = spans[np.argsort(spans[:,0])]
                    s,e = spans[0]
                    for s2,e2 in spans[1:]:
                        if s2 <= e + tol: e = max(e, e2)
                        else: merged.append([x_avg,s,x_avg,e]); s,e = s2,e2
                    merged.append([x_avg,s,x_avg,e])
            return merged

        def detect_cells(h_lines, v_lines, tol=2.0):
            merged_h = merge_lines(h_lines, "h", tol)
            merged_v = merge_lines(v_lines, "v", tol)
            intersections, edges = [], set()
            for hx0,hy0,hx1,hy1 in merged_h:
                for vx0,vy0,vx1,vy1 in merged_v:
                    if (vx0 >= hx0 - tol and vx0 <= hx1 + tol) and (hy0 >= vy0 - tol and hy0 <= vy1 + tol):
                        intersections.append((round(vx0,2), round(hy0,2)))
            intersections = sorted(set(intersections))
            if not intersections:
                return []

            X = np.array([p[0] for p in intersections], float)
            Y = np.array([p[1] for p in intersections], float)
            for hx0,hy0,hx1,hy1 in merged_h:
                idx = np.where((X >= hx0 - tol) & (X <= hx1 + tol) & (np.abs(Y - hy0) <= tol))[0]
                if idx.size >= 2:
                    pts = [intersections[i] for i in idx[np.argsort(X[idx])]]
                    for i in range(len(pts)-1): edges.add((pts[i], pts[i+1]))
            for vx0,vy0,vx1,vy1 in merged_v:
                idx = np.where((Y >= vy0 - tol) & (Y <= vy1 + tol) & (np.abs(X - vx0) <= tol))[0]
                if idx.size >= 2:
                    pts = [intersections[i] for i in idx[np.argsort(Y[idx])]]
                    for i in range(len(pts)-1): edges.add((pts[i], pts[i+1]))
            edges = {(min(a,b), max(a,b)) for a,b in edges}
            adj = {}
            for e1,e2 in edges:
                adj.setdefault(e1,set()).add(e2); adj.setdefault(e2,set()).add(e1)

            def has_long_edge(a,b,d='h',tol=3.0):
                if d == 'v' and abs(a[0]-b[0])>tol: return False
                if d == 'h' and abs(a[1]-b[1])>tol: return False
                visited={a}; q=[a]
                while q:
                    cur=q.pop(0)
                    if cur==b: return True
                    for nxt in adj.get(cur,()):
                        if nxt in visited: continue
                        if (d=='h' and abs(cur[1]-nxt[1])<=tol) or (d=='v' and abs(cur[0]-nxt[0])<=tol):
                            visited.add(nxt); q.append(nxt)
                return False

            cells=set(); points=intersections.copy()
            while points:
                p0=points[0]; points=points[1:]
                same_x=[p for p in points if abs(p[0]-p0[0])<=tol]
                same_y=[p for p in points if abs(p[1]-p0[1])<=tol]
                rest=[p for p in points if abs(p[1]-p0[1])>tol and abs(p[0]-p0[0])>tol]
                rest.sort(key=lambda p: hypot(p[0]-p0[0], p[1]-p0[1]))
                for (rx,ry) in rest:
                    p1=(rx,ry)
                    p2=next(((rx,y2) for (xx,y2) in same_y if abs(xx-rx)<=tol and has_long_edge(p1,(rx,y2),'v')), None)
                    p3=next(((x2,ry) for (x2,yy) in same_x if abs(yy-ry)<=tol and has_long_edge(p1,(x2,ry),'h')), None)
                    if p2 and p3 and has_long_edge(p0,p2,'h') and has_long_edge(p0,p3,'v'):
                        xs=[p0[0],p1[0],p2[0],p3[0]]; ys=[p0[1],p1[1],p2[1],p3[1]]
                        cells.add((min(xs),min(ys),max(xs),max(ys))); break
            return list(cells)

        page_cells = defaultdict(list)
        for page in sorted(set(page_to_h) | set(page_to_v)):
            horiz, vert = page_to_h[page], page_to_v[page]
            if not horiz or not vert: continue
            zones = detect_cells(horiz, vert)
            page_cells[page].extend(zones)

        # ---- Fill cells with paragraphs from words confined to each cell ----
        def center_inside(b: Block, cell_bbox: Tuple[float,float,float,float], tol: float = 0.5) -> bool:
            x0,y0,x1,y1 = cell_bbox
            return (b.cx >= x0 - tol) and (b.cx <= x1 + tol) and (b.cy >= y0 - tol) and (b.cy <= y1 + tol)

        cell_bboxes_by_page: Dict[int, List[Tuple[float,float,float,float]]] = defaultdict(list)
        cells_by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        for page, cells in page_cells.items():
            page_words = by_page_words.get(page, [])
            for (x0,y0,x1,y1) in cells:
                cell_bbox = (x0,y0,x1,y1)
                in_cell_words = [w for w in page_words if center_inside(w, cell_bbox)]

                # words -> lines -> paragraphs (scoped to this cell)
                lines = cls.__group_words_into_lines(in_cell_words, page_sizes=page_sizes) if in_cell_words else []
                paras = cls.__group_lines_into_paragraphs(lines, table_lines=None) if lines else []
                text = " ".join(p.text.strip() for p in paras) if paras else ""

                cells_by_page[page].append({"bbox": cell_bbox, "text": text})
                cell_bboxes_by_page[page].append(cell_bbox)

        # ---- cluster cells into tables and build table JSON ----
        def connected_tables(cells):
            def share_one(a,b):
                ax0,ay0,ax1,ay1 = a["bbox"]; bx0,by0,bx1,by1 = b["bbox"]
                A={(ax0,ay0),(ax0,ay1),(ax1,ay0),(ax1,ay1)}
                B={(bx0,by0),(bx0,by1),(bx1,by0),(bx1,by1)}
                return len(A & B) >= 1
            tables, visited = [], set()
            for i,c in enumerate(cells):
                if i in visited: continue
                cluster=[c]; queue=[i]; visited.add(i)
                while queue:
                    idx = queue.pop()
                    for j,other in enumerate(cells):
                        if j in visited: continue
                        if share_one(cells[idx], other):
                            cluster.append(other); visited.add(j); queue.append(j)
                tables.append(cluster)
            return tables

        def cluster_to_table_json(cluster, page, table_id, tol=2.0):
            if not cluster: return None
            x_ticks, y_ticks = infer_grid_from_cells(cluster, tol=tol)
            M, _ = build_span_matrix(cluster, x_ticks, y_ticks, tol=tol)
            x0 = min(c["bbox"][0] for c in cluster); y0 = min(c["bbox"][1] for c in cluster)
            x1 = max(c["bbox"][2] for c in cluster); y1 = max(c["bbox"][3] for c in cluster)
            rows=[]
            for r,row in enumerate(M):
                out_row=[]; c=0
                while c < len(row):
                    cell = row[c]
                    if cell is None: c+=1; continue
                    item = {
                        "r": r, "c": c,
                        "rowspan": int(cell["rowspan"]),
                        "colspan": int(cell["colspan"]),
                        "text": cell.get("text",""),
                    }
                    if include_bbox: item["bbox"] = [float(v) for v in cell["bbox"]]
                    out_row.append(item)
                    c += int(cell["colspan"])
                rows.append(out_row)
            tbl = {
                "type": "table",
                "doc_page": int(page), "table_id": int(table_id),
                "n_rows": len(M), "n_cols": (len(M[0]) if M else 0),
                "rows": rows,
            }
            if include_bbox:
                tbl["bbox"] = [float(x0),float(y0),float(x1),float(y1)]
            return tbl

        out_blocks: List[Block] = []
        for page, cells in cells_by_page.items():
            clusters = connected_tables(cells)
            for i, cluster in enumerate(clusters):
                tbl = cluster_to_table_json(cluster, page=page, table_id=i+1, tol=2.0)
                if not tbl: continue
                x0,y0,x1,y1 = tbl["bbox"]
                json_text = json.dumps(tbl, ensure_ascii=False)
                out_blocks.append(Block(page=page, text=json_text, bbox=(x0,y0,x1,y1), kind="table_json", children=[], font_size=8))

        return out_blocks, cell_bboxes_by_page

    # ---------- Step 5: order by proximity ----------
    @classmethod
    def __order_by_proximity(cls, blocks: List[Block]) -> List[Block]:
        if not blocks: return []
        page_to_idxs: Dict[int, List[int]] = defaultdict(list)
        for idx,b in enumerate(blocks): page_to_idxs[b.page].append(idx)
        ordered_global: List[int] = []
        for page, idxs in sorted(page_to_idxs.items()):
            centers = {i:(blocks[i].cx,blocks[i].cy) for i in idxs}
            adj: Dict[int, List[int]] = {i:[] for i in idxs}
            for i in idxs:
                xi,yi = centers[i]
                for j in idxs:
                    if j <= i: continue
                    xj,yj = centers[j]
                    pair_ref = max(6.0, 0.5*(blocks[i].font_size + blocks[j].font_size))
                    radius = cls.params.neighbor_radius_mult * pair_ref
                    if math.hypot(xi-xj, yi-yj) <= radius:
                        adj[i].append(j); adj[j].append(i)
            visited=set(); components=[]
            for i in idxs:
                if i in visited: continue
                stack=[i]; visited.add(i); comp=[]
                while stack:
                    u=stack.pop(); comp.append(u)
                    for v in adj[u]:
                        if v not in visited: visited.add(v); stack.append(v)
                components.append(comp)

            def comp_key(comp):
                bb = min((blocks[k] for k in comp), key=lambda b: (b.y0, b.x0))
                return (bb.y0, bb.x0)
            components.sort(key=comp_key)
            for comp in components:
                ordered_global.extend(greedy_natural_order(comp, blocks))
        return [blocks[i] for i in ordered_global] if ordered_global else blocks

    # ---------- Step 6: JSON emit helpers ----------
    @staticmethod
    def __paragraph_block_to_json(p: Block, include_bbox: bool = True) -> dict:
        obj = {
            "type": "paragraph",
            # "version": 1,
            "doc_page": int(p.page),
            "text": p.text,
        }
        if include_bbox:
            obj["bbox"] = [float(p.x0),float(p.y0),float(p.x1),float(p.y1)]
        return obj

    # ---------- Orchestrator ----------
    @staticmethod
    def build_text_from_pdf_words(
        pdf_bytes: bytes,
        *,
        visualize_tables: bool = False,
        use_bbox: bool = False,
    ) -> Dict[str, Any]:

        # 1) extract (raw words + page sizes + geometry lines)
        items, page_sizes, table_lines = PDFTextBuilder.extract_text_items(pdf_bytes)

        # words -> Block
        words: List[Block] = []
        for it in items:
            x0, y0, x1, y1 = it["bbox"]
            h = y1 - y0
            sz = float(it.get("font_size") or it.get("size") or h or 12.0)
            words.append(
                Block(
                    page=int(it["page_index"]),
                    text=str(it["text"]),
                    bbox=(x0, y0, x1, y1),
                    kind="word",
                    font_size=max(1.0, sz),
                )
            )
        words.sort(key=lambda b: (b.page, b.y0, b.x0))

        # 2) TABLES FIRST: words→lines→paragraphs INSIDE each detected cell
        tables_only_blocks, cell_bboxes_by_page = PDFTextBuilder.__tables_from_words(
            words,
            table_lines=table_lines,
            page_sizes=page_sizes,
            include_bbox=True,
        )

        # 3) LEFTOVER WORDS (outside any table cell) → lines → paragraphs
        def center_inside_bbox_list(b: Block, bbox_list: List[Tuple[float,float,float,float]], tol: float = 0.5) -> bool:
            for (x0,y0,x1,y1) in bbox_list:
                if (b.cx >= x0 - tol) and (b.cx <= x1 + tol) and (b.cy >= y0 - tol) and (b.cy <= y1 + tol):
                    return True
            return False

        leftover_words: List[Block] = []
        if not cell_bboxes_by_page:
            leftover_words = words[:]  # no tables found; everything leftover
        else:
            for w in words:
                bboxes = cell_bboxes_by_page.get(w.page, [])
                if not center_inside_bbox_list(w, bboxes):
                    leftover_words.append(w)

        # leftover: words -> lines
        lines_left = PDFTextBuilder.__group_words_into_lines(leftover_words, page_sizes=page_sizes) if leftover_words else []
        if not lines_left and leftover_words:
            lines_left = [Block(page=w.page, text=w.text, bbox=w.bbox, kind="line", children=[], font_size=w.font_size) for w in leftover_words]

        # leftover: lines -> paragraphs (respect ruled lines to avoid crossing)
        paras_left = PDFTextBuilder.__group_lines_into_paragraphs(lines_left, table_lines=table_lines) if lines_left else []

        # 4) MERGE + ORDER
        merged = (tables_only_blocks + paras_left)
        merged.sort(key=lambda b: (b.page, b.y0, b.x0))

        ordered_blocks = PDFTextBuilder.__order_by_proximity(merged)
        if not ordered_blocks and merged:
            ordered_blocks = merged

        # 5) JSON output (final) — keep bbox always
        out_blocks_json: List[Dict[str, Any]] = []
        for b in ordered_blocks:
            if b.kind == "table_json":
                tbl = json.loads(b.text)
                out_blocks_json.append({"type": "table", **tbl})
            elif b.kind == "paragraph":
                out_blocks_json.append(PDFTextBuilder.__paragraph_block_to_json(b, include_bbox=True))
            else:
                tmp_p = Block(page=b.page, text=b.text, bbox=b.bbox, kind="paragraph", children=[], font_size=b.font_size)
                out_blocks_json.append(PDFTextBuilder.__paragraph_block_to_json(tmp_p, include_bbox=True))

        if not use_bbox:
            clean_blocks = [strip_bbox(b) for b in out_blocks_json]
        else:
            clean_blocks = out_blocks_json
        result = {"size": len(clean_blocks), "blocks": clean_blocks}

        # Visualization
        if visualize_tables:
            paras_blocks_for_viz = paras_left if paras_left else []
            tables_only_json = [b for b in out_blocks_json if b.get("type") == "table"]
            try:
                PDFTextBuilder.__visualize_tables_debug(
                    words_blocks=words,
                    line_blocks=lines_left,
                    paragraph_blocks=paras_blocks_for_viz,
                    table_lines=table_lines or [],
                    tables_json=tables_only_json,   # function filters type=="table"
                )
            except Exception as e:
                print(f"Fail to display - Error: {e}")

        return result
    

    
    # ---------- Orchestrator ----------
    @staticmethod
    def build_text_from_img_words(
        items,
        page_sizes,
        table_lines,
        *,
        visualize_tables: bool = False,
        use_bbox: bool = False,
    ) -> Dict[str, Any]:

        # words -> Block
        words: List[Block] = []
        for it in items:
            x0, y0, x1, y1 = it["bbox"]
            h = y1 - y0
            sz = float(it.get("font_size") or it.get("size") or h or 12.0)
            words.append(
                Block(
                    page=int(it["page_index"]),
                    text=str(it["text"]),
                    bbox=(x0, y0, x1, y1),
                    kind="word",
                    font_size=max(1.0, sz),
                )
            )
        words.sort(key=lambda b: (b.page, b.y0, b.x0))

        # 2) TABLES FIRST: words→lines→paragraphs INSIDE each detected cell
        tables_only_blocks, cell_bboxes_by_page = PDFTextBuilder.__tables_from_words(
            words,
            table_lines=table_lines,
            page_sizes=page_sizes,
            include_bbox=True,
        )

        # 3) LEFTOVER WORDS (outside any table cell) → lines → paragraphs
        def center_inside_bbox_list(b: Block, bbox_list: List[Tuple[float,float,float,float]], tol: float = 0.5) -> bool:
            for (x0,y0,x1,y1) in bbox_list:
                if (b.cx >= x0 - tol) and (b.cx <= x1 + tol) and (b.cy >= y0 - tol) and (b.cy <= y1 + tol):
                    return True
            return False

        leftover_words: List[Block] = []
        if not cell_bboxes_by_page:
            leftover_words = words[:]  # no tables found; everything leftover
        else:
            for w in words:
                bboxes = cell_bboxes_by_page.get(w.page, [])
                if not center_inside_bbox_list(w, bboxes):
                    leftover_words.append(w)

        # leftover: words -> lines
        lines_left = PDFTextBuilder.__group_words_into_lines(leftover_words, page_sizes=page_sizes) if leftover_words else []
        if not lines_left and leftover_words:
            lines_left = [Block(page=w.page, text=w.text, bbox=w.bbox, kind="line", children=[], font_size=w.font_size) for w in leftover_words]

        # leftover: lines -> paragraphs (respect ruled lines to avoid crossing)
        paras_left = PDFTextBuilder.__group_lines_into_paragraphs(lines_left, table_lines=table_lines) if lines_left else []

        # 4) MERGE + ORDER
        merged = (tables_only_blocks + paras_left)
        merged.sort(key=lambda b: (b.page, b.y0, b.x0))

        ordered_blocks = PDFTextBuilder.__order_by_proximity(merged)
        if not ordered_blocks and merged:
            ordered_blocks = merged

        # 5) JSON output (final) — keep bbox always
        out_blocks_json: List[Dict[str, Any]] = []
        for b in ordered_blocks:
            if b.kind == "table_json":
                tbl = json.loads(b.text)
                out_blocks_json.append({"type": "table", **tbl})
            elif b.kind == "paragraph":
                out_blocks_json.append(PDFTextBuilder.__paragraph_block_to_json(b, include_bbox=True))
            else:
                tmp_p = Block(page=b.page, text=b.text, bbox=b.bbox, kind="paragraph", children=[], font_size=b.font_size)
                out_blocks_json.append(PDFTextBuilder.__paragraph_block_to_json(tmp_p, include_bbox=True))

        if not use_bbox:
            clean_blocks = [strip_bbox(b) for b in out_blocks_json]
        else:
            clean_blocks = out_blocks_json
        result = {"size": len(clean_blocks), "blocks": clean_blocks}

        # Visualization
        if visualize_tables:
            paras_blocks_for_viz = paras_left if paras_left else []
            tables_only_json = [b for b in out_blocks_json if b.get("type") == "table"]
            try:
                PDFTextBuilder.__visualize_tables_debug(
                    words_blocks=words,
                    line_blocks=lines_left,
                    paragraph_blocks=paras_blocks_for_viz,
                    table_lines=table_lines or [],
                    tables_json=tables_only_json,   # function filters type=="table"
                )
            except Exception as e:
                print(f"Fail to display - Error: {e}")

        return result

    # ---------- Visualization (optional) ----------
    @staticmethod
    def __visualize_tables_debug(
        words_blocks=None,                 # List[Block] | None
        line_blocks=None,                  # List[Block] | None
        paragraph_blocks=None,             # List[Block] | None
        table_lines=None,                  # List[Dict[str,Any]] | None
        tables_json=None,                  # List[Dict[str,Any]] | None
    ):
        table_lines   = table_lines   or []
        tables_json   = tables_json   or []
        words_blocks  = words_blocks  or []
        line_blocks   = line_blocks   or []
        paragraph_blocks = paragraph_blocks or []

        # collect pages present in any layer
        pages = sorted(set(
            [b.page for b in words_blocks] +
            [b.page for b in line_blocks] +
            [b.page for b in paragraph_blocks] +
            [tl["page_index"] for tl in table_lines] +
            [t.get("doc_page", 0) for t in tables_json if t.get("type") == "table"]
        ))

        # index by page
        by_page_words = defaultdict(list)
        by_page_lines = defaultdict(list)
        by_page_paras = defaultdict(list)
        for b in words_blocks:      by_page_words[b.page].append(b)
        for b in line_blocks:       by_page_lines[b.page].append(b)
        for b in paragraph_blocks:  by_page_paras[b.page].append(b)

        table_colors = list(mcolors.TABLEAU_COLORS.values())

        for page in pages:
            fig, ax = plt.subplots(figsize=(10, 14))  # modest size to avoid RAM spikes
            ax.set_aspect("equal")

            # set limits first
            x0,y0,x1,y1 = PDFTextBuilder.__page_bbox_for_limits(page, by_page_words, by_page_lines, by_page_paras, table_lines, tables_json)
            pad = max(4, 0.02*(x1-x0))  # small padding
            ax.set_xlim(x0 - pad, x1 + pad)
            ax.set_ylim(y1 + pad, y0 - pad)  # flip Y by ordering

            # 1) Paragraphs (purple fill, label “P#”)
            for i, p in enumerate(by_page_paras.get(page, []), start=1):
                x0, y0, x1, y1 = p.bbox
                ax.add_patch(patches.Rectangle(
                    (x0, y0), p.w, p.h, edgecolor="#7e57c2", facecolor="#7e57c2", alpha=0.12, lw=0.8, zorder=4
                ))
                ax.add_patch(patches.Rectangle(
                    (x0, y0), p.w, p.h, edgecolor="#7e57c2", facecolor="none", lw=0.8, zorder=5
                ))
                ax.text(x0+2, y0+8, f"P{i}", color="#7e57c2", fontsize=7, va="bottom", zorder=6)

            # 2) Table grid lines from geometry (red)
            for l in [ln for ln in table_lines if ln["page_index"] == page]:
                x0, y0, x1, y1 = l["points"]
                ax.plot([x0, x1], [y0, y1], color="red", lw=0.6, alpha=0.7, zorder=7)

            # 3) Tables from JSON (per-cell colors + outer bbox + “T#”)
            t_i = 0
            cmap = plt.get_cmap("tab20")  # 20 distinct colors to cycle per table

            for tbl in tables_json:
                if tbl.get("type") != "table" or tbl.get("doc_page") != page:
                    continue

                table_color = table_colors[t_i % len(table_colors)]  # keep outer bbox color per table
                t_i += 1

                # color each cell differently by cycling the colormap
                cell_idx = 0
                for row in tbl.get("rows", []):
                    for c in row:
                        if "bbox" not in c:
                            continue
                        cx0, cy0, cx1, cy1 = c["bbox"]
                        cw, ch = (cx1 - cx0), (cy1 - cy0)

                        # pick a distinct-ish color for this cell
                        color = cmap(cell_idx % cmap.N)  # RGBA
                        cell_idx += 1

                        ax.add_patch(patches.Rectangle(
                            (cx0, cy0), cw, ch,
                            edgecolor=color, facecolor=color, alpha=0.22, lw=0.8, zorder=8
                        ))

                # outer table bbox + label with a highlighted border
                if "bbox" in tbl:
                    x0, y0, x1, y1 = tbl["bbox"]
                    tw, th = (x1 - x0), (y1 - y0)

                    # 1) soft halo for emphasis
                    ax.add_patch(patches.Rectangle(
                        (x0, y0), tw, th,
                        edgecolor=table_color, facecolor="none",
                        lw=6, alpha=0.18, zorder=9
                    ))

                    # 2) solid main outline
                    ax.add_patch(patches.Rectangle(
                        (x0, y0), tw, th,
                        edgecolor=table_color, facecolor="none",
                        lw=2.2, alpha=0.95, zorder=10
                    ))

                    # 3) subtle inner dashed stroke for contrast (optional)
                    ax.add_patch(patches.Rectangle(
                        (x0, y0), tw, th,
                        edgecolor="black", facecolor="none",
                        lw=0.8, linestyle=(0, (3, 2)), alpha=0.35, zorder=11
                    ))

                    # optional: corner ticks for readability on busy pages
                    tick = max(6, 0.06 * min(tw, th))
                    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                    for (cx, cy) in corners:
                        # horizontal tick
                        x2 = cx + (tick if cx == x0 else -tick)
                        ax.plot([cx, x2], [cy, cy], lw=1.2, color=table_color, alpha=0.9, zorder=12)
                        # vertical tick
                        y2 = cy + (tick if cy == y0 else -tick)
                        ax.plot([cx, cx], [cy, y2], lw=1.2, color=table_color, alpha=0.9, zorder=12)

                    # label
                    ax.text(x0 + 2, y0 + 10, f"T{t_i}", color=table_color, fontsize=9, va="bottom", zorder=13)

            # 4) Text lines (blue outline)
            for i, ln in enumerate(by_page_lines.get(page, []), start=1):
                x0, y0, x1, y1 = ln.bbox
                ax.add_patch(patches.Rectangle(
                    (x0, y0), ln.w, ln.h, edgecolor="blue", facecolor="none", lw=0.5, alpha=0.7, zorder=2
                ))
                # optional tiny id on the left
                # ax.text(x0+1, y0+min(ln.h, 8), f"L{i}", color="blue", fontsize=6, va="top", zorder=3)

            # 5) Words (light gray outline) – “span-like” feel
            for i, w in enumerate(by_page_words.get(page, []), start=1):
                x0, y0, x1, y1 = w.bbox
                ax.add_patch(patches.Rectangle(
                    (x0, y0), w.w, w.h, edgecolor="#000000", facecolor="none", lw=0.3, alpha=0.8, zorder=1
                ))

            plt.show()

    @staticmethod
    def __page_bbox_for_limits(
        page, by_page_words, by_page_lines, by_page_paras, table_lines, tables_json
    ):
        xs, ys = [], []
        for b in by_page_words.get(page, []):
            x0,y0,x1,y1 = b.bbox; xs += [x0,x1]; ys += [y0,y1]
        for b in by_page_lines.get(page, []):
            x0,y0,x1,y1 = b.bbox; xs += [x0,x1]; ys += [y0,y1]
        for b in by_page_paras.get(page, []):
            x0,y0,x1,y1 = b.bbox; xs += [x0,x1]; ys += [y0,y1]
        for l in (tl for tl in table_lines if tl.get("page_index")==page):
            x0,y0,x1,y1 = l["points"]; xs += [x0,x1]; ys += [y0,y1]
        for tbl in (t for t in tables_json if t.get("type")=="table" and t.get("doc_page")==page):
            if "bbox" in tbl:
                x0,y0,x1,y1 = tbl["bbox"]; xs += [x0,x1]; ys += [y0,y1]
            for row in tbl.get("rows", []):
                for c in row:
                    if "bbox" in c:
                        x0,y0,x1,y1 = c["bbox"]; xs += [x0,x1]; ys += [y0,y1]
        if not xs:
            return (0,0,1,1)  # fallback so we never get 0–1 by accident
        return (min(xs), min(ys), max(xs), max(ys))

    # ---------- Plain-text conversion for embeddings ----------
    @staticmethod
    def to_embedding_text(obj: Any) -> str:
        def table_to_lines(tbl: Dict[str,Any]) -> List[str]:
            lines=[]
            tid = tbl.get("table_id", 0); page = tbl.get("doc_page", 0)
            for row in tbl.get("rows", []):
                cells = sorted(row, key=lambda c: c["c"])
                parts = [c["text"].strip() for c in cells if c.get("text")]
                if parts: lines.append(f"[page={page} table={tid}] " + " | ".join(parts))
            return lines

        if isinstance(obj, list):
            parts=[]
            for it in obj:
                if it.get("type")=="table": parts.extend(table_to_lines(it))
                elif it.get("type")=="paragraph": parts.append(it.get("text","").strip())
            return "\n".join(p for p in parts if p)

        if isinstance(obj, dict):
            if obj.get("type")=="table": return "\n".join(table_to_lines(obj))
            if obj.get("type")=="paragraph": return obj.get("text","").strip()
            if "blocks" in obj: return PDFTextBuilder.to_embedding_text(obj["blocks"])

        return str(obj).strip()

    # ---------- Page extraction ----------
    @staticmethod
    def filter_page(obj: Any, page: int = 0) -> Any:
        def correct_page(block: dict):
            if "doc_page" in block:
                return block["doc_page"] == page
            return False
        
        if isinstance(obj, list):
                parts=[]
                for it in obj:
                    if correct_page(it):
                        parts.append(it)
                return parts

        if isinstance(obj, dict):
            if "blocks" in obj:
                parts = []
                for it in obj["blocks"]:
                    if correct_page(it):
                        parts.append(it)
                return {"size": len(parts), "blocks": parts}
            if correct_page(obj):
                return obj
        
        return None