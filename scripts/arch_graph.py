#!/usr/bin/env python3
"""Derive pyali's pipeline architecture from the source of ``process_fov`` and draw it.

A *stage* is one call that ``pyali.pipeline.process_fov`` makes into the package, in source
order. For each stage the tool records the function, its module, the ``if`` it sits under, what
it consumes and produces (array shapes come from the callee's numpydoc docstring), the files it
reads or writes, and, for a stage that only saves or plots, the names it takes in. That ordered
list is the whole representation; edges are drawn only into stages that compute something. It is
written as a Markdown table and drawn twice with straight, axis-aligned edges:

    docs/architecture/architecture.md               the stage table
    docs/architecture/architecture_vertical.svg     one column; data that skips stages runs in
                                                    labelled lanes on the right, nothing on edges
    docs/architecture/architecture_horizontal.svg   rows of five, left to right, later rows lower;
                                                    labels sit on the edges

Every coordinate is a function of the stage index, so the outputs are deterministic and the
renderer asserts that no box, label or edge overlaps anything it should not. ``--check``
regenerates everything in memory and exits 1 if the committed files differ. Stdlib only.
"""
import ast
import os
import re
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "src")
ENTRY = "pyali.pipeline.process_fov"
OUT_DIR = os.path.join(REPO, "docs", "architecture")
DATA_EXT = (".bin", ".mat", ".json", ".png", ".html", ".csv", ".txt")
MONO = 6.7            # px per character at 11 px monospace


# ------------------------------------------------------------------ source model
class Package:
    """Modules, import aliases, top-level functions and docstrings of ``src/pyali``."""

    def __init__(self, src, pkg="pyali"):
        self.mods, self.alias, self.local, self.fn, self.fn_mod, self.doc = {}, {}, {}, {}, {}, {}
        for dp, dns, fns in os.walk(os.path.join(src, pkg)):
            dns[:] = sorted(d for d in dns if not d.startswith((".", "__")))
            for f in sorted(fns):
                if f.endswith(".py"):
                    rel = os.path.relpath(os.path.join(dp, f), src)[:-3].replace(os.sep, ".")
                    self.mods[rel[:-9] if rel.endswith(".__init__") else rel] = os.path.join(dp, f)
        for m in self.mods:
            self._parse(m)

    def _parse(self, m):
        path = self.mods[m]
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        alias, local = {}, {}
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    alias[a.asname or a.name.split(".")[0]] = a.name
            elif isinstance(n, ast.ImportFrom):
                parts = m.split(".") + (["__init__"] if path.endswith("__init__.py") else [])
                base = ".".join(parts[:-n.level] + ([n.module] if n.module else [])) if n.level \
                    else n.module
                for a in n.names:
                    alias[a.asname or a.name] = f"{base}.{a.name}"
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fq = f"{m}.{n.name}"
                local[n.name] = fq
                self.fn[fq], self.fn_mod[fq], self.doc[fq] = n, m, ast.get_docstring(n) or ""
        self.alias[m], self.local[m] = alias, local

    def callee(self, func, m):
        """Fully qualified name of a call target defined in the package, or None."""
        chain = []
        while isinstance(func, ast.Attribute):
            chain.append(func.attr)
            func = func.value
        if not isinstance(func, ast.Name):
            return None
        head = self.local[m].get(func.id) if not chain else None
        head = head or self.alias[m].get(func.id)
        fq = ".".join([head] + chain[::-1]) if head else None
        return fq if fq in self.fn else None

    def files(self, fq, depth=0, seen=None):
        """Data-file literals in a function body and, two levels down, in the package functions it
        calls, keyed 'read'/'write' by the name of the function that holds the literal."""
        seen = seen if seen is not None else set()
        out = defaultdict(set)
        if fq in seen or depth > 2:
            return out
        seen.add(fq)
        node, m = self.fn[fq], self.fn_mod[fq]
        kind = "read" if re.search(r"read|load|open", fq.rsplit(".", 1)[1]) else "write"
        body = node.body[1:] if node.body and isinstance(node.body[0], ast.Expr) \
            and isinstance(node.body[0].value, ast.Constant) else node.body      # skip the docstring
        for n in [d for d in node.args.defaults + node.args.kw_defaults if d] + body:
            for c in ast.walk(n):
                if isinstance(c, ast.Constant) and isinstance(c.value, str) and c.value.endswith(DATA_EXT):
                    out[kind].add(os.path.basename(c.value))
                elif isinstance(c, ast.Call):
                    sub = self.callee(c.func, m)
                    if sub:
                        for k, v in self.files(sub, depth + 1, seen).items():
                            out[k] |= v
        return out

    def shapes(self, fq):
        """{name: '(H, W)'} and a single return shape, parsed from the numpydoc docstring."""
        doc, named, single = self.doc[fq], {}, None
        for mt in re.finditer(r"``(\w+)``\s*:\s*\(([^)]*)\)", doc):
            named.setdefault(mt.group(1), f"({mt.group(2).strip()})")
        for mt in re.finditer(r"^\s*(\w+)\s*:\s*\(([^)]*)\)", doc, re.M):
            named.setdefault(mt.group(1), f"({mt.group(2).strip()})")
        mt = re.search(r"Returns\s*\n\s*-+\s*\n\s*\(([^)]*)\)\s*:\s*(.*)", doc)
        if mt:
            names = [s.strip() for s in mt.group(1).split(",") if s.strip()]
            shp = re.findall(r"\(([^)]*)\)", mt.group(2))
            shp = shp * len(names) if "tuple of" in mt.group(2) and len(shp) == 1 else shp
            if len(shp) == len(names):
                for nm, sh in zip(names, shp):
                    named.setdefault(nm, f"({sh.strip()})")
        else:
            mt = re.search(r"Returns\s*\n\s*-+\s*\n\s*\(([^)]*)\)\s*(?:ndarray|array)", doc) \
                or re.search(r"``\[([^\]]*)\]``", doc.split("\n", 1)[0])
            single = f"({mt.group(1).strip()})" if mt else None
        if not named and not single:                        # a dispatcher: the shape of what it returns
            for r in ast.walk(self.fn[fq]):
                if isinstance(r, ast.Return) and isinstance(r.value, ast.Call):
                    sub = self.callee(r.value.func, self.fn_mod[fq])
                    if sub:
                        single = self.shapes(sub)[1]
                        break
        return named, single


def stages(pk):
    """The calls ``process_fov`` makes into the package, in source order, with their dataflow."""
    node, m = pk.fn[ENTRY], pk.fn_mod[ENTRY]
    parent = {c: p for p in ast.walk(node) for c in ast.iter_child_nodes(p)}
    calls = [s for s in ast.walk(node) if isinstance(s, (ast.Assign, ast.Expr))
             and isinstance(s.value, ast.Call)]
    out, producer, shape, count = [], {}, {}, defaultdict(int)
    for s in sorted(calls, key=lambda s: (s.lineno, s.col_offset)):
        call = s.value
        fq = pk.callee(call.func, m)
        if not fq:
            continue
        outputs = [e.id for t in getattr(s, "targets", []) for e in
                   (t.elts if isinstance(t, ast.Tuple) else [t]) if isinstance(e, ast.Name)]
        args = call.args + [k.value for k in call.keywords]
        inputs = sorted({n.id for a in args for n in ast.walk(a) if isinstance(n, ast.Name)})
        if not outputs and call.args and isinstance(call.args[0], ast.Name) \
                and call.args[0].id in producer:                  # mutates its first argument
            outputs = [call.args[0].id]
        inplace = [o for o in outputs if o in inputs]
        cond, child, p = None, s, parent.get(s)
        while p is not None and not isinstance(p, ast.FunctionDef):
            if isinstance(p, ast.If):
                cond = ast.unparse(p.test)
                cond = cond if child in p.body else f"not ({cond})"
                break
            child, p = p, parent.get(p)
        cond = re.sub(r"getattr\(p, '(\w+)'(?:, [^)]*)?\)", r"p.\1", cond) if cond else None
        named, single = pk.shapes(fq)
        for o in outputs:
            sh = named.get(o) or (single if len(outputs) == 1 else None)
            if sh:
                shape[o] = sh
        movie = next((ast.unparse(a) for a in args if isinstance(a, ast.Subscript)
                      and isinstance(a.value, ast.Name) and a.value.id == "movie"), None)
        files = pk.files(fq)
        for a in args:                                            # literals passed at the call site
            for c in ast.walk(a):
                if isinstance(c, ast.Constant) and isinstance(c.value, str) and c.value.endswith(DATA_EXT):
                    files["read" if "read" in fq.rsplit(".", 1)[1] else "write"].add(c.value)
        count[fq] += 1
        st = dict(id=f"{fq}#{count[fq]}", fq=fq, module=pk.fn_mod[fq], line=s.lineno,
                  outputs=outputs, inputs=inputs, inplace=inplace, cond=cond, movie_expr=movie,
                  reads=sorted(files["read"]), writes=sorted(files["write"]), sink=not outputs,
                  deps=defaultdict(list))
        for v in inputs:
            if v in producer:
                st["deps"][producer[v]].append(v)
        # a sink (a stage that produces nothing: it saves, writes or plots) lists what it takes in
        # its own box, so no edge is drawn into it; save_mat_v73 names the datasets it stores
        st["stores"] = sorted(k.arg for k in call.keywords if k.arg) if "save_mat" in fq else \
            sorted({v for vs in st["deps"].values() for v in vs}) if st["sink"] else []
        out.append(st)
        for o in outputs:
            producer[o] = st["id"]
    return out, shape


# ------------------------------------------------------------------ text of a stage box
def _wrap(items, prefix, width):
    lines, cur = [], prefix
    for it in items:
        piece = it if cur == prefix else ", " + it
        if len(cur) + len(piece) > width and cur != prefix:
            lines.append(cur + ",")
            cur = " " * len(prefix) + it
        else:
            cur += piece
    return lines + [cur]


def _split(text, width):
    """Break a label at ', ' or ' ' into lines of at most ``width`` characters (at most three)."""
    lines, cur = [], ""
    for w in text.replace(", ", ",\0").split("\0"):           # first between variables ...
        if cur and len(cur) + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur += w
    lines.append(cur)
    out = []
    for ln in lines:                                            # ... then at spaces if still too long
        while len(ln) > width and " " in ln[:width]:
            cut = ln[:width].rfind(" ")
            out.append(ln[:cut])
            ln = ln[cut + 1:]
        out.append(ln)
    return out[:3]


def box_lines(st, shape, width=46):
    """[(style, text)] shown inside a stage box: title, module/condition, the movie line, what the
    stage produces, the files it reads/writes, and the arrays it stores."""
    fn, mod = st["fq"].rsplit(".", 1)[1], st["module"].rsplit(".", 1)[1] + ".py"
    lines = [("title", fn), ("sub", mod + (f"   if {st['cond']}" if st["cond"] else ""))]
    if "movie" in st["inputs"] or "movie" in st["outputs"]:
        sh = shape.get("movie", "")
        if "movie" in st["inplace"]:
            txt = f"movie {sh} in place"
        elif "movie" in st["outputs"]:
            txt = f"→ movie {sh}"
        else:
            txt = f"{st['movie_expr'] or 'movie'} {sh}"
        lines.append(("body", txt))
    prod = [o for o in st["outputs"] if o != "movie" and not o.startswith("_")]
    if prod:
        items = [f"{o} {shape.get(o, '')}".strip() + (" in place" if o in st["inplace"] else "")
                 for o in prod]
        lines += [("body", t) for t in _wrap(items, "→ ", width)]
    for kind, names in (("reads", st["reads"]), ("writes", st["writes"])):
        if names:
            txt = ", ".join(names) if len(names) <= 2 else \
                f"{len(names)} files ({', '.join(sorted({n.rsplit('.', 1)[-1] for n in names}))})"
            lines.append(("file", f"{kind} {txt}"))
    if st["stores"]:
        lines += [("body", t) for t in _wrap(st["stores"], "← ", width)]
    return lines


# ------------------------------------------------------------------ geometry
class Canvas:
    """Collects SVG elements plus the rectangles and segments used by the overlap check."""

    def __init__(self):
        self.el, self.rects, self.segs = [], [], []

    def box(self, x, y, w, h, lines, dashed):
        self.rects.append((x, y, x + w, y + h, "box"))
        dash = ' stroke-dasharray="6 3"' if dashed else ""
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="#fff" '
                       f'stroke="#1f2933" stroke-width="1.2"{dash}/>')
        ty = y + 15
        for style, text in lines:
            cls = {"title": 'font-weight="bold"', "sub": 'fill="#5d6670" font-size="10"',
                   "file": 'fill="#0f766e"', "body": ""}[style]
            self.el.append(f'<text x="{x + 9}" y="{ty}" {cls}>{_esc(text)}</text>')
            ty += 14

    def label(self, x, y, text, anchor="start"):
        """Text with a white halo; (x, y) is the baseline start (or end for anchor='end')."""
        w = len(text) * MONO + 6
        x0 = x - 3 if anchor == "start" else x - w + 3
        self.rects.append((x0, y - 11, x0 + w, y + 3, "label"))
        self.el.append(f'<rect x="{x0}" y="{y - 11}" width="{w}" height="14" fill="#fff"/>')
        self.el.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="#374151">{_esc(text)}</text>')

    def path(self, pts, arrow=True, dotted=False, dot=False):
        for a, b in zip(pts, pts[1:]):
            self.segs.append((a, b))
        d = "M " + " L ".join(f"{x} {y}" for x, y in pts)
        extra = (' marker-end="url(#arrow)"' if arrow else "") + (' stroke-dasharray="2 3"' if dotted else "")
        self.el.append(f'<path d="{d}" fill="none" stroke="#1f2933" stroke-width="1.2"{extra}/>')
        if dot:
            self.el.append(f'<circle cx="{pts[-1][0]}" cy="{pts[-1][1]}" r="2.6" fill="#1f2933"/>')

    def check(self):
        """No two boxes/labels overlap; no segment crosses the inside of a box or label."""
        for i, a in enumerate(self.rects):
            for b in self.rects[i + 1:]:
                assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1], \
                    f"overlap: {a} vs {b}"
        for (x1, y1), (x2, y2) in self.segs:
            for (rx1, ry1, rx2, ry2, kind) in self.rects:
                lo_x, hi_x, lo_y, hi_y = min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)
                if hi_x <= rx1 + 0.5 or lo_x >= rx2 - 0.5 or hi_y <= ry1 + 0.5 or lo_y >= ry2 - 0.5:
                    continue
                assert False, f"segment ({x1},{y1})-({x2},{y2}) crosses {kind} {(rx1, ry1, rx2, ry2)}"

    def svg(self, w, h, title):
        self.check()
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
                f'role="img" aria-label="{_esc(title)}" font-family="ui-monospace, Menlo, Consolas, monospace" '
                f'font-size="11" fill="#1f2933">\n<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
                f'markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" '
                f'fill="#1f2933"/></marker></defs>\n<rect width="{w}" height="{h}" fill="#fff"/>\n')
        return head + "\n".join(self.el) + "\n</svg>\n"


def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _edges(sts):
    """(src index, dst index, [vars]) for every dataflow dependency."""
    idx = {st["id"]: i for i, st in enumerate(sts)}
    return [(idx[src], j, vars_) for j, st in enumerate(sts) if not st["sink"]
            for src, vars_ in sorted(st["deps"].items())]


# ------------------------------------------------------------------ figure 1: one column + lanes
def svg_vertical(sts, shape):
    BW, X0, GAP, LW = 330, 16, 30, 26
    edges = _edges(sts)
    lanes = []                                                  # variables that skip a stage
    for i, j, vars_ in edges:
        if j != i + 1:
            lanes += [v for v in vars_ if v not in lanes]
    lanes.sort(key=lambda v: next(k for k, st in enumerate(sts) if v in st["outputs"]))
    conns = defaultdict(list)                                   # stage -> [(lane, 'in'|'out')]
    for i, j, vars_ in edges:
        if j != i + 1:
            for v in vars_:
                conns[j].append((v, "in"))
    for k, st in enumerate(sts):
        for v in st["outputs"]:
            if v in lanes and any(e[0] == k and e[1] != k + 1 and v in e[2] for e in edges):
                conns[k].append((v, "out"))
            elif v in lanes and v in st["inplace"]:
                conns[k].append((v, "out"))
    top = 14 + int(max((len(v) for v in lanes), default=0) * MONO) + 10
    cv, y, ys, hs = Canvas(), top, [], []
    for k, st in enumerate(sts):
        lines = box_lines(st, shape)
        h = max(18 + 14 * len(lines), 20 + 13 * len(conns[k]))
        cv.box(X0, y, BW, h, lines, bool(st["cond"]))
        ys.append(y)
        hs.append(h)
        y += h + GAP
    lane_x = {v: X0 + BW + 38 + i * LW for i, v in enumerate(lanes)}
    lane_y = defaultdict(list)
    for k, st in enumerate(sts):
        cs = sorted(conns[k], key=lambda c: lanes.index(c[0]))
        for n, (v, kind) in enumerate(cs):
            cy = ys[k] + 14 + 13 * n
            lane_y[v].append(cy)
            if kind == "out":
                cv.path([(X0 + BW, cy), (lane_x[v], cy)], arrow=False, dot=True)
            else:
                cv.path([(lane_x[v], cy), (X0 + BW, cy)])
    for v in lanes:
        x = lane_x[v]
        cv.el.append(f'<text x="{x + 4}" y="{min(lane_y[v]) - 8}" transform="rotate(-90 {x + 4} {min(lane_y[v]) - 8})" '
                     f'fill="#5d6670">{_esc(v)}</text>')
        cv.el.append(f'<path d="M {x} {min(lane_y[v])} L {x} {max(lane_y[v])}" stroke="#9aa5b1" stroke-width="1.2"/>')
    direct = {(i, j) for i, j, _ in edges if j == i + 1}
    for k in range(len(sts) - 1):
        cv.path([(X0 + BW // 2, ys[k] + hs[k]), (X0 + BW // 2, ys[k + 1])], dotted=(k, k + 1) not in direct)
    return cv.svg(X0 + BW + 38 + len(lanes) * LW + 16, y - GAP + 16, "pyali process_fov, stages top to bottom")


# ------------------------------------------------------------------ figure 2: rows of five
def svg_horizontal(sts, shape, cols=5):
    BW, GX, M, CH = 236, 150, 16, 30
    edges = _edges(sts)
    lines = [box_lines(st, shape, width=33) for st in sts]
    nrows = (len(sts) + cols - 1) // cols
    row_h = [18 + 14 * max(len(lines[i]) for i in range(r * cols, min(len(sts), (r + 1) * cols)))
             for r in range(nrows)]
    row, col = lambda i: i // cols, lambda i: i % cols
    # channel bookkeeping: band r sits above row r; each edge owns one channel per band it uses
    band, trunk, routes = defaultdict(list), [], []
    for i, j, vars_ in edges:
        if j == i + 1 and row(i) == row(j):
            routes.append(("direct", i, j, vars_))
        elif row(i) == row(j):
            routes.append(("same", i, j, vars_, len(band[row(i) + 1])))
            band[row(i) + 1].append((i, j))
        elif row(j) == row(i) + 1 and j != i + 1:
            routes.append(("next", i, j, vars_, len(band[row(j)])))
            band[row(j)].append((i, j))
        else:                                                   # via the trunk on the right
            routes.append(("trunk", i, j, vars_, len(band[row(i) + 1]), len(band[row(j)]), len(trunk)))
            band[row(i) + 1].append((i, j))
            band[row(j)].append((i, j))
            trunk.append((i, j))
    for i in range(cols - 1, len(sts) - 1, cols):              # end of a row -> start of the next
        vars_ = next((v for a, b, v in edges if (a, b) == (i, i + 1)), [])
        routes.append(("wrap", i, i + 1, vars_, len(band[row(i + 1)]), len(trunk)))
        band[row(i + 1)].append((i, i + 1))
        trunk.append((i, i + 1))
    band_h = {r: 24 + CH * len(band[r]) for r in range(nrows + 1)}
    row_y, y = [], M
    for r in range(nrows):
        y += band_h[r]
        row_y.append(y)
        y += row_h[r]
    width = M + cols * BW + (cols - 1) * GX + 24 + 14 * len(trunk) + M
    cv = Canvas()
    bx = lambda i: M + col(i) * (BW + GX)
    cx = lambda i: bx(i) + BW // 2
    by = lambda i: row_y[row(i)]
    bot = lambda i: row_y[row(i)] + row_h[row(i)]
    chan = lambda r, n: row_y[r] - band_h[r] + 16 + CH * n       # channel n of the band above row r
    tx = lambda n: M + cols * BW + (cols - 1) * GX + 18 + 14 * n
    for i, st in enumerate(sts):
        cv.box(bx(i), by(i), BW, row_h[row(i)], lines[i], bool(st["cond"]))
    for rt in routes:
        kind, i, j, vars_ = rt[:4]
        text = ", ".join(f"{v} {shape.get(v, '')}".strip() for v in vars_)
        text = text if len(text) <= 36 else text[:33] + "..."
        ym = by(i) + row_h[row(i)] // 2
        # a channel label goes right of the drop unless it would run into the trunk area
        side = ("start", cx(i) + 8) if cx(i) + 8 + len(text) * MONO + 6 < tx(0) - 6 else ("end", cx(i) - 8)
        if kind == "direct":                                    # label stacked above the arrow
            cv.path([(bx(i) + BW, ym), (bx(j), ym)])
            parts = _split(text, int((GX - 12) / MONO))
            for n, part in enumerate(parts):
                cv.label(bx(i) + BW + 6, ym - 6 - 14 * (len(parts) - 1 - n), part)
        elif kind == "same":
            yc = chan(row(i) + 1, rt[4]) if row(i) + 1 < nrows else bot(i) + 16 + CH * rt[4]
            cv.path([(cx(i), bot(i)), (cx(i), yc), (cx(j), yc), (cx(j), bot(j))])
            cv.label(side[1], yc - 4, text, side[0])
        elif kind == "next":
            yc = chan(row(j), rt[4])
            cv.path([(cx(i), bot(i)), (cx(i), yc), (cx(j), yc), (cx(j), by(j))])
            cv.label(side[1], yc - 4, text, side[0])
        elif kind == "trunk":
            y1, y2, x = chan(row(i) + 1, rt[4]), chan(row(j), rt[5]), tx(rt[6])
            cv.path([(cx(i), bot(i)), (cx(i), y1), (x, y1), (x, y2), (cx(j), y2), (cx(j), by(j))])
            cv.label(side[1], y1 - 4, text, side[0])
        else:                                                   # wrap
            yc, x = chan(row(j), rt[4]), tx(rt[5])
            cv.path([(bx(i) + BW, ym), (x, ym), (x, yc), (cx(j), yc), (cx(j), by(j))])
            if text:
                cv.label(cx(j) + 8, yc - 4, text)
    direct = {(i, j) for i, j, _ in edges if j == i + 1}
    for k in range(len(sts) - 1):                               # order-only arrows where no data flows
        if (k, k + 1) not in direct and row(k) == row(k + 1):
            ym = by(k) + row_h[row(k)] // 2
            cv.path([(bx(k) + BW, ym), (bx(k + 1), ym)], dotted=True)
    height = row_y[-1] + row_h[-1] + band_h.get(nrows, 24) + M
    return cv.svg(width, height, "pyali process_fov, stages left to right then down")


# ------------------------------------------------------------------ table + driver
def render_md(sts, shape):
    rows = ["# pyali architecture", "",
            "The calls `process_fov` makes into the package, in order. Generated by "
            "`scripts/arch_graph.py`; `--check` fails when this no longer matches the source.", "",
            "| # | stage | when | movie | produces | files | stores |", "|---|---|---|---|---|---|---|"]
    for n, st in enumerate(sts, 1):
        movie = ("in place" if "movie" in st["inplace"] else "produced" if "movie" in st["outputs"]
                 else st["movie_expr"] or ("consumed" if "movie" in st["inputs"] else ""))
        prod = ", ".join(f"{o} {shape.get(o, '')}".strip() for o in st["outputs"]
                         if o != "movie" and not o.startswith("_"))
        files = "; ".join(f"{k} {', '.join(v)}" for k, v in (("reads", st["reads"]), ("writes", st["writes"])) if v)
        rows.append(f"| {n} | `{st['fq'][len('pyali.'):]}` | {st['cond'] or ''} | {movie} | {prod} | {files} | "
                    f"{', '.join(st['stores'])} |")
    return "\n".join(rows) + "\n"


def build():
    pk = Package(SRC)
    sts, shape = stages(pk)
    return {"architecture.md": render_md(sts, shape),
            "architecture_vertical.svg": svg_vertical(sts, shape),
            "architecture_horizontal.svg": svg_horizontal(sts, shape)}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out = build()
    if "--check" in argv:
        stale = [n for n, s in out.items()
                 if not os.path.exists(os.path.join(OUT_DIR, n)) or open(os.path.join(OUT_DIR, n)).read() != s]
        print("[arch] " + ("STALE: " + ", ".join(stale) + " -- run scripts/arch_graph.py" if stale
                           else f"up to date: {OUT_DIR}"))
        return 1 if stale else 0
    os.makedirs(OUT_DIR, exist_ok=True)
    for n, s in out.items():
        with open(os.path.join(OUT_DIR, n), "w") as f:
            f.write(s)
    print(f"[arch] wrote {', '.join(out)} to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
