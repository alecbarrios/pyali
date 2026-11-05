#!/usr/bin/env python3
"""Derive pyali's architecture from its source and render it -- stdlib only, no graphviz.

Walks every ``*.py`` under ``pyali/``, ``scripts/`` and ``snr_analysis/`` with the
standard-library ``ast`` module and records, deterministically:

* every module, its definitions (line ranges, first docstring line), its imports -- including
  the lazy in-function imports a top-level grep misses -- and the resolved function -> function
  call edges;
* the **core** set: everything reachable from ``pyali.pipeline.process_fov`` over *non-lazy*
  calls. That is the extraction algorithm proper; figures, metadata and metrics hang off lazy
  imports and so are pruned without a hand-kept list;
* the def-use **dataflow** through ``process_fov``: which stage produces which arrays and which
  stage consumes them, with shapes read from the numpydoc ``Returns`` blocks where present;
* the **QC pass** (the A/B benchmark harness under ``snr_analysis/``) as script -> file ->
  script edges. Library primitives that take a path
  (``open``, ``h5py.File``, ``savefig``, pandas ``read_*``/``to_*``, ``np.fromfile``) are the
  only seeds; any repo function that passes one of its parameters into a seed inherits the role
  at a fixpoint, so ``load_v73``, ``save_mat_v73`` and the A/B harness's path resolver are
  all derived, not listed. File-name literals are resolved through
  module constants, imported constants, local assignments, ``for``-loop unpacking and default
  arguments;
* which ``Params`` field each function reads.

Outputs (default ``docs/architecture/``; committed; no timestamps so they are byte-stable):

    arch.json            the whole graph
    core_dataflow.svg    figure 1 -- voltage extraction, standalone
    qc_dataflow.svg      figure 2 -- QC checks, standalone
    core_dataflow.mmd    the same two graphs as Mermaid text (GitHub renders these)
    qc_dataflow.mmd
    architecture.html    self-contained page: both figures, per-module call trees, Params map

    python scripts/arch_graph.py            # regenerate docs/architecture/
    python scripts/arch_graph.py --check    # exit 1 if the committed outputs are stale
"""
import argparse
import ast
import html
import json
import os
import re
import sys
from collections import defaultdict, deque

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

SCAN_DIRS = ("src/pyali", "scripts", "snr_analysis")
SKIP_DIRS = {"bin", "lib", "include", "share", "site-packages", "__pycache__"}
ENTRY = "pyali.pipeline.process_fov"
CORE_ORDER = ("pyali.pipeline", "pyali.io", "pyali.preprocess", "pyali.segmentation",
              "pyali.extract", "pyali.utils", "pyali.params")
QC_SCRIPTS = ("snr_analysis.snr.snr_compare",)
DATA_EXT = (".bin", ".csv", ".parquet", ".mat", ".json", ".jsonl", ".png", ".html", ".txt")
SEED_WRITERS = {"savefig", "to_csv", "to_parquet", "to_json"}
SEED_READERS = {"read_csv", "read_parquet", "read_json", "fromfile", "memmap", "loadtxt"}
SEED_MODAL = {"open", "File"}                 # the mode argument decides: r => read, w/a/x => write
PER_FOV_NAMES = {"fov", "fov_dir", "dir_name", "read_root"}
PARAM_RECEIVERS = ("p", "params")
OUTPUTS = ("arch.json", "core_dataflow.svg", "qc_dataflow.svg", "core_dataflow.mmd",
           "qc_dataflow.mmd", "architecture.html")


def _first_line(doc):
    doc = (doc or "").strip()
    return doc.splitlines()[0].strip() if doc else ""


def _names(node):
    return sorted({n.id for n in ast.walk(node) if isinstance(n, ast.Name)})


def _basename(lit):
    return lit.rsplit("/", 1)[-1]


# =========================================================================================
# Extraction
# =========================================================================================
class Repo:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.mods = {}          # dotted -> path
        self.bare = {}          # bare script name -> dotted (scripts import each other bare)
        self.info = {}          # dotted -> serialisable module record
        self.trees = {}
        self.alias = {}         # dotted -> {name: (target, lazy)}
        self.consts = {}        # dotted -> {NAME: [literal]}
        self.local = {}         # dotted -> {name: fq} for top-level defs
        self.fn = {}            # fq -> FunctionDef
        self.fn_module = {}     # fq -> dotted
        self.params_of = {}     # fq -> [param names]
        self.defaults_of = {}   # fq -> {param: default expr}
        self.docs = {}          # fq -> full docstring
        self.classes = set()
        self.call_edges = []
        self.import_edges = []
        self.params_reads = {}  # fq -> [field]
        self.param_fields = []  # Params dataclass fields in declaration order
        self.roles = {}
        self.io_edges = []
        self.core = set()

    # ---------------------------------------------------------------- discovery
    def module_name(self, path):
        rel = os.path.relpath(path, self.root)[:-3].replace(os.sep, ".")
        rel = rel[4:] if rel.startswith("src.") else rel          # src layout: src/pyali -> pyali
        return rel[:-9] if rel.endswith(".__init__") else rel

    def discover(self):
        for d in SCAN_DIRS:
            base = os.path.join(self.root, d)
            if not os.path.isdir(base):
                continue
            for dp, dns, fns in os.walk(base):
                dns[:] = sorted(x for x in dns if x not in SKIP_DIRS and not x.startswith("."))
                for fn in sorted(fns):
                    if fn.endswith(".py"):
                        p = os.path.join(dp, fn)
                        if os.path.abspath(p) == os.path.abspath(__file__):
                            continue                      # the extractor is not part of the architecture
                        self.mods[self.module_name(p)] = p
        for m in self.mods:
            if m.startswith(("scripts.", "snr_analysis.")):
                self.bare.setdefault(m.rsplit(".", 1)[-1], m)

    def group_of(self, m):
        if m.startswith("pyali.registration"):
            return "registration"
        if m.startswith("pyali"):
            return "core"
        return m.split(".")[0]

    def _module_of(self, target):
        """Longest dotted prefix of ``target`` that is a parsed module, or None."""
        t = target
        while t and t not in self.mods:
            t = t.rpartition(".")[0]
        return t or None

    # ---------------------------------------------------------------- parsing
    def _resolve_rel(self, mod, level, name):
        parts = mod.split(".")
        if self.mods[mod].endswith("__init__.py"):
            parts = parts + ["__init__"]
        base = parts[:-level] if level else parts
        return ".".join(base + ([name] if name else []))

    def parse_all(self):
        for m in sorted(self.mods):
            self._parse(m)
        pt = self.trees.get("pyali.params")
        if pt is not None:
            for n in pt.body:
                if isinstance(n, ast.ClassDef) and n.name == "Params":
                    self.param_fields = [s.target.id for s in n.body
                                         if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)]
        for m in sorted(self.mods):
            self._resolve_module_calls(m)

    def _parse(self, m):
        path = self.mods[m]
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src, filename=path)
        self.trees[m] = tree
        rec = dict(path=os.path.relpath(path, self.root), group=self.group_of(m),
                   doc=_first_line(ast.get_docstring(tree)), loc=src.count("\n") + 1,
                   defs=[], imports=[])
        alias = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                lazy = node.col_offset > 0
                if isinstance(node, ast.Import):
                    for a in node.names:
                        tgt = self.bare.get(a.name, a.name)
                        alias[a.asname or a.name.split(".")[0]] = (tgt, lazy)
                        rec["imports"].append(dict(target=tgt, lazy=lazy, line=node.lineno))
                else:
                    if node.level:
                        base = self._resolve_rel(m, node.level, node.module)
                    else:
                        base = self.bare.get(node.module, node.module)
                    for a in node.names:
                        tgt = f"{base}.{a.name}"
                        alias[a.asname or a.name] = (tgt, lazy)
                        rec["imports"].append(dict(target=tgt, lazy=lazy, line=node.lineno))
        self.alias[m] = alias
        # module-level string constants (NAME = "x.csv", REQUIRED = (RESULT, META))
        consts = {}
        for n in tree.body:
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                vals = []
                for c in ast.walk(n.value):
                    if isinstance(c, ast.Constant) and isinstance(c.value, str):
                        vals.append(c.value)
                    elif isinstance(c, ast.Name) and c.id in consts:
                        vals.extend(consts[c.id])
                if vals:
                    consts[n.targets[0].id] = vals
        self.consts[m] = consts
        # definitions
        local = {}

        def collect(container, prefix):
            for n in container.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    q = prefix + n.name
                    fq = f"{m}.{q}"
                    kind = "class" if isinstance(n, ast.ClassDef) else "function"
                    rec["defs"].append(dict(name=q, kind=kind, line=n.lineno, end=n.end_lineno,
                                            doc=_first_line(ast.get_docstring(n))))
                    if not prefix:
                        local[n.name] = fq
                    if kind == "class":
                        self.classes.add(fq)
                        collect(n, q + ".")
                    else:
                        self.fn[fq] = n
                        self.fn_module[fq] = m
                        self.docs[fq] = ast.get_docstring(n) or ""
                        a = n.args
                        names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
                        self.params_of[fq] = names
                        d = {}
                        pos = a.posonlyargs + a.args
                        for p, dv in zip(pos[len(pos) - len(a.defaults):], a.defaults):
                            d[p.arg] = dv
                        for p, dv in zip(a.kwonlyargs, a.kw_defaults):
                            if dv is not None:
                                d[p.arg] = dv
                        self.defaults_of[fq] = d
        collect(tree, "")
        self.local[m] = local
        self.info[m] = rec
        for imp in rec["imports"]:
            tm = self._module_of(imp["target"])
            if tm and tm != m:
                self.import_edges.append(dict(src=m, dst=tm, lazy=imp["lazy"], line=imp["line"],
                                              name=imp["target"]))

    def resolve_callee(self, func, m):
        """(fq, lazy) for a call target, or (None, False). Internal targets only."""
        local, alias = self.local[m], self.alias[m]
        if isinstance(func, ast.Name):
            if func.id in local:
                return local[func.id], False
            if func.id in alias:
                return alias[func.id]
            return None, False
        if isinstance(func, ast.Attribute):
            chain, cur = [], func
            while isinstance(cur, ast.Attribute):
                chain.append(cur.attr)
                cur = cur.value
            chain.reverse()
            if isinstance(cur, ast.Name):
                if cur.id in alias:
                    tgt, lazy = alias[cur.id]
                    return tgt + "." + ".".join(chain), lazy
                if cur.id in PARAM_RECEIVERS and f"pyali.params.Params.{chain[0]}" in self.fn:
                    return f"pyali.params.Params.{chain[0]}", False
        return None, False

    def _resolve_module_calls(self, m):
        seen = set()
        for fq in sorted(f for f, mm in self.fn_module.items() if mm == m):
            node = self.fn[fq]
            reads = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    tgt, lazy = self.resolve_callee(sub.func, m)
                    if tgt and self._module_of(tgt):
                        key = (fq, tgt)
                        if key not in seen:
                            seen.add(key)
                            self.call_edges.append(dict(
                                src=fq, dst=tgt, line=sub.lineno, lazy=lazy,
                                src_module=m, dst_module=self._module_of(tgt),
                                defined=tgt in self.fn or tgt in self.classes))
                    if (isinstance(sub.func, ast.Name) and sub.func.id == "getattr" and sub.args
                            and isinstance(sub.args[0], ast.Name) and sub.args[0].id in PARAM_RECEIVERS
                            and len(sub.args) > 1 and isinstance(sub.args[1], ast.Constant)):
                        reads.add(str(sub.args[1].value))
                elif (isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)
                      and sub.value.id in PARAM_RECEIVERS):
                    reads.add(sub.attr)
            self.params_reads[fq] = sorted(f for f in reads if f in self.param_fields)

    # ---------------------------------------------------------------- reachability
    def compute_core(self):
        adj = defaultdict(set)
        for e in self.call_edges:
            if not e["lazy"]:
                adj[e["src"]].add(e["dst"])
        seen, q = {ENTRY}, deque([ENTRY])
        while q:
            u = q.popleft()
            for v in sorted(adj[u]):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        self.core = seen

    # ---------------------------------------------------------------- file I/O roles
    @staticmethod
    def _bind_for(target, it, bind):
        if isinstance(it, ast.Name) and bind.get(it.id):
            it = bind[it.id][-1]
        if not isinstance(it, (ast.Tuple, ast.List)):
            return
        for e in it.elts:
            if isinstance(target, ast.Name):
                bind[target.id].append(e)
            elif isinstance(target, ast.Tuple) and isinstance(e, (ast.Tuple, ast.List)) \
                    and len(e.elts) == len(target.elts):
                for t, v in zip(target.elts, e.elts):
                    if isinstance(t, ast.Name):
                        bind[t.id].append(v)

    def _bindings(self, node):
        bind = defaultdict(list)
        for n in ast.walk(node):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        bind[t.id].append(n.value)
                    elif isinstance(t, ast.Tuple) and isinstance(n.value, ast.Tuple) \
                            and len(t.elts) == len(n.value.elts):
                        for a, b in zip(t.elts, n.value.elts):
                            if isinstance(a, ast.Name):
                                bind[a.id].append(b)
            elif isinstance(n, ast.For):
                self._bind_for(n.target, n.iter, bind)
        return bind

    def _const(self, m, name):
        vals = []
        if name in self.consts.get(m, {}):
            vals = self.consts[m][name]
        elif name in self.alias[m]:
            tgt = self.alias[m][name][0]
            tm, _, nm = tgt.rpartition(".")
            if tm in self.consts and nm in self.consts[tm]:
                vals = self.consts[tm][nm]
        return sorted({_basename(v) for v in vals if v.endswith(DATA_EXT)})

    def _literals(self, expr, m, bind, depth=0):
        """Data-file literals an expression can evaluate to (f-strings collapse to ``*.ext``)."""
        out = []

        def visit(n):
            if isinstance(n, ast.JoinedStr) and any(isinstance(v, ast.FormattedValue) for v in n.values):
                for v in n.values:
                    if isinstance(v, ast.Constant) and isinstance(v.value, str):
                        ext = next((e for e in DATA_EXT if v.value.endswith(e)), None)
                        if ext:
                            out.append("*" + ext)
                    else:
                        visit(v)
                return
            if isinstance(n, ast.Constant):
                if isinstance(n.value, str) and n.value.endswith(DATA_EXT):
                    out.append(_basename(n.value))
                return
            if isinstance(n, ast.Name):
                if depth < 4:
                    for v in bind.get(n.id, []):
                        out.extend(self._literals(v, m, bind, depth + 1))
                out.extend(self._const(m, n.id))
                return
            if isinstance(n, ast.Call):
                fq, _ = self.resolve_callee(n.func, m)
                if fq in self.roles:
                    out.extend(self.roles[fq]["returns"])
            for c in ast.iter_child_nodes(n):
                visit(c)

        visit(expr)
        return sorted(set(out))

    def _deep_names(self, expr, bind, depth=0):
        out = set(_names(expr))
        if depth < 3:
            for nm in list(out):
                for v in bind.get(nm, []):
                    out |= self._deep_names(v, bind, depth + 1)
        return out

    def _call_io(self, call, m):
        """[(kind, [path exprs present], [(default expr, module)])] for one call."""
        func = call.func
        attr = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else None)
        fq, _ = self.resolve_callee(func, m)
        out = []
        if fq in self.roles and (self.roles[fq]["read"] or self.roles[fq]["write"]):
            params = self.params_of[fq]
            present = {}
            for i, a in enumerate(call.args):
                if i < len(params):
                    present[params[i]] = a
            for kw in call.keywords:
                if kw.arg:
                    present[kw.arg] = kw.value
            cm = self.fn_module[fq]
            for kind in ("read", "write"):
                ps = sorted(self.roles[fq][kind])
                if not ps:
                    continue
                exprs = [present[p] for p in ps if p in present]
                dflt = [(self.defaults_of[fq][p], cm) for p in ps
                        if p not in present and p in self.defaults_of[fq]]
                out.append((kind, exprs, dflt))
            return out
        if not call.args:
            return out
        if attr in SEED_WRITERS:
            out.append(("write", [call.args[0]], []))
        elif attr in SEED_READERS:
            out.append(("read", [call.args[0]], []))
        elif attr in SEED_MODAL:
            mode = None
            if len(call.args) > 1 and isinstance(call.args[1], ast.Constant):
                mode = str(call.args[1].value)
            for kw in call.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = str(kw.value.value)
            kind = "write" if mode and mode[0] in "wax" else "read"
            out.append((kind, [call.args[0]], []))
        return out

    def _role_of(self, fq):
        node, m = self.fn[fq], self.fn_module[fq]
        params = self.params_of[fq]
        bind = self._bindings(node)
        origin = {p: {p} for p in params}
        for _ in range(4):
            for name, exprs in bind.items():
                o = set()
                for e in exprs:
                    for nm in _names(e):
                        o |= origin.get(nm, set())
                if o - origin.get(name, set()):
                    origin[name] = origin.get(name, set()) | o
        role = dict(read=set(), write=set(), returns=[], read_int=[], write_int=[])
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            for kind, exprs, _dflt in self._call_io(call, m):
                for e in exprs:
                    for nm in _names(e):
                        role[kind] |= origin.get(nm, set())
                    role[kind + "_int"].extend(self._literals(e, m, bind))
        for r in ast.walk(node):
            if isinstance(r, ast.Return) and r.value is not None:
                role["returns"].extend(self._literals(r.value, m, bind))
        for k in ("returns", "read_int", "write_int"):
            role[k] = sorted(set(role[k]))
        return role

    def compute_io(self):
        self.roles = {fq: dict(read=set(), write=set(), returns=[], read_int=[], write_int=[])
                      for fq in self.fn}
        for _ in range(8):
            changed = False
            for fq in sorted(self.fn):
                r = self._role_of(fq)
                if r != self.roles[fq]:
                    self.roles[fq] = r
                    changed = True
            if not changed:
                break
        seen = set()
        for fq in sorted(self.fn):
            node, m = self.fn[fq], self.fn_module[fq]
            bind = self._bindings(node)
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                for kind, exprs, dflt in self._call_io(call, m):
                    lits, names = [], set()
                    for e in exprs:
                        lits.extend(self._literals(e, m, bind))
                        names |= self._deep_names(e, bind)
                    for e, cm in dflt:
                        lits.extend(self._literals(e, cm, {}))
                    loc = "per FOV" if names & PER_FOV_NAMES else "summary"
                    for lit in sorted(set(lits)):
                        key = (m, lit, kind)
                        if key in seen:
                            continue
                        seen.add(key)
                        self.io_edges.append(dict(module=m, func=fq, file=lit, kind=kind,
                                                  loc=loc, line=call.lineno))

    def call_site_files(self, call, m, bind):
        """Files a call writes/reads: role-param literals + omitted defaults + callee internals."""
        files = dict(read=set(), write=set())
        for kind, exprs, dflt in self._call_io(call, m):
            for e in exprs:
                files[kind] |= set(self._literals(e, m, bind))
            for e, cm in dflt:
                files[kind] |= set(self._literals(e, cm, {}))
        fq, _ = self.resolve_callee(call.func, m)
        if fq in self.roles:
            files["read"] |= set(self.roles[fq]["read_int"])
            files["write"] |= set(self.roles[fq]["write_int"])
        return {k: sorted(v) for k, v in files.items()}

    # ---------------------------------------------------------------- dataflow
    @staticmethod
    def _shapes(doc):
        """(named, positional_names, single) parsed from a numpydoc docstring."""
        named, positional, single = {}, [], None
        for mt in re.finditer(r"``(\w+)``\s*:\s*\(([^)]*)\)", doc):
            named.setdefault(mt.group(1), "(" + mt.group(2).strip() + ")")
        for mt in re.finditer(r"^\s*(\w+)\s*:\s*\(([^)]*)\)", doc, re.M):
            named.setdefault(mt.group(1), "(" + mt.group(2).strip() + ")")
        mt = re.search(r"Returns\s*\n\s*-+\s*\n\s*\(([^)]*)\)\s*:\s*(.*)", doc)
        if mt:
            positional = [s.strip() for s in mt.group(1).split(",") if s.strip()]
            rest = mt.group(2)
            shapes = re.findall(r"\(([^)]*)\)", rest)
            if "tuple of" in rest and len(shapes) == 1:
                shapes = shapes * len(positional)
            if len(shapes) == len(positional):
                for nm, sh in zip(positional, shapes):
                    named.setdefault(nm, "(" + sh.strip() + ")")
        else:
            mt = re.search(r"Returns\s*\n\s*-+\s*\n\s*\(([^)]*)\)\s*(?:ndarray|array|int|bool|float)", doc)
            if mt:
                single = "(" + mt.group(1).strip() + ")"
            else:
                mt = re.search(r"``\[([^\]]*)\]``", _first_line(doc))
                if mt:
                    single = "(" + mt.group(1).strip() + ")"
            mt2 = re.search(r"Returns\s*\n\s*-+\s*\n\s*\(([^)]*)\)\s*$", doc, re.M)
            if mt2 and not single:
                positional = [s.strip() for s in mt2.group(1).split(",") if s.strip()]
        return named, positional, single

    def _return_shape(self, fq, depth=0):
        """Single return shape of a function whose docstring has none, via ``return other(...)``."""
        if fq not in self.fn or depth > 2:
            return None
        m = self.fn_module[fq]
        for r in ast.walk(self.fn[fq]):
            if isinstance(r, ast.Return) and isinstance(r.value, ast.Call):
                callee, _ = self.resolve_callee(r.value.func, m)
                if callee in self.fn:
                    _named, _pos, single = self._shapes(self.docs.get(callee, ""))
                    return single or self._return_shape(callee, depth + 1)
        return None

    @staticmethod
    def _pretty_cond(text):
        text = re.sub(r"getattr\(p, '(\w+)'(?:, [^)]*)?\)", r"p.\1", text)
        return text

    def dataflow(self):
        node, m = self.fn[ENTRY], self.fn_module[ENTRY]
        parent = {}
        for p in ast.walk(node):
            for c in ast.iter_child_nodes(p):
                parent[c] = p
        bind = self._bindings(node)
        stmts = [s for s in ast.walk(node) if isinstance(s, (ast.Assign, ast.Expr))
                 and isinstance(s.value, ast.Call)]
        stmts.sort(key=lambda s: (s.lineno, s.col_offset))
        producer, var_shape, stages, edges, counts = {}, {}, [], [], defaultdict(int)
        for s in stmts:
            call = s.value
            fq, _ = self.resolve_callee(call.func, m)
            if not fq or not fq.startswith("pyali.") or not (fq in self.fn or fq in self.classes):
                continue
            counts[fq] += 1
            outputs = []
            if isinstance(s, ast.Assign):
                for t in s.targets:
                    outputs += [e.id for e in (t.elts if isinstance(t, ast.Tuple) else [t])
                                if isinstance(e, ast.Name)]
            inputs = sorted({n.id for a in call.args + [k.value for k in call.keywords]
                             for n in ast.walk(a) if isinstance(n, ast.Name)})
            inplace = []
            if not outputs and call.args and isinstance(call.args[0], ast.Name) \
                    and call.args[0].id in producer:
                outputs, inplace = [call.args[0].id], [call.args[0].id]
            else:
                inplace = [o for o in outputs if o in inputs]
            cond, child = None, s
            p = parent.get(s)
            while p is not None and not isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if isinstance(p, ast.If):
                    cond = self._pretty_cond(ast.unparse(p.test))
                    if child in p.orelse:
                        cond = "not (" + cond + ")"
                    break
                child, p = p, parent.get(p)
            named, positional, single = self._shapes(self.docs.get(fq, ""))
            if not named and not single:                 # a dispatcher: take the shape of what it returns
                single = self._return_shape(fq)
            shapes = {}
            for i, o in enumerate(outputs):
                sh = named.get(o)
                if sh is None and i < len(positional):
                    sh = named.get(positional[i])
                if sh is None and single and len(outputs) == 1:
                    sh = single
                if sh:
                    shapes[o] = sh
            for k, v in named.items():
                var_shape.setdefault(k, v)
            files = self.call_site_files(call, m, bind)
            sid = fq if counts[fq] == 1 else f"{fq}@{s.lineno}"
            st = dict(id=sid, fq=fq, line=s.lineno, module=self.fn_module.get(fq, "pyali.params"),
                      outputs=outputs, inputs=inputs, inplace=inplace, cond=cond, shapes=shapes,
                      reads=files["read"], writes=files["write"],
                      kwargs=sorted(k.arg for k in call.keywords if k.arg))
            upstream = defaultdict(list)
            for v in inputs:
                if v in producer:
                    upstream[producer[v]].append(v)
            for src in sorted(upstream):
                edges.append(dict(src=src, dst=sid, vars=sorted(upstream[src])))
            stages.append(st)
            for o in outputs:
                producer[o] = sid
                if o in shapes:
                    var_shape[o] = shapes[o]
        # second pass: ids may have been renamed by a later duplicate; rename consistently
        for st in stages:
            if counts[st["fq"]] > 1 and st["id"] == st["fq"]:
                st["id"] = f"{st['fq']}@{st['line']}"
        rename = {}
        for st in stages:                               # the first duplicate once carried the bare fq
            if counts[st["fq"]] > 1:
                rename.setdefault(st["fq"], st["id"])
        for e in edges:
            e["src"] = rename.get(e["src"], e["src"])
            e["dst"] = rename.get(e["dst"], e["dst"])
        return dict(stages=stages, edges=edges, var_shape=var_shape)

    # ---------------------------------------------------------------- serialisation
    def to_json(self):
        return dict(
            entry=ENTRY,
            modules=self.info,
            import_edges=sorted(self.import_edges, key=lambda e: (e["src"], e["dst"], e["name"])),
            call_edges=sorted(self.call_edges, key=lambda e: (e["src"], e["dst"])),
            params_fields=self.param_fields,
            params_reads={k: v for k, v in sorted(self.params_reads.items()) if v},
            core=sorted(self.core),
            io_roles={fq: dict(read=sorted(r["read"]), write=sorted(r["write"]), returns=r["returns"])
                      for fq, r in sorted(self.roles.items()) if r["read"] or r["write"] or r["returns"]},
            io_edges=sorted(self.io_edges, key=lambda e: (e["module"], e["kind"], e["file"])),
            dataflow=self.dataflow(),
        )


# =========================================================================================
# Layout: longest-path layering + barycenter ordering (top-to-bottom)
# =========================================================================================
MONO_W, SANS_W = 6.7, 5.4          # px per char at 11px mono / 10px sans


def node_size(n):
    w = max(len(n["label"]) * MONO_W, len(n.get("sub", "")) * SANS_W) + 22
    return max(96, int(round(w))), 40


def label_lines(lab):
    return lab if isinstance(lab, list) else ([lab] if lab else [])


def label_box(x, y, lines):
    """Bounding box of an edge label whose first baseline sits at (x, y)."""
    w = max(len(ln) for ln in lines) * SANS_W + 4
    return (x, y - (len(lines) - 1) * 6 - 9, w, 12 * len(lines) + 2)


def layout(nodes, edges, gap_x=34, gap_y=78, margin=18):
    """Sugiyama-style layered layout, top to bottom, with every choice tie-broken by id.

    Long edges get one dummy node per crossed layer, as wide as the edge's label, so the
    label and the line reserve a channel instead of landing on whatever sits in between.
    Returns ``(boxes, W, H, layer, routes)``; ``routes[(src, dst)]`` lists the waypoints a
    multi-layer edge passes through.
    """
    by_id = {n["id"]: n for n in nodes}
    ids = [n["id"] for n in nodes]
    real = set(ids)
    E, first_edge = [], {}
    for e in edges:
        key = (e["src"], e["dst"])
        if key[0] in real and key[1] in real and key[0] != key[1] and key not in first_edge:
            E.append(key)
            first_edge[key] = e
    pred, succ = defaultdict(list), defaultdict(list)
    for s, d in E:
        succ[s].append(d)
        pred[d].append(s)
    indeg = {i: len(pred[i]) for i in ids}
    q, layer = deque(sorted(i for i in ids if indeg[i] == 0)), {}
    while q:
        u = q.popleft()
        layer[u] = max((layer[p] + 1 for p in pred[u] if p in layer), default=0)
        for v in sorted(succ[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    for i in ids:                                   # cycle fallback: deterministic
        if i not in layer:
            layer[i] = max(layer.values(), default=-1) + 1
    for i in sorted(ids):                           # an input-less node sits just above its first consumer
        if not pred[i] and succ[i]:
            layer[i] = min(layer[s] for s in succ[i]) - 1
    chain, dummy_w = {}, {}
    for s, d in E:
        if layer[d] - layer[s] > 1:
            lines = label_lines(first_edge[(s, d)].get("label"))
            w = int(max(len(ln) for ln in lines) * SANS_W + 18) if lines else 12
            ds = []
            for L in range(layer[s] + 1, layer[d]):
                did = f"dummy:{s}>{d}:{L}"
                layer[did], dummy_w[did] = L, w
                ds.append(did)
            chain[(s, d)] = ds
    opred, osucc = defaultdict(list), defaultdict(list)
    for s, d in E:
        path = [s] + chain.get((s, d), []) + [d]
        for a, b in zip(path, path[1:]):
            osucc[a].append(b)
            opred[b].append(a)
    all_ids = ids + sorted(dummy_w)
    nl = max(layer.values()) + 1
    layers = [sorted(i for i in all_ids if layer[i] == L) for L in range(nl)]
    pos = {i: k for L in layers for k, i in enumerate(L)}

    def bary(i, nbrs):
        ps = [pos[p] for p in nbrs if p in pos]
        return sum(ps) / len(ps) if ps else pos[i]

    for _ in range(4):
        for L in range(1, nl):
            layers[L].sort(key=lambda i: (bary(i, opred[i]), i))
            pos.update({i: k for k, i in enumerate(layers[L])})
        for L in range(nl - 2, -1, -1):
            layers[L].sort(key=lambda i: (bary(i, osucc[i]), i))
            pos.update({i: k for k, i in enumerate(layers[L])})

    def size(i):
        return (dummy_w[i], 40) if i in dummy_w else node_size(by_id[i])

    # x placement: start packed and centred, then pull every node toward the mean x of its
    # neighbours while keeping the layer's order and gaps (isotonic regression per layer)
    widths = [sum(size(i)[0] for i in L) + gap_x * (len(L) - 1) for L in layers]
    xs = {}
    for L, row in enumerate(layers):
        x = (max(widths) - widths[L]) / 2
        for i in row:
            xs[i] = x
            x += size(i)[0] + gap_x

    def centre(i):
        return xs[i] + size(i)[0] / 2

    def pav(d):                                     # non-decreasing least-squares fit
        blocks = []
        for v in d:
            blocks.append([v, 1, v])
            while len(blocks) > 1 and blocks[-2][2] > blocks[-1][2]:
                b = blocks.pop()
                a = blocks[-1]
                a[0] += b[0]
                a[1] += b[1]
                a[2] = a[0] / a[1]
        out = []
        for s, c, v in blocks:
            out += [v] * c
        return out

    def place(row, desired):
        off, acc = [], 0.0
        for i in row:
            off.append(acc)
            acc += size(i)[0] + gap_x
        fitted = pav([desired[i] - size(i)[0] / 2 - off[k] for k, i in enumerate(row)])
        for k, i in enumerate(row):
            xs[i] = fitted[k] + off[k]

    def mean_centre(nbrs, fallback):
        cs = [centre(n) for n in nbrs]
        return sum(cs) / len(cs) if cs else fallback

    for _ in range(4):
        for L in range(1, nl):
            place(layers[L], {i: mean_centre(opred[i], centre(i)) for i in layers[L]})
        for L in range(nl - 2, -1, -1):
            place(layers[L], {i: mean_centre(osucc[i], centre(i)) for i in layers[L]})
    for L in range(nl):
        place(layers[L], {i: mean_centre(opred[i] + osucc[i], centre(i)) for i in layers[L]})
    shift = margin - min(xs.values())
    boxes, y = {}, margin
    for L, row in enumerate(layers):
        for i in row:
            w, h = size(i)
            boxes[i] = (round(xs[i] + shift, 1), y, w, h)
        y += 40 + gap_y
    W = max(b[0] + b[2] for b in boxes.values()) + margin
    H = y - gap_y + margin
    routes = {}
    for key, ds in chain.items():
        routes[key] = [(boxes[d][0] + boxes[d][2] / 2, boxes[d][1] + boxes[d][3] / 2) for d in ds]
    real_boxes = {i: boxes[i] for i in ids}
    return real_boxes, int(round(W)), int(H), {i: layer[i] for i in ids}, routes


def _bezier(p0, p1, p2, p3, t):
    u = 1 - t
    return (u ** 3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t ** 3 * p3[0],
            u ** 3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t ** 3 * p3[1])


def _segments(points):
    """Cubic segments through consecutive waypoints, controls vertical at each end."""
    segs = []
    for a, b in zip(points, points[1:]):
        dy = (b[1] - a[1]) / 2
        segs.append((a, (a[0], a[1] + dy), (b[0], b[1] - dy), b))
    return segs


def _intersects(a, b):
    return not (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])


# =========================================================================================
# SVG
# =========================================================================================
def _esc(s):
    return html.escape(str(s), quote=True)


def render_svg(graph, fig_id, aria, standalone=False):
    nodes, edges = graph["nodes"], graph["edges"]
    boxes, W, H, layer, routes = layout(nodes, edges)
    out = []
    obstacles = [boxes[i] for i in sorted(boxes)]       # nodes first, then every placed label
    # edges first so nodes paint over them
    for e in edges:
        if e["src"] not in boxes or e["dst"] not in boxes:
            continue
        x1, y1, w1, h1 = boxes[e["src"]]
        x2, y2, w2, h2 = boxes[e["dst"]]
        sx, sy = x1 + w1 / 2, y1 + h1
        tx, ty = x2 + w2 / 2, y2
        if ty < sy:                                    # back edge: leave from the top, enter from below
            sy, ty = y1, y2 + h2
        points = [(sx, sy)] + routes.get((e["src"], e["dst"]), []) + [(tx, ty)]
        segs = _segments(points)
        d = f"M {sx:.1f} {sy:.1f} " + " ".join(
            f"C {c1[0]:.1f} {c1[1]:.1f}, {c2[0]:.1f} {c2[1]:.1f}, {b[0]:.1f} {b[1]:.1f}" for _a, c1, c2, b in segs)
        dash = ' stroke-dasharray="5 4"' if e.get("dashed") else ""
        out.append(f'<path d="{d}" fill="none" stroke="currentColor" stroke-width="1.4"{dash} '
                   f'marker-end="url(#arrow-{fig_id})"><title>{_esc(e.get("title") or e.get("label") or "")}</title></path>')
        lines = label_lines(e.get("label"))
        if lines:
            # candidate anchors: each reserved waypoint first, then points along the first segment
            cands = [(wx + 7, wy) for wx, wy in routes.get((e["src"], e["dst"]), [])]
            for seg in (segs[0], segs[-1]):
                for t in (0.5, 0.35, 0.65, 0.25, 0.75, 0.15, 0.85):
                    px, py = _bezier(*seg, t)
                    cands.append((px + 7, py))
            anchor = cands[0]
            for cx, cy in cands:
                if not any(_intersects(label_box(cx, cy, lines), ob) for ob in obstacles):
                    anchor = (cx, cy)
                    break
            lb = label_box(anchor[0], anchor[1], lines)
            obstacles.append(lb)
            W = max(W, int(lb[0] + lb[2]) + 18)             # the canvas grows to keep labels inside
            y0 = anchor[1] - (len(lines) - 1) * 6
            for k, ln in enumerate(lines):
                out.append(f'<text x="{anchor[0]:.1f}" y="{y0 + 12 * k + 3.5:.1f}" font-size="10" '
                           f'font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif" '
                           f'fill="currentColor" style="paint-order:stroke;stroke:var(--bg,#fff);'
                           f'stroke-width:4px;stroke-linejoin:round">{_esc(ln)}</text>')
    for n in nodes:
        x, y, w, h = boxes[n["id"]]
        kind = n.get("kind", "stage")
        title = n.get("title") or n["label"]
        stroke = "var(--accent,#0f766e)" if n.get("dashed") else "currentColor"
        dash = ' stroke-dasharray="6 3"' if n.get("dashed") else ""
        out.append(f'<g><title>{_esc(title)}</title>')
        if kind == "file":
            f = 9
            d = (f"M {x} {y} H {x + w - f} L {x + w} {y + f} V {y + h} H {x} Z "
                 f"M {x + w - f} {y} V {y + f} H {x + w}")
            out.append(f'<path d="{d}" fill="var(--surface,#fff)" stroke="{stroke}" stroke-width="1.2"{dash} '
                       f'stroke-linejoin="round"/>')
        elif kind == "module":
            out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="var(--surface,#fff)" '
                       f'stroke="{stroke}" stroke-width="1.2"{dash}/>')
        else:
            out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="var(--surface,#fff)" '
                       f'stroke="{stroke}" stroke-width="1.4"{dash}/>')
        cx = x + w / 2
        out.append(f'<text x="{cx:.1f}" y="{y + 17}" font-size="11" text-anchor="middle" '
                   f'fill="currentColor">{_esc(n["label"])}</text>')
        if n.get("sub"):
            out.append(f'<text x="{cx:.1f}" y="{y + 31}" font-size="10" text-anchor="middle" '
                       f'font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif" '
                       f'fill="var(--muted,#5d6670)">{_esc(n["sub"])}</text>')
        out.append("</g>")
    out.append("</svg>")
    root_style = 'style="color:#1f2933;--bg:#ffffff;--surface:#ffffff;--muted:#5d6670;--accent:#0f766e"' \
        if standalone else ""
    head = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
            f'role="img" aria-label="{_esc(aria)}" {root_style} '
            f'font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace">',
            f'<defs><marker id="arrow-{fig_id}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"/></marker></defs>']
    if standalone:
        head.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
    return "\n".join(head + out) + "\n", W


# =========================================================================================
# Figure graphs
# =========================================================================================
def core_graph(data):
    df = data["dataflow"]
    nodes, edges, files = [], [], {}
    for st in df["stages"]:
        short = st["fq"][len("pyali."):]
        mod, fn = short.rsplit(".", 1)
        sub = f"{mod}.py" + (f"  ·  if {st['cond']}" if st["cond"] else "")
        tip = f"{st['fq']} (pipeline.py L{st['line']})"
        if st["cond"]:
            tip += f"\nonly when {st['cond']}"
        if st["outputs"]:
            tip += "\n-> " + ", ".join(f"{o} {st['shapes'].get(o, '')}".strip() for o in st["outputs"])
        if not st["outputs"] and st["writes"] and not st["reads"]:
            # an output writer: one node that is both the call and the file(s) it leaves behind
            names = st["writes"]
            label = names[0] if len(names) == 1 else f"{len(names)} result figures"
            tip += "\nwrites " + ", ".join(names)
            if st["kwargs"]:
                tip += "\nstores " + ", ".join(st["kwargs"])
            nodes.append(dict(id=st["id"], label=label, sub=f"{short}" + (f"  ·  if {st['cond']}" if st["cond"] else ""),
                              kind="file", dashed=bool(st["cond"]), title=tip))
            continue
        nodes.append(dict(id=st["id"], label=fn, sub=sub, kind="stage", dashed=bool(st["cond"]), title=tip))
        for kind, names in (("read", st["reads"]), ("write", st["writes"])):
            if not names:
                continue
            if len(names) > 3:
                fid = f"files:{st['id']}"
                files[fid] = dict(id=fid, label=f"{len(names)} result figures", sub="out_dir/",
                                  kind="file", title="\n".join(names))
                ids = [fid]
            else:
                ids = []
                for nm in names:
                    fid = f"file:{nm}"
                    files.setdefault(fid, dict(id=fid, label=nm, kind="file",
                                               sub="fov_dir/" if kind == "read" else "out_dir/"))
                    ids.append(fid)
            for fid in ids:
                if kind == "read":
                    edges.append(dict(src=fid, dst=st["id"], label="reads", dashed=bool(st["cond"])))
                else:
                    edges.append(dict(src=st["id"], dst=fid, label="writes", dashed=bool(st["cond"])))
    by_id = {st["id"]: st for st in df["stages"]}
    for e in df["edges"]:
        src, dst = by_id[e["src"]], by_id[e["dst"]]
        full = []
        for v in e["vars"]:
            sh = df["var_shape"].get(v, "")
            tag = " in place" if v in dst["inplace"] else ""
            full.append(f"{v} {sh}{tag}".strip())
        lines = full
        if len(e["vars"]) > 3:                          # many arrays at once: name three, count the rest
            shapes = {df["var_shape"].get(v, "") for v in e["vars"]}
            common = shapes.pop() if len(shapes) == 1 else ""
            lines = [", ".join(e["vars"][:3]) + ",", f"+{len(e['vars']) - 3} more" + (f", each {common}" if common else "")]
        edges.append(dict(src=e["src"], dst=e["dst"], label=lines, title=", ".join(full),
                          dashed=bool(src["cond"] or dst["cond"])))
    return dict(nodes=nodes + [files[k] for k in sorted(files)], edges=edges)


def qc_graph(data):
    nodes, edges, files = [], [], {}
    for s in QC_SCRIPTS:
        rec = data["modules"][s]
        folder = rec["path"].rsplit("/", 1)[0] + "/"
        nodes.append(dict(id=s, label=rec["path"].rsplit("/", 1)[-1], sub=folder, kind="stage",
                          title=f"{rec['path']}\n{rec['doc']}"))
    io = [e for e in data["io_edges"] if e["module"] in QC_SCRIPTS]
    readers, loc_of = defaultdict(set), {}
    for e in io:
        if e["kind"] == "read":
            readers[e["file"]].add(e["module"])
        if e["loc"] == "per FOV" or e["file"] not in loc_of:
            loc_of[e["file"]] = e["loc"]
    # a script's terminal outputs (nothing downstream reads them) collapse into one node past three
    collapsed = {}
    for s in QC_SCRIPTS:
        terminal = sorted({e["file"] for e in io if e["module"] == s and e["kind"] == "write"
                           and not e["file"].startswith("*") and not readers[e["file"]]})
        if len(terminal) > 3:
            gid = f"files:{s}"
            files[gid] = dict(id=gid, label=f"{len(terminal)} output files", sub="summary", kind="file",
                              title="\n".join(terminal))
            for f in terminal:
                collapsed[(s, f)] = gid
    seen = set()
    for e in io:
        if (e["module"], e["file"]) in collapsed:
            fid = collapsed[(e["module"], e["file"])]
        else:
            shared = not e["file"].startswith("*")
            fid = f"file:{e['file']}" if shared else f"file:{e['module']}:{e['file']}"
            files.setdefault(fid, dict(id=fid, label=e["file"], sub=loc_of[e["file"]], kind="file"))
        key = (e["module"], fid, e["kind"])
        if key in seen:
            continue
        seen.add(key)
        if e["kind"] == "write":
            edges.append(dict(src=e["module"], dst=fid, title=f"{e['module']} writes {e['file']}"))
        else:
            edges.append(dict(src=fid, dst=e["module"], title=f"{e['module']} reads {e['file']}"))
    per_script = defaultdict(set)
    for e in data["import_edges"]:
        if e["src"] in QC_SCRIPTS and e["dst"].startswith("pyali"):
            per_script[(e["dst"], e["src"])].add(e["name"].rsplit(".", 1)[-1])
    for mdl in sorted({k[0] for k in per_script}):
        nodes.append(dict(id=f"mod:{mdl}", label=mdl, sub="package module", kind="module"))
    for (mdl, s), names in sorted(per_script.items()):
        edges.append(dict(src=f"mod:{mdl}", dst=s, label=", ".join(sorted(names)), dashed=True,
                          title=f"{s} imports {', '.join(sorted(names))} from {mdl}"))
    return dict(nodes=nodes + [files[k] for k in sorted(files)], edges=edges)


# =========================================================================================
# Mermaid
# =========================================================================================
def _mid(s):
    return re.sub(r"[^A-Za-z0-9_]", "_", s)


def render_mermaid(graph):
    out = ["flowchart TD"]
    for n in graph["nodes"]:
        label = n["label"] + (f"<br/><i>{n['sub']}</i>" if n.get("sub") else "")
        shape = ('[/"', '"/]') if n.get("kind") == "file" else (('(["', '"])') if n.get("kind") == "module" else ('["', '"]'))
        out.append(f"  {_mid(n['id'])}{shape[0]}{label}{shape[1]}")
        if n.get("dashed"):
            out.append(f"  style {_mid(n['id'])} stroke-dasharray: 6 3")
    for e in graph["edges"]:
        lab = e.get("label")
        lab = "<br/>".join(lab) if isinstance(lab, list) else lab
        arrow = "-.->" if e.get("dashed") else "-->"
        mid = f"|{lab}|" if lab else ""
        out.append(f"  {_mid(e['src'])} {arrow}{mid} {_mid(e['dst'])}")
    return "\n".join(out) + "\n"


# =========================================================================================
# HTML page
# =========================================================================================
CSS = """
/* Layout: one reading column (~72ch) for prose and tables; each figure breaks out into a
   full-width strip that scrolls sideways when the drawing is wider than the viewport. */
:root {
  --bg: #f3f5f7; --fg: #1f2933; --muted: #5d6670; --surface: #ffffff; --line: #d3d9df;
  --accent: #0f766e; --accent-soft: #d8efec; --code-bg: #e9edf1;
  --mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #15191e; --fg: #e3e7eb; --muted: #9aa4ad; --surface: #1d2329; --line: #2f3841;
  --accent: #4fc3b8; --accent-soft: #163a37; --code-bg: #242c34; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #15191e; --fg: #e3e7eb; --muted: #9aa4ad; --surface: #1d2329; --line: #2f3841;
  --accent: #4fc3b8; --accent-soft: #163a37; --code-bg: #242c34; color-scheme: dark; }
body { background: var(--bg); color: var(--fg); font-family: var(--sans); line-height: 1.5;
  margin: 0; padding-block: 32px 48px;
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.col { max-width: 1400px; margin: 0 auto; }
.col > header, p, h1, h2, h3, .note, footer { max-width: 72ch; }
details { max-width: 96ch; }
h1 { font-size: 1.7rem; font-weight: 650; letter-spacing: -0.01em; margin: 0 0 4px; text-wrap: balance; }
h2 { font-size: 1.25rem; font-weight: 650; margin: 44px 0 8px; text-wrap: balance; }
h3 { font-size: 1rem; font-weight: 650; margin: 28px 0 6px; }
p { margin: 0 0 12px; }
.sub { color: var(--muted); margin-bottom: 4px; }
.facts { display: flex; flex-wrap: wrap; gap: 8px 18px; color: var(--muted); font-size: 0.92rem;
  font-variant-numeric: tabular-nums; margin: 10px 0 0; padding: 0; list-style: none; }
.facts b { color: var(--fg); font-weight: 600; }
code, .mono { font-family: var(--mono); font-size: 0.92em; }
code { background: var(--code-bg); padding: 1px 5px; border-radius: 3px; }
pre { background: var(--code-bg); padding: 10px 12px; border-radius: 4px; overflow-x: auto; margin: 8px 0 14px; }
pre code { background: none; padding: 0; }
figure { margin: 18px 0 6px; }
.strip { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 6px;
  padding: 10px; color: var(--fg); }
.strip svg { display: block; height: auto; max-width: none; margin: 0 auto; }
figcaption { color: var(--muted); font-size: 0.92rem; margin-top: 8px; line-height: 1.45; }
figcaption b { color: var(--fg); font-weight: 600; }
details { border-top: 1px solid var(--line); padding: 8px 0; }
details:last-of-type { border-bottom: 1px solid var(--line); }
summary { cursor: pointer; font-weight: 600; }
summary .mono { font-weight: 500; }
summary .n { color: var(--muted); font-weight: 400; font-size: 0.9rem; }
ul.tree, ul.tree ul { list-style: none; margin: 4px 0 0; padding-left: 0; }
ul.tree ul { padding-left: 18px; border-left: 1px solid var(--line); margin-left: 6px; }
ul.tree li { margin: 4px 0; }
.fn { font-family: var(--mono); font-size: 0.92em; }
.loc { color: var(--muted); font-size: 0.82rem; font-variant-numeric: tabular-nums; margin-left: 6px; }
.doc { color: var(--muted); font-size: 0.9rem; display: block; margin-left: 2px; }
.tag { color: var(--muted); font-size: 0.78rem; margin-left: 6px; }
.chip { display: inline-block; font-family: var(--mono); font-size: 0.74rem; background: var(--accent-soft);
  color: var(--fg); border-radius: 3px; padding: 0 5px; margin: 2px 4px 0 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 6px 0 14px; }
th, td { text-align: left; vertical-align: top; padding: 6px 8px; border-top: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; font-size: 0.8rem; letter-spacing: 0.03em; text-transform: uppercase; }
.tablewrap { overflow-x: auto; }
.note { color: var(--muted); font-size: 0.9rem; }
footer { color: var(--muted); font-size: 0.85rem; margin-top: 48px; border-top: 1px solid var(--line); padding-top: 12px; }
a { color: var(--accent); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
"""


def _call_tree(data, roots, mod, visited, depth=0):
    reads = data["params_reads"]
    defs = {f"{m}.{d['name']}": d for m, rec in data["modules"].items() for d in rec["defs"]}
    callees = defaultdict(list)
    for e in data["call_edges"]:
        if not e["lazy"] and e["dst"] in data["core"]:
            callees[e["src"]].append(e)
    items = []
    for fq in roots:
        d = defs.get(fq, {})
        name = fq.rsplit(".", 1)[-1]
        fmod = fq.rsplit(".", 1)[0]
        tag = "" if fmod == mod else f'<span class="tag">[{_esc(fmod[len("pyali."):])}.py]</span>'
        chips = "".join(f'<span class="chip">p.{_esc(f)}</span>' for f in reads.get(fq, []))
        loc = f'<span class="loc">L{d.get("line")}–{d.get("end")}</span>' if d else ""
        doc = f'<span class="doc">{_esc(d["doc"])}</span>' if d.get("doc") else ""
        kids = ""
        if fmod == mod and fq not in visited and depth < 8:
            visited = visited | {fq}
            ch = sorted(callees[fq], key=lambda e: (e["line"], e["dst"]))
            ch = [e["dst"] for e in ch if e["dst"] != fq]
            if ch:
                kids = _call_tree(data, ch, mod, visited, depth + 1)
        items.append(f'<li><span class="fn">{_esc(name)}</span>{tag}{loc} {chips}{doc}{kids}</li>')
    return "<ul class=\"tree\">" + "".join(items) + "</ul>"


def render_html(data, core_svg, qc_svg, stats):
    df = data["dataflow"]
    core_by_mod = defaultdict(set)
    for fq in data["core"]:
        core_by_mod[fq.rsplit(".", 1)[0]].add(fq)
    same_mod_callee = set()
    for e in data["call_edges"]:
        if not e["lazy"] and e["src_module"] == e["dst_module"] and e["dst"] in data["core"] \
                and e["src"] != e["dst"]:
            same_mod_callee.add(e["dst"])
    first_call = {}
    for e in data["call_edges"]:
        if e["src"] == ENTRY:
            first_call.setdefault(e["dst"], e["line"])
    sections = []
    for mod in CORE_ORDER:
        if mod == "pyali.pipeline":
            continue
        fns = sorted(core_by_mod.get(mod, set()))
        roots = [f for f in fns if f not in same_mod_callee]
        roots.sort(key=lambda f: (first_call.get(f, 10 ** 6), f))
        if not roots:
            continue
        rec = data["modules"][mod]
        sections.append(
            f'<details open><summary><span class="mono">{_esc(rec["path"])}</span> '
            f'<span class="n">· {len(fns)} function{"s" if len(fns) != 1 else ""} in the core set · '
            f'{_esc(rec["doc"])}</span></summary>{_call_tree(data, roots, mod, set())}</details>')
    # Params map
    rows = []
    for f in data["params_fields"]:
        users = sorted(fq for fq, fl in data["params_reads"].items() if f in fl and fq in data["core"])
        cells = ", ".join(f'<span class="mono">{_esc(u[len("pyali."):])}</span>' for u in users) or "—"
        rows.append(f"<tr><td><code>{_esc(f)}</code></td><td>{cells}</td></tr>")
    unreached = []
    for mod in CORE_ORDER:
        names = [d["name"] for d in data["modules"][mod]["defs"]
                 if f"{mod}.{d['name']}" not in data["core"] and d["kind"] == "function"
                 and "." not in d["name"]]
        if names:
            unreached.append(f'<b>{_esc(mod[len("pyali."):])}.py</b>: ' + ", ".join(f"<code>{_esc(n)}</code>" for n in names))
    # QC table
    qrows = []
    for s in QC_SCRIPTS:
        rec = data["modules"][s]
        rd = sorted({e["file"] for e in data["io_edges"] if e["module"] == s and e["kind"] == "read"})
        wr = sorted({e["file"] for e in data["io_edges"] if e["module"] == s and e["kind"] == "write"})
        imps = sorted({e["name"][len("pyali."):] for e in data["import_edges"]
                       if e["src"] == s and e["dst"].startswith("pyali")})
        qrows.append(f'<tr><td><span class="mono">{_esc(rec["path"].rsplit("/", 1)[-1])}</span>'
                     f'<span class="doc">{_esc(rec["doc"])}</span></td>'
                     f'<td class="mono">{"<br>".join(map(_esc, rd)) or "—"}</td>'
                     f'<td class="mono">{"<br>".join(map(_esc, wr)) or "—"}</td>'
                     f'<td class="mono">{"<br>".join(map(_esc, imps)) or "—"}</td></tr>')
    n_stage = sum(1 for s in df["stages"])
    n_cond = sum(1 for s in df["stages"] if s["cond"])
    head = f"""<title>pyali Architecture</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>{CSS}</style>"""
    body = f"""<div class="col">
<header>
<h1>pyali architecture</h1>
<p class="sub">Derived from the source by <code>scripts/arch_graph.py</code> with the standard-library
<code>ast</code> module. Nothing here is hand-drawn; regenerate after any change.</p>
<ul class="facts">
<li><b>{stats['modules']}</b> modules</li><li><b>{stats['defs']}</b> definitions</li>
<li><b>{stats['imports']}</b> import edges</li><li><b>{stats['calls']}</b> resolved call edges</li>
<li><b>{stats['core']}</b> functions reachable from <code>process_fov</code> over non-lazy calls</li>
</ul>
</header>

<section id="extraction">
<h2>Voltage extraction</h2>
<p>Every cross-module call in the algorithm goes through <code>pipeline.process_fov</code>; inside each
module the functions form a tree. So the figure that shows the mechanism is the dataflow through the
orchestrator: each box is one stage call, each arrow carries the array that stage hands on, and the
shape beside it comes from that function's docstring. Two branches leave the raw movie. The reference
branch uses the first <code>n_ref</code> frames to build the sharpened image that segmentation runs on;
the movie branch is background-subtracted and motion-corrected in place. They rejoin at footprint
extraction, and the pseudoinverse turns footprints into traces.</p>
<figure>
<div class="strip">{core_svg}</div>
<figcaption><b>Figure 1.</b> Dataflow through <code>process_fov</code>: {n_stage} stage calls,
{n_cond} of them conditional. A dashed box in the accent colour runs only when the condition under its
name holds, so the whitened-GLS path and the output writers are optional; solid boxes are the
unconditional algorithm. Document shapes are files on disk. "in place" marks a stage that mutates the
array it receives rather than returning a new one.</figcaption>
</figure>

<h3>Call trees by module</h3>
<p class="note">Functions in the core set, grouped by file, in the order the orchestrator first calls
them. A function is a root of its file's tree when nothing else in that file calls it. Chips list the
<code>Params</code> fields the function reads. Private helpers are included.</p>
{"".join(sections)}

<h3>Params fields and the functions that read them</h3>
<div class="tablewrap"><table>
<thead><tr><th>field</th><th>read by (core set)</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>

<p class="note">Defined in the core files but not reachable from <code>process_fov</code>, so not part of
the extraction algorithm: {"; ".join(unreached)}. These are entry-point helpers, alternate loaders and
reference implementations.</p>
</section>

<section id="qc">
<h2>QC checks</h2>
<p>The post-hoc QC pass never re-runs the extraction: the A/B harness under <code>snr_analysis/</code>
reads the <code>ALI_Result.mat</code> files two pipeline runs left on disk and writes its report
beside them. The coupling to the package is two imports, the <code>.mat</code> reader in
<code>pyali.io</code> and the SNR definitions in <code>pyali.metrics</code>. The figure therefore
shows scripts and the files between them.</p>
<figure>
<div class="strip">{qc_svg}</div>
<figcaption><b>Figure 2.</b> The QC pass as script → file → script. An arrow into a script is a read,
an arrow out of it a write; dashed arrows are imports from the package. <code>*.png</code>
stands for every figure a script writes. The A/B harness <code>snr_compare.py</code> reads two
<code>ALI_Result.mat</code> files, one per run, and writes its report files.</figcaption>
</figure>

<h3>What each script reads and writes</h3>
<div class="tablewrap"><table>
<thead><tr><th>script</th><th>reads</th><th>writes</th><th>from pyali</th></tr></thead>
<tbody>{"".join(qrows)}</tbody></table></div>
</section>

<footer>Regenerate with <code>python scripts/arch_graph.py</code>; <code>--check</code> exits non-zero
when the committed outputs no longer match the source. The underlying graph is
<code>docs/architecture/arch.json</code>; the two figures are also written as standalone SVG and
as Mermaid text.</footer>
</div>
"""
    fragment = head + "\n" + body
    full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            + head + "\n</head>\n<body>\n" + body + "</body>\n</html>\n")
    return full, fragment


# =========================================================================================
# main
# =========================================================================================
def build(root):
    repo = Repo(root)
    repo.discover()
    repo.parse_all()
    repo.compute_core()
    repo.compute_io()
    data = repo.to_json()
    stats = dict(modules=len(data["modules"]),
                 defs=sum(len(v["defs"]) for v in data["modules"].values()),
                 imports=len(data["import_edges"]), calls=len(data["call_edges"]),
                 core=len(data["core"]))
    cg, qg = core_graph(data), qc_graph(data)
    core_svg, _ = render_svg(cg, "core", "Dataflow of pyali.pipeline.process_fov: stage calls and the arrays "
                                        "passed between them, plus the files read and written.")
    qc_svg, _ = render_svg(qg, "qc", "The post-hoc QC pass as script, file, script: which SNR script reads and "
                                    "writes which file, and which pyali modules they import.")
    core_svg_sa, _ = render_svg(cg, "core", "Dataflow of pyali.pipeline.process_fov", standalone=True)
    qc_svg_sa, _ = render_svg(qg, "qc", "pyali QC pass: scripts and the files between them", standalone=True)
    page, fragment = render_html(data, core_svg, qc_svg, stats)
    files = {
        "arch.json": json.dumps(data, indent=1, sort_keys=True) + "\n",
        "core_dataflow.svg": core_svg_sa,
        "qc_dataflow.svg": qc_svg_sa,
        "core_dataflow.mmd": render_mermaid(cg),
        "qc_dataflow.mmd": render_mermaid(qg),
        "architecture.html": page,
    }
    return files, stats, fragment


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=_REPO, help="repo root (default: the parent of scripts/)")
    ap.add_argument("--out", default=None, help="output dir (default: <root>/docs/architecture)")
    ap.add_argument("--check", action="store_true",
                    help="regenerate into a temp dir and exit 1 if any output differs from --out")
    ap.add_argument("--fragment", default=None, help=argparse.SUPPRESS)   # page body only, for embedding
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.root, "docs", "architecture")
    files, stats, fragment = build(a.root)
    print(f"[arch] modules={stats['modules']} defs={stats['defs']} import_edges={stats['imports']} "
          f"call_edges={stats['calls']} core={stats['core']}")
    if a.check:
        stale = []
        for name, text in files.items():
            p = os.path.join(out, name)
            cur = open(p, encoding="utf-8").read() if os.path.isfile(p) else None
            if cur != text:
                stale.append(name)
        if stale:
            print(f"[arch] STALE: {', '.join(stale)} -- run scripts/arch_graph.py and commit {out}")
            return 1
        print(f"[arch] up to date: {out}")
        return 0
    os.makedirs(out, exist_ok=True)
    for name, text in files.items():
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            f.write(text)
    if a.fragment:
        with open(a.fragment, "w", encoding="utf-8") as f:
            f.write(fragment)
    print(f"[arch] wrote {', '.join(OUTPUTS)} to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
