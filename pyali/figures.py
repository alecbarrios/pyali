"""Presentation-quality result figures for a processed field of view.

Produces four static plots (PNG) plus one interactive plot (HTML):

  * detected_regions.png       - segmentation mask with region centroids + bounding boxes
  * coms.png                   - action-potential centers of mass on the reference image
  * cell_traces.png            - normalized per-cell traces, stacked
  * center_of_cell_regions.png - footprint centers on the reference image
  * cell_traces.html           - interactive cell traces (zoom/pan, click a legend entry to
                                 hide/isolate a trace, hover for values); needs ``plotly``

Region centroids/bounding boxes are 1-indexed; 1 is subtracted when drawing on the
0-indexed image axes.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")                                 # render to files, no display needed
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np


def _rescale(x):
    """Linearly map an array to [0, 1] (a constant array maps to zeros)."""
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _title(ax, text):
    ax.set_title(text, fontsize=15, fontweight="bold")


def fig_detected_regions(binary_map, regions, path, dpi=150):
    """Binary segmentation mask with red centroids, green bounding boxes, and region labels.

    binary_map : (H, W) bool/uint8 array
    regions    : list of dicts with 'Centroid' [col, row] and 'BoundingBox' [x, y, w, h]
    """
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.imshow(np.asarray(binary_map), cmap="gray", interpolation="nearest")
    for b, r in enumerate(regions, 1):
        cx, cy = float(r["Centroid"][0]) - 1, float(r["Centroid"][1]) - 1
        x_ul, y_ul, w, h = (float(v) for v in r["BoundingBox"])
        ax.plot(cx, cy, "r*", ms=6)
        ax.add_patch(Rectangle((x_ul - 1, y_ul - 1), w, h, ec="lime", fc="none", lw=1))
        ax.text(cx + 5, cy + 5, str(b), color="r", fontsize=7)
    _title(ax, "Detected Regions with Centroids and Bounding Boxes")
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return path


def fig_coms(reference_image, COMs, path, dpi=150):
    """Action-potential centers of mass scattered on the reference image.

    reference_image : (H, W) float array
    COMs            : (K, >=2) array; column 0 = row, column 1 = column (1-indexed)
    """
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.imshow(np.asarray(reference_image), cmap="gray", interpolation="nearest")
    if len(COMs):
        C = np.asarray(COMs)
        ax.scatter(C[:, 1] - 1, C[:, 0] - 1, s=15, c="r", edgecolors="k", linewidths=0.5)
    _title(ax, "COMs")
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return path


def fig_cell_traces(cell_traces, path, fps=800.0, drop_last=100, dpi=150):
    """Per-cell traces, each rescaled to [0, 1] and offset by its index (jet colormap).

    cell_traces : (N, T) float array
    fps         : frames per second, for the time axis
    drop_last   : number of final frames to omit from the plot
    """
    ct = np.asarray(cell_traces)
    n = ct.shape[0]
    fig, ax = plt.subplots(figsize=(13, max(4.0, min(0.16 * n + 2, 40))))
    if n:
        T = ct.shape[1]
        time = np.arange(T) / fps
        cmap = plt.cm.jet(np.linspace(0, 1, n))
        end = T - drop_last if T > drop_last else T
        for i in range(n):
            ax.plot(time[:end], _rescale(ct[i, :end]) + i - 0.5, color=cmap[i], lw=1.2)
        ax.set_xlim(0, time.max()); ax.set_ylim(0, n + 1)
        ax.set_yticks(np.arange(0.5, n, 5))
        ax.set_yticklabels([str(k) for k in range(1, n + 1, 5)])
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Cell Traces")
    _title(ax, "Normalized Cell Traces from Original Video")
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return path


def _plotly_script_tag():
    """A <script> tag with plotly.js: embedded (offline) if plotly is installed, else CDN."""
    try:
        from plotly.offline import get_plotlyjs
        return "<script type='text/javascript'>" + get_plotlyjs() + "</script>", "embedded"
    except Exception:
        return ("<script src='https://cdn.plot.ly/plotly-2.35.2.min.js' charset='utf-8'>"
                "</script>", "cdn")


def _footprint_crops(footprint, pad=2):
    """Per-cell footprint cropped to its nonzero bounding box (+pad), in absolute pixel coords.

    ``footprint`` is ``(H, W, N)``. Returns a list (len N) of ``{z, r0, c0}`` dicts (z = the
    cropped weight image; r0/c0 = 0-indexed top-left in the full frame) or ``None`` for an
    empty footprint. Small crops keep the embedded HTML light.
    """
    H, W, N = footprint.shape
    out = []
    for k in range(N):
        fp = footprint[:, :, k]
        nz = np.argwhere(fp != 0)
        if nz.size == 0:
            out.append(None); continue
        r0, c0 = nz.min(0); r1, c1 = nz.max(0)
        r0 = max(0, int(r0) - pad); c0 = max(0, int(c0) - pad)
        r1 = min(H - 1, int(r1) + pad); c1 = min(W - 1, int(c1) + pad)
        crop = fp[r0:r1 + 1, c0:c1 + 1]
        out.append(dict(z=[[round(float(v), 4) for v in row] for row in crop],
                        r0=r0, c0=c0))
    return out


def fig_cell_explorer(cell_traces, path, fps=800.0, metrics=None, footprint=None,
                      drop_last=100, max_points=3500):
    """Self-contained interactive cell explorer (HTML).

    Filter cells by SNR-metric cutoffs (max ``noise_sigma``, min ``snr_median``, min
    ``spectral_hf_snr``), then step through the matching cells one at a time (zoom/pan, optional
    spike-isolating detrend) or overlay them all stacked. In single-cell mode it also shows that
    cell's spatial ``footprint`` (the pixel-weight map the pseudoinverse uses to extract the
    displayed trace) as a heatmap with an intensity colorbar and pixel-coordinate axes. Works
    offline when ``plotly`` is installed (its JS is embedded); otherwise loads plotly.js from CDN.

    cell_traces : (N, T) float array
    metrics     : optional dict from :func:`pyali.metrics.per_cell_snr` (else computed here)
    footprint   : optional (H, W, N) footprint stack; if given, its per-cell map is shown
    max_points  : cap on samples per cell (traces are strided to fit, keeping the file small)

    Returns ``(path, "embedded"|"cdn")`` or ``None`` if there are no cells.
    """
    ct = np.asarray(cell_traces, float)
    n = ct.shape[0]
    if n == 0:
        return None
    T = ct.shape[1]
    end = T - drop_last if T > drop_last else T
    stride = max(1, int(np.ceil(end / max_points)))
    t = np.arange(0, end, stride) / fps
    Y = ct[:, :end:stride]

    if metrics is None:
        from .metrics import per_cell_snr
        metrics = per_cell_snr(ct[:, :end], fps)

    def _clean(a):
        return [None if not np.isfinite(v) else round(float(v), 6) for v in a]

    fp_crops = _footprint_crops(np.asarray(footprint, float)) if footprint is not None else None

    data = dict(
        t=[round(float(v), 4) for v in t],
        Y=[[round(float(v), 3) for v in row] for row in Y],
        ns=_clean(metrics["noise_sigma"]),
        sm=_clean(metrics["snr_median"]),
        sh=_clean(metrics["spectral_hf_snr"]),
        nsp=[int(v) for v in metrics["n_spikes"]],
        fp=fp_crops,
    )
    plotly_tag, src = _plotly_script_tag()
    html = (_EXPLORER_TEMPLATE
            .replace("/*__DATA__*/", json.dumps(data))
            .replace("<!--__PLOTLY__-->", plotly_tag))
    with open(path, "w") as fh:
        fh.write(html)
    return path, src


def fig_center_of_regions(reference_image, footprint_center, path, dpi=150):
    """Footprint centers as numbered circles (jet colormap) on the reference image.

    footprint_center : (N, 2) array of [row, column] (1-indexed)
    """
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.imshow(np.asarray(reference_image), cmap="gray", interpolation="nearest")
    fc = np.asarray(footprint_center)
    if len(fc):
        cmap = plt.cm.jet(np.linspace(0, 1, len(fc)))
        for t, (row, col) in enumerate(fc):
            ax.add_patch(Circle((col - 1, row - 1), 3, ec=cmap[t], fc="none", lw=1.5))
            ax.text(col - 1 + 5, row - 1 + 5, str(t + 1), color=cmap[t], fontsize=7)
    _title(ax, "Center of cell regions")
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return path


def save_result_figures(out_dir, reference_image, regions, binary_map, COMs,
                        footprint_center, cell_traces, fps=800.0, dpi=150, footprint=None):
    """Write all result figures into ``out_dir``; returns the list of output paths.

    Produces the four PNGs and the interactive cell_traces.html explorer. If ``footprint``
    ``(H, W, N)`` is given, each cell's footprint map is shown in the explorer's single-cell view.
    """
    os.makedirs(out_dir, exist_ok=True)
    j = lambda name: os.path.join(out_dir, name)
    paths = [
        fig_detected_regions(binary_map, regions, j("detected_regions.png"), dpi),
        fig_coms(reference_image, COMs, j("coms.png"), dpi),
        fig_cell_traces(cell_traces, j("cell_traces.png"), fps, dpi=dpi),
        fig_center_of_regions(reference_image, footprint_center, j("center_of_cell_regions.png"), dpi),
    ]
    explorer = fig_cell_explorer(cell_traces, j("cell_traces.html"), fps, footprint=footprint)
    if explorer:
        html_path, src = explorer
        paths.append(html_path)
        if src == "cdn":
            print("[pyali] cell_traces.html written using the plotly.js CDN (needs internet to "
                  "view). `pip install plotly==5.24.1` to embed it for offline use.")
        else:
            print("[pyali] cell_traces.html written (interactive explorer, offline-ready).")
    return paths


_EXPLORER_TEMPLATE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pyali cell explorer</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:16px;color:#222}
 h2{margin:0 0 2px;font-weight:500} .sub{color:#666;font-size:13px;margin-bottom:12px}
 .panel{display:flex;flex-wrap:wrap;gap:14px;align-items:flex-end;background:#f6f6f4;border:1px solid #ddd;border-radius:8px;padding:12px;margin-bottom:10px}
 .grp{display:flex;flex-direction:column;gap:3px} .grp label{font-size:12px;color:#555}
 input[type=number]{width:96px;padding:4px 6px;border:1px solid #ccc;border-radius:6px}
 button{padding:5px 10px;border:1px solid #bbb;border-radius:6px;background:#fff;cursor:pointer;font-size:13px}
 button:hover{background:#eee} button.on{background:#2b6cb0;color:#fff;border-color:#2b6cb0}
 select{padding:4px 6px;border:1px solid #ccc;border-radius:6px;max-width:160px}
 .readout{font-size:13px;color:#333;margin:6px 0 10px} .readout b{color:#000}
 #plot{width:100%;height:52vh;min-height:360px}
 #fpwrap{margin-top:10px} .cap{font-size:12px;color:#555;margin:2px 0 4px}
 #fp{width:480px;height:400px;max-width:100%}
</style>
<!--__PLOTLY__-->
</head><body>
<h2>pyali cell explorer</h2>
<div class="sub">Filter cells by SNR metrics, then step through the matching cells one at a time (zoom / pan) or overlay them all. Leave a filter blank for "no limit".</div>
<div class="panel">
 <div class="grp"><label>mode</label><div><button id="mSingle" class="on" onclick="setMode('single')">single cell</button> <button id="mOverlay" onclick="setMode('overlay')">overlay filtered</button></div></div>
 <div class="grp"><label>max noise_sigma</label><input type="number" id="fNS" step="0.001" placeholder="any"></div>
 <div class="grp"><label>min snr_median</label><input type="number" id="fSM" step="0.5" placeholder="any"></div>
 <div class="grp"><label>min spectral_hf_snr</label><input type="number" id="fSH" step="0.1" placeholder="any"></div>
 <div class="grp"><label>&nbsp;</label><button onclick="applyFilters()">apply filters</button></div>
 <div class="grp"><label>cell</label><div><button onclick="step(-1)">&lsaquo; prev</button> <select id="sel" onchange="pick(this.selectedIndex)"></select> <button onclick="step(1)">next &rsaquo;</button></div></div>
 <div class="grp"><label><input type="checkbox" id="detr" onchange="render()"> isolate spikes (detrend)</label></div>
</div>
<div class="readout" id="ro"></div>
<div id="plot"></div>
<div id="fpwrap"><div class="cap">cell footprint &mdash; the pixel-weight map the pseudoinverse uses to extract this trace (axes = pixel coordinates in the full frame; colorbar = weight intensity)</div><div id="fp"></div></div>
<script>
const D = /*__DATA__*/;
let mode='single', pos=0, filt=[];
function val(id){const v=parseFloat(document.getElementById(id).value);return isNaN(v)?null:v;}
function ok(i){
 const a=val('fNS'), b=val('fSM'), c=val('fSH');
 if(a!==null && !(D.ns[i]!==null && D.ns[i]<=a)) return false;
 if(b!==null && !(D.sm[i]!==null && D.sm[i]>=b)) return false;
 if(c!==null && !(D.sh[i]!==null && D.sh[i]>=c)) return false;
 return true;
}
function applyFilters(){
 filt=[]; for(let i=0;i<D.Y.length;i++) if(ok(i)) filt.push(i);
 const sel=document.getElementById('sel'); sel.innerHTML='';
 filt.forEach(i=>{const o=document.createElement('option');o.text='cell '+(i+1);sel.add(o);});
 pos=0; render();
}
function setMode(m){mode=m;
 document.getElementById('mSingle').className=(m==='single')?'on':'';
 document.getElementById('mOverlay').className=(m==='overlay')?'on':''; render();}
function step(d){if(!filt.length)return; pos=(pos+d+filt.length)%filt.length; document.getElementById('sel').selectedIndex=pos; render();}
function pick(k){pos=k; render();}
function detrend(y,win){const n=y.length,out=new Array(n);let acc=0;
 for(let i=0;i<n;i++){acc+=y[i]; if(i>=win)acc-=y[i-win]; out[i]=y[i]-acc/Math.min(i+1,win);} return out;}
function f3(v){return v===null?'n/a':(+v).toFixed(3);}
function renderFootprint(i){
 const w=document.getElementById('fpwrap');
 const fp=(D.fp && D.fp[i])?D.fp[i]:null;
 if(!fp){w.style.display='none'; Plotly.purge('fp'); return;}
 w.style.display='';
 const xs=fp.z[0].map((_,k)=>fp.c0+k), ys=fp.z.map((_,k)=>fp.r0+k);
 Plotly.react('fp',[{z:fp.z,x:xs,y:ys,type:'heatmap',colorscale:'Viridis',
   colorbar:{title:{text:'intensity',side:'right'}}}],
   {margin:{t:6,r:10,l:54,b:44},xaxis:{title:'column (px)',constrain:'domain'},
    yaxis:{title:'row (px)',autorange:'reversed',scaleanchor:'x'}},
   {responsive:true});
}
function render(){
 const ro=document.getElementById('ro');
 if(!filt.length){ro.innerHTML='no cells match the current filters'; Plotly.purge('plot'); document.getElementById('fpwrap').style.display='none'; return;}
 const dt=document.getElementById('detr').checked;
 if(mode==='single'){
  const i=filt[pos]; let y=D.Y[i]; if(dt)y=detrend(y,60);
  Plotly.react('plot',[{x:D.t,y:y,type:'scattergl',mode:'lines',line:{width:1,color:'#2b6cb0'},name:'cell '+(i+1)}],
   {margin:{t:8,r:12},xaxis:{title:'time (s)'},yaxis:{title:dt?'detrended (a.u.)':'trace (a.u.)'}},
   {responsive:true,scrollZoom:true});
  ro.innerHTML='showing <b>cell '+(i+1)+'</b> ('+(pos+1)+' of '+filt.length+' matching) &nbsp;&nbsp; noise_sigma=<b>'+f3(D.ns[i])+'</b> &nbsp; snr_median=<b>'+f3(D.sm[i])+'</b> &nbsp; spectral_hf_snr=<b>'+f3(D.sh[i])+'</b> &nbsp; spikes=<b>'+D.nsp[i]+'</b>';
  renderFootprint(i);
 }else{
  document.getElementById('fpwrap').style.display='none';
  const traces=[];
  filt.forEach((i,k)=>{let y=D.Y[i]; let lo=Infinity,hi=-Infinity;
   for(const v of y){if(v<lo)lo=v; if(v>hi)hi=v;}
   const r=(hi>lo)?y.map(v=>(v-lo)/(hi-lo)+k):y.map(()=>k);
   traces.push({x:D.t,y:r,type:'scattergl',mode:'lines',line:{width:1},name:'cell '+(i+1),
    hovertemplate:'cell '+(i+1)+'<br>t=%{x:.3f}s<extra></extra>'});});
  Plotly.react('plot',traces,{margin:{t:8,r:12},showlegend:filt.length<=40,
   xaxis:{title:'time (s)'},yaxis:{title:'cells (rescaled 0-1, stacked)'}},
   {responsive:true,scrollZoom:true});
  ro.innerHTML='overlaying <b>'+filt.length+'</b> matching cells (rescaled + stacked); scroll to zoom, drag to pan'+(filt.length<=40?', click legend to toggle':'')+'.';
 }
}
applyFilters();
</script>
</body></html>"""
