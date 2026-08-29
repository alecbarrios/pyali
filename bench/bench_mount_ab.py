#!/usr/bin/env python3
"""Head-to-head: s3fs vs mount-s3, identical access pattern, distinct cold files per point."""
import csv, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

KEEP="/home/jovyan/spatial-technology-platform/AB/pyali_c27fc46_outputs/keep.csv"
ROOTS={"s3fs":"/home/jovyan/spatial-technology-platform","mount-s3":"/mnt/s3ab"}
CHUNK=8<<20

def read_head(p, n):
    got=0
    with open(p,"rb",buffering=0) as f:
        while got<n:
            b=f.read(min(CHUNK,n-got))
            if not b: break
            got+=len(b)
    return got

day=sys.argv[1]; gb=float(sys.argv[2]); streams=[int(x) for x in sys.argv[3:]]
rows=[r for r in csv.DictReader(open(KEEP)) if r["day"]==day]
rows.sort(key=lambda r:r["fov_path"])
rel=[r["fov_path"]+"/frames1.bin" for r in rows]
nb=int(gb*1e9); cur=0; out=[]
for n in streams:
    for name,root in ROOTS.items():
        batch=[os.path.join(root,"AB",x) for x in rel[cur:cur+n]]
        cur+=n
        t0=time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as ex:
            got=sum(ex.map(lambda p: read_head(p,nb), batch))
        dt=time.perf_counter()-t0
        mbs=got/1e6/dt
        out.append(dict(client=name,streams=n,mb_s=mbs,gb=got/1e9,seconds=dt))
        print(f"  {name:>9s}  {n:3d} streams: {mbs:7.0f} MB/s  ({got/1e9:.1f} GB in {dt:.1f}s)",flush=True)
json.dump(out,open("/home/jovyan/bench/mount_ab.json","w"),indent=2)
tot=sum(o["gb"] for o in out); print(f"\n  total read: {tot:.1f} GB")
for n in streams:
    a=[o for o in out if o["streams"]==n]
    d={o["client"]:o["mb_s"] for o in a}
    print(f"  {n:3d} streams -> mount-s3 {d['mount-s3']:.0f} vs s3fs {d['s3fs']:.0f} MB/s  = {d['mount-s3']/d['s3fs']:.1f}x")
