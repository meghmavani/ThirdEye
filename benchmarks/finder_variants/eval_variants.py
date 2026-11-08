"""Layer-finder camera-combination variants on every labelled pull (this produced Table 2 of the paper).
Recovered from the session that ran it.   .venv/bin/python benchmarks/finder_variants/eval_variants.py [after_window_s]
Truth = the player's typed correction, else the finder's own answer when it was confident and not corrected (optimistic)."""
import json,sys,os,glob,collections,numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0,ROOT+"/tracker"); os.chdir(ROOT)
import move_infer as mi
orig=mi._rows
def per_cam_scores(R,mk,prev,start,after):
    out={}
    for cam in ("iphone","android"):
        mi._rows=lambda rd,c=cam:[r for r in orig(rd) if r["cam"]==c]
        try: sc=mi.block_scores(R,mk,prev,start,after=after)
        except Exception: sc=None
        out[cam]=sc
    mi._rows=orig
    return out
moves=[]
for run in sorted(glob.glob("dataset/live_*")):
    p=run+"/labels.jsonl"
    if not os.path.exists(p): continue
    recs=[json.loads(l) for l in open(p)]
    marks=sorted([r for r in recs if "mark_ms" in r],key=lambda r:r["mark_ms"])
    corr={r["seq"]:r["correction"] for r in recs if "correction" in r}
    try: rows=[json.loads(l) for l in open(run+"/index.jsonl")]
    except Exception: continue
    if not rows: continue
    start=rows[0]["ts_ms"]; prev=None; layers=max(r["layer"] for r in rows[:2000])+1
    for m in marks:
        res=m["result"]; truth=corr.get(m["seq"]) or m.get("user_layer")
        if truth is None and res.get("ok") and res.get("confident"): truth=res["layer"]   # confident and not corrected: taken as right
        if truth is not None: moves.append((run,m,truth,start,prev,layers))
        prev=m["mark_ms"]
print("moves with a known layer:",len(moves),"in",len({m[0] for m in moves}),"games")
after=float(sys.argv[1]) if len(sys.argv)>1 else 3.0
rows_out=[]
for run,m,truth,start,prev,layers in moves:
    pc=per_cam_scores(run,m["mark_ms"],prev,start,after)
    rows_out.append((run,m["seq"],truth,pc))
def layer_of(sc):
    if not sc: return None
    (l,s),z=max(sc.items(),key=lambda kv:kv[1]); return l+1
def comb(pc,how):
    a,b=pc["iphone"],pc["android"]
    if how=="iphone": return layer_of(a)
    if how=="android": return layer_of(b)
    keys=set(a or {})|set(b or {})
    if not keys: return None
    def get(sc,k,d=-5.0): return (sc or {}).get(k,d)
    if how=="mean": sc={k:(get(a,k)+get(b,k))/2 for k in keys}
    elif how=="max": sc={k:max(get(a,k),get(b,k)) for k in keys}
    elif how=="weighted":
        pa=max((a or {0:0}).values()); pb=max((b or {0:0}).values())
        wa,wb=max(pa,1.0),max(pb,1.0)
        sc={k:(wa*get(a,k)+wb*get(b,k))/(wa+wb) for k in keys}
    elif how=="sqrtprod":     # both must see it; strong where both agree, otherwise falls back to the stronger one
        sc={k:(max(get(a,k),0)*max(get(b,k),0))**0.5 if min(get(a,k),get(b,k))>0 else max(get(a,k),get(b,k))*0.5 for k in keys}
    (l,s),z=max(sc.items(),key=lambda kv:kv[1]); return l+1
print("after window",after,"s")
for how in ("mean","iphone","android","max","weighted","sqrtprod"):
    ex=w1=n=0
    for run,seq,truth,pc in rows_out:
        p=comb(pc,how)
        if p is None: continue
        n+=1; ex+= (p==truth); w1+= abs(p-truth)<=1
    print(f"  {how:9s} exact {ex}/{n} = {100*ex/max(n,1):.0f}%   within one layer {w1}/{n} = {100*w1/max(n,1):.0f}%")
json.dump([(r,s,t,{c:({f"{k[0]},{k[1]}":v for k,v in (sc or {}).items()}) for c,sc in pc.items()}) for r,s,t,pc in rows_out],open(os.path.join(ROOT,"benchmarks/finder_variants/variant_scores.json"),"w"))
