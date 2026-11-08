import glob,json,os,sys,cv2,numpy as np
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0,ROOT+"/tracker"); os.chdir(ROOT)
import face_crops as fc, move_infer as mi
run="dataset/"+sys.argv[1]; L=int(sys.argv[2])
recs=[json.loads(l) for l in open(run+"/labels.jsonl")]
marks=[r for r in recs if "mark_ms" in r]; truth={r["seq"]:r["correction"] for r in recs if "correction" in r}
rows=[json.loads(l) for l in open(run+"/index.jsonl")]; start=rows[0]["ts_ms"]
ts=sorted(int(os.path.basename(f).split("_")[0]) for f in glob.glob("iphone_saved/*_rgb.jpg"))
t0=min(t for t in ts if t>=start-2000)
img=cv2.imread(f"iphone_saved/{t0}_rgb.jpg"); meta=json.load(open(f"iphone_saved/{t0}_intrinsics.json"))
K=fc.iphone_K(meta,img.shape[1]); R,t,err=fc.solve_pose(img,K)
orig=mi._rows
def use(cams): mi._rows=lambda rd:[r for r in orig(rd) if r["cam"] in cams]
print("truth",truth,"typed",[(m["seq"],m["pos"]) for m in marks])
for rot in (0,90,180,270):
    slots=mi.slots_for_positions(R,t,K,L,rot)
    for name,cams in (("both",("iphone","android")),("iphone",("iphone",)),("android",("android",))):
        use(cams); prev=None; out=[]
        for m in marks:
            r=mi.infer(run,m["mark_ms"],prev,start,m["pos"],slots,list(range(L-1))); prev=m["mark_ms"]
            out.append((r["layer"],round(r["z"],1)) if r["ok"] else None)
        right=sum(1 for m,o in zip(marks,out) if o and o[0]==truth.get(m["seq"]))
        print(f"rot {rot:3d} {name:7s} right {right}/{len(marks)}",out)
