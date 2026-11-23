"""Export static, shareable C24 figures from preserved parent summary files."""
import argparse
import hashlib
import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLICIES=["PLAIN_STOP","PLAIN_ALWAYS","FAST_ONCE","FAST_BLOCK","OUTWARD_BLOCK","RESIDUAL"]
GATES=["FAST_ONCE","FAST_BLOCK","OUTWARD_BLOCK"]
COLORS={"PLAIN_STOP":"#929aa5","PLAIN_ALWAYS":"#27394f","FAST_ONCE":"#008a7a",
        "FAST_BLOCK":"#db7a25","OUTWARD_BLOCK":"#8752aa","RESIDUAL":"#3b74b7"}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split",choices=["TRAIN","CAL","TEST","UNIT"],default="TEST")
    ap.add_argument("--evidence-dir",type=pathlib.Path)
    args=ap.parse_args()
    doc=pathlib.Path(__file__).resolve().parents[1];ev=args.evidence_dir or doc/"evidence"
    source=ev/f"C24_{args.split}_SUMMARY.json";data=json.loads(source.read_text())
    png=ev/f"C24_{args.split}_COST_QUALITY.png";pdf=png.with_suffix(".pdf")
    manifest=ev/f"C24_{args.split}_FIGURE.json"
    if any(p.exists() for p in [png,pdf,manifest]):raise FileExistsError("preserved figure exists")
    ss=data["summaries"];cap=data["primary_additional"]
    primary=next(s for s in ss if s["additional"]==cap)
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
        "pdf.fonttype":42,"ps.fonttype":42})
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    ax=axes[0,0]
    for name in GATES:
        ax.plot([s["additional"] for s in ss],
            [s["policies"]["PLAIN_ALWAYS"]["mean_parent_median_us"]/s["policies"][name]["mean_parent_median_us"] for s in ss],
            marker="o",label=name,color=COLORS[name])
    ax.axhline(1,color="black",lw=.8,ls="--");ax.set_xscale("log",base=2)
    ax.set_xticks([s["additional"] for s in ss],[s["additional"] for s in ss])
    ax.set_xlabel("Additional CG cap per action");ax.set_ylabel("PLAIN_ALWAYS time / policy time")
    ax.set_title("A. End-to-end speed ratio");ax.legend(fontsize=8)

    ax=axes[0,1];sub=[r for r in primary["subgroups"] if r["field"]=="side"]
    x=np.arange(len(sub));width=.13
    for i,name in enumerate(POLICIES):
        ax.bar(x+(i-2.5)*width,[r["policies"][name]["mean_parent_median_us"] for r in sub],
            width*.94,color=COLORS[name],label=name)
    ax.set_yscale("log");ax.set_xticks(x,[str(r["value"]) for r in sub])
    ax.set_xlabel("Grid side m (unknowns = m²)");ax.set_ylabel("Mean parent median time (µs)")
    ax.set_title(f"B. Size dependence, cap {cap}")

    ax=axes[0,2];x=np.arange(len(ss))
    for i,name in enumerate(POLICIES):
        ax.bar(x+(i-2.5)*width,[s["policies"][name]["wrong_decision_parents"] for s in ss],
            width*.94,color=COLORS[name],label=name)
    ax.set_xticks(x,[s["additional"] for s in ss]);ax.set_xlabel("Additional CG cap per action")
    ax.set_ylabel(f"Parents with positive exact regret / {data['unique_parents']}")
    ax.set_title("C. Exact wrong-decision count");ax.legend(fontsize=7,ncol=2)

    ax=axes[1,0];x=np.arange(len(GATES))
    cert=[primary["policies"][p]["certified_parents"] for p in GATES]
    early=[primary["policies"][p]["actual_early_stop_parents"] for p in GATES]
    ax.bar(x-.18,cert,.34,color="#3d596f",label="Certified")
    ax.bar(x+.18,early,.34,color="#df9345",label="Fewer actual CG calls")
    ax.set_xticks(x,GATES,rotation=15);ax.set_ylabel("Parent count")
    ax.set_title(f"D. Certificate versus work saved, cap {cap}");ax.legend(fontsize=8)

    ax=axes[1,1];x=np.arange(len(POLICIES));width=.38
    ax.bar(x-width/2,[primary["policies"][p]["mean_parent_median_us"] for p in POLICIES],
        width,color=[COLORS[p] for p in POLICIES],label="Total elapsed")
    ax.bar(x+width/2,[primary["policies"][p]["mean_bounds_us"] for p in POLICIES],
        width,color=[COLORS[p] for p in POLICIES],alpha=.45,hatch="//",label="All readouts / enclosures")
    ax.set_xticks(x,POLICIES,rotation=35,ha="right");ax.set_ylabel("Mean parent median time (µs)")
    ax.set_title(f"E. Total and evaluator costs, cap {cap}");ax.legend(fontsize=8)

    ax=axes[1,2]
    for name in GATES:
        priced=primary["policies"][name]["priced"]
        means=np.asarray([r["mean_net_difference_vs_PLAIN_ALWAYS"] for r in priced])
        lows=np.asarray([r["paired_parent_bootstrap_net_difference_vs_PLAIN_ALWAYS_95"][0] for r in priced])
        highs=np.asarray([r["paired_parent_bootstrap_net_difference_vs_PLAIN_ALWAYS_95"][1] for r in priced])
        xx=np.asarray([r["lambda_utility_per_us"] for r in priced])
        ax.plot(xx,means,marker="o",color=COLORS[name],label=name)
        ax.fill_between(xx,lows,highs,color=COLORS[name],alpha=.12)
    ax.axhline(0,color="black",lw=.8);ax.set_xscale("symlog",linthresh=1e-8)
    ax.set_xlabel("Frozen price λ (utility / µs)");ax.set_ylabel("Net utility difference vs PLAIN_ALWAYS")
    ax.set_title(f"F. Paired parent bootstrap 95%, cap {cap}")
    fig.suptitle(f"Quartz C24 · {args.split} · {data['unique_parents']} parents · cheap decision certificates",fontsize=14)
    fig.savefig(png,dpi=180);fig.savefig(pdf);plt.close(fig)
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    record={"source":source.name,"source_sha256":sha(source),"plotter_sha256":sha(pathlib.Path(__file__)),
        "scope":"static summaries of preserved measurements; uncertainty is paired across independent parents",
        "matplotlib":matplotlib.__version__,"artifacts":{p.name:{"sha256":sha(p),"bytes":p.stat().st_size} for p in [png,pdf]}}
    with manifest.open("x") as out:out.write(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record))
if __name__=="__main__":
    main()
