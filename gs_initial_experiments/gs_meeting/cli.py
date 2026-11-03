import argparse
import json
import time
from pathlib import Path
from .data import make_data, audit_data, write_json
from .engine import train, hardware
from .evaluate import evaluate
from .report import summarize
from .methods import VARIANTS


def main():
    p = argparse.ArgumentParser(description="Synthetic Gaussian meeting experiment starter")
    sub = p.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor"); doctor.add_argument("--out", default="machine.json")
    data = sub.add_parser("make-data"); data.add_argument("--config", required=True)
    audit = sub.add_parser("audit-data"); audit.add_argument("--config", required=True)
    tr = sub.add_parser("train"); tr.add_argument("--config", required=True)
    tr.add_argument("--variant", choices=VARIANTS, required=True); tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--device", default="auto"); tr.add_argument("--coarse"); tr.add_argument("--resume", action="store_true")
    tr.add_argument("--steps", type=int); tr.add_argument("--max-hours", type=float)
    ev = sub.add_parser("evaluate"); ev.add_argument("--config", required=True); ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--variant", choices=VARIANTS); ev.add_argument("--coarse"); ev.add_argument("--device", default="auto")
    ev.add_argument("--split", choices=["val", "test", "train"], default="test")
    ev.add_argument("--nfe", type=int, default=8); ev.add_argument("--samples", type=int, default=8)
    ev.add_argument("--lpips", action="store_true"); ev.add_argument("--overwrite", action="store_true")
    su = sub.add_parser("suite"); su.add_argument("--config", required=True); su.add_argument("--seed", type=int, default=0)
    su.add_argument("--device", default="auto"); su.add_argument("--nfe", type=int, nargs="+", default=[4, 8, 16])
    su.add_argument("--samples", type=int, default=8); su.add_argument("--split", choices=["val", "test"], default="val")
    su.add_argument("--extra-controls", action="store_true"); su.add_argument("--resume", action="store_true")
    su.add_argument("--max-hours", type=float, default=8.)
    re = sub.add_parser("summarize"); re.add_argument("--root", required=True); re.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = p.parse_args()
    if args.command=="doctor":
        h = hardware(); write_json(args.out, h); print(json.dumps(h, indent=2)); return
    if args.command=="summarize": summarize(args.root, args.split); return
    cfg = json.loads(Path(args.config).read_text())
    if args.command=="audit-data": print(json.dumps(audit_data(cfg), indent=2)); return
    if args.command=="make-data": make_data(cfg); print(cfg["data_root"]); return
    if args.command=="train":
        if args.steps is not None: cfg["steps"] = args.steps
        train(cfg, args.variant, args.seed, args.device, args.coarse, args.resume, args.max_hours); return
    if args.command=="evaluate":
        evaluate(cfg, args.checkpoint, args.split, args.nfe, args.samples, args.device,
                 args.coarse, args.variant, args.lpips, args.overwrite); return
    # Suite budget includes this invocation's training and evaluation, excluding preparation.
    # Enforced between training updates/evaluation jobs, not a preemptive GPU scheduler.
    root = Path(cfg["output_root"]); begin = time.perf_counter()
    audit = audit_data(cfg)
    for warning in audit["warnings"]: print("DATA AUDIT:", warning)
    if not cfg["debug_only"] and any(v["mean_disoccluded_foreground_fraction"]<=0 for v in audit["splits"].values()):
        raise ValueError("Pilot completion experiment has empty disocclusion masks; fix data first")
    coarse = root/f"A0_seed{args.seed}"/"checkpoint.pt"
    order = ["A0", "A2", "B0", "B2", "A1", "B1"]
    if args.extra_controls: order += ["B0_anchor", "B3_anchor"]
    for v in order:
        remaining = args.max_hours-(time.perf_counter()-begin)/3600
        if remaining<=0: print("Budget exhausted. Rerun suite --resume to continue."); return
        path = train(cfg, v, args.seed, args.device, str(coarse) if v.startswith("B") else None,
                     args.resume and (root/f"{v}_seed{args.seed}"/"checkpoint.pt").exists(), remaining)
        status = json.loads((path.parent/"manifest.json").read_text())["status"]
        if status!="completed": print("Paused before dependent comparisons; continue with --resume."); return
    for v in order+["B3"]:
        ck = root/f"{'A1' if v=='B3' else v}_seed{args.seed}"/"checkpoint.pt"
        for nfe in ([1] if v=="A0" else args.nfe):
            if (time.perf_counter()-begin)/3600>=args.max_hours:
                print("Budget exhausted during evaluation. Resume suite."); return
            folder = root/"evaluations"/f"{v}_seed{args.seed}_{args.split}_nfe{nfe}_k{args.samples}"
            if args.resume and (folder/"evaluation.json").exists(): continue
            evaluate(cfg, ck, args.split, nfe, args.samples, args.device,
                     str(coarse) if v.startswith("B") else None, "B3" if v=="B3" else None,
                     overwrite=args.resume)
    summarize(root, args.split)


if __name__ == "__main__": main()
