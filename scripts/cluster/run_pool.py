#!/usr/bin/env python
"""Run scripts/cluster/jobs.tsv on one machine without a scheduler, never using
more than --cores cores at once.

  nohup python scripts/cluster/run_pool.py --cores 45 > logs/pool.log 2>&1 &

* Each job declares how many worker processes it uses; the pool starts a job only
  when that many cores are free, highest priority first and, within a priority,
  the most expensive first; smaller jobs backfill the remaining cores.
* A job whose summary already exists (results/<tag>_summary.json, or the output
  file of the E4 scripts) is skipped, so the pool can be stopped and restarted.
* Logs: logs/<experiment>/<tag>.log; progress: logs/pool_status.tsv.
* --only E1,E4 restricts the run to some experiments; --max-priority 5 skips the
  optional tier (n = 768 and 1024 on K_n).
"""
import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--cores", type=int, default=45)
ap.add_argument("--jobs", default="scripts/cluster/jobs.tsv")
ap.add_argument("--only", default=None, help="comma-separated experiments, e.g. E1,E4")
ap.add_argument("--max-priority", type=int, default=99)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

rows = []
for line in Path(a.jobs).read_text().splitlines()[1:]:
    pri, w, cost, tag, cmd = line.split("\t")
    rows.append(dict(pri=int(pri), workers=int(w), cost=float(cost), tag=tag, cmd=cmd))
if a.only:
    keep = set(a.only.split(","))
    rows = [r for r in rows if r["tag"].split("/")[0] in keep]
rows = [r for r in rows if r["pri"] <= a.max_priority]


def done(r):
    exp, tag = r["tag"].split("/", 1)
    if exp == "E4":
        parts = shlex.split(r["cmd"])
        f = Path(parts[parts.index("--out") + 1])
        if not f.exists():
            return False
        return tag != "ansatz_bound" or "\n4096\t" in f.read_text()
    return (Path("results") / exp / f"{tag}_summary.json").exists()


todo = [r for r in rows if not done(r)]
for r in todo:
    r["workers"] = min(r["workers"], a.cores)
todo.sort(key=lambda r: (r["pri"], -r["cost"]))
print(f"{len(rows)} jobs selected, {len(rows) - len(todo)} already done, {len(todo)} to run "
      f"on {a.cores} cores", flush=True)
if a.dry_run:
    for r in todo:
        print(r["pri"], r["workers"], f"{r['cost']:.2f}", r["tag"])
    sys.exit()

env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
           VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
running = {}
t0 = time.time()
failed = []


def status():
    with open("logs/pool_status.tsv", "w") as f:
        f.write("state\ttag\tworkers\tstarted_s\n")
        for p, (r, ts, _) in running.items():
            f.write(f"running\t{r['tag']}\t{r['workers']}\t{ts - t0:.0f}\n")
        for r in todo:
            f.write(f"queued\t{r['tag']}\t{r['workers']}\t\n")
        for r in failed:
            f.write(f"failed\t{r['tag']}\t{r['workers']}\t\n")


Path("logs").mkdir(exist_ok=True)
while todo or running:
    used = sum(r["workers"] for r, _, _ in running.values())
    free = a.cores - used
    started = False
    min_pri = min((r["pri"] for r in todo), default=None)
    for r in list(todo):
        # strict priority, except that lower-priority jobs may backfill cores that
        # no job of the current priority can use
        if r["workers"] <= free and (r["pri"] == min_pri or
                                     all(x["workers"] > free for x in todo if x["pri"] == min_pri)):
            log = Path("logs") / f"{r['tag']}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            fh = open(log, "w")
            p = subprocess.Popen(shlex.split(r["cmd"]), stdout=fh, stderr=subprocess.STDOUT, env=env)
            running[p.pid] = (r, time.time(), (p, fh))
            todo.remove(r)
            free -= r["workers"]
            started = True
            print(f"[{(time.time() - t0) / 3600:6.2f} h] start {r['tag']} ({r['workers']} cores, "
                  f"{free} free, {len(todo)} queued)", flush=True)
    for pid, (r, ts, (p, fh)) in list(running.items()):
        if p.poll() is not None:
            fh.close()
            del running[pid]
            ok = p.returncode == 0
            if not ok:
                failed.append(r)
            print(f"[{(time.time() - t0) / 3600:6.2f} h] {'done' if ok else 'FAILED'} {r['tag']} "
                  f"after {(time.time() - ts) / 60:.1f} min", flush=True)
    status()
    if not started:
        time.sleep(5)
print(f"all done in {(time.time() - t0) / 3600:.2f} h; {len(failed)} failed", flush=True)
for r in failed:
    print("FAILED:", r["tag"], r["cmd"])
