#!/usr/bin/env python3
"""Run every TensorMcEliece harness, verify, and summarize the throughput.

Usage:  python3 benchmark_all.py [--out DIR] [--timeout SEC] [--skip NAME ...]

Build first with `make` (see README.md). Results go to
results/<gpu>_<timestamp>/ : raw/ (full output of every binary), env.txt
(GPU, driver, clocks, toolchain), summary.txt and summary.json.
"""
import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BATCHES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

# (name, binary relative to ROOT). McEliece348864 encapsulation engines, incl. the full INT8 family.
BINARIES = [
    ("bmma", "mceliece-gpu-bmma/mceliece-gpu-bmma"),
]
# Multi-parameter harness (one binary per parameter set, generated from libmceliece vec).
PARAM_SETS = ["348864", "460896", "6688128", "6960119", "8192128"]
PARAM_BINARIES = [("params-%s" % ps, "mceliece-gpu-params/mceliece-gpu-params-%s" % ps) for ps in PARAM_SETS]
PARAM_ENC = ["CC", "BMMA", "BS2-M4", "BS2-BLK3"]
PARAM_DEC = ["CC-PKE", "TC-PKE", "CC-KEM", "TC-KEM"]
INT8_COLUMNS = ["INT8 WMMA", "INT8 BS2", "BS2-M4", "BS2-W4", "BS2-BLK"]


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return "(%s failed: %s)" % (cmd, exc)


def gpu_info():
    q = sh("nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit "
           "--format=csv,noheader")
    name = q.split(",")[0].strip() if q and "failed" not in q else "unknown-gpu"
    return name, q


def snapshot_env(path):
    with open(path, "w") as f:
        for title, cmd in [
            ("date", "date -Is"),
            ("host", "uname -a"),
            ("cpu", "lscpu | grep -E 'Model name|^CPU\\(s\\)'"),
            ("nvcc", "nvcc --version | tail -2"),
            ("gpu", "nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.sm,clocks.max.sm,"
                    "clocks.mem,power.draw,power.limit,temperature.gpu,pstate --format=csv"),
            ("clocks", "nvidia-smi -q -d CLOCK"),
        ]:
            f.write("### %s\n%s\n\n" % (title, sh(cmd)))


def run(path, raw_dir, timeout, mirror_dir=None):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        print("  %-62s MISSING (not built?)" % path)
        return ""
    print("  %-62s ... " % path, end="", flush=True)
    t0 = datetime.datetime.now()
    try:
        res = subprocess.run([full], capture_output=True, text=True, timeout=timeout,
                             cwd=os.path.dirname(full))
        out = res.stdout + res.stderr
        rc = res.returncode
    except Exception as exc:  # noqa: BLE001
        out, rc = "FAILED: %s\n" % exc, -1
    dt = (datetime.datetime.now() - t0).total_seconds()
    name = os.path.basename(path) + ".txt"
    with open(os.path.join(raw_dir, name), "w") as f:
        f.write(out)
    if mirror_dir:                       # flat copy, the layout of the older runs
        os.makedirs(mirror_dir, exist_ok=True)
        with open(os.path.join(mirror_dir, name), "w") as f:
            f.write(out)
    print("%s (%.0f s)" % ("DONE" if rc == 0 else "EXIT %d" % rc, dt))
    return out


def parse_bmma(out, section):
    """section: "PKE" (pre-generated e) or "KEM" (fixed-weight sampling timed)."""
    bmma, fam, verified, seen = {}, {c: {} for c in INT8_COLUMNS}, True, False
    marker = "[%s]" % section
    if marker in out:
        out = out.split(marker, 1)[1]
        out = out.split("\n[", 1)[0]
    for line in out.splitlines():
        f = [c.strip() for c in line.split("|")]
        if len(f) != 8 or not f[0].isdigit():
            continue
        seen = True
        n = int(f[0])
        vals = []
        for c in f[1:7]:
            try:
                vals.append(float(c))
            except ValueError:
                vals.append(0.0)
        bmma[n] = vals[0]
        for c, v in zip(INT8_COLUMNS, vals[1:]):
            if v > 0:
                fam[c][n] = v
        if f[7] != "PASSED":
            verified = False
    return bmma, fam, verified and seen


def parse_params(out):
    """sections [ENCAP-PKE], [ENCAP-KEM], [DECAP]: rows 'N | v | v ... | PASSED'"""
    res = {"encap_pke": {c: {} for c in PARAM_ENC}, "encap_kem": {c: {} for c in PARAM_ENC},
           "decap": {c: {} for c in PARAM_DEC}, "breakdown": {}}
    section = None
    for line in out.splitlines():
        m = re.match(r"^\[(ENCAP-PKE|ENCAP-KEM|DECAP|BREAKDOWN)\]", line)
        if m:
            section = m.group(1)
            continue
        if section == "BREAKDOWN":
            mb = re.match(r"^\s+([\w/]+)\s*:\s*([\d.]+) ms", line)
            if mb:
                res["breakdown"][mb.group(1)] = float(mb.group(2))
            continue
        f = [c.strip() for c in line.split("|")]
        if section is None or len(f) < 3 or not f[0].isdigit():
            continue
        n = int(f[0])
        cols = PARAM_DEC if section == "DECAP" else PARAM_ENC
        key = {"ENCAP-PKE": "encap_pke", "ENCAP-KEM": "encap_kem", "DECAP": "decap"}[section]
        for c, v in zip(cols, f[1:1 + len(cols)]):
            res[key][c][n] = float(v)
    ok = "ALL CHECKS PASSED" in out and bool(res["decap"]["TC-KEM"])
    return res, ok


def table(lines, title, columns):
    lines.append("")
    lines.append("=" * 96)
    lines.append(title.center(96))
    lines.append("=" * 96)
    head = "%8s | " % "Batch N" + " | ".join("%11s" % c for c, _ in columns)
    lines.append(head)
    lines.append("-" * len(head))
    best = {c: (0, 0.0) for c, _ in columns}
    for n in BATCHES:
        if all(data.get(n) is None for _, data in columns):
            continue
        cells = []
        for c, data in columns:
            v = data.get(n)
            if v is None:
                cells.append("%11s" % "N/A")
            else:
                cells.append("%11.0f" % v)
                if v > best[c][1]:
                    best[c] = (n, v)
        lines.append("%8d | " % n + " | ".join(cells))
    lines.append("-" * len(head))
    lines.append("%8s | " % "PEAK" + " | ".join("%11.0f" % best[c][1] for c, _ in columns))
    lines.append("%8s | " % "at N" + " | ".join("%11s" % best[c][0] for c, _ in columns))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="output directory (default results/<gpu>_<timestamp>)")
    ap.add_argument("--timeout", type=int, default=7200, help="per-binary timeout in seconds")
    ap.add_argument("--skip", nargs="*", default=[], help="harness names to skip: " +
                    " ".join(n for n, _ in BINARIES + PARAM_BINARIES))
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these harnesses (e.g. --only params for all parameter sets)")
    ap.add_argument("--raw-mirror", default="", metavar="DIR",
                    help="also write every binary's output flat into DIR (relative to the suite; "
                         "default: disabled)")
    args = ap.parse_args()

    gpu, gpu_line = gpu_info()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", gpu).strip("-")
    out_dir = args.out or os.path.join(ROOT, "results", "%s_%s" % (slug, stamp))
    raw_dir = os.path.join(out_dir, "raw")
    mirror_dir = os.path.join(ROOT, args.raw_mirror) if args.raw_mirror else None
    os.makedirs(raw_dir, exist_ok=True)
    snapshot_env(os.path.join(out_dir, "env.txt"))

    print("=" * 96)
    print("TensorMcEliece GPU benchmark suite".center(96))
    print(("GPU: " + gpu_line).center(96))
    print(("results -> " + os.path.relpath(out_dir, ROOT)).center(96))
    print("=" * 96)

    all_names = [n for n, _ in BINARIES + PARAM_BINARIES]
    if args.only is not None:
        keep = set()
        for o in args.only:
            keep |= {n for n in all_names if n == o or (o == "params" and n.startswith("params-"))}
        args.skip += [n for n in all_names if n not in keep]
    status = []
    raw = {}
    for name, path in BINARIES + PARAM_BINARIES:
        if name in args.skip:
            print("  %-62s SKIPPED" % path)
            continue
        raw[name] = run(path, raw_dir, args.timeout, mirror_dir)

    sm = None
    for out in raw.values():
        m = re.search(r"SMs: (\d+)", out)
        if m:
            sm = int(m.group(1))
            break

    enc_cols, fam_cols, kfam_cols, penc_cols = [], [], [], []
    if raw.get("bmma"):
        b, fam, v = parse_bmma(raw["bmma"], "PKE")
        kb, kfam, kv = parse_bmma(raw["bmma"], "KEM")
        enc_cols.append(("bmma", kb))
        penc_cols.append(("bmma", b))
        enc_cols.append(("bs2-m4", kfam["BS2-M4"]))
        penc_cols.append(("bs2-m4", fam["BS2-M4"]))
        for c in INT8_COLUMNS:
            fam_cols.append((c, fam[c]))
            kfam_cols.append((c, kfam[c]))
        status.append(("bmma+int8-encap", v and kv))

    params = {}
    for name, _ in PARAM_BINARIES:
        if raw.get(name):
            res, v = parse_params(raw[name])
            params[name[len("params-"):]] = res
            status.append((name, v))

    lines = ["GPU: %s" % gpu_line, "SMs: %s" % sm, "date: %s" % stamp]
    if penc_cols:
        table(lines, "ENCAPSULATION, PKE setting: pre-generated e (ops/s)", penc_cols)
    if enc_cols:
        table(lines, "ENCAPSULATION, KEM setting: incl. fixed-weight sampling (ops/s)", enc_cols)
    if fam_cols:
        table(lines, "INT8 ENCAPSULATION FAMILY, PKE setting (ops/s)", fam_cols)
    if kfam_cols:
        table(lines, "INT8 ENCAPSULATION FAMILY, KEM setting (ops/s)", kfam_cols)
    if params:
        def peak(d):
            return max(d.values()) if d else 0.0
        for key, cols, title in [("encap_pke", PARAM_ENC, "ALL PARAMETER SETS: ENCAPSULATION PEAKS, PKE setting (ops/s)"),
                                 ("encap_kem", PARAM_ENC, "ALL PARAMETER SETS: ENCAPSULATION PEAKS, KEM setting (ops/s)"),
                                 ("decap", PARAM_DEC, "ALL PARAMETER SETS: DECAPSULATION PEAKS (ops/s)")]:
            lines.append("")
            lines.append("=" * 96)
            lines.append(title.center(96))
            lines.append("=" * 96)
            head = "%10s | " % "Set" + " | ".join("%11s" % c for c in cols)
            lines.append(head)
            lines.append("-" * len(head))
            for ps in PARAM_SETS:
                if ps in params:
                    lines.append("%10s | " % ps + " | ".join("%11.0f" % peak(params[ps][key][c]) for c in cols))
        for ps in PARAM_SETS:
            if ps in params:
                for key, cols, title in [("encap_pke", PARAM_ENC, "ENCAPSULATION, PKE setting"),
                                         ("encap_kem", PARAM_ENC, "ENCAPSULATION, KEM setting"),
                                         ("decap", PARAM_DEC, "DECAPSULATION")]:
                    table(lines, "mceliece%s: %s (ops/s)" % (ps, title), [(c, params[ps][key][c]) for c in cols])

    lines.append("")
    lines.append("=" * 96)
    for name, ok in status:
        lines.append("  %-16s %s" % (name, "PASSED" if ok else "FAILED"))
    lines.append("=" * 96)
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "gpu": gpu_line, "sm_count": sm, "timestamp": stamp,
            "encap": {c: d for c, d in enc_cols},
            "encap_pke": {c: d for c, d in penc_cols},
            "int8_family": {c: d for c, d in fam_cols},
            "int8_family_kem": {c: d for c, d in kfam_cols},
            "params": params,
            "status": {n: ok for n, ok in status},
        }, f, indent=1)
    latest = os.path.join(ROOT, "results", "latest")
    if args.out is None:
        try:
            if os.path.islink(latest) or os.path.exists(latest):
                os.remove(latest)
            os.symlink(os.path.basename(out_dir), latest)
        except OSError:
            shutil.copytree(out_dir, latest, dirs_exist_ok=True)
    return 0 if status and all(ok for _, ok in status) else 1


if __name__ == "__main__":
    sys.exit(main())
