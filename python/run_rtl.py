"""Run XSim regression and optional routed core checks from a fresh local directory."""

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile

from hexsight.fixed_point_io import load_bundle, sha256, write_json
from hexsight.fixed_point_eval import verify_vectors

REPO = Path(__file__).resolve().parents[1]
RTL = ["hexsight_mac.v", "hexsight_relu.v", "hexsight_argmax.v", "hexsight_cnn.v"]


def run_tool(command, work, label, marker=None, allowed_codes=(0,)):
    print(f"Running {label}...", flush=True)
    result = subprocess.run([str(arg) for arg in command], cwd=work, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=1200)
    output = result.stdout + result.stderr
    (work / (label + ".console.txt")).write_text(output, encoding="utf-8")
    if result.returncode not in allowed_codes or (marker is not None and marker not in output):
        print(output[-16000:], flush=True)
        raise RuntimeError(f"{label} failed; inspect {work}")
    print(f"{label}: PASS", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sim", "all"), default="all")
    parser.add_argument("--vivado", type=Path, help="Vivado launcher (vivado.bat on Windows)")
    parser.add_argument("--work-dir", type=Path, help="New/empty local directory; defaults to a temporary directory")
    parser.add_argument("--results-dir", type=Path, default=REPO / "results/rtl-inference")
    args = parser.parse_args()
    settings_path = REPO / "local.settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    launcher = args.vivado or Path(settings.get("vivado", shutil.which("vivado") or "vivado"))
    if not launcher.is_file():
        raise ValueError("Supply --vivado with the installed Vivado launcher path")
    bin_dir = launcher.resolve().parent
    suffix = ".bat" if os.name == "nt" else ""
    for name in ("xvlog", "xelab", "xsim"):
        if not (bin_dir / (name + suffix)).is_file():
            raise ValueError(f"Missing Vivado tool: {name}")
    if args.results_dir.exists() and any(args.results_dir.iterdir()):
        raise ValueError("Use a new results directory; completed evidence is never overwritten")
    work = args.work_dir.resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix="hexsight-rtl-"))
    if os.name == "nt" and str(work).startswith("\\\\"):
        raise ValueError("Vivado batch launchers need a local drive working directory on Windows")
    if work.exists() and any(work.iterdir()):
        raise ValueError("Use a new or empty work directory")
    work.mkdir(parents=True, exist_ok=True)
    print(f"Local build directory: {work}", flush=True)

    model_dir, vector_dir = REPO / "model/cnn-fixed-point", REPO / "test_vectors/cnn-fixed-point"
    model, model_manifest = load_bundle(model_dir)
    vector_check = verify_vectors(vector_dir, model, model_dir)
    vectors = json.loads((vector_dir / "manifest.json").read_text())
    if len(vectors["cases"]) != 24:
        raise ValueError("The current regression testbench expects 24 frozen vectors")
    source_paths = ([REPO / "rtl" / name for name in RTL]
                    + [REPO / "tb/tb_fixed_arithmetic.sv", REPO / "tb/tb_hexsight_cnn.sv",
                       REPO / "scripts/synth_core.tcl", REPO / "constraints/core_ooc.xdc"])
    source_hashes = {}
    for source in source_paths:
        # Stable source identity across LF and CRLF checkouts.
        content = source.read_bytes().replace(b"\r\n", b"\n")
        (work / source.name).write_bytes(content)
        source_hashes[source.relative_to(REPO).as_posix()] = hashlib.sha256(content).hexdigest()
    for source in [*model_dir.glob("*.mem"), model_dir / "fixed_point.vh"]:
        shutil.copyfile(source, work / source.name)
    for source in vector_dir.glob("*.mem"):
        shutil.copyfile(source, work / ("vectors_" + source.name))

    version = run_tool([launcher, "-version"], work, "vivado_version", "vivado v", (0, 1)).strip()
    run_tool([bin_dir / ("xvlog" + suffix), "--sv", "-i", ".", *RTL,
              "tb_fixed_arithmetic.sv", "tb_hexsight_cnn.sv"], work, "compile")
    for top, snapshot, marker in (
        ("tb_fixed_arithmetic", "arithmetic_snapshot", "ARITHMETIC_PASS"),
        ("tb_hexsight_cnn", "cnn_snapshot", "CNN_PASS"),
    ):
        run_tool([bin_dir / ("xelab" + suffix), "--debug", "typical", "--mt", "2",
                  top, "-s", snapshot], work, top + "_elaborate")
        run_tool([bin_dir / ("xsim" + suffix), snapshot, "-runall"], work, top + "_simulate", marker)

    arithmetic = json.loads((work / "arithmetic_results.json").read_text())
    cnn = json.loads((work / "cnn_results.json").read_text())
    with (work / "rtl_cases.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not arithmetic["passed"] or not cnn["passed"] or len(rows) != 25:
        raise ValueError("Incomplete testbench results")
    if cnn["words_compared"] != 88625 or cnn["reset_abort_checks"] != 3:
        raise ValueError("Missing intermediate comparisons or reset checks")
    for row, expected_case in zip(rows, [*range(24), 0]):
        case = vectors["cases"][expected_case]
        if (int(row["case_index"]) != expected_case or int(row["prediction"]) != case["expected_prediction"]
                or int(row["cycles"]) != 69433 or int(row["conv_outputs"]) != 1568
                or int(row["pool_outputs"]) != 392 or int(row["scores"]) != 16):
            raise ValueError(f"Unexpected per-case RTL report: {row}")

    synthesis = None
    reports = ["arithmetic_results.json", "cnn_results.json", "rtl_cases.csv"]
    if args.mode == "all":
        synthesis_log = run_tool([launcher, "-mode", "batch", "-source", "synth_core.tcl", "-nolog", "-nojournal"],
                                 work, "synthesis", "CORE_INTERNAL_TIMING_PASS")
        warnings = [line for line in synthesis_log.splitlines() if "WARNING:" in line or "CRITICAL WARNING:" in line]
        (work / "tool_warnings.txt").write_text("\n".join(warnings) + "\n", encoding="utf-8", newline="\n")
        timing_check = (work / "check_timing.rpt").read_text()
        for name in ("no_clock", "unconstrained_internal_endpoints", "multiple_clock", "loops"):
            match = re.search(r"checking " + name + r" \((\d+)\)", timing_check)
            if not match or int(match.group(1)):
                raise ValueError(f"Incomplete internal timing constraints: {name}")
        synthesis = json.loads((work / "synthesis_results.json").read_text())
        reports += ["synthesis_results.json", "synthesis_utilization.rpt", "routed_utilization.rpt",
                    "routed_timing.rpt", "internal_timing.rpt", "routed_drc.rpt", "check_timing.rpt", "tool_warnings.txt"]

    args.results_dir.mkdir(parents=True, exist_ok=True)
    for name in reports:
        shutil.copyfile(work / name, args.results_dir / name)
    source_hashes["python/run_rtl.py"] = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    summary = {
        "vivado_version": version, "mode": args.mode,
        "model_manifest_sha256": sha256(model_dir / "manifest.json"),
        "vector_manifest_sha256": sha256(vector_dir / "manifest.json"),
        "source_sha256_lf": source_hashes,
        "source_hash_note": "Source CRLF is normalized to LF before hashing and staging.",
        "python_vector_verification": vector_check,
        "arithmetic": arithmetic, "cnn": cnn, "synthesis": synthesis,
        "report_sha256": {name: sha256(args.results_dir / name) for name in reports},
        "test_set_evaluated": False, "board_programmed": False,
        "timing_note": "Internal register-to-register timing passes. Unlocated OOC boundary timing is not closed; full reports and warnings are retained. No board timing or CPU speed claim.",
    }
    write_json(args.results_dir / "verification.json", summary)
    print(json.dumps({"arithmetic": arithmetic, "cnn": cnn, "synthesis": synthesis}, indent=2), flush=True)
    print(f"Saved evidence: {args.results_dir}", flush=True)


if __name__ == "__main__":
    main()
