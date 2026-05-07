import os
import sys
import argparse
import subprocess
import re
import time
import glob
import shutil
import json

def get_project_root():
    """
    Get the project root directory assuming this script is in src/utils/
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up 2 levels: src/utils/ -> src/ -> root/
    return os.path.abspath(os.path.join(current_dir, "..", ".."))

def get_orfs_root(root_dir):
    return os.path.join(root_dir, "thirdparty", "OpenROAD-flow-scripts")


def pick_design_config_file(design_dir):
    candidates = [
        "config_eval.mk",
        "config_flow.mk",
        "config.mk",
    ]
    for name in candidates:
        p = os.path.join(design_dir, name)
        if os.path.isfile(p):
            return p
    return None


def parse_config_exports(config_path):
    exports = {}
    pattern = re.compile(r"^\s*export\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
    try:
        with open(config_path, "r") as f:
            for line in f:
                m = pattern.match(line)
                if m:
                    exports[m.group(1)] = m.group(2).strip()
    except Exception:
        pass
    return exports


def find_synth_seed_dir(orfs_root, platform, output_design, preferred_variant=None):
    base = os.path.join(orfs_root, "results", platform, output_design)
    if not os.path.isdir(base):
        return None

    candidates = []
    if preferred_variant:
        candidates.append(preferred_variant)
    candidates.extend(["eval", "base"])

    seen = set()
    ordered = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)

    for v in ordered:
        p = os.path.join(base, v)
        if os.path.isdir(p):
            if os.path.exists(os.path.join(p, "1_synth.v")) and os.path.exists(os.path.join(p, "1_synth.sdc")):
                return p

    # Fallback: any variant that has synthesis artifacts.
    for p in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.isdir(p):
            if os.path.exists(os.path.join(p, "1_synth.v")) and os.path.exists(os.path.join(p, "1_synth.sdc")):
                return p

    return None


def seed_synth_artifacts(orfs_root, flow_work_home, platform, output_design, variant):
    dst_dir = os.path.join(flow_work_home, "results", platform, output_design, variant)
    os.makedirs(dst_dir, exist_ok=True)

    dst_v = os.path.join(dst_dir, "1_synth.v")
    dst_sdc = os.path.join(dst_dir, "1_synth.sdc")
    if os.path.exists(dst_v) and os.path.exists(dst_sdc):
        return True

    seed_dir = find_synth_seed_dir(orfs_root, platform, output_design, preferred_variant=variant)
    if not seed_dir:
        return False

    src_v = os.path.join(seed_dir, "1_synth.v")
    src_sdc = os.path.join(seed_dir, "1_synth.sdc")
    try:
        shutil.copy2(src_v, dst_v)
        shutil.copy2(src_sdc, dst_sdc)
        return True
    except Exception:
        return False


def mark_isolated_workdir(flow_work_home):
    marker = os.path.join(flow_work_home, ".orfs_eval_isolated")
    try:
        with open(marker, "w") as f:
            f.write("isolated_orfs_eval_workdir\n")
    except Exception:
        pass


def cleanup_flow_work_home(flow_work_home):
    marker = os.path.join(flow_work_home, ".orfs_eval_isolated")
    if not os.path.exists(marker):
        print(f"Warning: skip cleanup because marker is missing: {flow_work_home}")
        return

    # Keep logs for post-run debugging; cleanup heavy intermediates only.
    for name in ["objects", "reports", "results"]:
        p = os.path.join(flow_work_home, name)
        if os.path.isdir(p):
            try:
                shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass


def discover_design_meta(orfs_root, platform):
    """
    Scan designs/<platform>/*/<config>.mk and return metadata entries.
    """
    base = os.path.join(orfs_root, "designs", platform)
    metas = []
    if not os.path.isdir(base):
        return metas

    for d in os.listdir(base):
        design_dir = os.path.join(base, d)
        cfg = pick_design_config_file(design_dir)
        if not cfg:
            continue
        exports = parse_config_exports(cfg)
        metas.append(
            {
                "dir_name": d,
                "config_path": cfg,
                "config_file": os.path.basename(cfg),
                "design_name": exports.get("DESIGN_NAME", d),
                "design_nickname": exports.get("DESIGN_NICKNAME", d),
            }
        )
    return metas


def resolve_design(orfs_root, platform, design_input):
    """
    Resolve user-provided design alias (short/full/dir) to config directory and output directory name.
    """
    metas = discover_design_meta(orfs_root, platform)
    if not metas:
        return None

    key = design_input.strip().lower()
    matched = []
    for m in metas:
        candidates = {
            m["dir_name"].lower(),
            str(m.get("design_name", "")).lower(),
            str(m.get("design_nickname", "")).lower(),
        }
        if key in candidates:
            matched.append(m)

    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        # Prefer exact dir name match if ambiguous.
        for m in matched:
            if m["dir_name"].lower() == key:
                return m
        return matched[0]

    return None


def parse_metrics_from_files(stdout_content, flow_work_home, orfs_root, platform, output_design, variant):
    """
    Parse the OpenROAD logs and reports to extract metrics.
    Prioritizes reading from generated report/log files. Fallbacks to stdout if needed.
    """
    metrics = {
        "GRT_WL": None,
        "DRT_WL": None,
        "DRC": None,
        "StdCellArea": None,
        "WNS": None,
        "TNS": None,
        "Power": None
    }  # type: dict[str, float | None]
    
    base = flow_work_home if flow_work_home else orfs_root
    report_dir = os.path.join(base, "reports", platform, output_design, variant)
    log_dir = os.path.join(base, "logs", platform, output_design, variant)
    
    finish_rpt_path = os.path.join(report_dir, "6_finish.rpt")
    grt_log_path = os.path.join(log_dir, "5_1_grt.log")
    drt_log_path = os.path.join(log_dir, "5_3_route.log") # Sometimes named 5_3_route.log or similar
    route_json_path = os.path.join(log_dir, "5_2_route.json")
    finish_json_path = os.path.join(log_dir, "6_report.json")

    # 0. Parse machine-friendly stage JSON first (preferred for stability).
    if os.path.exists(route_json_path):
        try:
            with open(route_json_path, "r") as f:
                route_json = json.load(f)
            if "detailedroute__route__wirelength" in route_json:
                metrics["DRT_WL"] = float(route_json["detailedroute__route__wirelength"])
            if "detailedroute__route__drc_errors" in route_json:
                metrics["DRC"] = float(route_json["detailedroute__route__drc_errors"])
        except Exception as e:
            print(f"Warning: Failed to read {route_json_path}: {e}")

    if os.path.exists(finish_json_path):
        try:
            with open(finish_json_path, "r") as f:
                finish_json = json.load(f)
            if "finish__timing__setup__tns" in finish_json:
                metrics["TNS"] = float(finish_json["finish__timing__setup__tns"])
            if "finish__timing__setup__ws" in finish_json:
                metrics["WNS"] = float(finish_json["finish__timing__setup__ws"])
            if "finish__power__total" in finish_json:
                metrics["Power"] = float(finish_json["finish__power__total"])
            if "finish__design__instance__area__stdcell" in finish_json:
                metrics["StdCellArea"] = float(finish_json["finish__design__instance__area__stdcell"])
        except Exception as e:
            print(f"Warning: Failed to read {finish_json_path}: {e}")
    
    # 1. Parse WNS, TNS, Power from 6_finish.rpt
    if os.path.exists(finish_rpt_path):
        try:
            with open(finish_rpt_path, 'r') as f:
                content = f.read()
                
                # Parse TNS
                # Format in file:
                # tns -3896.76
                # Use multiline and start-of-line anchor to avoid matching "finish report_tns"
                # and strict float regex to avoid matching separator lines like "-----------"
                match = re.search(r"^tns\s+(-?\d+\.?\d*)", content, re.MULTILINE)
                if match:
                    metrics["TNS"] = float(match.group(1))
                
                # Parse WNS
                # Format in file:
                # wns -1.38
                match = re.search(r"^wns\s+(-?\d+\.?\d*)", content, re.MULTILINE)
                if match:
                    metrics["WNS"] = float(match.group(1))

                # Parse Power (if available)
                # Look for "Total ... ... ... <TotalPower> ..." pattern in report_power section
                if "report_power" in content:
                    lines = content.splitlines()
                    in_power = False
                    for line in lines:
                        if "finish report_power" in line:
                            in_power = True
                        if in_power and line.strip().startswith("Total") and "%" in line:
                            parts = line.split()
                            # Example: Total 1.23e-01 4.56e-02 7.89e-03 1.76e-01 100.0%
                            # Target is typically the 4th value (Total Power) which is at index 4
                            if len(parts) >= 5:
                                try:
                                    metrics["Power"] = float(parts[4])
                                except ValueError:
                                    pass
                            break
        except Exception as e:
            print(f"Warning: Failed to read {finish_rpt_path}: {e}")

    # 2. Parse GRT WL from 5_1_grt.log
    if os.path.exists(grt_log_path):
        try:
            with open(grt_log_path, 'r') as f:
                content = f.read()
                # [INFO GRT-0018] Total wirelength: 9102046 um
                match = re.search(r"Total wirelength:\s+([\d\.]+)\s+um", content)
                if match:
                    metrics["GRT_WL"] = float(match.group(1))
        except Exception as e:
             print(f"Warning: Failed to read {grt_log_path}: {e}")
             
    # 3. Parse DRT WL from route log
    if os.path.exists(drt_log_path):
        try:
            with open(drt_log_path, 'r') as f:
                content = f.read()
                # [INFO DRT-0198] Complete detail routing.
                # Total wire length = 7762983 um.
                match = re.search(r"Total wire length =\s+([\d\.]+)\s+um", content)
                if match:
                    metrics["DRT_WL"] = float(match.group(1))
        except Exception as e:
             print(f"Warning: Failed to read {drt_log_path}: {e}")

    # Fallback alternative naming used in some ORFS flows.
    if metrics["DRT_WL"] is None:
        drt_log_alt_path = os.path.join(log_dir, "5_2_route.log")
        if os.path.exists(drt_log_alt_path):
            try:
                with open(drt_log_alt_path, 'r') as f:
                    content = f.read()
                    match = re.search(r"Total wire length =\s+([\d\.]+)\s+um", content)
                    if match:
                        metrics["DRT_WL"] = float(match.group(1))
            except Exception as e:
                print(f"Warning: Failed to read {drt_log_alt_path}: {e}")

    # Fallback to stdout if metrics are still None
    if metrics["GRT_WL"] is None:
        match = re.search(r"Total wirelength:\s+([\d\.]+)\s+um", stdout_content)
        if match: metrics["GRT_WL"] = float(match.group(1))

    if metrics["DRT_WL"] is None:
        match = re.search(r"Total wire length =\s+([\d\.]+)\s+um", stdout_content)
        if match: metrics["DRT_WL"] = float(match.group(1))
        
    # Fallback for WNS/TNS (if report file didn't exist or failed)
    if metrics["TNS"] is None:
        match = re.search(r"tns\s+([\-\d\.]+)", stdout_content)
        if match: metrics["TNS"] = float(match.group(1))
        
    if metrics["WNS"] is None:
        match = re.search(r"wns\s+([\-\d\.]+)", stdout_content)
        if match: metrics["WNS"] = float(match.group(1))

    return metrics

def run_evaluation(
    def_path,
    design,
    platform,
    variant,
    work_dir,
    root_dir,
    flow_work_home=None,
    cleanup_flow_work="success",
    resolve_only=False,
    reuse_existing_logs=False,
):
    start_time = time.time()
    orfs_root = get_orfs_root(root_dir)
    if not os.path.exists(orfs_root):
        print(f"Error: OpenROAD-flow-scripts not found at {orfs_root}")
        return None
        
    abs_def_path = os.path.abspath(def_path)
    if not os.path.exists(abs_def_path):
        print(f"Error: DEF file not found: {abs_def_path}")
        return None
        
    resolved = resolve_design(orfs_root, platform, design)
    if resolved is None:
        print(f"Error: cannot resolve design alias '{design}' under designs/{platform}")
        return None

    config_design = resolved["dir_name"]
    output_design = resolved["design_nickname"]
    design_name = resolved["design_name"]
    config_file = f"designs/{platform}/{config_design}/{resolved.get('config_file', 'config.mk')}"

    if flow_work_home is None:
        flow_work_home = os.path.join(work_dir, "orfs_work")
    flow_work_home = os.path.abspath(flow_work_home)
    os.makedirs(flow_work_home, exist_ok=True)
    mark_isolated_workdir(flow_work_home)

    print(f"Evaluating {design} ({platform}) using Make flow...")
    print(f"Resolved design: dir={config_design}, DESIGN_NAME={design_name}, DESIGN_NICKNAME={output_design}")
    print(f"DEF Path: {abs_def_path}")
    print(f"Variant: {variant}")

    if resolve_only:
        return {
            "resolved": True,
            "config_design": config_design,
            "design_name": design_name,
            "output_design": output_design,
        }

    if reuse_existing_logs:
        existing_metrics = parse_metrics_from_files(
            "",
            flow_work_home,
            orfs_root,
            platform,
            output_design,
            variant,
        )
        if any(
            existing_metrics.get(k) is not None
            for k in ["GRT_WL", "DRT_WL", "DRC", "WNS", "TNS", "Power", "StdCellArea"]
        ):
            existing_metrics["resolved_design_dir"] = config_design
            existing_metrics["resolved_output_design"] = output_design
            existing_metrics["flow_work_home"] = flow_work_home
            print("[INFO] Reusing existing ORFS logs/reports and skipping make execution.")
            print("\n=== Evaluation Results ===")
            for k, v in existing_metrics.items():
                print(f"{k}: {v}")
            report_file = os.path.join(work_dir, "metrics.txt")
            with open(report_file, "w") as rf:
                for k, v in existing_metrics.items():
                    rf.write(f"{k}: {v}\n")
            print(f"Report saved to {report_file}")
            return existing_metrics

    # Populate isolated WORK_HOME with synthesis seeds so run_wo_synth can start from floorplan.
    synth_seed_ok = seed_synth_artifacts(orfs_root, flow_work_home, platform, output_design, variant)
    if not synth_seed_ok:
        print(
            "Warning: synthesis seed (1_synth.v/.sdc) not found in ORFS cache; "
            "run_wo_synth may fail if no local synthesis artifacts exist."
        )

    # Build Make Command for Macro Placement (run_mp)
    # This step converts the input DEF into a macro placement file (macro_out)
    # which is then consumed by the main flow.
    
    cmd_mp = [
        "make",
        "run_mp",
        f"DESIGN_CONFIG={config_file}",
        f"MACRO_DEF={abs_def_path}",
        f"FLOW_VARIANT={variant}",
        f"WORK_HOME={flow_work_home}",
    ]
    
    log_path_mp = os.path.join(work_dir, "make_mp.log")
    print(f"Running Macro Placement: {' '.join(cmd_mp)}")
    print(f"Logging to: {log_path_mp}")

    try:
        with open(log_path_mp, "w") as log_file:
            process = subprocess.Popen(
                cmd_mp,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Merge stderr to stdout
                text=True,
                cwd=orfs_root
            )
            
            full_output = []
            if process.stdout:
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break
                    log_file.write(line)
                    full_output.append(line)
            
            process.wait()
            
            if process.returncode != 0:
                print("Error: Make run_mp command failed.")
                print("Last 20 lines of log:")
                for l in full_output[-20:]:
                    print(l.strip())
                return None
                
    except Exception as e:
        print(f"Exception running Make run_mp: {e}")
        return None

    # Build Make Command for Main Flow (run_wo_synth)
    # We execute: make run_wo_synth DESIGN_CONFIG=... MACRO_DEF=... FLOW_VARIANT=...
    # Note: MACRO_DEF must be passed as an absolute path or relative to ORFS root.
    # We use absolute path to be safe.
    
    cmd = [
        "make",
        "run_wo_synth",
        f"DESIGN_CONFIG={config_file}", # Config file path relative to ORFS root
        f"MACRO_DEF={abs_def_path}",
        f"FLOW_VARIANT={variant}",
        f"WORK_HOME={flow_work_home}",
    ]
    
    
    log_path = os.path.join(work_dir, "make.log")
    print(f"Running Flow: {' '.join(cmd)}")
    print(f"Logging to: {log_path}")
    
    try:
        with open(log_path, "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Merge stderr to stdout
                text=True,
                cwd=orfs_root
            )
            
            full_output = []
            if process.stdout:
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break
                    # We can print progress if needed, but Make is noisy
                    # print(line, end="") 
                    log_file.write(line)
                    full_output.append(line)
            
            process.wait()
            
            stdout_content = "".join(full_output)
            
            if process.returncode != 0:
                print("Error: Make command failed.")
                # print tail of log?
                print("Last 20 lines of log:")
                for l in full_output[-20:]:
                    print(l.strip())
                return None
            
            # Use file-based parsing
            metrics = parse_metrics_from_files(stdout_content, flow_work_home, orfs_root, platform, output_design, variant)
            
            end_time = time.time()
            runtime = end_time - start_time
            
            # Print metrics to stdout for caller
            if metrics:
                metrics['runtime'] = runtime
                metrics['resolved_design_dir'] = config_design
                metrics['resolved_output_design'] = output_design
                metrics['flow_work_home'] = flow_work_home
                print("\n=== Evaluation Results ===")
                for k, v in metrics.items():
                    print(f"{k}: {v}")
                    
                report_file = os.path.join(work_dir, "metrics.txt")
                with open(report_file, "w") as rf:
                    for k, v in metrics.items():
                        rf.write(f"{k}: {v}\n")
                print(f"Report saved to {report_file}")

                if cleanup_flow_work in {"success", "always"}:
                    cleanup_flow_work_home(flow_work_home)
            else:
                print("Warning: No metrics parsed from log.")
                if cleanup_flow_work == "always":
                    cleanup_flow_work_home(flow_work_home)
                
            return metrics

    except Exception as e:
        print(f"Exception running Make flow: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate OpenROAD Metrics for a DEF file using Make flow")
    parser.add_argument("--def_path", required=True, help="Path to the DEF file")
    parser.add_argument("--design", required=True, help="Design alias/name/dir (e.g. ariane133, bp, bp_be_top)")
    parser.add_argument("--platform", default="nangate45", help="Platform Name (e.g. nangate45)")
    parser.add_argument("--variant", default="eval", help="Flow Variant Name (default: eval)")
    parser.add_argument("--work_dir", default=None, help="Working Directory for logs/reports")
    parser.add_argument("--flow_work_home", default=None, help="Isolated ORFS WORK_HOME for this evaluation")
    parser.add_argument(
        "--cleanup_flow_work",
        default="success",
        choices=["never", "success", "always"],
        help="Cleanup isolated ORFS intermediates in flow_work_home",
    )
    parser.add_argument(
        "--reuse_existing_logs",
        action="store_true",
        help="If existing ORFS logs/reports are found, parse and skip rerun",
    )
    parser.add_argument("--resolve_only", action="store_true", help="Only resolve design mapping, do not run make")
    
    args = parser.parse_args()
    
    root_dir = get_project_root()
    def_path = os.path.abspath(args.def_path)
    
    if args.work_dir:
        work_dir = args.work_dir
    else:
        # Default work dir in current directory
        def_dir = os.path.dirname(def_path)
        work_dir = os.path.join(def_dir, "eval_" + args.variant)
        
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        
    result = run_evaluation(
        def_path,
        args.design,
        args.platform,
        args.variant,
        work_dir,
        root_dir,
        flow_work_home=args.flow_work_home,
        cleanup_flow_work=args.cleanup_flow_work,
        resolve_only=args.resolve_only,
        reuse_existing_logs=args.reuse_existing_logs,
    )
    if not result:
        sys.exit(1)

if __name__ == "__main__":
    main()
