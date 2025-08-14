import os
import subprocess
import json


def run_r_script(
    r_script_path: str,
    r_code: str,
    output_json: str,
    td: str,
    Rscript_path: str | None = None,
    timeout_seconds: int = 300,
) -> dict | None:
    """
    Run an R script and return the output.
    r_script_path: Path to the R script to run.
    r_code: R code to run.
    output_json: Path to the output JSON file.
    td: Path to the temporary directory.
    Rscript_path: Path to the Rscript executable.
    timeout_seconds: Timeout in seconds.
    """
    print(f"Writing R script to {r_script_path}")
    with open(r_script_path, "w", encoding="utf-8") as f:
        f.write(r_code)

    # Determine Rscript command
    chosen_rscript = None
    if Rscript_path:
        # Auto-correct if a path to R.exe was provided instead of Rscript.exe
        base = os.path.basename(Rscript_path).lower()
        if base in ("r.exe", "r"):
            candidate = os.path.join(os.path.dirname(Rscript_path), "Rscript.exe")
            if os.path.exists(candidate):
                chosen_rscript = candidate
            else:
                # Try x64 subfolder
                candidate2 = os.path.join(
                    os.path.dirname(Rscript_path), "x64", "Rscript.exe"
                )
                if os.path.exists(candidate2):
                    chosen_rscript = candidate2
        else:
            chosen_rscript = Rscript_path
    if not chosen_rscript:
        chosen_rscript = "Rscript"

    cmd = [chosen_rscript]
    # Use forward slashes to avoid quoting issues on Windows
    r_script_path_arg = r_script_path.replace("\\", "/")
    cmd += ["--vanilla", r_script_path_arg]

    print(f"Running Rscript with command: {cmd}")
    # Run with timeout and captured output; set cwd to temp dir to avoid permission issues
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_seconds, cwd=td
    )
    print(f"Rscript output: {proc.stdout}")
    print(f"Rscript error: {proc.stderr}")
    if proc.returncode != 0:
        print("Rscript failed:")
        print(proc.stderr)
        return None

    # Read JSON from file written by R
    if not os.path.exists(output_json):
        print("R did not produce output JSON. Stdout/Stderr:")
        print(proc.stdout)
        print(proc.stderr)
        return None

    with open(output_json, "r", encoding="utf-8") as jf:
        res = json.load(jf)

    if "error" in res:
        print(f"R fixest error: {res['error']}")
        return None

    return res