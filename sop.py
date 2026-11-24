"""
Run the retraining flow in its fixed order, stopping at the first stage that fails:

    python sop.py --data-dir data --csv-left-slot left,Down=right --seeds 0 1 2 3 4

Stages: preflight.py's checks, then run_experiments.py. Any other option is passed on to
run_experiments.py (--seeds, --only, --out-dir is used by both).

--csv-left-slot has no default and is never taken from check_data.py's suggestion: a wrong
value trains every sample on the other hand's label without any error, so a person confirms
it from check_data.py's preview images first (LOCAL_SETUP.md). check_data.py is therefore not
a stage here.
"""
import argparse
import os
import subprocess
import sys

from preflight import print_findings, run_checks

HERE = os.path.dirname(os.path.abspath(__file__))


def stage_preflight(args, extra):
    errors, warnings, info = run_checks(args.data_dir, args.out_dir)
    print_findings(errors, warnings, info)
    return not errors


def stage_experiments(args, extra):
    # preflight just ran (or was skipped on purpose), so run_experiments does not repeat it.
    command = [sys.executable, os.path.join(HERE, "run_experiments.py"), "--data-dir", args.data_dir,
               "--csv-left-slot", args.csv_left_slot, "--out-dir", args.out_dir, "--skip-preflight", *extra]
    return subprocess.run(command).returncode == 0


STAGES = [("preflight", stage_preflight), ("experiments", stage_experiments)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--csv-left-slot", required=True, help="as for train.py; confirm it with check_data.py first")
    parser.add_argument("--out-dir", default="experiments")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="skip the preflight stage (e.g. to train on only some of the videos)")
    args, extra = parser.parse_known_args(argv)

    for name, stage in STAGES:
        if name == "preflight" and args.skip_preflight:
            print(f"[sop] {name}: skipped")
            continue
        print(f"[sop] {name}")
        if not stage(args, extra):
            print(f"[sop] stopped: stage '{name}' failed, later stages did not run.")
            return 1
    print("[sop] all stages finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
