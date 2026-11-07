"""CPU check of gate_eval.py's verdicts on synthetic paired-seed trajectories.

Locks the two properties the statistical gate relies on: a candidate whose
trajectories match the baseline to within float noise is judged PASS, and one
shifted by 0.05 at every step is judged INCORRECT. Input lines use the same
"<tag> <arm> s<seed> TRAJ {...}" format that multihost_split.py --traj produces.

Usage (from the repo root, stdlib only):  python -m tpu_smoke.test_gate_eval
"""
import json
import os
import random
import subprocess
import sys
import tempfile

STEPS, SEEDS = 120, (0, 1, 2)


def traj(seed, shift, jitter):
    r = random.Random(seed)
    offset = r.gauss(0, 0.01)                      # seed-to-seed spread sets delta
    noise = random.Random(1000 + seed)
    out = []
    for t in range(STEPS):
        loss = 2.0 - 0.004 * t + offset + r.gauss(0, 0.01)
        out.append([loss + shift + noise.gauss(0, jitter), None])
    return out


def main():
    arms = {"up": (0.0, 0.0), "same": (0.0, 1e-5), "shift": (0.05, 1e-5)}
    lines = []
    for arm, (shift, jitter) in arms.items():
        for s in SEEDS:
            body = json.dumps(dict(seed=s, traj=traj(s, shift, jitter)))
            lines.append(f"host0 {arm} s{s} TRAJ {body}\n")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.writelines(lines)
        path = f.name
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        res = subprocess.run([sys.executable, "-m", "tpu_smoke.gate_eval", path],
                             cwd=root, capture_output=True, text=True, check=True)
    finally:
        os.unlink(path)
    print(res.stdout, end="")
    verdict = {ln.split()[0]: ln.split()[2] for ln in res.stdout.splitlines()
               if ln.split() and ln.split()[0] in arms}
    assert verdict.get("same") == "PASS", verdict
    assert verdict.get("shift") == "INCORRECT", verdict
    print("PASS gate_eval: matching candidate PASS, shifted candidate INCORRECT")


if __name__ == "__main__":
    main()
