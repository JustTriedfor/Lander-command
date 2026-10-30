"""The PID test pilot (src/sim/pid_pilot.py) flown like a training run: train.py hands over here when the brain to start from is "PID".
Nothing is trained. It flies the same environments at the chosen level with every rule on, writes episodes.csv and streams flights to the
live viewer exactly as a training run does, until the requested number of steps."""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

from src.sim.pid_pilot import PIDPilot
from src.training.env import make_env
from src.utils import config as C


def run_pid(args, name: str, run_dir: Path) -> None:
    if C.START_PROFILE != "spacex":
        sys.exit("The PID test pilot flies one lander: set 'What to fly' to One lander.")
    from src.training.train import EP_FIELDS
    run_dir.mkdir(parents=True, exist_ok=True)
    live_dir = run_dir / "live"
    live = not args.no_live
    if live:
        shutil.rmtree(live_dir, ignore_errors=True)
        live_dir.mkdir(parents=True, exist_ok=True)
    level, stage = args.start_level, C.RULES_STAGE_MAX
    for fname, head, row in (("levels.csv", ["timesteps", "level", "reason", "success_rate"], [0, level, "start", ""]),
                             ("stages.csv", ["timesteps", "stage", "reason", "survival_rate"], [0, stage, "start", ""])):
        with open(run_dir / fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(head)
            w.writerow(row)
    C.MODEL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    (C.MODEL_CONFIG_DIR / f"{name}.json").write_text(json.dumps(dict(vars(args), pilot="PID", fuel_budget=C.FUEL_BUDGET, levels=C.LEVELS,
                                                                     overrides=C.OVERRIDES), indent=2))
    n = max(1, args.n_envs)
    envs, pilots = [make_env(level=level, record=live) for _ in range(n)], [PIDPilot() for _ in range(n)]
    for i, env in enumerate(envs):
        env.set_level(level)
        env.set_rules_stage(stage)
        env.reset(seed=args.seed + i)
    print(f"[init] PID test pilot (no network, nothing is trained): {n} worlds at level {level}, every rule on", flush=True)
    steps, live_n, last, paths, ep_w, soft, eps = 0, 0, {}, [], None, 0, 0
    with open(run_dir / "episodes.csv", "w", newline="") as ep_f:
        while steps < args.total_steps:
            for i, (env, pilot) in enumerate(zip(envs, pilots)):
                s = env.s
                _, _, term, trunc, info = env.step(pilot.action(s, env.wind.at(s.t, s.y)))
                steps += 1
                if not (term or trunc):
                    continue
                m = info.get("episode_metrics")
                if m is not None:
                    if ep_w is None:
                        ep_w = csv.DictWriter(ep_f, EP_FIELDS + sorted(k for k in m if k.startswith("term_")), extrasaction="ignore")
                        ep_w.writeheader()
                    ep_w.writerow({**m, "timesteps": steps})
                    eps, soft = eps + 1, soft + int(bool(m.get("soft")))
                    if eps % 50 == 0:
                        ep_f.flush()
                        print(f"[pid] {eps} flights, {soft} soft landings ({100 * soft / eps:.0f}%)", flush=True)
                traj = info.get("trajectory")
                now = time.time()
                if live and traj is not None and now - last.get(i, 0.0) >= 1.5:       # like training: at most one flight per world every 1.5 s
                    last[i] = now
                    live_n += 1
                    traj.update(id=live_n, env=i, timesteps=steps)
                    try:
                        tmp = live_dir / f"tmp_e{i}.json"
                        tmp.write_text(json.dumps(traj, separators=(",", ":")))
                        path = live_dir / f"ep_{live_n:08d}_e{i}.json"
                        os.replace(tmp, path)
                        paths.append(path)
                        while len(paths) > 80:
                            paths.pop(0).unlink(missing_ok=True)
                    except OSError:
                        pass
                env.reset()
                pilot.reset()
    print(f"done: {args.total_steps} steps requested, ran {steps}; {eps} flights, {soft} soft landings", flush=True)
