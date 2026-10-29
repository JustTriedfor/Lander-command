# Lander command

An open-source 2-D rocket-landing simulator for training and testing AI pilots.

A booster comes in fast from the side, thousands of metres up, and has to brake with one throttleable engine, steer with a gimbal, grid fins and gas thrusters, and touch down softly on a small pad. You write (or train) the pilot. The simulator tells you how it went.

It is written in Python on its own rigid-body physics (no game engine), exposes a standard [Gymnasium](https://gymnasium.farama.org/) environment, and comes with a scripted pilot, a trained neural-network pilot, training tools, and a browser app for watching and monitoring flights.

> **Early days.** This is the second version of a hobby project, and it is shared so other people can use it, break it and improve it. Some files, comments and the app's title bar still use the project's old name, **SpaceXperiment**.

## What you get

- **A real-scale landing problem.** A Falcon 9-class stage (about 22 t dry, one engine that throttles 40 to 100 % and cannot hover), starting 3 km up and falling at up to about 300 m/s, with drag, wind that gets stronger with height, burning propellant and spring-loaded landing legs.
- **A Gymnasium environment**, `SpaceXEnv`: 18 observations, 5 actions (throttle, gimbal, grid fins, two gas thrusters), 20 decisions per second.
- **Four difficulty levels**, from no wind and a gentle start (level 0) to 15 m/s wind and a fast, shallow, sideways start (level 3).
- **A scripted pilot** (`src/sim/expert.py`), a hand-written controller you can read, copy and beat.
- **A way to train a neural-network pilot**: behaviour cloning from the scripted pilot, then DAgger, then evaluation, as one pipeline. PPO training is included too.
- **Trained pilots included** (small, about 0.6 MB each) in `models/checkpoints/`, so you can fly something on the first day.
- **A browser app**: a live viewer with telemetry panels, a replay *showcase* that loops saved landings without simulating anything, and a **monitoring ground** under the picture: a dense dashboard of readings, combined charts, gauges and landing margins.
- **Two-lander modes**: two landers, two pads, either one shared network or one network each.
- **157 tests.**

## Quick start

You need Python 3.11. A GPU is optional.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows. On Linux or macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The versions in `requirements.txt` are the ones the project was built and tested with.

**Run the tests** (about two minutes):

```bash
python -m pytest -q
```

**Fly one landing with the scripted pilot.** Save this as `fly_one.py` in the project folder and run `python fly_one.py`:

```python
from src.sim.env import SpaceXEnv
from src.sim.expert import Expert
from src.sim.expert_data import _label

env = SpaceXEnv(level=1)          # level 0 (easy) to 3 (hard)
env.set_rules_stage(2)            # the full rules
obs, _ = env.reset(seed=122)      # the same seed gives the same start and wind
pilot = Expert(); pilot.reset()
done = False
while not done:
    obs, _, terminated, truncated, info = env.step(_label(pilot, env, False))
    done = terminated or truncated
m = info["episode_metrics"]
print("success:", m["success"], "| touchdown speed (m/s):", round(m["speed_mps"], 2), "| pad offset (m):", round(m["dist_m"], 2))
```

**Fly the same landing with a trained network instead:**

```python
from stable_baselines3 import PPO
from src.sim.env import SpaceXEnv
from src.training.weights_io import resolve_checkpoint

model = PPO.load(str(resolve_checkpoint("rt1_final")), device="cpu")
env = SpaceXEnv(level=1)
env.set_rules_stage(2)
obs, _ = env.reset(seed=122)
done = False
while not done:
    action = model.predict(obs, deterministic=True)[0]     # 5 numbers in [-1, 1]
    obs, _, terminated, truncated, info = env.step(action)
    done = terminated or truncated
print(info["episode_metrics"]["success"])
```

To plug in your own pilot, replace the `action` line with anything that returns five numbers between -1 and 1.

**Score a pilot on many landings:**

```bash
python -m src.sim.evaluate rt1_final --episodes 50 --levels 0,1,2,3
```

## The app

```bash
python src/app/server.py --open          # or double-click run_app.bat on Windows
```

It opens at <http://127.0.0.1:8767> and only listens on your own computer. Use `--port` to change the port.

- **Replay showcase:** <http://127.0.0.1:8767/?showcase=1>. Plays eleven saved landings in a loop with the numbers behind them. Nothing is simulated while it plays, so it is light on the computer and suits a screen left running. `?recorder=1` records a new set.
- **Live view:** the main page shows the flights of a training run. It starts empty until you start a run from the **Startup** tab, which runs the whole pipeline for you.
- **Monitoring ground:** scroll below the picture. Tabs: Overview, Telemetry, 3D View, AI / NN, Graphs, Logs, Settings. The Overview packs eight key readings with trend lines, a flight-phase bar, the live view, system gauges, landing margins and combined charts (solid, dashed and dotted lines with a legend, some on a second axis). Its panels are small templates, so adding another layout is one entry in `OPS_LAYOUTS` in `src/app/index.html`. It only reads what the main screen already has, and it stops drawing when it is off screen. Everything on it comes from the simulator's own numbers: there is no engine-temperature, camera-vision or ship-motion panel because the simulator has none of those.

## How a landing is judged

A landing counts as a **success** only if all four hold: touchdown speed at most **3 m/s**, tilt at most **0.15 rad**, landing point within **10 m** of the pad centre, and the fuel never ran out.

The environment ends an episode when the stage settles, crashes, tips over, stalls, runs out of time or leaves the area. Observations, actions and the exact rules are written at the top of `src/sim/env.py`; the reward is in `REWARD.md`.

## How well do the included pilots do?

Successful landings out of 300 test flights per level (seeds 0 to 299, no sensor noise). The wind level is a scale of 0, 5, 10 and 15 m/s, and each flight draws its own wind within it.

| Pilot | Level 0 | Level 1 | Level 2 | Level 3 |
|---|---|---|---|---|
| `rt1_final` (trained network) | 300 | 259 | 161 | 90 |
| Scripted pilot | 300 | 258 | 161 | 95 |

Levels 2 and 3 are unsolved: there is a lot of room to do better, and that is the point of the project.

## Train your own pilot

The usual route is **imitation**: the scripted pilot flies, a network copies it, then DAgger lets the network fly and the scripted pilot correct it.

```bash
python -m src.sim.pipeline --help        # clone, DAgger and evaluation as one command
python -m src.training.train --help      # PPO training
```

Or press **Start** on the app's Startup tab and watch it work. Training data and logs go into `data/` and `logs/` (not in the repository) and checkpoints into `models/checkpoints/`.

Honest status: **learning to land from scratch with PPO did not work** in my tests (no soft landings after about a million steps), and PPO fine-tuning of an imitation network tends to make it worse at the default settings. A network trained by imitation can only be as good as its teacher. A pilot that learns by itself is the main open problem here.

## Make it yours

- **The vehicle** is one file, `src/sim/vehicle.py`: mass, engine, throttle floor, gimbal, fins, gas thrusters, legs, air drag. Every number is a plain constant, and the ones that are my own guesses are marked "estimate".
- **The start, wind and difficulty levels** are in `src/utils/config.py` and `src/sim/spawn.py`.
- **The reward** is in `src/training/reward.py`.
- **The scripted pilot's constants** can be re-tuned with `tools/expert_tuning/`.

### Model rockets

The long-term idea is for anyone to be able to describe a different vehicle, such as a model rocket, and train a pilot for it. **That is not done yet.** Today everything is sized for a Falcon 9-class booster: the start is 3 km up at hundreds of metres per second, and the engine model assumes a throttle that cannot go below 40 %. Changing the constants gets you part of the way, but small hobby rockets also use solid motors that cannot be throttled or restarted, which would need a different kind of action. If you want to work on this, it is a good first project.

## Limits

- **Simplified.** The world is 2-D (no sideways dimension), the vehicle numbers are approximate public figures or estimates, and the wind is a simple model. It has **not been checked against real flight data**. Do not read the results as a statement about how a real rocket lands, and **never use this as flight software**.
- **Fragile to sensor noise.** The included network was trained with perfect sensors. In a test in the original project, adding noise equal to 1 % of each reading's range to its inputs made it fail every landing. It has not been retested here.
- **Not affiliated with SpaceX.** Nothing here is SpaceX code or data.

## What is where

| Folder or file | What is in it |
|---|---|
| `src/sim/` | the simulator: physics, vehicle, wind and start states, the environment, the scripted pilot, the imitation pipeline, evaluation |
| `src/training/` | reward, cloning and DAgger, PPO wrappers |
| `src/app/` | the browser app (`server.py` and a single-page `index.html`) |
| `src/utils/config.py` | settings: levels, rules, network size |
| `models/checkpoints/` | the trained pilots |
| `replays/` | the landings the showcase plays |
| `tests/` | the tests |
| `tools/` | pilot tuning and the replay recorder |
| `RUNBOOK.md` | how to run, watch, stop and troubleshoot everything |
| `SPACEXPERIMENT.md`, `REWARD.md` | notes on the physics model and the reward |
| `paper_assets/retrain/` | evaluation tables from the original project; a few tests compare against them |

## Contributing

Issues and pull requests are welcome. Please run `python -m pytest -q` before you send changes. Good places to start: a pilot that gets level 2 or 3 right more often, a way to describe other vehicles, a three-dimensional version, or better documentation.

## License

MIT, see `LICENSE`.
