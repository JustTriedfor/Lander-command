"""The PID test pilot (src/sim/pid_pilot.py) and its run mode (init_from = "PID")."""
import numpy as np

from src.app.schema import build_run
from src.sim.env import ON_THRESHOLD, SpaceXEnv, throttle_from_action
from src.sim.pid_pilot import PIDPilot
from src.utils import config as C


def test_form_accepts_pid_for_one_lander_only():
    r = build_run({"init_from": "PID", "start_profile": "spacex"})
    assert "errors" not in r and r["name"].endswith("_pid") and r["argv"][r["argv"].index("--init-from") + 1] == "PID"
    assert "errors" in build_run({"init_from": "PID", "start_profile": "dual"})


def test_actions_stay_in_range_and_keep_the_engine_lit():
    env = SpaceXEnv(level=0)
    env.reset(seed=1)
    pilot, lit = PIDPilot(), False
    for _ in range(400):
        a = pilot.action(env.s, 0.0)
        assert all(-1.0 <= v <= 1.0 for v in a)
        if pilot.burning:                                     # once the fire cue lights it, the command never drops to the engine-off line
            lit = True
            assert a[0] > ON_THRESHOLD and throttle_from_action(a[0]) >= 0.4
        _, _, term, trunc, _ = env.step(np.array(a))
        if term or trunc:
            break
    assert lit


def test_a_level0_flight_comes_to_rest():
    env = SpaceXEnv(level=0)
    env.set_rules_stage(C.RULES_STAGE_MAX)
    env.reset(seed=3)
    pilot = PIDPilot()
    while True:
        s = env.s
        _, _, term, trunc, info = env.step(pilot.action(s, env.wind.at(s.t, s.y)))
        if term or trunc:
            break
    m = info["episode_metrics"]
    assert m["end"] == "rest" and m["ignitions"] <= C.MAX_IGNITIONS
