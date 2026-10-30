"""A test pilot made of plain PID loops plus one extra rule, the main-engine fire cue. It is here to compare with the networks and the
scripted expert (src/sim/expert.py), not to land well.

  fire cue   the engine lights when the stage falls faster than the braking curve v(h) = sqrt(2 a h + v0^2): the speed from which
             a steady burn still stops it at the deck (a = A_FRAC of the deceleration the engine has at full throttle). It cuts if the
             stage stops above the deck and starts to climb (the engine cannot throttle low enough to hover).
  throttle   PID on the descent-speed error against that curve
  tilt goal  PD on the sideways position and speed (lean toward the pad)
  actuators  PID on the tilt error; the same command goes to the gimbal, the grid fins and cold gas system A (no scheduling by airspeed)

The gains were tuned by a random hill-climb inside the environment with every rule on (levels 0-2, 300 tries). On 300 fresh starts
per level it made 91 / 23 / 0 / 0% soft landings at levels 0-3 (the scripted expert, measured without the environment's rules:
100 / 77 / 52 / 24%). Plain PID loops without the fire cue landed none: holding any constant descent rate runs the tank dry high up.
The engine stays lit once the cue lights it (the environment allows three ignitions a flight); the PID only sets 40-100% thrust.
"""
from __future__ import annotations

import math

from src.sim import vehicle as V
from src.utils import config as C

GAINS = dict(
    V_SET=0.4565,        # m/s: the touchdown speed the braking curve aims at
    A_FRAC=0.3202,      # the share of the full-throttle deceleration the curve plans with
    THR0=0.4917,      # throttle feed-forward
    KP_V=0.3265, KI_V=0.07788, KD_V=0.004936,                  # throttle PID on the descent-speed error (m/s)
    KX=0.0007818, KVX=0.01241, TILT_MAX=0.2036,                # lean toward the pad: rad per m, rad per m/s, the most it leans (rad)
    KP_A=2.762, KI_A=0.186, KD_A=7.169,                      # attitude PID (normalised actuator command per rad, rad s, rad/s)
)
DT = 1.0 / C.FPS


class PIDPilot:
    def __init__(self, gains: dict | None = None):
        self.p = dict(GAINS, **(gains or {}))
        self.reset()

    def reset(self) -> None:
        self.iv = self.ia = 0.0
        self.prev_ev = None
        self.burning = False

    def command(self, s, wind_x: float = 0.0) -> tuple:
        """The physical commands for state s: (throttle 0 or 0.4..1, gimbal rad, fin rad, cold gas A, cold gas B)."""
        p = self.p
        h = max(s.foot_height(), 0.0)
        vd = -s.vy                                                       # descent speed (+ = falling)
        a = max(p["A_FRAC"] * (V.thrust_full(h) / s.mass - C.GRAVITY_MPS2), 2.0)
        v_ref = math.sqrt(2.0 * a * h + p["V_SET"] ** 2)                 # the braking curve
        if not self.burning and vd >= v_ref and s.prop > 0.0:           # the fire cue
            self.burning = True
        if self.burning and s.vy > 0.3 and s.thr > 0.0:                  # stopped above the deck and climbing: cut, fall, burn again
            self.burning = False
        ev = vd - v_ref                                                  # + = falling faster than the curve
        dev = 0.0 if self.prev_ev is None else (ev - self.prev_ev) / DT
        self.prev_ev = ev
        u = p["THR0"] + p["KP_V"] * ev + p["KI_V"] * self.iv + p["KD_V"] * dev
        if 0.0 < u < 1.0:
            self.iv += ev * DT                                           # (no wind-up while saturated)
        throttle = max(V.THROTTLE_MIN, min(1.0, u)) if self.burning else 0.0          # only the fire cue switches the engine on and off (three ignitions a flight)
        th_d = max(-p["TILT_MAX"], min(p["TILT_MAX"], p["KX"] * s.x + p["KVX"] * s.vx))      # positive tilt pushes toward -x
        ea = th_d - s.th
        self.ia = max(-2.0, min(2.0, self.ia + ea * DT))
        c = max(-1.0, min(1.0, p["KP_A"] * ea + p["KI_A"] * self.ia - p["KD_A"] * s.om))
        return throttle, c * V.GIMBAL_MAX, -c * V.FIN_MAX, -c, 0.0

    def action(self, s, wind_x: float = 0.0) -> list:
        """The same commands as the environment's actions (each -1..1, as a network would send them)."""
        from src.sim.env import ON_THRESHOLD
        throttle, gim, fin, ra, rb = self.command(s, wind_x)
        a0 = -1.0 if throttle <= 0.0 else ON_THRESHOLD + (1.0 - ON_THRESHOLD) * (max(throttle, V.THROTTLE_MIN) - V.THROTTLE_MIN) / (1.0 - V.THROTTLE_MIN)
        a0 = max(a0, ON_THRESHOLD + 1e-6) if throttle > 0.0 else a0      # (40% must stay above the engine-off line)
        return [a0, gim / V.GIMBAL_MAX, fin / V.FIN_MAX, ra, rb]
