"""Pick N well-spread can positions that are valid for the whole reBot -> Panda -> G1 chain.

    MUJOCO_GL=egl python3 scripts/plan_can_poses.py [N] [MIN_APPROACHES]

A position is valid only if all four hold:

  1. the reBot's left arm can reach it, IK-verified against the grasp orientations the operator
     actually used. FK sampling only says "reachable" if a random configuration happened to land
     there, so it under-reports;
  2. the reBot's right arm cannot reach it. If the right arm can grasp the can directly then the
     handover is unmotivated and the demonstrations carry contradictory evidence about which arm
     should pick, so a kinematically ideal pose can still be task-invalid;
  3. a G1 arm can reach it, so the data is not wasted for the destination embodiment. Note the
     mirror: with the G1 facing the table at yaw 180 its left arm covers world -y, so the
     reBot's left arm maps to the G1's right arm;
  4. it is on the table with clearance for the can's radius.

Among the valid cells the N returned are chosen by farthest-point sampling, so they spread over
the reachable region instead of clustering where the IK is most comfortable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "zero_control"))
import zero_layout as L
from zero_control.action import rot_from_6d
from zero_control.ik import ArmIK

G1_XML = "~/mujoco_menagerie/unitree_g1/g1_with_hands.xml"
G1_PALM = {"right": (0.1152, 0.0845, -0.0024)}
G1_BASE = np.array([0.75, 0.0, 0.79])          # pelvis, yaw 180; reach_gate.py's verdict
GRID = 0.04
CAN_R = 0.033


def grasp_orientations() -> list[np.ndarray]:
    """The distinct approach orientations the operator actually used, from the v1 recording."""
    import glob
    import pandas as pd
    files = sorted(glob.glob(str(Path.home() / "zero_data/rebot_pick_place/data/chunk-*/file-*.parquet")))
    if not files:
        return []
    df = pd.concat([pd.read_parquet(f, columns=["observation.state", "action", "episode_index"])
                    for f in files], ignore_index=True)
    S = np.stack(df["observation.state"].to_numpy()).astype(float)
    A = np.stack(df["action"].to_numpy()).astype(float)
    ep = df["episode_index"].to_numpy()
    keep: list[np.ndarray] = []
    for e in np.unique(ep):
        m = ep == e
        a, s = A[m], S[m]
        c = np.convolve((a[:, 9] < 0.5).astype(int), np.ones(10, int), "valid")
        i = int(np.argmax(c == 10))
        if c[i] != 10:
            continue
        R = rot_from_6d(s[i, 3:9])
        if not any(np.degrees(np.arccos(np.clip((np.trace(R.T @ K) - 1) / 2, -1, 1))) < 12
                   for K in keep):
            keep.append(R)
    return keep


def g1_band(z: float) -> np.ndarray:
    """400k FK samples of the G1's right arm. Cached: it is the slow part of this script and the
    G1 does not move between runs, so re-planning with different filters should be seconds."""
    cache = Path(__file__).resolve().parent / ".g1_band_cache.npy"
    if cache.exists():
        return np.load(cache)
    m = mujoco.MjModel.from_xml_path(G1_XML)
    d = mujoco.MjData(m)
    names = [f"right_{j}_joint" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                                          "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    adr = [m.jnt_qposadr[i] for i in jid]
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    wy = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    off = np.array(G1_PALM["right"])
    rng = np.random.default_rng(0)
    pts = np.empty((400_000, 3))
    for i in range(len(pts)):
        d.qpos[adr] = rng.uniform(lo, hi)
        mujoco.mj_kinematics(m, d)
        pts[i] = d.xpos[wy] + d.xmat[wy].reshape(3, 3) @ off - d.xpos[pel]
    Rz = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1.0]])       # yaw 180
    w = pts @ Rz.T + G1_BASE
    band = w[(w[:, 2] > z - 0.05) & (w[:, 2] < z + 0.05)]
    np.save(cache, band)
    return band


def main() -> None:
    n_want = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    # Reject positions only a couple of wrist angles can reach: the operator has to hit a specific
    # orientation, which is awkward to teleoperate and tends to produce demos with a large IK
    # residual. Pass 1 to see every valid cell.
    min_free = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    Rs = grasp_orientations()
    print(f"{len(Rs)} distinct grasp orientations recovered from the v1 recording")
    z = 0.834                                     # measured mean grasp height

    r = L.ROBOTS["rebot"]
    urdf = str(L.PKG / "urdf" / "zero_rebot.urdf")
    iks = {}
    for side in L.SIDES:
        ik = ArmIK(urdf, f"{side}_gripper_end",
                   [L.prefixed(side, j) for j in r["arm_joints"]], r["eef_offset"])
        q = np.zeros(ik.model.nq)
        for i, j in enumerate(r["arm_joints"]):
            q[ik.qidx[i]] = r["home"][side][i]
        iks[side] = (ik, q)
    if not Rs:
        Rs = [iks["left"][0].fk(iks["left"][1]).rotation]

    band = g1_band(z)
    print(f"G1 right-arm samples in the grasp height band: {len(band)}")

    cx, cy = L.TABLE_CENTER_XY
    hx, hy, _ = L.TABLE_HALF
    xs = np.arange(0.16, cx + hx - CAN_R, GRID)
    ys = np.arange(0.04, cy + hy - CAN_R, GRID)

    valid = []
    for x in xs:
        for y in ys:
            n_left = 0
            reach_right = False
            for side, ok in (("left", None), ("right", None)):
                ik, q0 = iks[side]
                for R in Rs:
                    res = ik.solve(q0, pin.SE3(R, np.array([x, y, z])), iters=300, tol=1e-4)
                    if res.pos_err < 5e-3 and np.degrees(res.rot_err) < 8.0:
                        if side == "left":
                            n_left += 1
                        else:
                            reach_right = True
                            break
            if n_left < min_free or reach_right:
                continue
            if ((np.abs(band[:, 0] - x) < 0.05) & (np.abs(band[:, 1] - y) < 0.05)).sum() < 3:
                continue
            valid.append((x, y, n_left))
    V = np.array([(x, y) for x, y, _ in valid])
    print(f"{len(valid)} valid cells (left-only, G1-reachable, on the table)")
    if not len(V):
        return

    # farthest-point sampling: start from the most approach-free cell, then repeatedly take the cell
    # furthest from everything chosen so far
    free = np.array([n for _, _, n in valid])
    picked = [int(np.argmax(free))]
    while len(picked) < min(n_want, len(V)):
        dmin = np.min(np.linalg.norm(V[:, None, :] - V[picked][None, :, :], axis=2), axis=1)
        picked.append(int(np.argmax(dmin)))

    print(f"\n{len(picked)} positions, {10} episodes each:\n")
    print(f"  {'#':>2}  {'can_x':>6} {'can_y':>6}  approaches   nearest other")
    for k, i in enumerate(picked, 1):
        x, y = V[i]
        others = np.delete(V[picked], k - 1, axis=0)
        near = np.linalg.norm(others - V[i], axis=1).min() if len(others) else float("nan")
        print(f"  {k:>2}  {x:6.2f} {y:6.2f}  {free[i]}/{len(Rs)}         {near*100:5.1f} cm")
    print("\ncommands:")
    for i in picked:
        x, y = V[i]
        print(f"  ros2 launch zero_bringup rebot.launch.py can_x:={x:.2f} can_y:={y:.2f}")


if __name__ == "__main__":
    main()
