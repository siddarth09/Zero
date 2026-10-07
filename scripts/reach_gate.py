"""Phase 0.5 go/no-go gate: can a task built for two reBots be performed by a G1?

Three questions, answered by measurement before any policy code exists:

  A. Aperture. What object sizes can both a reBot parallel jaw and a Dex3 hold?
  B. Overlap.  Is there table-surface volume reachable by both reBots and both G1 arms?
  C. Verdict.  The concrete box to place objects in.

This runs first because the previous project spent days on a policy that could not succeed: its
target was kinematically unreachable for a grasped object, and the tell (closest approach pinned
at 0.069-0.086 m against a 0.05 m threshold, identical across every tilt cap, every curriculum
rung and every checkpoint) was read as a learning failure for too long. A learning failure moves
when you change a knob. A kinematic one does not.

Two rules the script follows:

  1. Do not optimise the placement. A first version swept the G1 pelvis to maximise overlap
     litres and returned pelvis x=-0.60, z=0.60, both pinned to their sweep bounds, with the G1
     facing away from the table and reaching backwards over its own shoulders. Maximising volume
     is not the objective. Fix the robot in a stance it can physically hold, then verify.
  2. Exclude the obstacles. Before the keep-out below, the "best" task volume was centred on
     x in [-0.40,-0.08], exactly where the reBot bases are bolted to the table. Both robots can
     reach their own hardware; that is not a workspace.

Run:  MUJOCO_GL=egl python scripts/reach_gate.py
"""

from __future__ import annotations

import numpy as np
import mujoco

import zero_layout as L

G1_XML = "~/mujoco_menagerie/unitree_g1/g1_with_hands.xml"
VOXEL = 0.04            # 4 cm, matching the earlier handover reachability study
N_SAMPLES = 60_000      # per arm
SLAB = 0.25             # task volume height above the table top
BASE_KEEPOUT = 0.18     # radius round each reBot base, a physical obstacle
G1_PELVIS_Z = 0.79      # the G1's real standing pelvis height
G1_YAW = np.pi          # faces -x: stands across the table, looking at the arms
DOWN_CONE_DEG = 35.0

# Neither the reBot nor the G1 model ships an end-effector site. These G1 palm offsets on
# *_wrist_yaw_link were derived in the previous project as the midpoint of the thumb pad and the
# index+middle pad centroid at a ~7 cm aperture, verified as strictly bracketed by all three pads.
# Reused here rather than re-guessed.
G1_PALM_OFFSET = {"left": (0.1152, -0.0845, 0.0024), "right": (0.1152, 0.0845, -0.0024)}

# Voxel coords are packed into one int64 so set ops become np.intersect1d. A Python set of tuples
# needed ~1000 x 60k tuple constructions and never finished the sweep.
_OFF, _SPAN = 1 << 12, 1 << 13


def keys(pts: np.ndarray) -> np.ndarray:
    v = np.floor(pts / VOXEL).astype(np.int64) + _OFF
    return np.unique((v[:, 0] * _SPAN + v[:, 1]) * _SPAN + v[:, 2])


def unkey(k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, dtype=np.int64)
    return np.stack([k // (_SPAN * _SPAN), (k // _SPAN) % _SPAN, k % _SPAN], 1) - _OFF


def inter(*ks: np.ndarray) -> np.ndarray:
    out = ks[0]
    for k in ks[1:]:
        out = np.intersect1d(out, k, assume_unique=True)
    return out


def _ids(m, obj, names):
    out = []
    for n in names:
        i = mujoco.mj_name2id(m, obj, n)
        assert i >= 0, f"missing {n}"
        out.append(i)
    return out


def measure_apertures() -> None:
    print("=" * 76)
    print("A. GRIPPER APERTURE, measured from the models, not from datasheets")
    print("=" * 76)
    m = mujoco.MjModel.from_xml_path(str(L.MENAGERIE_ARM))
    d = mujoco.MjData(m)
    bl, br = _ids(m, mujoco.mjtObj.mjOBJ_BODY, ["gripper_left", "gripper_right"])
    jl, jr = _ids(m, mujoco.mjtObj.mjOBJ_JOINT, ["gripper_joint1", "gripper_joint2"])
    span = []
    for q in (0.0, 0.05):
        d.qpos[:] = 0
        d.qpos[m.jnt_qposadr[jl]] = d.qpos[m.jnt_qposadr[jr]] = q
        mujoco.mj_forward(m, d)
        span.append(float(np.linalg.norm(d.xpos[bl] - d.xpos[br])))
    rebot = max(span)
    print(f"  reBot jaw : finger separation {span[0]*1000:.0f} mm (closed) -> "
          f"{span[1]*1000:.0f} mm (open)")
    g = mujoco.MjModel.from_xml_path(G1_XML)
    dg = mujoco.MjData(g)
    th, ix = _ids(g, mujoco.mjtObj.mjOBJ_BODY,
                  ["right_hand_thumb_2_link", "right_hand_index_1_link"])
    mujoco.mj_forward(g, dg)
    dex3 = float(np.linalg.norm(dg.xpos[th] - dg.xpos[ix]))
    print(f"  Dex3 hand : thumb-tip to index-tip at open pose {dex3*1000:.0f} mm")
    lim = min(rebot, dex3)
    print(f"\n  -> binding constraint is the {'reBot jaw' if rebot < dex3 else 'Dex3 hand'} "
          f"at {lim*1000:.0f} mm.  Use a ~{lim*0.55*1000:.0f} mm object for grasp margin.")


def rebot_cloud(rng):
    m = mujoco.MjModel.from_xml_path(str(L.MENAGERIE_ARM))
    d = mujoco.MjData(m)
    ge = _ids(m, mujoco.mjtObj.mjOBJ_BODY, ["gripper_end"])[0]
    jid = _ids(m, mujoco.mjtObj.mjOBJ_JOINT, list(L.ARM_JOINTS))
    adr = [m.jnt_qposadr[i] for i in jid]
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    pts, down = np.empty((N_SAMPLES, 3)), np.empty(N_SAMPLES)
    for i in range(N_SAMPLES):
        d.qpos[adr] = rng.uniform(lo, hi)
        mujoco.mj_forward(m, d)
        pts[i] = d.xpos[ge]
        down[i] = d.xmat[ge].reshape(3, 3)[:, 2] @ np.array([0, 0, -1.0])
    return {s: pts + np.array(mt) for s, mt in L.MOUNTS.items()}, down


def g1_cloud_pelvis(rng):
    m = mujoco.MjModel.from_xml_path(G1_XML)
    d = mujoco.MjData(m)
    pel = _ids(m, mujoco.mjtObj.mjOBJ_BODY, ["pelvis"])[0]
    out = {}
    for side in ("left", "right"):
        names = [f"{side}_{j}_joint" for j in
                 ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                  "wrist_roll", "wrist_pitch", "wrist_yaw")]
        jid = _ids(m, mujoco.mjtObj.mjOBJ_JOINT, names)
        adr = [m.jnt_qposadr[i] for i in jid]
        lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
        wy = _ids(m, mujoco.mjtObj.mjOBJ_BODY, [f"{side}_wrist_yaw_link"])[0]
        off = np.array(G1_PALM_OFFSET[side])
        pts = np.empty((N_SAMPLES, 3))
        for i in range(N_SAMPLES):
            d.qpos[adr] = rng.uniform(lo, hi)
            mujoco.mj_forward(m, d)
            pts[i] = d.xpos[wy] + d.xmat[wy].reshape(3, 3) @ off - d.xpos[pel]
        out[side] = pts
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    measure_apertures()

    print("\n" + "=" * 76)
    print("B. WORKSPACE OVERLAP")
    print("=" * 76)
    rb, down = rebot_cloud(rng)
    g1 = g1_cloud_pelvis(rng)
    hx, hy, _ = L.TABLE_HALF
    cx, cy = L.TABLE_CENTER_XY

    both = inter(keys(rb["left"]), keys(rb["right"]))
    print(f"  reBot, either arm : {len(keys(rb['left']))} vox each")
    print(f"  reBot, BOTH arms  : {len(both)} vox ({len(both)*VOXEL**3*1000:.1f} L, free space)")

    c = (unkey(both) + 0.5) * VOXEL
    on = ((c[:, 0] >= cx - hx) & (c[:, 0] <= cx + hx) & (c[:, 1] >= cy - hy)
          & (c[:, 1] <= cy + hy) & (c[:, 2] >= L.TABLE_TOP_Z)
          & (c[:, 2] <= L.TABLE_TOP_Z + SLAB))
    for mx, my, _ in L.MOUNTS.values():
        on &= np.hypot(c[:, 0] - mx, c[:, 1] - my) > BASE_KEEPOUT
    both = both[on]
    print(f"  ...restricted to the {SLAB*100:.0f} cm table slab, minus a "
          f"{BASE_KEEPOUT*100:.0f} cm keep-out round each base: "
          f"{len(both)} vox ({len(both)*VOXEL**3*1000:.2f} L)")
    if len(both) == 0:
        print("\n  NO-GO: the two reBots share no usable table volume. Widen BASE_SEP.")
        return

    cw, sw = np.cos(G1_YAW), np.sin(G1_YAW)
    R = np.array([[cw, -sw, 0], [sw, cw, 0], [0, 0, 1.0]])
    gl, gr = g1["left"] @ R.T, g1["right"] @ R.T
    edge = cx + hx
    print(f"\n  G1 FIXED at pelvis z={G1_PELVIS_Z} m, yaw={np.degrees(G1_YAW):.0f}° "
          f"(facing the table). Sweeping only the standoff:")
    best = None
    for so in np.arange(0.10, 0.80, 0.05):
        base = np.array([edge + so, 0.0, G1_PELVIS_Z])
        k = inter(both, keys(gl + base), keys(gr + base))
        mark = ""
        if best is None or len(k) > best[0]:
            best = (len(k), so, k)
            mark = "   <-- best"
        print(f"    standoff {so:.2f} m (pelvis x={base[0]:+.2f}) : {len(k):4d} vox "
              f"{len(k)*VOXEL**3*1000:6.2f} L{mark}")
    n, so, k = best

    print("\n" + "=" * 76)
    print("C. VERDICT")
    print("=" * 76)
    if n == 0:
        print("  NO-GO: no common table volume at any standoff.")
        print("    Levers: lower TABLE_TOP_Z, move the reBot mounts forward (BASE_X),")
        print("            or let the G1 stand at a non-zero y.")
        return
    P = unkey(k) * VOXEL
    ctr = P.mean(axis=0) + VOXEL / 2
    print(f"  GO: {n} vox = {n*VOXEL**3*1000:.2f} L shared, at standoff {so:.2f} m")
    print(f"    G1 pelvis      : [{edge+so:+.2f}, 0.00, {G1_PELVIS_Z}] yaw "
          f"{np.degrees(G1_YAW):.0f}°")
    for i, ax in enumerate("xyz"):
        print(f"    {ax} extent       : [{P[:,i].min():+.3f}, {P[:,i].max()+VOXEL:+.3f}] m")
    print(f"    place object at: [{ctr[0]:+.3f}, {ctr[1]:+.3f}, {ctr[2]:+.3f}]  "
          f"({(ctr[2]-L.TABLE_TOP_Z)*100:+.0f} cm above the table)")
    frac = float((down > np.cos(np.deg2rad(DOWN_CONE_DEG))).mean())
    print(f"\n  reBot poses within {DOWN_CONE_DEG:.0f}° of straight-down approach: {frac*100:.1f}% "
          f"of random configs")
    print("  G1 approach-axis convention not asserted; needs its own check before P1.")


if __name__ == "__main__":
    main()
