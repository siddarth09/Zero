"""Measure each gripper's true grasp centre and print the eef_offset to put in zero_layout.

    python scripts/measure_tcp.py
"""

from __future__ import annotations

import numpy as np
import mujoco

import zero_layout as L


def pinch_point(m, d, r, side: str) -> tuple:
    """World position where the two finger pads MEET, with the gripper fully closed.

    THIS is the grasp centre. The fingers travel symmetrically, so the point where they touch when
    closed is the point an object's centre occupies when they are open around it.

    The previous version averaged the two finger bodies' collision-AABB centres, which sounds
    equivalent and is not: these finger meshes are asymmetric (one carries its knuckle), so the
    average sat 39 mm (reBot) / 22.7 mm (Panda) off the line between the pads. The jaws open
    100 mm, so an object would still land between them sometimes; grasps succeeded often enough
    to look like a tuning problem, while the failure mode was the gripper closing beside the can
    and flicking it away. The giveaway was measuring each finger's clearance separately: +9.5 mm
    on one pad, +99 mm on the other. Symmetric clearances are the acceptance test for this value.
    """
    from scipy.spatial.distance import cdist

    joints = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, L.prefixed(side, j))
              for j in r["gripper_joints"]]
    # Close the jaw by COMMANDING it and stepping, not by writing qpos. Writing qpos assumes every
    # gripper joint takes the same value, which holds for the reBot's two same-signed slides and
    # fails for a mirrored pair: the vx300s runs left_finger [0.021, 0.057] against right_finger
    # [-0.057, -0.021], so writing 0.021 to both drives one finger out of range and puts the
    # measured pinch point 28.7 mm off, with a tell-tale asymmetric y. Commanding the actuator lets
    # the <equality> place the follower correctly, and it works the same way for a four-bar linkage
    # as for a slide.
    for a_ in range(m.nu):
        if m.actuator_trntype[a_] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        if m.actuator_trnid[a_, 0] in joints:
            d.ctrl[a_] = r["grip_range"][0]
    for _ in range(int(1.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)

    # Which bodies carry the gripping surfaces. On a slide jaw that is the body the gripper joint
    # sits on, but a four-bar gripper like the 2F-85 drives its pads through two more links, so the
    # joint's own body is a knuckle 89 mm short of the pads. `pad_bodies` names them when they
    # differ.
    if r.get("pad_bodies"):
        bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, L.prefixed(side, b))
                  for b in r["pad_bodies"]]
        assert all(b >= 0 for b in bodies), f"pad_bodies not in the model: {r['pad_bodies']}"
    else:
        bodies = [m.jnt_bodyid[j] for j in joints]

    clouds = []
    for b in bodies:
        pts = []
        for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            X, P = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                V = m.mesh_vert[m.mesh_vertadr[mid]:
                                m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].reshape(-1, 3)
            else:
                # A primitive pad has no vertex list. Use its six FACE CENTRES, not its corners:
                # the closest pair between two facing boxes is then the two facing face centres,
                # so their midpoint is the grasp centre exactly. Corners put the closest pair on
                # an edge instead, which threw the measured offset 11 mm off in y.
                h = m.geom_size[g]
                V = np.array([[h[0], 0, 0], [-h[0], 0, 0],
                              [0, h[1], 0], [0, -h[1], 0],
                              [0, 0, h[2]], [0, 0, -h[2]]])
            pts.append((X @ V.T).T + P)
        assert pts, f"{side} gripper body {b} has no collision geometry"
        clouds.append(np.vstack(pts))

    # Average over every NEAR-minimal pair, not just argmin. On two flat parallel pads the closest
    # pair is degenerate: the vx300s's fingers face each other across 15 mm over a surface spanning
    # 70 mm in x and 61 mm in z, so thousands of pairs tie and argmin returns an arbitrary corner.
    # That made the measured offset jump between (0.1065, 0, 0.000) and (0.0815, 0, -0.020)
    # depending only on iteration order. The centroid of the tied pairs is the pad centre, which is
    # what the tool point should be, and it degrades gracefully to argmin when the closest pair is
    # genuinely unique, as on the reBot's asymmetric fingers.
    D = cdist(clouds[0], clouds[1])
    dmin = float(D.min())
    # 0.01 mm band. Tight on purpose: on parallel pads every facing pair sits at exactly dmin, so
    # a tight band still captures the whole surface, while on the reBot's asymmetric fingers it
    # captures only the genuinely closest pair and reproduces the value the recorded dataset was
    # measured against. That value must not move: it is baked into all 82 episodes.
    ii, kk = np.where(D <= dmin + 1e-5)
    mids = (clouds[0][ii] + clouds[1][kk]) / 2.0
    return mids.mean(axis=0), dmin


def measure(key: str, side: str = "left") -> tuple:
    r = L.ROBOTS[key]
    m = mujoco.MjModel.from_xml_path(str(L.PKG / "mjcf" / f"zero_{key}.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(m, d)

    eb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, L.prefixed(side, r["eef_body"]))
    R, P = d.xmat[eb].reshape(3, 3), d.xpos[eb]
    pinch, gap = pinch_point(m, d, r, side)
    measured = R.T @ (pinch - P)                       # into the eef body's local frame
    current = P + R @ np.array(r["eef_offset"], dtype=float)
    return measured, float(np.linalg.norm(current - pinch))


def main() -> None:
    for key in L.ROBOTS:
        off, err = measure(key)
        print(f"\n{key}:")
        print(f"    eef_body        {L.ROBOTS[key]['eef_body']!r}")
        print(f"    current offset  {tuple(L.ROBOTS[key]['eef_offset'])}")
        print(f"    MEASURED offset ({off[0]:.4f}, {off[1]:.4f}, {off[2]:.4f})")
        print(f"    tool point is   {err * 1000:.1f} mm from the real grasp centre")
        if err > 0.005:
            print(f"    -> paste the measured triple into ROBOTS[{key!r}]['eef_offset']")


if __name__ == "__main__":
    main()
