"""Generate a dual-arm URDF (plus ros2_control block) for any registered embodiment.

    python scripts/gen_urdf.py rebot
    python scripts/gen_urdf.py panda
"""

from __future__ import annotations

import copy
import math

import numpy as np
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import zero_layout as L


def _rpy_to_mat(rpy):
    r, pi_, y = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(pi_),
                              math.sin(pi_), math.cos(y), math.sin(y))
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp,   cp*sr,            cp*cr]])


def _mat_to_rpy(R):
    sp = -R[2, 0]
    sp = max(-1.0, min(1.0, sp))
    p_ = math.asin(sp)
    if abs(abs(sp) - 1.0) < 1e-9:                     # gimbal lock: fold yaw into roll
        return [math.atan2(-R[1, 2], R[1, 1]), p_, 0.0]
    return [math.atan2(R[2, 1], R[2, 2]), p_, math.atan2(R[1, 0], R[0, 0])]


def _axis_angle_to_mat(a, ang):
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def _frozen_angles(r: dict) -> dict:
    """{joint name: angle} for every joint gen_scene freezes, read from the same keyframe it uses.

    Returns {} when the robot freezes nothing, so the caller's `if frozen` short-circuits.
    """
    pats = r.get("freeze_joints", ())
    if not pats:
        return {}
    import mujoco
    src = mujoco.MjModel.from_xml_path(str(r["mjcf"]))
    kid = mujoco.mj_name2id(src, mujoco.mjtObj.mjOBJ_KEY, r.get("base_keyframe", ""))
    out = {}
    for j in range(src.njnt):
        name = mujoco.mj_id2name(src, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if any(pat in name for pat in pats):
            out[name] = float(src.key_qpos[kid, src.jnt_qposadr[j]]) if kid >= 0 else 0.0
    return out


def _remap_part(part: str) -> str:
    """Converter mesh name -> the menagerie asset that actually exists.

    Three cases, all discovered by checking which references failed to resolve rather than by
    assuming a scheme:
      link0_0   -> link0_0.obj            visual, keeps its index
      link0_c   -> link0.stl              collision; menagerie ships these as .stl
      link5_c0  -> link5_collision_0.obj  link5's collision is split into 3 convex pieces
    """
    m = re.fullmatch(r"(link\d+)_c(\d+)", part)
    if m:
        return f"{m.group(1)}_collision_{m.group(2)}.obj"
    if part.endswith("_c"):
        return f"{part[:-2]}.stl"
    return f"{part}.obj"


def prefix_side(arm: ET.Element, side: str, r: dict) -> list[ET.Element]:
    """Deep-copy links+joints, prefix every name AND every back-reference.

    A name can hide in four places: the link's `name`, the joint's `name`, and the joint's
    <parent link=> / <child link=>. Miss one and the URDF still parses but describes a
    disconnected tree, which `check_urdf` reports as multiple roots.
    """
    # side=None means a single-body embodiment: copy the tree without renaming anything, because
    # its links and joints already carry left_/right_ where it matters.
    pre = (lambda n: n) if side is None else (lambda n: L.prefixed(side, n))
    frozen = _frozen_angles(r)
    skip = set(r["urdf_skip_links"])
    remap = re.compile(r["mesh_remap"]) if r["mesh_remap"] else None
    out: list[ET.Element] = []

    for link in arm.findall("link"):
        if link.get("name") in skip:
            continue
        el = copy.deepcopy(link)
        el.set("name", pre(link.get("name")))
        for tag in ("visual", "collision"):
            for g in el.findall(f"{tag}/geometry/mesh"):
                fn = g.get("filename").rsplit("/", 1)[-1]
                if remap:
                    m = remap.fullmatch(fn)
                    if m:
                        part = m.group(1)
                        # `mesh_remap_to` is a template for converters whose naming does not match
                        # menagerie's Panda scheme. The G1's URDF spells its meshes
                        # converted_<link>_<rrggbbaa>.obj, where the hex is a COLOUR, so several
                        # .obj files collapse onto one .STL: the per-colour split is lost and RViz
                        # renders one colour per link. The physics is unaffected; the MJCF, which is
                        # what the sim runs, keeps its own assets.
                        if r.get("mesh_remap_to"):
                            g.set("filename",
                                  "package://zero_description/meshes/"
                                  + r["mesh_remap_to"].format(part=part))
                            continue
                        # A trailing `_c` marks a collision mesh in the converter's naming, and menagerie ships those
                        # as .stl (`link0.stl`), not `_c.obj`. Visual meshes keep their index and stay .obj
                        # (`link0_0.obj`).
                        fn = _remap_part(part)
                g.set("filename", f"package://zero_description/meshes/{fn}")
        out.append(el)

    for joint in arm.findall("joint"):
        p = joint.find("parent")
        c = joint.find("child")
        # A joint touching a skipped link (Panda's world->link0 weld) is dropped; our own mount joint
        # replaces it.
        if (p is not None and p.get("link") in skip) or (c is not None and c.get("link") in skip):
            continue
        el = copy.deepcopy(joint)
        el.set("name", pre(joint.get("name")))
        # Mirror gen_scene's `freeze_joints`. The MJCF deletes those joints, so if the URDF left
        # them movable the two descriptions would disagree on DOF count and robot_state_publisher
        # would draw the G1's legs straight while MuJoCo holds them crouched. Made `fixed` with the
        # keyframe angle BAKED INTO the origin, so the rendered pose matches the simulated one.
        if frozen and any(pat in joint.get("name") for pat in frozen):
            ang = frozen.get(joint.get("name"), 0.0) if isinstance(frozen, dict) else 0.0
            ax = el.find("axis")
            org = el.find("origin")
            if ax is not None and abs(ang) > 1e-9:
                a = np.array([float(v) for v in ax.get("xyz").split()], dtype=float)
                a /= np.linalg.norm(a)
                rpy0 = [float(v) for v in (org.get("rpy", "0 0 0").split())] if org is not None \
                    else [0.0, 0.0, 0.0]
                R = _rpy_to_mat(rpy0) @ _axis_angle_to_mat(a, ang)
                if org is None:
                    org = ET.SubElement(el, "origin", {"xyz": "0 0 0"})
                org.set("rpy", " ".join(f"{v:.9f}" for v in _mat_to_rpy(R)))
            el.set("type", "fixed")
            for tag in ("axis", "limit", "dynamics", "mimic"):
                sub = el.find(tag)
                if sub is not None:
                    el.remove(sub)
        for tag in ("parent", "child"):
            ref = el.find(tag)
            if ref is not None and ref.get("link"):
                ref.set("link", pre(ref.get("link")))
        mim = el.find("mimic")
        if mim is not None and mim.get("joint"):
            mim.set("joint", pre(mim.get("joint")))
        out.append(el)
    return out


def add_ros2_control(robot: ET.Element, key: str) -> None:
    r = L.ROBOTS[key]
    rc = ET.SubElement(robot, "ros2_control",
                       {"name": f"Zero{key.capitalize()}System", "type": "system"})
    hw = ET.SubElement(rc, "hardware")
    ET.SubElement(hw, "plugin").text = "mujoco_ros2_control/MujocoSystemInterface"

    # `$(find ...)`, not `package://`. The plugin does not resolve package://; it hands the string
    # straight to the MuJoCo loader and dies with "MuJoCo model file 'package://...' does not exist!".
    # xacro expands $(find pkg) into an absolute path before the plugin sees it, so the launch must
    # pipe this URDF through xacro. That is how Panda_mujoco does it, and it beats baking in
    # $HOME/... which breaks the moment the project moves.
    ET.SubElement(hw, "param", {"name": "mujoco_model"}).text = (
        f"$(find zero_description)/mjcf/zero_{key}.xml")
    # Faster-than-realtime stepping, for data collection and eval sweeps.
    ET.SubElement(hw, "param", {"name": "sim_speed_factor"}).text = "1.0"
    ET.SubElement(hw, "param", {"name": "camera_publish_rate"}).text = f"{L.CAM_RATE_HZ}"
    # Start-position override: the launch writes this file from its can_x/can_y/can_yaw arguments
    # before the sim starts. The path is fixed here because hardware params cannot be set from the
    # command line; only the file's contents vary per run.
    ET.SubElement(hw, "param", {"name": "override_start_position_file"}).text = str(
        L.start_override_path(key))
    # Without this the sim starts at qpos=0 and the arms bolt upright. gen_scene.py writes a `home`
    # keyframe holding the searched per-side poses plus the object's placed pose.
    # (mujoco_system_interface.cpp:1353 reads this param.)
    ET.SubElement(hw, "param", {"name": "initial_keyframe"}).text = "home"

    for name in L.robot_prefixed_ros2_joints(key):
        j = ET.SubElement(rc, "joint", {"name": name})
        ET.SubElement(j, "command_interface", {"name": "position"})
        for iface in ("position", "velocity"):
            ET.SubElement(j, "state_interface", {"name": iface})

    # Cameras publish only if a <sensor> of the same name as the MJCF <camera> appears here. No extra
    # plugin: image/info/depth publishing lives in the core system interface.
    #
    # Fingertip F/T sensors: mujoco_ros2_control maps a <sensor mujoco_type="fts"> named X onto MJCF
    # sensors X_force / X_torque, and exports exactly six state interfaces named force.x/y/z and
    # torque.x/y/z. That spelling is checked literally in the plugin, so a typo drops the axis rather
    # than erroring. force_torque_sensor_broadcaster reads those and republishes a WrenchStamped.
    for sensor_name, _ in L.ft_sensors(key):
        sen = ET.SubElement(rc, "sensor", {"name": sensor_name})
        ET.SubElement(sen, "param", {"name": "mujoco_type"}).text = "fts"
        for axis in ("force.x", "force.y", "force.z", "torque.x", "torque.y", "torque.z"):
            ET.SubElement(sen, "state_interface", {"name": axis})

    for cam in L.all_cameras():
        sen = ET.SubElement(rc, "sensor", {"name": cam})
        for k, v in (("frame_name", f"{cam}_optical_frame"),
                     ("image_topic", f"{L.CAM_NS}/{cam}/image_raw"),
                     ("info_topic", f"{L.CAM_NS}/{cam}/camera_info"),
                     ("depth_topic", f"{L.CAM_NS}/{cam}/depth")):
            ET.SubElement(sen, "param", {"name": k}).text = v


def add_graft_urdf(robot: ET.Element, side: str, r: dict) -> None:
    """Append the grafted gripper's URDF side: base frame, one commanded joint, visuals.

    Mirrors `graft_gripper` on the MJCF side. `robotiq_85_base_link` must land exactly where the
    MJCF body of the same name sits, because ik.py composes `eef_offset` onto this frame; a
    discrepancy is a constant bias in every recorded pose that no test other than
    check_parity.check_eef_frame would catch.
    """
    g = r["graft_urdf"]
    p3 = lambda v: " ".join(str(x) for x in v)

    j = ET.SubElement(robot, "joint",
                      {"name": L.prefixed(side, "robotiq_85_base_joint"), "type": "fixed"})
    ET.SubElement(j, "origin", {"xyz": p3(g["origin_xyz"]), "rpy": p3(g["origin_rpy"])})
    ET.SubElement(j, "parent", {"link": L.prefixed(side, g["parent"])})
    ET.SubElement(j, "child", {"link": L.prefixed(side, g["base_link"])})

    base = ET.SubElement(robot, "link", {"name": L.prefixed(side, g["base_link"])})
    vis = ET.SubElement(base, "visual")
    ET.SubElement(vis, "origin", {"xyz": p3(g["base_visual_xyz"]), "rpy": p3(g["base_visual_rpy"])})
    ET.SubElement(ET.SubElement(vis, "geometry"), "mesh",
                  {"filename": f"package://zero_description/meshes/{g['base_mesh']}"})
    ine = ET.SubElement(base, "inertial")
    ET.SubElement(ine, "origin", {"xyz": p3(g["base_com"]), "rpy": "0 0 0"})
    ET.SubElement(ine, "mass", {"value": str(g["base_mass"])})
    ET.SubElement(ine, "inertia", {"ixx": "2.6e-4", "ixy": "0", "ixz": "0",
                                   "iyy": "2.3e-4", "iyz": "0", "izz": "1.5e-4"})

    k = ET.SubElement(robot, "joint",
                      {"name": L.prefixed(side, g["knuckle_joint"]), "type": "revolute"})
    ET.SubElement(k, "origin", {"xyz": p3(g["knuckle_xyz"]), "rpy": "0 0 0"})
    ET.SubElement(k, "parent", {"link": L.prefixed(side, g["base_link"])})
    ET.SubElement(k, "child", {"link": L.prefixed(side, g["knuckle_link"])})
    ET.SubElement(k, "axis", {"xyz": p3(g["knuckle_axis"])})
    lo, hi = g["knuckle_limit"]
    ET.SubElement(k, "limit", {"lower": str(lo), "upper": str(hi),
                               "effort": "50", "velocity": "0.5"})

    kn = ET.SubElement(robot, "link", {"name": L.prefixed(side, g["knuckle_link"])})
    kvis = ET.SubElement(kn, "visual")
    ET.SubElement(kvis, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(ET.SubElement(kvis, "geometry"), "mesh",
                  {"filename": f"package://zero_description/meshes/{g['knuckle_mesh']}"})
    kine = ET.SubElement(kn, "inertial")
    ET.SubElement(kine, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(kine, "mass", {"value": "0.009"})
    ET.SubElement(kine, "inertia", {"ixx": "1.7e-6", "ixy": "0", "ixz": "0",
                                    "iyy": "1.6e-6", "iyz": "0", "izz": "3.2e-7"})


def build(key: str) -> ET.Element:
    r = L.ROBOTS[key]
    if not r["urdf"].exists():
        raise SystemExit(f"missing {r['urdf']}")
    arm = ET.parse(r["urdf"]).getroot()
    robot = ET.Element("robot", {"name": f"zero_{key}"})
    for mat in arm.findall("material"):
        robot.append(copy.deepcopy(mat))
    ET.SubElement(robot, "link", {"name": "world"})

    if r.get("single_body"):
        # A humanoid is one body carrying both arms, so it is mounted once and keeps its own names.
        # The G1 already spells its joints left_shoulder_pitch_joint and right_..., so a per-side
        # prefix would give left_left_shoulder_pitch_joint and break the by-name binding with the
        # MJCF. Copied verbatim apart from the mesh paths.
        x, y, z = r["base_pos"]
        w, qx, qy, qz = r["base_quat"]
        yaw = 2.0 * math.atan2(qz, w)          # base_quat is a yaw-only turn to face the table
        m = ET.SubElement(robot, "joint", {"name": "pelvis_mount", "type": "fixed"})
        ET.SubElement(m, "origin", {"xyz": f"{x} {y} {z}", "rpy": f"0 0 {yaw}"})
        ET.SubElement(m, "parent", {"link": "world"})
        ET.SubElement(m, "child", {"link": r["urdf_root"]})
        for el in prefix_side(arm, None, r):
            robot.append(el)
    else:
        mounts = L.robot_mounts(key)
        for side in L.SIDES:
            x, y, z = mounts[side]
            m = ET.SubElement(robot, "joint", {"name": f"{side}_mount", "type": "fixed"})
            ET.SubElement(m, "origin", {"xyz": f"{x} {y} {z}", "rpy": "0 0 0"})
            ET.SubElement(m, "parent", {"link": "world"})
            ET.SubElement(m, "child", {"link": L.prefixed(side, r["urdf_root"])})
            for el in prefix_side(arm, side, r):
                robot.append(el)
            if r.get("graft_urdf"):
                add_graft_urdf(robot, side, r)

    add_ros2_control(robot, key)
    return robot


def copy_meshes(key: str) -> int:
    r = L.ROBOTS[key]
    dst = L.PKG / "meshes"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for pattern in r["mesh_glob"]:
        for f in sorted(r["mesh_src"].glob(pattern)):
            shutil.copy2(f, dst / f.name)
            n += 1
    # .mtl siblings, else the .obj materials resolve to nothing in RViz
    for f in sorted(r["mesh_src"].glob("*.mtl")):
        shutil.copy2(f, dst / f.name)
    # The grafted gripper's meshes come from a different tree (see scripts/fetch_vendor.sh), and
    # only the two the URDF actually references are copied.
    g = r.get("graft_urdf")
    if g:
        for name in (g["base_mesh"], g["knuckle_mesh"]):
            src = g["mesh_src"] / name
            if not src.exists():
                raise SystemExit(f"missing {src}; run: bash scripts/fetch_vendor.sh")
            shutil.copy2(src, dst / name)
            n += 1
    return n


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "rebot"
    if key not in L.ROBOTS:
        raise SystemExit(f"unknown robot {key!r}; known: {list(L.ROBOTS)}")
    robot = build(key)
    out = L.PKG / "urdf" / f"zero_{key}.urdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(out, encoding="utf-8", xml_declaration=True)

    n = copy_meshes(key)
    joints = [(j.get("name"), j.get("type")) for j in robot.findall("joint")]
    movable = [x for x, t in joints if t != "fixed"]
    rcj = [x.get("name") for rc in robot.findall("ros2_control") for x in rc.findall("joint")]
    sens = [x.get("name") for rc in robot.findall("ros2_control") for x in rc.findall("sensor")]
    print(f"[{key}] wrote {out}")
    print(f"  links {len(robot.findall('link'))}  joints {len(joints)}  movable {len(movable)}")
    print(f"  ros2_control joints ({len(rcj)}): {rcj}")
    print(f"  camera sensors ({len(sens)}): {sens}")
    print(f"  copied {n} meshes")


if __name__ == "__main__":
    main()
