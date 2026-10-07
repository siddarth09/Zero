"""Assert the URDF, the MJCF and the controller config agree on joint names."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

import mujoco
import yaml

import zero_layout as L


KEY = sys.argv[1] if len(sys.argv) > 1 else "rebot"
URDF = L.PKG / "urdf" / f"zero_{KEY}.urdf"
MJCF = L.PKG / "mjcf" / f"zero_{KEY}.xml"


def urdf_joints() -> set[str]:
    root = ET.parse(URDF).getroot()
    return {j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"}


def urdf_ros2_control_joints() -> set[str]:
    root = ET.parse(URDF).getroot()
    return {j.get("name") for rc in root.findall("ros2_control") for j in rc.findall("joint")}


def mjcf_joints() -> set[str]:
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)}


def mjcf_actuated_joints() -> set[str]:
    """Joints that some actuator actually drives, not actuator names.

    Comparing actuator names only works by luck: menagerie names the reBot's actuators after
    their joints, but Panda's are `actuator1..8`. ros2_control needs every commanded joint to
    have an actuator behind it, so resolve each actuator's transmission target instead.
    `actuator_trnid[:, 0]` is the joint id for trntype JOINT.
    """
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    out: set[str] = set()
    for i in range(m.nu):
        if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            out.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0]))
    return out


def controller_joints() -> set[str]:
    path = L.PKG.parent / "zero_bringup" / "config" / f"{KEY}_controllers.yaml"
    if not path.exists():
        return set()
    cfg = yaml.safe_load(path.read_text())
    out: set[str] = set()
    for key, val in cfg.items():
        if isinstance(val, dict) and "ros__parameters" in val:
            out |= set(val["ros__parameters"].get("joints", []) or [])
    return out


def mjcf_cameras() -> set[str]:
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)}


def urdf_camera_sensors() -> set[str]:
    """Camera sensors only. The <sensor> block also carries the fingertip force/torque sensors,
    which are distinguished by a `mujoco_type` param; cameras have none. Matching on the tag
    alone made this check fail the moment F/T was added, reporting the F/T names as 'unexpected'
    cameras."""
    root = ET.parse(URDF).getroot()
    out = set()
    for rc in root.findall("ros2_control"):
        for x in rc.findall("sensor"):
            types = {p.get("name") for p in x.findall("param")}
            if "mujoco_type" not in types:
                out.add(x.get("name"))
    return out


def urdf_ft_sensors() -> set[str]:
    root = ET.parse(URDF).getroot()
    out = set()
    for rc in root.findall("ros2_control"):
        for x in rc.findall("sensor"):
            for p in x.findall("param"):
                if p.get("name") == "mujoco_type" and p.text == "fts":
                    out.add(x.get("name"))
    return out


def mjcf_ft_sensors() -> set[str]:
    """MJCF force/torque sensor pairs, reported by the base name the URDF must declare.

    mujoco_ros2_control resolves a `<sensor mujoco_type="fts">` named X to MJCF sensors X_force
    and X_torque. If only one of the pair exists the plugin logs an error and skips the sensor:
    it publishes nothing and the arm still works, so this is the same by-name failure as the
    cameras and the joints.
    """
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(m.nsensor)}
    return {n[:-6] for n in names if n and n.endswith("_force") and f"{n[:-6]}_torque" in names}


def check_kinematics(n_samples: int = 500, tol_mm: float = 0.5) -> bool:
    """Same joint angles in the URDF and the MJCF must give the same end-effector pose.

    Name parity is not enough. menagerie inverted the rotation axis on all six arm joints
    relative to Seeed's official URDF: names matched, geometry matched, the zero pose matched,
    and forward kinematics disagreed by 766 mm on average (1448 mm worst case) because every
    joint turned the wrong way. mujoco_ros2_control binds the two descriptions by name, so it
    would have commanded joints happily while the sim mirrored every motion.
    `scripts/fix_mjcf_conventions.py` corrects it; this asserts it stays corrected.

    Compiles the single-arm URDF in MuJoCo (MuJoCo reads URDF natively) and compares against the
    single-arm MJCF rather than the bimanual pair: the prefixing is already covered above and a
    single arm isolates the convention question.
    """
    import numpy as np

    if KEY != "rebot":
        print(f"SKIP  kinematic check is reBot-specific (other targets ship URDF and MJCF from one source, "
              f"verified at 0.0000 mm); skipped for {KEY}")
        return True
    if not L.SEEED_URDF.exists():
        print("SKIP  Seeed URDF not on disk, kinematic check skipped")
        return True

    txt = L.SEEED_URDF.read_text().replace(
        "package://rebotarm_bringup/description/meshes_rs/", "")
    txt = txt.replace(
        "<robot",
        f'<robot><mujoco><compiler meshdir="{L.SEEED_MESHES}" balanceinertia="true" '
        'discardvisual="false" fusestatic="false"/></mujoco>', 1)
    mu = mujoco.MjModel.from_xml_string(txt)
    mm = mujoco.MjModel.from_xml_path(str(L.MENAGERIE_ARM))
    du, dm = mujoco.MjData(mu), mujoco.MjData(mm)

    def _id(m, obj, n):
        i = mujoco.mj_name2id(m, obj, n)
        assert i >= 0, f"{n} missing"  # -1 would silently index the last element
        return i

    bu = _id(mu, mujoco.mjtObj.mjOBJ_BODY, "gripper_end")
    bm = _id(mm, mujoco.mjtObj.mjOBJ_BODY, "gripper_end")
    rng = np.random.default_rng(0)
    lo, hi = mm.jnt_range[:6, 0], mm.jnt_range[:6, 1]
    worst = 0.0
    for _ in range(n_samples):
        q = rng.uniform(lo, hi)
        for k, j in enumerate(L.ARM_JOINTS):
            du.qpos[mu.jnt_qposadr[_id(mu, mujoco.mjtObj.mjOBJ_JOINT, j)]] = q[k]
            dm.qpos[mm.jnt_qposadr[_id(mm, mujoco.mjtObj.mjOBJ_JOINT, j)]] = q[k]
        mujoco.mj_forward(mu, du)
        mujoco.mj_forward(mm, dm)
        worst = max(worst, float(np.linalg.norm(du.xpos[bu] - dm.xpos[bm])))
    ok = worst * 1000 <= tol_mm
    print(f"{'OK  ' if ok else 'FAIL'}  {'URDF vs MJCF kinematics':28} "
          f"max {worst * 1000:.3f} mm over {n_samples} configs (tol {tol_mm})")
    if not ok:
        print("        run scripts/fix_mjcf_conventions.py")
    return ok


def check_eef_frame(n_samples: int = 300, tol_mm: float = 0.5) -> bool:
    """The point the IK servos must be the point the MJCF calls the end effector.

    Runs for every robot, unlike check_kinematics(). Joint-level FK agreeing does not mean the
    tool point agrees: the recorded action is the eef site's pose, and `eef_offset` shifts it off
    the gripper body in both descriptions independently. Getting that composition wrong is
    silent, since IK still converges cleanly onto the wrong point, so the arm reaches confidently
    past the object and it reads as a bad grasp pose or a calibration problem.

    Caught once already: pinocchio's `Frame` places its `placement` relative to the parent joint,
    not the parent frame, so the natural-looking SE3(I, offset) anchored the tool 119.9 mm
    (reBot) / 107.0 mm (Panda) from the real gripper.
    """
    import numpy as np
    sys.path.insert(0, str(L.PKG.parent / "zero_control"))
    from zero_control.ik import ArmIK

    r = L.ROBOTS[KEY]
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    d = mujoco.MjData(m)
    off = np.array(r["eef_offset"], dtype=float)
    worst = 0.0
    for side in L.SIDES:
        jn = [L.prefixed(side, j) for j in r["arm_joints"]]
        ik = ArmIK(str(URDF), r["urdf_eef_frame"].format(side=side), jn, r["eef_offset"],
                   r.get("eef_quat", (1.0, 0.0, 0.0, 0.0)))
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, L.prefixed(side, r["eef_body"]))
        qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in jn]
        jr = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                       for n in jn])
        rng = np.random.default_rng(0)
        for _ in range(n_samples):
            q = rng.uniform(jr[:, 0], jr[:, 1])
            d.qpos[:] = 0
            for i, a in enumerate(qadr):
                d.qpos[a] = q[i]
            mujoco.mj_forward(m, d)
            p_mj = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ off
            qp = np.zeros(ik.model.nq)
            qp[ik.qidx] = q
            worst = max(worst, float(np.linalg.norm(p_mj - ik.fk(qp).translation)))
    ok = worst * 1000 <= tol_mm
    print(f"{'OK  ' if ok else 'FAIL':4}  {'IK tool point vs MJCF':28} "
          f"max {worst * 1000:.4f} mm over {n_samples * len(L.SIDES)} configs (tol {tol_mm})")
    return ok


def check_tcp(tol_mm: float | None = None) -> bool:
    """The tool point must sit where the fingers actually close.

    `eef_offset` moves the IK's target, and the recorded action, from the gripper body to the
    pinch point. Nothing detects a wrong one: IK still converges onto a point in mid-air next to
    the gripper, so the arm reports arriving at the object and the fingers shut on nothing. The
    reBot shipped 113 mm out (a guessed 0.10 along +z when its fingers close 49 mm along -x) and
    it presented as "the gripper cannot grasp anything"; Panda was 17.9 mm out, which biased
    every grasp. scripts/measure_tcp.py prints the correct value.
    """
    # Per-robot, because the reBot's offset is locked by the recorded dataset and differs from a
    # fresh measurement by a known 15.6 mm. See ROBOTS["rebot"]["eef_offset"].
    if tol_mm is None:
        tol_mm = float(L.ROBOTS[KEY].get("tcp_tol_mm", 5.0))
    sys.path.insert(0, str(L.ROOT / "scripts"))
    import measure_tcp

    off, err = measure_tcp.measure(KEY)
    ok = err * 1000 <= tol_mm
    print(f"{'OK  ' if ok else 'FAIL':4}  {'tool point vs fingers':28} "
          f"{err * 1000:.2f} mm from the grasp centre (tol {tol_mm})")
    if not ok:
        print(f"        measured offset: ({off[0]:.4f}, {off[1]:.4f}, {off[2]:.4f})"
              f"  (run scripts/measure_tcp.py)")
    return ok


def report_superset(name: str, got: set[str], want: set[str]) -> bool:
    """MJCF may legitimately hold more joints than ros2_control commands.

    Panda's second finger joint is driven by an <equality> constraint, not by its own actuator,
    so it exists in the MJCF but must not be declared as a command interface. Assert containment,
    not equality.
    """
    missing = want - got
    ok = not missing
    print(f"{'OK  ' if ok else 'FAIL'}  {name:28} {len(got):>3} present, "
          f"{len(want)} required")
    if missing:
        print(f"        missing: {sorted(missing)}")
    return ok


def report(name: str, got: set[str], want: set[str]) -> bool:
    missing, extra = want - got, got - want
    ok = not missing and not extra
    print(f"{'OK  ' if ok else 'FAIL'}  {name:28} {len(got):>3} items")
    if missing:
        print(f"        missing: {sorted(missing)}")
    if extra:
        print(f"        unexpected: {sorted(extra)}")
    return ok


def main() -> int:
    want = set(L.robot_prefixed_ros2_joints(KEY))
    print(f"expected {len(want)} joints from zero_layout "
          f"for {KEY} ({len(L.SIDES)} sides x "
          f"{len(L.ROBOTS[KEY]['ros2_control_joints'])})\n")

    checks = [
        # Superset: the URDF may hold more movable joints than are commanded. Panda's
        # `finger_joint2` is driven by a mimic/equality, not its own command interface.
        report_superset("URDF movable joints", urdf_joints(), want),
        report("URDF <ros2_control> joints", urdf_ros2_control_joints(), want),
        report_superset("MJCF joints", mjcf_joints(), want),
        report_superset("MJCF actuated joints", mjcf_actuated_joints(), want),
    ]
    # Cameras bind by name like the joints, with the same silent failure.
    checks.append(report("MJCF cameras", mjcf_cameras(), set(L.all_cameras())))
    checks.append(report("URDF camera <sensor>s", urdf_camera_sensors(), set(L.all_cameras())))
    checks.append(check_kinematics())
    checks.append(check_eef_frame())
    checks.append(check_tcp())
    want_ft = {n for n, _ in L.ft_sensors(KEY)}
    checks.append(report("URDF <sensor> fts", urdf_ft_sensors(), want_ft))
    checks.append(report("MJCF force/torque pairs", mjcf_ft_sensors(), want_ft))
    ctrl = controller_joints()
    if ctrl:
        # Controllers may legitimately cover a subset (arm + gripper controller split), so
        # assert containment rather than equality.
        stray = ctrl - want
        ok = not stray
        print(f"{'OK  ' if ok else 'FAIL'}  {'controller yaml joints':28} {len(ctrl):>3} joints"
              f" (subset check)")
        if stray:
            print(f"        not in either description: {sorted(stray)}")
        checks.append(ok)
    else:
        print("SKIP  controller yaml not present yet")

    print()
    if all(checks):
        print("all descriptions agree")
        return 0
    print("DRIFT DETECTED. Fix before launching mujoco_ros2_control.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
