"""Build the bimanual UR5e + Robotiq 2F-85 scene, writing the MJCF and URDF directly.

    MUJOCO_GL=egl python3 scripts/build_ur5e.py

Deliberately standalone: it does NOT import zero_layout, and `ur5e` is not a registry entry, so
nothing in gen_scene/gen_urdf/check_parity touches the files this writes. The flip side is that
nothing checks them either, so the asserts at the end of this file are the only guard that the MJCF
and the URDF still agree about joint names.

Why the UR5e at all: the vx300s cannot mount a wrist camera anywhere good. Its pinch site is
mid-finger, so a reBot-like 90 mm standoff buries the lens in the gripper casting (100% of the
frame is the robot), while the only clear mount at 54 mm puts a grasped object 21 mm from the lens.
The 2F-85's fingers are long enough that the tool point sits out near the fingertips like the
reBot's, which is the geometry that makes a standoff exist at all.

Two changes are made to the upstream assets:

  * menagerie drives the 2F-85 from a TENDON actuator over both driver joints. eef_control_node
    sends a per-joint position command, so that is replaced with a position servo on the left
    driver joint. The gripper already ships a joint equality coupling the two drivers, so the right
    one follows without any extra plumbing.
  * the table is widened. Two UR5e's need ~1.2 m between bases to meet in the middle, against the
    reBot's 0.9 m, and the stock 1.4 m table leaves the bases hanging off the edge.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import shutil

import mujoco
import numpy as np

ROOT = Path("/home/sid/projects25/src/ZERO")
# Which gripper this UR5e wears. Two variants on purpose, so the pair is a controlled ablation:
# same arm, same IK, same controllers, same table, and ONLY the end effector differs. That isolates
# how much of the cross-embodiment gap is the gripper rather than the arm.
#
#   robotiq : Robotiq 2F-85, what a real UR5e cell is fitted with.
#   rebot   : the reBot's OWN gripper, grafted on. The wrist camera rides on the gripper, so this
#             variant sees a geometrically identical scene to the source robot and needs no
#             re-recording. Geometry recovered from commit b026d2b^, which built this scene and
#             measured it ("at 110 mm the near-black fraction is 51.5% against the reBot's 51.4%")
#             but never produced a URDF, controllers or a launch file, so it was never runnable.
VARIANT = "robotiq"
GRIPPERS = {
    "robotiq": {
        "suffix": "",
        "prefix": "rq_",
        "mount_body": "rq_base_mount",
        # Menagerie's own mount frame. Attaching at wrist_3_link's origin with identity instead
        # buries the gripper inside the wrist; see the note at the attach site.
        "attach": "attachment_site",
        "grip_joint": "left_driver_joint",
        "grip_joints": ("left_driver_joint",),
        "urdf_actuated": "robotiq_85_left_knuckle_joint",
        "eef_pos": (0.0, 0.0, 0.134),
        "eef_quat": (-0.5, 0.5, 0.5, 0.5),
        "eef_offset": (0.0, 0.0, 0.142),
        "urdf_eef_frame": "{side}_rq_robotiq_85_base_link",
        "urdf_mount_xyz": "0 0 0", "urdf_mount_rpy": "0 0 0",
        "cam_pos": (0.0, -0.065, 0.075),
        "cam_aim": (0.0, 0.0, 0.170),
        "cam_xyaxes": None,
        # The 2F-85 cannot use reBot's own 0.90 m separation: at |y|=0.45 the home target sits
        # almost directly over the shoulder and the solve stalls 30.8 mm out on a singularity.
        "base_x": -0.15, "base_sep": 1.20,
        "grip_range": (0.8, 0.0),
        "grip_kp": 100.0,
        "act_force": 60.0,
        "ft_fingers": ("left_follower", "right_follower"),
        "ft_no_load": 2.4,
        "squeeze_cap": 7.6,
    },
    "rebot": {
        "suffix": "_rg",
        "prefix": "rg_",
        "mount_body": "rg_gripper_end",
        # (host body, pos, quat) rather than a site: the reBot gripper has no mount site, and these
        # are the measured values from the earlier build. 0.21 along wrist_3's +y is the 100 mm
        # flange plus 110 mm of standoff, which is what keeps the wrist camera (100 mm BEHIND the
        # gripper) from sitting inside the UR5e's wrist.
        "attach": ("wrist_3_link", (0.0, 0.21, 0.0), (0.5, -0.5, -0.5, 0.5)),
        "grip_joint": "gripper_joint1",
        # Two independent slides, both commanded, exactly as on the reBot.
        "grip_joints": ("gripper_joint1", "gripper_joint2"),
        "urdf_actuated": None,
        # The reBot's own eef site and camera, in the same frame, so the recorded observations and
        # the tool point mean the same thing on both robots.
        "eef_pos": (-0.0109, 0.0, 0.0050),
        "eef_quat": (1.0, 0.0, 0.0, 0.0),
        "eef_offset": (-0.0109, 0.0, 0.0050),
        "urdf_eef_frame": "{side}_rg_gripper_end",
        # 110 mm along tool0's +z, then -pi/2 about y. See the note at the mount joint.
        "urdf_mount_xyz": "0 0 0.11", "urdf_mount_rpy": "0 -1.5707963 0",
        "cam_pos": (-0.1009, 0.0, 0.0050),
        "cam_aim": None,
        "cam_xyaxes": (0, 1, 0, 0, 0, -1),
        # NOT reBot's own -0.05 / 0.90, though this gripper can reach it (homes solve to
        # 0.24/0.01 mm there). Matching the source base does not improve the front view and
        # measured slightly worse: 17.7 against 15.6 mean |pixel diff| vs the training front
        # frame, because at reBot's placement the UR5e sits closer to the camera and fills more
        # of it. A UR5e does not look like a reBot wherever it is bolted, so the front view needs
        # cross-painting or multi-embodiment data, not repositioning.
        "base_x": -0.15, "base_sep": 1.20,
        "grip_range": (0.0, 0.05),
        # kp from the gripper model's own default class, force capped at the reBot's grip_force, so
        # the squeeze this gripper produces matches the one in the training data.
        "grip_kp": 5000.0,
        "act_force": 15.0,
        "ft_fingers": ("gripper_left", "gripper_right"),
        # Slide fingers, no linkage preload, so nothing to subtract and the cap is reBot's own.
        "ft_no_load": 0.0,
        "squeeze_cap": 15.0,
    },
}
if len(sys.argv) > 1:
    if sys.argv[1] not in GRIPPERS:
        raise SystemExit(f"unknown gripper {sys.argv[1]!r}; known: {list(GRIPPERS)}")
    VARIANT = sys.argv[1]
G = GRIPPERS[VARIANT]
SFX = G["suffix"]

OUT_MJCF = ROOT / f"zero_description/mjcf/zero_ur5e{SFX}.xml"
OUT_URDF = ROOT / f"zero_description/urdf/zero_ur5e{SFX}.urdf"
OUT_CTRL = ROOT / f"zero_bringup/config/ur5e{SFX}_control.yaml"
OUT_CTRLRS = ROOT / f"zero_bringup/config/ur5e{SFX}_controllers.yaml"
VENDOR = ROOT / "robots/_vendor"
ARM_URDF = VENDOR / "ur5e_flat.urdf"
GRIP_URDF = {"robotiq": VENDOR / "robotiq_2f85_flat.urdf",
             "rebot": ROOT / "robots" / "shared_gripper" / "rebot_gripper.urdf"}[VARIANT]
# package:// prefixes to rewrite, and where their files live. The composed URDF ships its meshes
# from zero_description so it resolves without ur_description or robotiq_description installed.
MESH_SRC = {
    "ur_description": VENDOR / "ur_description",
    "robotiq_description": VENDOR / "robotiq/robotiq_description",
}
URDF_MESH_OUT = ROOT / "zero_description/meshes/ur5e_urdf"
# The MJCF mounts the gripper on menagerie's attachment_site; the URDF must mount it somewhere that
# lands in the same place. Checked numerically: MJCF attachment_site and URDF tool0 agree to
# 1.08 mm at a test configuration, while the two wrist_3_link frames are 99 mm apart. So tool0 is
# the URDF anchor, and wrist_3_link would be wrong by the length of the wrist.
URDF_MOUNT_LINK = "tool0"
# Menagerie's ur5e.xml declares <body name="base" quat="0 0 0 -1">, a 180 degree rotation about z
# that the UR URDF's base_link does not have. Mounting both at the same place with identity leaves
# the two descriptions 1.59 m apart, with the telltale that z agrees to 1.5 mm while x is
# sign-flipped. The URDF mount carries the same yaw so the TF tree and the simulation agree.
URDF_MOUNT_YAW = np.pi
MESH_OUT = Path("/home/sid/projects25/src/ZERO/zero_description/meshes/ur5e")
MEN = Path("/home/sid/.cache/robot_descriptions/mujoco_menagerie")
ARM_XML = MEN / "universal_robots_ur5e" / "ur5e.xml"
GRIP_XML = {"robotiq": MEN / "robotiq_2f85" / "2f85.xml",
            "rebot": ROOT / "robots" / "shared_gripper" / "rebot_gripper.xml"}[VARIANT]

SIDES = ("left", "right")
TABLE_TOP_Z = 0.75
# The stock ZERO table, identical to what the reBot and vx300s scenes use (zero_layout.TABLE_*),
# so the three embodiments differ only in the arms. Bases at +/-0.60 still sit on a top that spans
# +/-0.70, with 100 mm to spare, so widening it was never necessary.
TABLE_HX, TABLE_HY, TABLE_HZ = 0.60, 0.70, 0.02   # the stock ZERO table, as reBot/vx300s use
TABLE_CENTER_XY = (0.05, 0.0)
LEG_R = 0.03
BASE_X = G["base_x"]
# Base height relative to the table top. 0.0 = bases ON the table, which is where they are, by
# request, and where reBot and the vx300s put theirs.
#
# This is NOT free of consequences, and the measurement is kept here because it will come up
# again. reBot's tool poses are near-horizontal (its eef +x is only 0.099 downward), so reaching a
# can at z=0.808 that way puts a UR5e's forearm and wrist BELOW the tool point. Scored on 624
# waypoints from three real demo episodes:
#     table hy 0.70, sep 1.20, z  0.00 :  9.5% of waypoints drive the arm >5 mm into the table
#                                         (worst 60 mm), 0.0% unreachable      <- this layout
#     table hy 0.70, sep 1.56, z -0.15 :  0.0% into the table, 9.1% unreachable (the handover
#                                         lands 850-1095 mm out against a 850 mm reach)
#     table hy 0.50, sep 1.16, z -0.10 :  0.0% and 0.0%, but needs a narrower table
# The IK cannot see any of this: it is collision-blind and reports 0 mm residual at every one of
# those waypoints, so the arm jams on the table and stalls while the log says the target was met.
# Standing the arms off the table removed the collisions and did NOT change the live behaviour,
# so it is not the whole story either.
BASE_Z = 0.0
# Moved back from 0.00. Neither home nor the task points constrain this: reBot's home pose solves
# to under 0.3 mm anywhere from 0.00 to -0.25, and pick/place/handover stay reachable throughout,
# because the UR5e's 850 mm reach is generous against this table. So it is a framing choice, not a
# kinematic one. The table top spans x -0.55..0.65, so a 75 mm base radius here keeps 325 mm of
# clearance to the back edge.
BASE_SEP = G["base_sep"]
CAN_R, CAN_HH = 0.033, 0.0575
PICK = (0.34, +0.45)
PLACE = (0.34, -0.45)
CAM_RES = (224, 224)
WRIST_FOVY = 70.0
FRONT_FOVY = 58.0
CAM_ZNEAR = 0.010       # must equal zero_layout.CAM_ZNEAR: it is the value the demos were recorded at
ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
# The two vendor sources decompose the 2F-85 linkage differently and share no joint names: the
# MJCF has driver/coupler/spring_link/follower (4 per finger), the URDF has
# knuckle/inner_knuckle/finger_tip (3 per finger, all but one a <mimic>). Only the ACTUATED joint
# has to carry the same name in both, since that is the one ros2_control declares. Both run 0..0.8
# with 0 open, so renaming the URDF's actuated knuckle onto the MJCF's driver name is sufficient.
URDF_GRIP_ACTUATED = G["urdf_actuated"]
GRIP_JOINT = G["grip_joint"]
GRIP_JOINTS = G["grip_joints"]   # every joint the gripper controller commands
GRIP_PREFIX = G["prefix"]
def grip_joint_names(side):
    """The commanded gripper joints for one side, as they are named in both descriptions."""
    return [f"{side}_{GRIP_PREFIX}{j}" for j in GRIP_JOINTS]
MOUNT_BODY = G["mount_body"]
# Fingertip F/T, one per finger rather than one per gripper: the signal that matters during a
# grasp is the wrench through each finger's own joint, which separates "both pads loaded" from
# "object resting against one pad". The name is a contract -- mujoco_ros2_control resolves a
# <sensor mujoco_type="fts"> named X to MJCF sensors X_force and X_torque -- so both ends are
# generated from this one list.
#
# The follower is the sensorised body: it is the last link before the pad, so the whole grasp load
# passes through its joint. Sensorising the driver instead would read the actuator through the
# 4-bar rather than the contact.
FT_FINGERS = G["ft_fingers"]
# Two different quantities that one constant used to conflate. The actuator cap is what the
# driver servo may exert; the squeeze cap is the divisor force.py uses to turn a finger's F/T
# reading into the 0..1 squeeze channel.
GRIP_ACT_FORCE = G["act_force"]    # N, forcerange on the gripper position servo
# CALIBRATED AGAINST reBOT at BOTH ends, not picked. force.py computes
#     squeeze = max(mean per-finger |F| - FT_NO_LOAD, 0) / FT_SQUEEZE_CAP
# In the training data the squeeze channel is sharply bimodal: ~0.045 when the hand is empty and
# ~1.0 when it holds the can, with only 1.3% of frames between 0.15 and 0.35. So both ends have to
# line up, not just the hold. Measured on the CAN:
#     open, nothing held   2.4 N   <- FT_NO_LOAD, the four-bar's own preload
#     holding the can     10.0 N   -> (10.0 - 2.4) / 7.6 = 1.00, matching reBot's 1.00
#     closed on nothing   46.8 N   -> 5.8, a state the demos also rarely visit
# Subtracting the no-load reading is what makes the channel measure grip instead of preload.
# Without it the UR5e idles at 2.4/10.0 = 0.24, which sits inside the empty band, so the policy
# reads "partly gripping" whenever the hand is actually open. reBot's slide fingers have no
# preload, so its FT_NO_LOAD is 0 and nothing about it changes.
#
# An earlier value of 353.0 here was measured with a broken free-joint lookup that closed the
# gripper on the TRAY instead of the can: both free joints were unnamed, so mj_name2id returned
# -1 and jnt_qposadr[-1] silently returned the tray's address. Both are named now.
FT_NO_LOAD = G["ft_no_load"]
FT_SQUEEZE_CAP = G["squeeze_cap"]
ARM_KP = (3000.0, 3000.0, 3000.0, 500.0, 500.0, 500.0)
GRIP_KP = G["grip_kp"]

# Measured on the compiled gripper, in the base_mount frame: the pads sit at z = 134 mm and
# x = +/-47 mm when open, so the approach axis is +z and the jaw axis is +/-x.
TOOL_Z = 0.134                 # tool point, i.e. where the pads meet
# Wrist camera, in the mount frame. Matched against the RECORDED TRAINING VIDEO, not against a
# render of zero_rebot.xml -- for most of this project those were different images, see
# zero_layout.CAM_ZNEAR.
#
# It cannot sit on the approach axis: the 2F-85's housing spans z = -7..153 mm with the pads
# meeting at 134, so any on-axis mount is inside the casting. From mesh vertices (geom_rbound
# overestimates an elongated hull) the housing narrows in y as z rises: 37.5 mm at z=30, 35.0 at
# z=50, 19.5 at z=70, 17.0 at z=90.
#
# Three things then have to hold at once, and the reference for all three is reBot's own wrist
# view: gripper 32.2% of frame at vertical centroid 0.329, NO housing, and the can at 57.3% during
# the grasp from a 90 mm standoff. Judged on body-keyed segmentation, as
# (wedge % / centroid / housing % / can at grasp %):
#     y -0.050 z 0.075 aim 0.164 :  9.5 / 0.644 /  0.0 / 75.9
#     y -0.045 z 0.085 aim 0.137 :  7.7 / 0.498 /  0.0 / 18.2   centred, but the can is clipped
#     y -0.050 z 0.075 aim 0.130 : 13.5 / 0.575 / 15.4 / 60.7   housing in frame
#     y -0.060 z 0.060 aim 0.164 : 15.3 / 0.736 /  8.5 / 49.4
#     y -0.065 z 0.075 aim 0.170 : 12.5 / 0.728 /  0.0 / 58.8   <- chosen, 88 mm standoff
# The lateral offset is what buys standoff without moving toward the pads, which is what lets the
# can clear the 26.5 mm near plane while the housing stays out of frame. Centring the wedges
# vertically is the one thing not achieved: pulling them up to reBot's 0.329 requires tilting down,
# which drags the housing in and clips the can. Coverage cannot match either -- a 2F-85 shows two
# thin wedge tips where reBot shows one broad flat face.
WRIST_CAM_POS = G["cam_pos"]
WRIST_CAM_AIM = G["cam_aim"]      # None when the camera uses explicit xyaxes instead
WRIST_CAM_XYAXES = G["cam_xyaxes"]
# The front camera is the stock ZERO one: zero_layout.SCENE_CAMS["front"] looking at HANDOVER_POS.
FRONT_CAM_EYE = (1.15, 0.0, 1.15)
LOOK_AT = (0.34, 0.0, TABLE_TOP_Z + 0.13)
# The site frame ZERO's 20-dim action is defined in: +x out of the jaw toward the object, +y along
# the jaw axis. In mount coordinates that is x_site=+z (the gripper extends along +z, pads meeting
# at z=134 mm), y_site=+x (the pads open to +/-x), z_site=+y.
#
# (0.5, 0.5, 0.5, 0.5) is the WRONG cycle for that and was here first: it gives x_site=+y,
# y_site=+z, z_site=+x, i.e. it names the jaw axis as the approach axis. Nothing catches this,
# because the home solve then matches reBot's SITE orientation perfectly while aiming the physical
# gripper 90 degrees off, sideways across the table instead of down it. Verified by measurement
# rather than by reading the quaternion: mount +z must be the axis the pads sit on.
EEF_QUAT = G["eef_quat"]
EEF_POS = G["eef_pos"]      # the eef site, in the gripper mount body frame
# MuJoCo cameras look along -z_cam. Pointing that at the object means z_cam = -mount_z, and putting
# image-right along the jaw gives x_cam = mount_x, hence y_cam = -mount_y.
WRIST_XYAXES = (1, 0, 0, 0, -1, 0)
# Home is SOLVED, not hand-guessed. Hand-picked joint angles put one tool point 1.31 m out in y,
# off the table entirely, and both 570 mm above it: a posture that compiles and renders while being
# useless to look at or servo from. The targets below are what the pose has to achieve.
#   * each tool point hovers over its own object, high enough not to be touching it
#   * the jaw points down at the table, which is how the demos approach
# The targets are reBot's OWN home, read off zero_rebot.xml's `home` keyframe by forward
# kinematics on its {side}_eef site, so the two scenes stage the same pose and a side-by-side
# render differs only in the arms. Copied as numbers rather than imported, since the UR5e path
# deliberately does not depend on zero_layout.
#
# Note the asymmetry is real and worth keeping: reBot's left gripper points nearly straight down
# the table (+x) while the right is yawed 38 degrees inboard, pre-aimed at the handover. Mirroring
# the left pose onto the right would look tidier and stage the wrong thing.
HOME_TARGET = {"left":  (0.3362,  0.4515, 0.9780),
               "right": (0.3406, -0.4072, 0.9526)}
# Columns are the eef site axes in world: +x out of the jaw toward the object, +y the jaw axis.
HOME_ROT = {
    "left":  np.array([[ 0.9869,  0.1452, -0.0703],
                       [ 0.1282, -0.9703, -0.2051],
                       [-0.0980,  0.1934, -0.9762]]),
    "right": np.array([[ 0.7774,  0.6250,  0.0707],
                       [ 0.6182, -0.7385, -0.2692],
                       [-0.1160,  0.2530, -0.9605]]),
}

# Seed the solve elbow-up so it does not find an elbow-down branch that folds under the table.
SEED = {"left": (-1.2, -1.2, 1.4, -1.75, -1.57, 0.0),
        "right": (1.2, -1.9, -1.4, -1.4, 1.57, 0.0)}


def write_bringup(m, d, home: dict) -> None:
    """Emit ur5e_control.yaml and ur5e_controllers.yaml from the model that was just built.

    These were hand-written, and drifted the moment anything moved: the gripper joint rename left
    `*_grip_joints` naming a joint that no longer existed, and correcting the eef frame left every
    `*_home` a solution to the old pose. Both are silent until runtime. Deriving them from the
    compiled model is the only version of this that stays true. Note this reads the MODEL, not
    zero_layout -- ur5e is deliberately outside that registry.
    """
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
    mujoco.mj_forward(m, d)

    ft_names, ft_rots, ft_sides = [], [], []
    for side in SIDES:
        for finger in FT_FINGERS:
            name = f"{side}_{finger}_ft"
            eb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_{MOUNT_BODY}")
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{name}_site")
            assert eb >= 0 and sid >= 0, f"missing body/site for {name}"
            # Rotation taking a wrench from the sensor site frame into the tool frame. The follower
            # bodies do rotate as the jaw closes, so this is the value at the home (open) pose.
            R = d.xmat[eb].reshape(3, 3).T @ d.site_xmat[sid].reshape(3, 3)
            ft_names.append(name)
            ft_rots += [float(v) for v in np.asarray(R).reshape(-1)]
            ft_sides.append(side)

    q = ", ".join(f"{v}" for v in EEF_QUAT)
    L = [
        "# GENERATED by scripts/build_ur5e.py from the compiled model. Do not hand-edit: every",
        "# value here is measured off zero_ur5e.xml/.urdf, and the two that were hand-written",
        "# (grip joint names, solved homes) both went stale the first time the model changed.",
        "#",
        "# `ur5e` is deliberately not a zero_layout registry entry, so gen_bringup.py does not",
        "# produce this and check_parity.py does not check it.",
        "",
        "/**:",
        "  ros__parameters:",
        f"    robot: ur5e{SFX}",
        "    # Tool point in the URDF gripper-base frame. The MJCF eef site sits at z=0.134 in the",
        "    # MJCF mount frame; the two gripper-base frames are orientation-identical (0.00 deg)",
        "    # and 7.98 mm apart, and 0.142 absorbs that so the IK tool point lands on the site.",
        f"    eef_offset: [{G['eef_offset'][0]}, {G['eef_offset'][1]}, {G['eef_offset'][2]}]",
        "    # +x out of the jaw toward the object, +y along the jaw axis: the ZERO 20-dim action",
        "    # convention. Must stay equal to build_ur5e.EEF_QUAT, which is why it is written here.",
        f"    eef_quat: [{q}]",
        "    # (value at closed, value at open). The Robotiq driver joint runs [0, 0.8] and closes",
        "    # as it INCREASES, so this pair is descending. eef_control_node computes a signed span,",
        "    # which is what makes that work; a magnitude would pin the channel at one end forever.",
        f"    grip_range: [{G['grip_range'][0]}, {G['grip_range'][1]}]",
        f"    grip_ctrl_n: {len(GRIP_JOINTS)}",
        f"    pick_xyz: [{PICK[0]:.5f}, {PICK[1]:.5f}, {TABLE_TOP_Z + CAN_HH:.5f}]",
        # Tray top plus the can's half height: this is where the TOOL point goes, i.e. where the
        # can's centre must end up, not the tray surface. The surface value was 58 mm too low.
        f"    place_xyz: [{PLACE[0]:.5f}, {PLACE[1]:.5f}, {TABLE_TOP_Z + 0.016 + CAN_HH:.5f}]",
        f"    ws_min: [{BASE_X - 0.55:.4f}, {-BASE_SEP / 2 - 0.35:.4f}, {TABLE_TOP_Z + 0.0575:.4f}]",
        f"    ws_max: [{BASE_X + 0.65:.4f}, {BASE_SEP / 2 + 0.35:.4f}, {TABLE_TOP_Z + 0.55:.4f}]",
        '    cameras: ["front", "left_wrist", "right_wrist"]',
        "    # Fingertip F/T, one per finger. See build_ur5e.FT_FINGERS.",
        "    ft_sensors: [" + ", ".join(f'"{n}"' for n in ft_names) + "]",
        "    ft_side: [" + ", ".join(f'"{s}"' for s in ft_sides) + "]",
        "    ft_rot: [" + ", ".join(f"{v:.6f}" for v in ft_rots) + "]",
        "    # The squeeze NORMALISER (force.py divides the mean per-finger |F| by it), not the",
        "    # actuator cap. See build_ur5e.FT_SQUEEZE_CAP for the measurement.",
        f"    grip_force: {FT_SQUEEZE_CAP}",
        "    # No-load reading subtracted before normalising, so squeeze measures grip and not the",
        "    # four-bar's own preload. See build_ur5e.FT_NO_LOAD.",
        f"    ft_bias: {FT_NO_LOAD}",
    ]
    for side in SIDES:
        js = ", ".join(f'"{side}_{j}"' for j in ARM_JOINTS)
        L += [f"    {side}_eef_frame: " + G["urdf_eef_frame"].format(side=side),
              f"    {side}_arm_joints: [{js}]",
              f"    {side}_grip_joints: [" + ", ".join(grip_joint_names(side)) + "]",
              f"    {side}_home: [" + ", ".join(f"{v:.4f}" for v in home[side]) + "]"]
    OUT_CTRL.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT_CTRL}")

    C = ["# GENERATED by scripts/build_ur5e.py. Do not hand-edit.",
         "controller_manager:", "  ros__parameters:", "    update_rate: 100", "",
         "    joint_state_broadcaster:",
         "      type: joint_state_broadcaster/JointStateBroadcaster", ""]
    # JointGroupPositionController, NOT JointTrajectoryController. This is dictated by
    # eef_control_node, which publishes a Float64MultiArray of joint positions to
    # /{side}_arm_controller/commands at 100 Hz. Only JointGroupPositionController exposes that
    # topic; a JointTrajectoryController takes /joint_trajectory or the follow_joint_trajectory
    # action and silently ignores /commands. Nothing errors, no controller reports a problem, and
    # the arm simply never moves -- the trace showed the measured pose bit-identical across 1478
    # ticks, 0 mm travelled, while the IK dutifully reported a ~77 mm residual against a target it
    # was never going to reach. The grippers were already this type, which is why they responded
    # while the arms did not.
    for side in SIDES:
        C += [f"    {side}_arm_controller:",
              "      type: position_controllers/JointGroupPositionController", ""]
    for side in SIDES:
        C += [f"    {side}_gripper_controller:",
              "      type: position_controllers/JointGroupPositionController", ""]
    for n in ft_names:
        C += [f"    {n}_broadcaster:",
              "      type: force_torque_sensor_broadcaster/ForceTorqueSensorBroadcaster", ""]
    for side in SIDES:
        js = ", ".join(f'"{side}_{j}"' for j in ARM_JOINTS)
        C += [f"{side}_arm_controller:", "  ros__parameters:",
              f"    joints: [{js}]",
              '    command_interfaces: ["position"]',
              '    state_interfaces: ["position", "velocity"]', ""]
    for side in SIDES:
        C += [f"{side}_gripper_controller:", "  ros__parameters:",
              "    joints: [" + ", ".join(f'"{j}"' for j in grip_joint_names(side)) + "]", ""]
    for n in ft_names:
        # force.py subscribes to /<sensor>_broadcaster/wrench, so the topic name is a contract too.
        C += [f"{n}_broadcaster:", "  ros__parameters:",
              f"    sensor_name: {n}",
              f"    frame_id: {n}_site",
              f"    topic_name: /zero/ft/{n}", ""]
    OUT_CTRLRS.write_text("\n".join(C) + "\n")
    print(f"wrote {OUT_CTRLRS} ({len(ft_names)} F/T broadcasters)")
    # Every controller eef_control_node drives over /commands must be able to receive it.
    text = "\n".join(C)
    for side in SIDES:
        for which in ("arm", "gripper"):
            i = text.index(f"    {side}_{which}_controller:")
            kind = text[i:].split("type:", 1)[1].splitlines()[0].strip()
            if kind != "position_controllers/JointGroupPositionController":
                raise SystemExit(
                    f"{side}_{which}_controller is {kind}, which does not subscribe to "
                    f"/{side}_{which}_controller/commands; eef_control_node would publish into "
                    f"the void and the joints would never move")
    print(f"  all {2 * len(SIDES)} commanded controllers accept /commands")


def lookat(eye, target) -> list[float]:
    """MuJoCo camera xyaxes that put `eye` looking at `target`, image-right kept horizontal."""
    fwd = np.asarray(target, float) - np.asarray(eye, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    return [float(v) for v in (*right, *np.cross(right, fwd))]


def stage_urdf_meshes() -> None:
    """Copy the URDF's meshes under zero_description and return nothing.

    The composed URDF is rewritten to reference package://zero_description/..., so the description
    resolves on a machine where ur_description and robotiq_description are not installed, which is
    this one. Keeping the vendor's directory layout under a per-package subdirectory means the
    rewrite is a plain string swap with no path bookkeeping.
    """
    n = 0
    for pkg, src in MESH_SRC.items():
        for f in (src / "meshes").rglob("*"):
            if not f.is_file() or f.suffix.lower() not in (".stl", ".dae", ".obj", ".png", ".jpg"):
                continue
            dst = URDF_MESH_OUT / pkg / f.relative_to(src / "meshes")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            n += 1
    print(f"  staged {n} URDF meshes into {URDF_MESH_OUT}")


def _prefixed(src: Path, prefix: str, rename: dict[str, str] | None = None) -> tuple[list, list]:
    """Read a URDF and return its links and joints with every name prefixed.

    rename maps joint names before prefixing, including <mimic joint=...> references.
    """
    import xml.etree.ElementTree as ET
    rename = rename or {}
    root = ET.parse(src).getroot()
    for el in root.iter():
        for attr in ("name", "link", "joint"):
            if attr in el.attrib and el.tag in ("link", "joint", "parent", "child", "mimic"):
                val = el.get(attr)
                if el.tag in ("joint", "mimic") and attr in ("name", "joint"):
                    val = rename.get(val, val)
                el.set(attr, prefix + val)
        if el.tag == "mesh" and "filename" in el.attrib:
            fn = el.get("filename")
            for pkg in MESH_SRC:
                fn = fn.replace(f"package://{pkg}/meshes/",
                                f"package://zero_description/meshes/ur5e_urdf/{pkg}/")
            el.set("filename", fn)
    # Top-level <material> definitions travel with the links that reference them. Dropping them
    # is not cosmetic-only noise: urdf_parser warns "material 'X' undefined" once per reference,
    # 20 lines per launch here, which buries real errors in the log.
    return root.findall("link"), root.findall("joint"), root.findall("material")


def build_urdf() -> None:
    """Compose the bimanual URDF and its <ros2_control> block, and write it out."""
    import xml.etree.ElementTree as ET

    stage_urdf_meshes()
    robot = ET.Element("robot", {"name": "zero_ur5e"})
    ET.SubElement(robot, "link", {"name": "world"})

    tl = ET.SubElement(robot, "link", {"name": "table"})
    vis = ET.SubElement(tl, "visual")
    ET.SubElement(vis, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(ET.SubElement(vis, "geometry"), "box",
                  {"size": f"{2*TABLE_HX} {2*TABLE_HY} {2*TABLE_HZ}"})
    tj = ET.SubElement(robot, "joint", {"name": "world_to_table", "type": "fixed"})
    ET.SubElement(tj, "parent", {"link": "world"})
    ET.SubElement(tj, "child", {"link": "table"})
    ET.SubElement(tj, "origin", {"xyz": f"0.05 0 {TABLE_TOP_Z - TABLE_HZ}", "rpy": "0 0 0"})

    seen_materials: set = set()
    for side in SIDES:
        y = (BASE_SEP / 2.0) * (1 if side == "left" else -1)
        rename = {URDF_GRIP_ACTUATED: GRIP_JOINT} if URDF_GRIP_ACTUATED else None
        for tag, (links, joints, mats) in (("arm", _prefixed(ARM_URDF, f"{side}_")),
                                           ("grip", _prefixed(GRIP_URDF, f"{side}_{GRIP_PREFIX}",
                                                              rename))):
            for e in links + joints:
                robot.append(e)
            # Materials are global by name and both sides carry the same set, so de-duplicate.
            for e in mats:
                if e.get("name") not in seen_materials:
                    seen_materials.add(e.get("name"))
                    robot.append(e)
        mj = ET.SubElement(robot, "joint", {"name": f"world_to_{side}_base", "type": "fixed"})
        ET.SubElement(mj, "parent", {"link": "world"})
        ET.SubElement(mj, "child", {"link": f"{side}_base_link"})
        ET.SubElement(mj, "origin", {"xyz": f"{BASE_X} {y} {TABLE_TOP_Z + BASE_Z}",
                                     "rpy": f"0 0 {URDF_MOUNT_YAW}"})
        gj = ET.SubElement(robot, "joint", {"name": f"{side}_gripper_mount", "type": "fixed"})
        ET.SubElement(gj, "parent", {"link": f"{side}_{URDF_MOUNT_LINK}"})
        ET.SubElement(gj, "child", {"link": G["urdf_eef_frame"].format(side=side)})
        # The MJCF grafts relative to wrist_3_link, the URDF relative to tool0. Those are NOT the
        # same frame -- MJCF and URDF wrist_3_link differ by 99 mm -- but MJCF attachment_site and
        # URDF tool0 agree to 1.08 mm, so the graft is re-expressed there and the two descriptions
        # agree by construction. Computed, not guessed: inv(T_attachment_site) * T_graft.
        ET.SubElement(gj, "origin", {"xyz": G["urdf_mount_xyz"], "rpy": G["urdf_mount_rpy"]})

    rc = ET.SubElement(robot, "ros2_control",
                       {"name": f"ZeroUr5e{SFX.upper().strip('_') or ''}System", "type": "system"})
    hw = ET.SubElement(rc, "hardware")
    ET.SubElement(hw, "plugin").text = "mujoco_ros2_control/MujocoSystemInterface"
    # $(find pkg), not $(find-pkg-share pkg). The launch pipes this URDF through xacro, and xacro
    # only knows ['find', 'env', 'optenv', 'dirname', 'arg'] -- find-pkg-share is a LAUNCH
    # substitution and xacro rejects it outright, so the whole launch dies before the sim starts.
    ET.SubElement(hw, "param", {"name": "mujoco_model"}).text = (
        f"$(find zero_description)/mjcf/zero_ur5e{SFX}.xml")
    ET.SubElement(hw, "param", {"name": "sim_speed_factor"}).text = "1.0"
    ET.SubElement(hw, "param", {"name": "camera_publish_rate"}).text = "30.0"
    ET.SubElement(hw, "param", {"name": "override_start_position_file"}).text = \
        f"/tmp/zero_ur5e{SFX}_start.xml"
    # Without this the sim starts at the model's zero configuration rather than the solved home,
    # so the arms boot folded through the table.
    ET.SubElement(hw, "param", {"name": "initial_keyframe"}).text = "home"
    for side in SIDES:
        for j in list(ARM_JOINTS) + [f"{GRIP_PREFIX}{g}" for g in GRIP_JOINTS]:
            jn = f"{side}_{j}"
            je = ET.SubElement(rc, "joint", {"name": jn})
            ET.SubElement(je, "command_interface", {"name": "position"})
            ET.SubElement(je, "state_interface", {"name": "position"})
            ET.SubElement(je, "state_interface", {"name": "velocity"})
    # Fingertip F/T. mujoco_ros2_control resolves a sensor named X with mujoco_type "fts" to the
    # MJCF sensors X_force and X_torque, so these names must be exactly the ones build() wrote.
    for side in SIDES:
        for finger in FT_FINGERS:
            se = ET.SubElement(rc, "sensor", {"name": f"{side}_{finger}_ft"})
            ET.SubElement(se, "param", {"name": "mujoco_type"}).text = "fts"
            for axis in ("force", "torque"):
                for c in "xyz":
                    ET.SubElement(se, "state_interface", {"name": f"{axis}.{c}"})

    for cam in ("front", "left_wrist", "right_wrist"):
        se = ET.SubElement(rc, "sensor", {"name": cam})
        for k, v in (("frame_name", f"{cam}_optical_frame"),
                     ("image_topic", f"/zero/{cam}/image_raw"),
                     ("info_topic", f"/zero/{cam}/camera_info"),
                     ("depth_topic", f"/zero/{cam}/depth")):
            ET.SubElement(se, "param", {"name": k}).text = v

    ET.indent(robot, space="  ")
    OUT_URDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_URDF.write_text('<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode"))
    print(f"wrote {OUT_URDF}")
    # The launch feeds this through xacro, so verify it survives that rather than only that it is
    # well-formed XML. A bad substitution is valid XML and kills the launch.
    import subprocess
    # $(find zero_description) needs the built workspace on AMENT_PREFIX_PATH, which is not set
    # when this builder runs from a bare shell. Point at the install tree so the check tests the
    # substitution rather than the caller's environment.
    env = dict(os.environ)
    inst = ROOT.parent.parent / "install"
    if inst.is_dir():
        env["AMENT_PREFIX_PATH"] = f"{inst/'zero_description'}:{env.get('AMENT_PREFIX_PATH','')}"
    r = subprocess.run(["xacro", str(OUT_URDF)], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise SystemExit(f"xacro rejected the emitted URDF:\n{r.stderr.strip()[:400]}")
    print(f"  xacro accepts it ({len(r.stdout)} chars expanded)")
    check_declared_joints(robot)


def check_declared_joints(robot) -> None:
    """Every <ros2_control> joint must exist as a movable URDF joint AND as an MJCF joint.

    controller_manager throws "Joint 'x' not found in URDF" and takes the whole launch down if the
    first holds; mujoco_ros2_control cannot bind a command interface if the second does. Comparing
    the block against the controller YAML is not this check -- both are written from the MJCF side,
    so they agree with each other while disagreeing with the URDF tree.
    """
    declared = [j.get("name") for j in robot.find("ros2_control").findall("joint")]
    tree = {j.get("name") for j in robot.findall("joint") if j.get("type") != "fixed"}
    mj = mujoco.MjModel.from_xml_path(str(OUT_MJCF))
    mjj = {mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(mj.njnt)}
    missing_urdf = [n for n in declared if n not in tree]
    missing_mjcf = [n for n in declared if n not in mjj]
    if missing_urdf or missing_mjcf:
        raise SystemExit(f"declared joints absent from URDF tree: {missing_urdf}\n"
                         f"declared joints absent from MJCF: {missing_mjcf}")
    print(f"  all {len(declared)} declared joints exist in both the URDF tree and the MJCF")
    # Same class of drift is possible for sensors: a <sensor mujoco_type="fts"> named X needs BOTH
    # X_force and X_torque in the MJCF, and a missing one is a runtime failure, not a load error.
    mjs = {mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(mj.nsensor)}
    fts = [se.get("name") for se in robot.find("ros2_control").findall("sensor")
           if (pe := se.find("param")) is not None and pe.text == "fts"]
    absent = [f"{n}_{k}" for n in fts for k in ("force", "torque") if f"{n}_{k}" not in mjs]
    if absent:
        raise SystemExit(f"F/T sensors declared with no MJCF counterpart: {absent}")
    print(f"  all {len(fts)} F/T sensors have both _force and _torque in the MJCF")


def _link_verts(m, geoms):
    """Local-frame vertex arrays per geom, computed once.

    Rebuilding these inside the IK loop made the home solve run for over nine minutes: it is one
    scan over ~100 mesh geoms per iteration, per descent target, per restart.
    """
    rng = np.random.default_rng(0)
    out = []
    for g in geoms:
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mi = m.geom_dataid[g]
            a = m.mesh_vertadr[mi]
            v = m.mesh_vert[a:a + m.mesh_vertnum[mi]].reshape(-1, 3).copy()
            if len(v) > 200:
                v = v[rng.choice(len(v), 200, replace=False)]
        else:
            sz = m.geom_size[g]
            v = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * sz
        out.append((g, v))
    return out


_TABLE_GEOMS: set = set()


def _table_geoms(m) -> set:
    """The table top and legs, i.e. what the arm must not reach through."""
    out = set()
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if n == "table_top" or n.startswith("leg"):
            out.add(g)
    return out


def _link_geoms(m, side: str):
    """Every geom of this arm except its base, which rests ON the table by design."""
    base = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in (f"{side}_base", f"{side}_base_mount")}
    out = []
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if n.startswith(side) and b not in base:
            out.append(g)
    return out


def _table_penetration(m, d, arm_geoms: set) -> float:
    """How deep any of this arm's geoms is inside the table, in metres. 0.0 if clear.

    Contacts, not a height test. The height version of this ("is any link below the table top")
    is wrong as soon as a base sits outside the table footprint, because a link hanging beside
    the table is harmless -- and it reported a constant -130 mm across three very different
    layouts, which is what gave it away. mj_collision gives the actual geometry.
    """
    mujoco.mj_collision(m, d)
    worst = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        pair = {int(c.geom1), int(c.geom2)}
        if pair & arm_geoms and pair & _TABLE_GEOMS:
            worst = max(worst, -float(c.dist))
    return worst


def _descent_clearance(m, d, side: str, q_home, geoms) -> float:
    """Worst clearance above the table while descending from home to the task poses.

    This is the check that decides the home pose, and it is not the same question as "does the IK
    converge". The IK is collision-blind: it reported 0.0 mm residual for a grasp pose whose
    solution puts wrist_1 92 mm and the forearm 53 mm THROUGH the table top. The arm then stalls
    on the table about 93 mm short, and because the residual is reported against a kinematically
    valid target it reads exactly like the policy failing.
    #
    The online IK is a local method seeded from the current configuration, so it stays in whatever
    branch home put it in. Choosing home by residual alone picks an arbitrary branch; of 243
    converged branches for one grasp pose only 70 keep the arm above the table. So home has to be
    chosen for the branch, which is what this scores.
    """
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{j}") for j in ARM_JOINTS]
    qadr = [m.jnt_qposadr[j] for j in jid]
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_eef")
    dofs = [m.jnt_dofadr[j] for j in jid]
    # The task poses this arm actually has to reach, which are NOT mirror images: the left arm
    # picks the can off the table, the right arm takes the handover mid-table and then places on
    # the tray, which is the lowest pose either arm visits. Mirroring the left target onto the
    # right scored a pose the right arm never goes to and rejected every branch.
    if side == "left":
        targets = [(PICK[0], PICK[1], TABLE_TOP_Z + CAN_HH),      # can, default placement
                   (0.47, 0.24, TABLE_TOP_Z + CAN_HH)]            # can, far corner of the demos
    else:
        targets = [(0.34, 0.0, TABLE_TOP_Z + 0.13),               # handover, mid-table
                   # Tray top plus the can's half height: the TOOL point goes to where the can's
                   # centre must end up, not to the tray surface. Aiming at the surface put the
                   # target 57 mm too low and rejected every branch.
                   (PLACE[0], PLACE[1], TABLE_TOP_Z + 0.016 + CAN_HH)]
    targets = [t for t in targets]
    worst = 0.0
    for tgt in targets:
        for k, a in enumerate(qadr):
            d.qpos[a] = q_home[k]
        t = np.asarray(tgt, float)
        for it in range(250):
            mujoco.mj_forward(m, d)
            if it % 5 == 0:         # sampled; a step moves at most 0.05 rad per joint
                worst = max(worst, _table_penetration(m, d, geoms))
            err = t - d.site_xpos[sid]
            cur = d.site_xmat[sid].reshape(3, 3)
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, (cur.T @ HOME_ROT[side]).flatten())
            n = np.linalg.norm(q[1:])
            ang = 2.0 * np.arctan2(n, q[0])
            rerr = cur @ (q[1:] / (n + 1e-12) * ang)
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, sid)
            jac = np.vstack([jp[:, dofs], jr[:, dofs]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.05 ** 2 * np.eye(6),
                                         np.concatenate([err, 0.6 * rerr]))
            for k, j in enumerate(jid):
                v = d.qpos[qadr[k]] + float(np.clip(dq[k], -0.05, 0.05))
                if m.jnt_limited[j]:
                    v = float(np.clip(v, *m.jnt_range[j]))
                d.qpos[qadr[k]] = v
    return worst


def solve_home(m, d, side: str) -> list[float]:
    """Damped least squares onto reBot's home pose, with restarts. Returns the joint angles.

    A single seed is not enough. The correct eef frame makes the right arm's target (yawed 38
    degrees inboard, matching reBot) reachable only on some IK branches, and the elbow-up seed
    lands on one that stalls 34 mm out against a wrist limit. Restarting from perturbed seeds and
    keeping the best result costs a second and removes the hand-tuning.
    """
    rng = np.random.default_rng(0)
    global _TABLE_GEOMS
    _TABLE_GEOMS = _table_geoms(m)
    geoms = set(_link_geoms(m, side))
    cands = []
    for attempt in range(40):
        seed = np.asarray(SEED[side], float)
        if attempt:
            seed = seed + rng.uniform(-np.pi, np.pi, 6) * (0.35 if attempt < 20 else 1.0)
        q, pos_e, ang_e = _solve_from(m, d, side, seed)
        if pos_e < 1e-3 and ang_e < 5e-3:
            cands.append((q, pos_e, ang_e))
    if not cands:
        raise SystemExit(f"{side} home: no restart reached reBot's pose")
    # Rank by how much table clearance the branch keeps on the way down to the task poses, not by
    # residual: every candidate here already matches reBot's pose to under a millimetre, and the
    # residual cannot distinguish a branch that reaches over the table from one that reaches
    # through it.
    # HOME ITSELF MUST BE COLLISION-FREE. This is a hard requirement and separate from how the
    # descent scores. Ranking only by descent penetration picked a "least bad" branch that was
    # embedded in the table at rest -- upper_arm 45 mm and forearm 37 mm into the top, at the home
    # keyframe. The sim then shoves the arm out, it settles ~100 mm from the commanded pose, and
    # the IK can never recover: |dq| pins at dq_max and `clamped` stays true forever, which is
    # exactly the "stalling short" signature. A home in collision is not a starting state.
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{j}") for j in ARM_JOINTS]
    qadr = [m.jnt_qposadr[j] for j in jid]
    clear = []
    for q, pe, ae in cands:
        for k, a in enumerate(qadr):
            d.qpos[a] = q[k]
        mujoco.mj_forward(m, d)
        if _table_penetration(m, d, geoms) <= 1e-4:
            clear.append((q, pe, ae))
    if not clear:
        raise SystemExit(f"{side}: no home branch is collision-free at rest")
    dropped = len(cands) - len(clear)
    scored = [(_descent_clearance(m, d, side, q, geoms), q, pe, ae) for q, pe, ae in clear]
    scored.sort(key=lambda t: t[0])
    pen, best, be, ba = scored[0]
    print(f"  {side:5} home solved to {be*1000:.2f} mm / {np.degrees(ba):.2f} deg of reBot's pose "
          f"({len(cands)}/40 starts converged, {dropped} dropped for touching the table at rest, "
          f"descent penetration {pen*1000:.1f} mm)")
    if pen > 0.005:
        # A warning, not a failure. With the bases on the table there is no penetration-free
        # branch, and refusing to emit a model would just make the scene unbuildable. Ranking by
        # penetration still picks the least bad branch, which is strictly better than ranking by
        # a residual that cannot tell the difference.
        print(f"  {side:5} WARNING: the best branch still reaches {pen*1000:.0f} mm into the "
              f"table at the task poses; expect the arm to stall short there")
    return best


def _solve_from(m, d, side: str, seed) -> tuple[list[float], float, float]:
    """One damped-least-squares run from one seed. Returns (q, position error, angle error)."""
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{j}") for j in ARM_JOINTS]
    qadr = [m.jnt_qposadr[j] for j in jid]
    dof = [m.jnt_dofadr[j] for j in jid]
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_eef")
    for i, a in enumerate(qadr):
        v = float(seed[i])
        if m.jnt_limited[jid[i]]:
            v = float(np.clip(v, *m.jnt_range[jid[i]]))
        d.qpos[a] = v
    tgt = np.asarray(HOME_TARGET[side])
    for _ in range(400):
        mujoco.mj_forward(m, d)
        err = tgt - d.site_xpos[sid]
        cur = d.site_xmat[sid].reshape(3, 3)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (cur.T @ HOME_ROT[side]).flatten())
        n = np.linalg.norm(q[1:])
        ang = 2.0 * np.arctan2(n, q[0])
        if np.linalg.norm(err) < 3e-4 and abs(ang) < 3e-3:
            break
        rerr = cur @ (q[1:] / (n + 1e-12) * ang)
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, sid)
        jac = np.vstack([jp[:, dof], jr[:, dof]])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.05 ** 2 * np.eye(6),
                                     np.concatenate([err, 0.6 * rerr]))
        for k, j in enumerate(jid):
            v = d.qpos[qadr[k]] + float(np.clip(dq[k], -0.05, 0.05))
            if m.jnt_limited[j]:
                v = float(np.clip(v, *m.jnt_range[j]))
            d.qpos[qadr[k]] = v
    mujoco.mj_forward(m, d)
    cur = d.site_xmat[sid].reshape(3, 3)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (cur.T @ HOME_ROT[side]).flatten())
    ang = abs(2.0 * np.arctan2(np.linalg.norm(q[1:]), q[0]))
    return ([float(d.qpos[a]) for a in qadr],
            float(np.linalg.norm(tgt - d.site_xpos[sid])), float(min(ang, 2 * np.pi - ang)))


def stage_meshes() -> Path:
    """Merge both asset sets into one directory and return it.

    A compiled MuJoCo model has a SINGLE global meshdir, but this scene draws meshes from two
    upstream packages. They currently have no filename in common, so a flat merge is safe; the
    assert makes a future upstream rename fail here rather than by silently serving the wrong mesh.
    """
    MESH_OUT.mkdir(parents=True, exist_ok=True)
    seen: dict[str, Path] = {}
    for src in (ARM_XML.parent / "assets", GRIP_XML.parent / "assets"):
        for f in sorted(src.iterdir()):
            if not f.is_file():
                continue
            if f.name in seen:
                raise SystemExit(
                    f"mesh filename collision: {f.name} in both {seen[f.name]} and {src}. "
                    f"MuJoCo resolves one meshdir, so these would silently shadow each other.")
            seen[f.name] = src
            shutil.copy2(f, MESH_OUT / f.name)
    print(f"  staged {len(seen)} meshes into {MESH_OUT}")
    return MESH_OUT.resolve()


def position_servo(spec, name, target, kp, ctrlrange, kv_ratio=0.081):
    """A position servo with an EXPLICIT control range.

    ctrlrange is not optional here. Left unset it compiles to (0, 0), so the actuator can only ever
    be commanded to zero: the arm holds at its home angles and the gripper can never close. Nothing
    errors, the joint simply never moves, which reads as a policy or controller problem.
    """
    a = spec.add_actuator()
    a.name = name
    a.target = target
    a.trntype = mujoco.mjtTrn.mjTRN_JOINT
    a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    a.gainprm = [kp] + [0.0] * 9
    a.biasprm = [0.0, -kp, -kp * kv_ratio] + [0.0] * 7
    a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    a.ctrlrange = list(ctrlrange)
    return a


def build() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = "zero_ur5e"
    spec.compiler.autolimits = True
    spec.visual.map.znear = CAM_ZNEAR

    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720

    # Scene lighting and backdrop, copied value-for-value from gen_scene.py so a UR5e frame is
    # photometrically the same as a reBot or vx300s frame. The arms are what should differ between
    # embodiments; appearance is variation a policy would otherwise have to learn around.
    #
    # Two DIRECTIONAL lights, not the spots that were here. Spots lit the table but left the
    # grippers in shadow, and the gripper is the one thing a wrist camera has to resolve; a spot
    # also falls off with distance, so the same gripper changed brightness as the arm moved.
    # Directional rays are parallel with no falloff. One casts shadows (key) and one does not
    # (fill), or every object gets two shadows and the table reads as clutter in the front view.
    spec.add_texture(name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.3, 0.5, 0.7], rgb2=[0, 0, 0], width=512, height=3072)
    spec.add_texture(name="groundplane", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     mark=mujoco.mjtMark.mjMARK_EDGE, rgb1=[0.2, 0.3, 0.4],
                     rgb2=[0.1, 0.2, 0.3], markrgb=[0.8, 0.8, 0.8], width=300, height=300)
    spec.add_material(name="groundplane", texrepeat=[5, 5], reflectance=0.2).textures[
        mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
    spec.add_material(name="table_mat", rgba=[0.72, 0.60, 0.44, 1])
    spec.add_material(name="leg_mat", rgba=[0.25, 0.25, 0.28, 1])

    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, 0.05], material="groundplane")
    for dr, dif, shadow in (((0.3, -0.3, -1.0), 0.45, True),
                            ((-0.4, 0.3, -1.0), 0.28, False)):
        spec.worldbody.add_light(dir=list(dr), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                                 castshadow=shadow,
                                 diffuse=[dif] * 3, specular=[0.05] * 3, ambient=[0.10] * 3)
    # The headlight travels with whatever camera is rendering, so it fills exactly what that camera
    # sees. That is the direct fix for a gripper that is dark in the wrist view but fine in the
    # scene view. Modest on purpose: much above this blows out the pale table top, and a clipped
    # frame is unusable as training data.
    spec.visual.headlight.ambient = [0.22, 0.22, 0.22]
    spec.visual.headlight.diffuse = [0.28, 0.28, 0.28]
    spec.visual.headlight.specular = [0.08, 0.08, 0.08]

    cx, cy = TABLE_CENTER_XY
    table = spec.worldbody.add_body(name="table", pos=[cx, cy, TABLE_TOP_Z - TABLE_HZ])
    table.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[TABLE_HX, TABLE_HY, TABLE_HZ], material="table_mat")
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        lz = (TABLE_TOP_Z - 2 * TABLE_HZ) / 2
        spec.worldbody.add_body(
            name=f"leg{i}", pos=[cx + sx * (TABLE_HX - 0.05), cy + sy * (TABLE_HY - 0.05), lz]
        ).add_geom(name=f"leg{i}_g", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[LEG_R, lz], material="leg_mat")

    # A plinth under each arm. The bases sit beside the table and below its top (see BASE_SEP), so
    # without these they float. Sized to the UR5e base and stopping at the base plate.
    for side in SIDES:
        by = (BASE_SEP / 2.0) * (1 if side == "left" else -1)
        top = TABLE_TOP_Z + BASE_Z
        spec.worldbody.add_body(name=f"{side}_plinth", pos=[BASE_X, by, top / 2.0]).add_geom(
            name=f"{side}_plinth_g", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.10, 0.10, top / 2.0], material="leg_mat")

    for side in SIDES:
        y = (BASE_SEP / 2.0) * (1 if side == "left" else -1)
        arm = mujoco.MjSpec.from_file(str(ARM_XML))
        for k in list(arm.keys):
            arm.delete(k)
        for lt in list(arm.lights):
            arm.delete(lt)
        for g in list(arm.worldbody.geoms):
            arm.delete(g)
        # Drop the arm's own actuators too. Menagerie ships six `general` servos whose gains come
        # from its own default classes; keeping them alongside ours gives 26 actuators for 14
        # joints, and ros2_control would bind to whichever it found first.
        for a in list(arm.actuators):
            arm.delete(a)
        # The gripper goes on before the arm is attached, so its bodies come along with the prefix.
        grip = mujoco.MjSpec.from_file(str(GRIP_XML))
        for k in list(grip.keys):
            grip.delete(k)
        for lt in list(grip.lights):
            grip.delete(lt)
        for g in list(grip.worldbody.geoms):
            grip.delete(g)
        # Drop the tendon actuator; a position servo on the driver joint replaces it below.
        for a in list(grip.actuators):
            grip.delete(a)
        for t in list(grip.tendons):
            grip.delete(t)
        # Mount on menagerie's own attachment_site, not on wrist_3_link's origin with identity.
        # The UR5e's tool face is 100 mm out along the link's +y and rotated; attaching at the
        # origin buries the whole gripper inside the wrist, which renders as a camera seeing 99%
        # robot from every mount position and looks like a camera-placement problem rather than a
        # mounting one. The URDF says the same thing: wrist_3 -> flange is xyz 0 0 0 with
        # rpy (0, -pi/2, -pi/2), i.e. a pure reorientation that identity throws away.
        if G["attach"] == "attachment_site":
            att = arm.site("attachment_site")
            host, apos, aquat = "wrist_3_link", list(att.pos), list(att.quat)
        else:
            host, apos, aquat = G["attach"]
            apos, aquat = list(apos), list(aquat)
        arm.attach(grip, prefix=GRIP_PREFIX,
                   frame=arm.body(host).add_frame(pos=apos, quat=aquat))
        spec.attach(arm, prefix=f"{side}_",
                    frame=spec.worldbody.add_frame(pos=[BASE_X, y, TABLE_TOP_Z + BASE_Z],
                                                   quat=[1, 0, 0, 0]))

    # Task objects: a can to pick and a tray to place it on.
    obj = spec.worldbody.add_body(name="obj_root",
                                  pos=[PICK[0], PICK[1], TABLE_TOP_Z + CAN_HH + 0.0005])
    obj.add_freejoint(name="obj_free")
    obj.add_geom(name="obj_can", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CAN_R, CAN_HH, 0],
                 rgba=[0.72, 0.11, 0.13, 1], mass=0.015,
                 friction=[2.0, 0.05, 0.002], solref=[0.005, 1])
    tray = spec.worldbody.add_body(name="plate_root",
                                   pos=[PLACE[0], PLACE[1], TABLE_TOP_Z + 0.008])
    tray.add_freejoint(name="plate_free")
    tray.add_geom(name="plate_object", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[0.13, 0.09, 0.008], rgba=[0.12, 0.12, 0.13, 1], mass=0.4)

    spec.worldbody.add_camera(name="front", pos=list(FRONT_CAM_EYE), fovy=FRONT_FOVY,
                              resolution=list(CAM_RES),
                              xyaxes=lookat(FRONT_CAM_EYE, LOOK_AT))
    return spec


def main() -> None:
    spec = build()
    # Actuators, after the attaches so the prefixed joint names exist. Ranges are read from the
    # compiled joints so they cannot drift from the model.
    probe = spec.compile()
    def jrange(name):
        j = mujoco.mj_name2id(probe, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert j >= 0, f"no joint {name}"
        return tuple(probe.jnt_range[j])
    for side in SIDES:
        for j, kp in zip(ARM_JOINTS, ARM_KP, strict=True):
            position_servo(spec, f"{side}_{j}", f"{side}_{j}", kp, jrange(f"{side}_{j}"))
        # One servo per commanded gripper joint. The 2F-85 has a single driver and the rest of the
        # four-bar follows by equality; the reBot gripper has two independent slides, both driven.
        for gi, gjn in enumerate(GRIP_JOINTS):
            gj = f"{side}_{GRIP_PREFIX}{gjn}"
            name = f"{side}_gripper" if len(GRIP_JOINTS) == 1 else f"{side}_gripper{gi + 1}"
            act = position_servo(spec, name, gj, GRIP_KP, jrange(gj))
            # Cap on the spec so it is compiled in; a post-compile write to actuator_forcerange is
            # silently ignored.
            act.forcerange = [-GRIP_ACT_FORCE, GRIP_ACT_FORCE]

    m = spec.compile()
    d = mujoco.MjData(m)

    # The eef site: the frame the 20-dim action is expressed in, and the IK target.
    for side in SIDES:
        body = spec.body(f"{side}_{MOUNT_BODY}")
        body.add_site(name=f"{side}_eef", pos=list(EEF_POS), quat=list(EEF_QUAT),
                      size=[0.008] * 3, group=4)
        # Wrist camera: ON TOP of the gripper, off the approach axis. See WRIST_CAM_POS for why
        # the on-axis mount is impossible on a 2F-85. Aim is derived from the mount and the aim
        # point rather than written out, so moving either keeps the camera pointed at the pads.
        if WRIST_CAM_AIM is None:
            # Explicit axes, as the reBot declares them: the camera looks straight down the
            # approach axis rather than converging on the tool point.
            xyaxes = [float(v) for v in WRIST_CAM_XYAXES]
        else:
            f = np.asarray(WRIST_CAM_AIM, float) - np.asarray(WRIST_CAM_POS, float)
            f /= np.linalg.norm(f)
            # Image-right along the jaw axis (mount +x), which is perpendicular to the aim by
            # construction since the mount is offset in y/z only. z_cam = -f, y_cam = z_cam x x_cam.
            x_cam = np.array([1.0, 0.0, 0.0])
            xyaxes = [float(v) for v in (*x_cam, *np.cross(-f, x_cam))]
        body.add_camera(name=f"{side}_wrist", pos=list(WRIST_CAM_POS), xyaxes=xyaxes,
                        fovy=WRIST_FOVY, resolution=list(CAM_RES))

        # Fingertip F/T: a site on each follower body plus a matching force+torque pair. The site
        # sits at the follower's origin, so the sensor reports the wrench through that finger's own
        # joint, which is what separates a two-pad grasp from an object resting on one pad.
        for finger in FT_FINGERS:
            name = f"{side}_{finger}_ft"
            spec.body(f"{side}_{GRIP_PREFIX}{finger}").add_site(name=f"{name}_site", pos=[0.0, 0.0, 0.0],
                                                      group=4)   # marker only, never rendered
            for kind, stype in (("force", mujoco.mjtSensor.mjSENS_FORCE),
                                ("torque", mujoco.mjtSensor.mjSENS_TORQUE)):
                sen = spec.add_sensor()
                sen.name = f"{name}_{kind}"
                sen.type = stype
                sen.objtype = mujoco.mjtObj.mjOBJ_SITE
                sen.objname = f"{name}_site"

    # Home keyframe: arms at HOME, grippers open, objects where they were placed.
    m = spec.compile()
    d = mujoco.MjData(m)
    home = {}
    for side in SIDES:
        home[side] = solve_home(m, d, side)
    # Both solves share one MjData, so re-apply both before snapshotting the keyframe.
    for side in SIDES:
        for i, j in enumerate(ARM_JOINTS):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{j}")
            d.qpos[m.jnt_qposadr[jid]] = home[side][i]
        # Open the gripper explicitly. Leaving it at qpos0 is not the same thing: grip_range is
        # (value at CLOSED, value at OPEN), and only by coincidence is the open value 0 for the
        # 2F-85. The reBot gripper opens at 0.05, so at qpos0 it came up CLOSED, which put the two
        # fingers together on the camera axis 59 mm ahead of the lens and filled 76.7% of the wrist
        # frame against the reBot's 3.2%. Every episode starts with the hand open.
        for gjn in grip_joint_names(side):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, gjn)
            assert jid >= 0, f"no gripper joint {gjn}"
            d.qpos[m.jnt_qposadr[jid]] = G["grip_range"][1]
    mujoco.mj_forward(m, d)
    key = spec.add_key()
    key.name = "home"
    key.qpos = list(d.qpos)
    key.ctrl = [0.0] * m.nu
    for side in SIDES:
        for i, j in enumerate(ARM_JOINTS):
            a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_{j}")
            key.ctrl[a] = home[side][i]
        # and hold it open, or the servo drives it shut on the first step
        for gi in range(len(GRIP_JOINTS)):
            nm = f"{side}_gripper" if len(GRIP_JOINTS) == 1 else f"{side}_gripper{gi + 1}"
            a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, nm)
            assert a >= 0, f"no actuator {nm}"
            key.ctrl[a] = G["grip_range"][1]

    # meshdir and texturedir are separate compiler paths; both must be absolute because the MJCF
    # is written to zero_description/mjcf/ while the meshes live under zero_description/meshes/.
    md = stage_meshes()
    spec.meshdir = str(md)
    spec.texturedir = str(md)
    m = spec.compile()
    OUT_MJCF.parent.mkdir(parents=True, exist_ok=True)
    OUT_MJCF.write_text(spec.to_xml())
    print(f"wrote {OUT_MJCF}")
    print(f"  nq={m.nq} nu={m.nu} nbody={m.nbody} ngeom={m.ngeom} ncam={m.ncam} nkey={m.nkey}")
    for side in SIDES:
        for j in list(ARM_JOINTS) + [f"{GRIP_PREFIX}{g}" for g in GRIP_JOINTS]:
            assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{j}") >= 0, \
                f"missing joint {side}_{j}"
    print("  all expected joints present")
    for a in range(m.nu):
        lo, hi = m.actuator_ctrlrange[a]
        assert hi > lo, (f"actuator {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)} has "
                         f"ctrlrange ({lo}, {hi}); it could never be commanded anywhere")
    print(f"  all {m.nu} actuators have a usable ctrlrange")
    # The written file is the artefact, so re-load it from disk. An in-process compile can succeed
    # while the emitted XML is unusable, which is exactly what bare mesh filenames did here.
    check = mujoco.MjModel.from_xml_path(str(OUT_MJCF))
    assert (check.nq, check.nu, check.ncam) == (m.nq, m.nu, m.ncam), "reloaded model differs"
    print(f"  re-loaded from disk OK: nq={check.nq} nu={check.nu} ncam={check.ncam}")
    build_urdf()
    write_bringup(check, mujoco.MjData(check), home)


if __name__ == "__main__":
    main()
