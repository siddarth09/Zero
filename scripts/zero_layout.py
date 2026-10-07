"""Shared layout constants for project ZERO's bimanual reBot workstation.

The URDF and the MJCF are two independent descriptions of the same robot, and
`mujoco_ros2_control` binds them together by joint name across two topics
(`/robot_description` for the URDF, `/mujoco_robot_description` for the MJCF). If the names or
the mount poses drift apart nothing throws: the plugin reports "Joint '%s' not found in the URDF
joint data" for the lucky cases and ignores the rest, so a gripper can end up never moving while
everything looks fine. Both generators import from here so drift is impossible.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- sources ---
MENAGERIE_ARM = ROOT / "robots" / "seeed_rebot_devarm" / "seeed_rebot_devarm.xml"
# Seeed's official RS (RobStride) URDF, the variant menagerie models, per its README's RS-06 /
# RS-00 motor torque limits. The DM variant uses Damiao actuators and different specs.
SEEED_URDF = Path.home() / "rebotarm_ros2/src/rebotarm_bringup/description/urdf/00-arm-rs_asm-v3.urdf"
SEEED_MESHES = Path.home() / "rebotarm_ros2/src/rebotarm_bringup/description/meshes_rs"

# --- outputs ---
PKG = ROOT / "zero_description"
OUT_URDF = PKG / "urdf" / "zero_bimanual.urdf"
OUT_MJCF = PKG / "mjcf" / "zero_bimanual.xml"
OUT_PNG = ROOT / "scenes" / "bimanual.png"

# `gripper_joint1/2` come from Seeed's official URDF. Menagerie ships the same two joints as
# `joint_left`/`joint_right`, so our copy of the MJCF was renamed to match the URDF rather than
# the other way round: upstream's MoveIt config and ros2_controllers.yaml already reference the
# URDF names, and keeping them lets those configs work unmodified.
ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
GRIPPER_JOINTS = ("gripper_joint1", "gripper_joint2")
ALL_JOINTS = ARM_JOINTS + GRIPPER_JOINTS
SIDES = ("left", "right")


def prefixed(side: str, name: str) -> str:
    return f"{side}_{name}"


def all_prefixed_joints() -> list[str]:
    return [prefixed(s, j) for s in SIDES for j in ALL_JOINTS]


# Measured from 20k random configurations of one arm (scripts/measure_reach.py):
#   points along +x at qpos=0; gripper_end at (0.302, 0, 0.218) from base
#   max reach 0.909 m, p95 0.828 m, reaches to z = -0.37 below its own base
# BASE_SEP 0.44 m puts the whole centreline inside both envelopes, which is what makes this
# genuinely bimanual instead of two arms working independently.
TABLE_TOP_Z = 0.75
TABLE_HALF = (0.60, 0.70, 0.02)  # widened 2026-08-20: two Pandas crowded the old 0.9x0.9 top
TABLE_CENTER_XY = (0.05, 0.0)
BASE_SEP = 0.44
# Measured, not chosen. Phase-0.5 sensitivity sweep (scripts/sens.py) over table height x mount
# position: table height barely matters, BASE_X dominates. Shared reBot+G1 table volume, at G1
# standoffs of 0.10 / 0.20 / 0.30 m:
#     BASE_X -0.25 : 11.6 /  5.4 / 1.3 L   original; the G1 must press against the table
#     BASE_X -0.15 : 19.7 / 10.9 / 4.7 L
#     BASE_X -0.05 : 27.7 / 16.3 / 7.6 L   chosen; 5.6x the margin at 0.30 m standoff
# Moving the arms forward costs work-area depth in front of them (still ~37 cm to the front edge)
# and buys the G1 room to stand without leaning over the table.
BASE_X = -0.05

# Mount pose per side, in world coordinates. Used by BOTH generators.
MOUNTS = {
    "left": (BASE_X, +BASE_SEP / 2, TABLE_TOP_Z),
    "right": (BASE_X, -BASE_SEP / 2, TABLE_TOP_Z),
}

# menagerie's "raised" keyframe, sign-flipped to match the URDF convention that
# fix_mjcf_conventions.py enforces. menagerie ships it as (0, -0.7, -1.1); with the six arm axes
# negated those same numbers would drive the arm down into the table instead of clear of it.
HOME_QPOS = (0.315, 1.139, 0.570, 0.734, -0.432, 3.047)  # aimed at LOOK_AT, see scripts/findhome.py
GRIPPER_OPEN = (0.0, 0.0)


# Cameras. Named to match the convention in siddarth09/Panda_mujoco, which uses RealSense D435i
# bodies at top / table / side / wrist. Scaled to this table: top at z=0.75, spans x[-0.40,0.50],
# y[-0.45,0.45]; the Phase-0.5 task volume is x[0.12,0.52] y[+/-0.36] z[0.76,1.00], centroid
# ~(0.36, 0, 0.90). LOOK_AT is the task centroid, so every scene camera frames the work area
# rather than the whole table.
#
# The wrist cams are the primary view for cross-embodiment, not the scene cams. A wrist view puts
# the base and torso out of frame, so the manipulator-vs-humanoid difference stops being visible;
# scene cams show two completely different robots and cannot transfer. That is why Mirage-style
# cross-painting was dropped, see research/README.md.
#
# Overwritten at the bottom of this file once PICK_POS/PLACE_POS exist: the cameras and the
# home-pose search must aim at the actual task, not at a hand-picked point above it.
LOOK_AT = (0.36, 0.0, 0.90)
# Every camera costs frame rate, and the dataset can never be faster than its slowest
# observation. With 5 cameras: 5.00 Hz at 640x480, 5.60 at 320x240, 6.50 at 224x224, so
# resolution is almost free and each extra camera is not. Down to front plus two wrists.
# `top` and `side` are one line each to restore if a policy turns out to need them, but they
# have to be paid for in fps, and 5 Hz demos do not contain an operator's corrections.
SCENE_CAMS = {
    # name:       (eye position,            fovy)
    "front":      ((1.15, 0.00, 1.15), 58),   # across the table, roughly the G1's viewpoint
    # "top":      ((0.30, 0.00, 1.60), 58),
    # "side":     ((0.30, -1.05, 1.10), 58),
}
WRIST_CAM_FOVY = 70          # wide: the object is only ~10 cm from the lens

# Near clipping plane, as a FRACTION of the model extent (MuJoCo multiplies the two). On a wrist
# camera the plane does two jobs at once: it hides the gripper's own casting, which sits ~14 mm in
# front of the lens, and it can also clip away the object being grasped.
#
# THIS MUST MATCH WHAT THE 82 EPISODES WERE RECORDED WITH, which is MuJoCo's default 0.010. That
# is not a preference, it is the training distribution. Verified by decoding the recorded
# left_wrist video and rendering zero_rebot.xml at the same pose with only znear varied:
#     0.007 -> 18.6 mm : mean |pixel diff| vs the training frame = 37.8
#     0.010 -> 26.5 mm : mean |pixel diff| vs the training frame =  6.4   <- what was recorded
#     0.015 -> 39.8 mm : mean |pixel diff| vs the training frame =  6.4
# At 0.010 the plane cuts the gripper casting away and the wrist view contains only the two wedge
# tips, which is what every training frame shows. At 0.007 the casting is no longer clipped and a
# broad grey gripper body appears in every frame, which appears in NO training image.
#
# 0.007 was previously chosen here to raise can coverage at the vx300s grasp from 0.1% to 78.9%,
# matching the 78.4% measured in the training video at that moment. That measurement was right
# but incomplete: it checked the grasp instant and never checked the approach, and it was taken on
# the vx300s giving camera, whose 50 mm standoff is what clipped the can in the first place. The
# reBot at its 100 mm standoff sees the can perfectly well at 0.010 -- the recorded video proves
# it. So a camera that cannot see the object has to be moved, not compensated for by opening the
# near plane on every robot at once and desynchronising them all from the data.
CAM_ZNEAR = 0.010


# Task objects whose live pose is exported as a ros2_control state interface. Named here because
# gen_scene writes the MJCF framepos/framequat pair and gen_urdf writes the matching <sensor>, and
# the two must agree; mujoco_ros2_control derives the MJCF names as <name>_pos and <name>_quat.
OBJECT_POSE_SENSORS = ("can_pose", "tray_pose")
WRIST_CAM_POS = (0.0, 0.0, 0.035)   # on gripper_end, just behind the finger roots
# 224x224, the native input size for essentially every vision backbone a policy would use, so
# nothing is resized at train time. Larger buys no fps back anyway, see SCENE_CAMS.
CAM_RES = (224, 224)

# Camera publish rate, which has to be set explicitly. mujoco_ros2_control defaults to 5 Hz
# (measured 4.6), fine for eyeballing the sim and far too slow to imitate from, since an
# operator's corrections are not in the data at 5 Hz. Read as a hardware parameter from the URDF's
# <ros2_control><hardware> block (mujoco_system_interface.cpp reads `camera_publish_rate` there),
# so this is ours to set without touching that package. It also caps the LeRobot dataset fps: the
# dataset can never be faster than its slowest observation.
CAM_RATE_HZ = 30.0


def all_cameras() -> list[str]:
    """Every camera name in the MJCF, which is also every <sensor> name the URDF must declare.

    mujoco_ros2_control matches cameras to sensors by name ("For cameras, the sensor name _must_
    match the camera name in the MJCF", from upstream's own demo URDF). A mismatch does not
    error: the camera never publishes, so RViz shows nothing and it looks like a networking or
    QoS problem. Same trap as the joints, same fix: one source of truth here, asserted by
    scripts/check_parity.py.
    """
    return list(SCENE_CAMS) + [f"{s}_wrist" for s in SIDES]


# Topic namespace per camera: /zero/<cam>/{image_raw,camera_info,depth}
CAM_NS = "/zero"


ROBOCASA = Path("/home/sid/projects25/src/robocasa/robocasa/models/assets/objects/lightwheel")
# An upright can, and specifically a uniform cylinder. Two separate requirements:
#
#   Height. The original lemon wedge lies flat, 23 mm tall with its grasp centre ~11 mm off the
#   table, so a top-down gripper has to reach almost to the surface, which put the reBot's left
#   arm on its workspace boundary. An upright object carries its grasp centre ~75 mm up and the
#   reBot reaches it comfortably.
#
#   Shape. Juice006 met the height requirement and still failed in teleop, because it is a bottle:
#   its diameter is a uniform 49 mm up to 58% of its height and then breaks into a shoulder and a
#   35 mm cap. Any grasp above mid-body closes on the neck, so the operator gets a hold on the cap
#   instead of the body. A summary "uniformity" statistic over the whole grasp band hid this; the
#   diameter profile showed it immediately.
#
# Paprika012 measures 64-67 mm across from 5% to 75% of its height: a real cylinder, so any grasp
# in that band is on the body. Checked on both embodiments (grasp at 75% height, then lifted and
# translated 80 mm to stress the hold):
#   reBot  lift 100.2 mm, in-jaw slip 6.2 mm     Panda  lift 87.7 mm, slip 12.6 mm
# The near-cylinders that are cleaner in profile are not usable: Cinnamon007/005 tip Panda over at
# every grasp height, GlassCup025 tips the reBot. This is the only candidate both arms hold.
PICK_OBJECT = ROBOCASA / "paprika" / "Paprika012" / "model.xml"
PLACE_TARGET = ROBOCASA / "tray" / "Tray001" / "model.xml"

# Handover geometry, solved rather than guessed (scripts/solve_handover.py). The task is
# pick -> handover -> place, so it is only a handover if the picking arm cannot reach the tray.
# Otherwise the policy skips the handover and the task degenerates into pick-and-place.
# Constraints: pick reachable by left only, place by right only, the handover pose by both, with a
# 6 cm margin on the "cannot reach" side.
# Solved separations:  reBot 0.75-1.20 m,  Panda 0.55-1.05 m
# The task must be identical across embodiments and only the mounts may differ, so take the
# intersection [0.75, 1.05] and use its midpoint for both robots.
HANDOVER_SEP = 0.90

PICK_POS = (0.34, +HANDOVER_SEP / 2, TABLE_TOP_Z + 0.03)    # in front of the LEFT arm
PLACE_POS = (0.34, -HANDOVER_SEP / 2, TABLE_TOP_Z + 0.015)  # in front of the RIGHT arm
HANDOVER_POS = (0.34, 0.0, TABLE_TOP_Z + 0.13)              # centreline, above the table

# Each arm aims at its own role's target. Aiming both at one shared point made them face each
# other, which is wrong for a handover and was visible immediately in the viewer.
HOME_TARGET = {"left": PICK_POS, "right": PLACE_POS}

LOOK_AT = HANDOVER_POS

# ======================================================================================
# EMBODIMENT REGISTRY
# ======================================================================================
# The cross-embodiment chain is reBot (source) -> Panda -> G1. Everything above this line is
# reBot-specific and is what the URDF generator, parity check and ros2_control path use, because
# only the source embodiment needs teleoperation and ros2_control. Targets only need to be stepped
# and IK'd, so they live here as plain MJCF specs.
#
# Measured per robot; do not edit by hand, see the printouts in gen_scene.py:
#   reBot : reach max 0.909 m (p95 0.83), gripper 100 mm open, 6 arm DoF
#   Panda : reach max 1.189 m (p95 1.12), gripper  80 mm open, 7 arm DoF
#   reBot : gripper 100 mm open (two 50 mm slide joints, coupled by an equality)
#   Dex3  : gripper 124 mm open (thumb-tip to index-tip)
# The binding aperture is Panda's 80 mm; the reBot opens wider, not narrower. Panda's throat at
# full open is 74 mm, so a 66 mm object leaves ~4 mm per side. It grasps reliably in test, but
# that is the tight constraint to respect when swapping the object, not the reBot.
#
# Gripper force limits, because the source models effectively have none: the reBot's gripper
# actuators ship forcerange +/-1904 N and Panda's is [0, 0], which in MuJoCo means unlimited. A
# position-controlled gripper commanded fully closed then drives straight through whatever it is
# holding: 11 mm of penetration into a 66 mm can, which crushes it, squirts it out of the jaws and
# makes contact forces meaningless. Real values make the grasp compliant, so it squeezes until the
# limit and then holds by friction, which is what the recorded demo should show. Applied by
# gen_scene to the gripper actuators only; the arm joints keep their own limits.
OBJECT_WIDTH = 0.066

# Mass of the pick object, forced rather than inherited. The robocasa assets are meshes with a
# default density, so a spice tin came out at 0.763 kg, roughly twice a full 330 ml can and heavy
# enough on a small arm that the grasp keeps failing under load. A real coke can is ~0.39 kg full
# and ~0.015 kg empty. Set in the MJCF at generation time: writing mass or friction onto an
# already-compiled model measures byte-identical, the same trap as body_gravcomp.
#
# This is an empty can (~15 g of aluminium), not a full one (~350 g). The asset it replaced weighed
# 1.19-1.51 kg (the mass is split across a nested body, so reading one body's own mass understates
# it), which is too much payload to keep hold of on a small arm.
#
# Being this light costs nothing in stability. The 7.6 g lemon mesh rocked through 24-58 deg
# because MuJoCo's convex-convex narrowphase gave it a single contact point, whereas this cylinder
# is a primitive and collides analytically: drift over 20 s is 0.00 mm / 0.00 deg at every mass
# from 15 g to 350 g. Set PICK_MASS = 0.35 for a full can if a heavier payload is ever wanted.
PICK_MASS = 0.015

# The pick object is a built primitive, not a robocasa asset. There is no drinks can in the
# 69-object lightwheel set, and every asset near the right size is the wrong shape or proportions:
#   Juice006      49 x 114  a bottle: uniform to 58% of height, then a shoulder and a 35 mm cap,
#                           so any grasp above mid-body closes on the neck
#   Paprika012    66 x 150  a true cylinder, but at a realistic 0.39 kg it is top-heavy; the
#                           usable grasp band shrank to the top 15% and below that it tips
#   Cinnamon007   48 x 125  the cleanest cylinder of all, and it tips Panda at every height
#   GlassCup025   48 x 114  tips the reBot
# A real 330 ml can is 66 x 115 mm, aspect 1.74, squatter than anything available, which is why it
# stands up to being grasped. Two further wins over a mesh: a cylinder primitive collides
# analytically instead of through a convex hull (no multiccd contact-count games), and its friction
# is ours to set, where every robocasa asset ships torsional friction 0.005 and a post-compile
# write to change that is ignored.
#
# PICK_SKIN is a mesh used purely for the can's appearance; collision and mass come from the
# cylinder in PICK_PRIMITIVE. Scaled onto that cylinder at generation time, so swapping it changes
# only how the object looks and never how it behaves. None gives the bare primitive.
PICK_SKIN = None

PICK_PRIMITIVE = {
    "radius": 0.033,          # 66 mm across
    "half_height": 0.0575,    # 115 mm tall
    "rgba": (0.72, 0.11, 0.13, 1.0),
    # Raised from (1.0, 0.02) after the can kept rotating out of the jaw mid-carry. The policy
    # grips ~5 mm below the rim, so the can hangs ~110 mm below the contact and the moment arm does
    # the damage; torsional friction is what resists that, not squeeze. Simulated over the policy's
    # own carry path, slip fell from 80.1 mm to 49.3 mm and then stopped improving, so this is the
    # knee of the curve rather than an arbitrary large number. See "pad_friction" on the vx300s.
    "friction": (2.0, 0.05, 0.002),
    # Contact time constant. MuJoCo's default 0.02 is deliberately soft for solver stability, and on
    # a grasped object that softness is visible: the pads sank 7.8 mm into a 66 mm can, which looks
    # like the gripper passing through it. 0.005 with a realistic gripper force brings that to 2.2 mm,
    # 3% of the diameter, and the lift is unaffected (76 mm either way).
    "solref": (0.005, 1.0),
}

ROBOTS = {
    "rebot": {
        "mjcf": ROOT / "robots" / "seeed_rebot_devarm" / "seeed_rebot_devarm.xml",
        "arm_joints": ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        "gripper_joints": ("gripper_joint1", "gripper_joint2"),
        "eef_body": "gripper_end",
        # Measured, not chosen: scripts/measure_tcp.py, asserted by check_parity.py. The offset from
        # `gripper_end` to where the two pads meet when closed. Two wrong values preceded this one,
        # both of which produced grasps that worked often enough to look like a tuning problem.
        # (0, 0, 0.10), copied from Panda's convention, put the commanded point 113 mm out in mid-air
        # beside the gripper; the midpoint of the finger collision AABBs put it 39 mm off, because
        # these finger meshes are asymmetric. The 39 mm version closed the gripper beside the can and
        # flicked it away, which reads as the gripper passing through the object. The acceptance test
        # is symmetric pad clearance.
        "eef_offset": (-0.0109, 0.0000, 0.0050),
        # LOCKED. All 82 recorded episodes were captured against this value, so it defines what
        # "the tool point" means for the whole dataset and cannot be changed without invalidating
        # them. scripts/measure_tcp.py now reports (-0.0256, 0, 0) instead, 15.6 mm away: this
        # gripper's pads are flat parallel slabs, so the closest-point search is degenerate and the
        # original number came from an arbitrary tie among thousands of equally-close pairs. The
        # committed value is the empirically validated one, since the trained policy grasps at
        # 3.3 mm closest approach with symmetric pad clearance. tcp_tol widens check_parity's
        # assertion to cover the known gap rather than letting it fail silently.
        "tcp_tol_mm": 20.0,
        # THE SHARED TOOL CONVENTION, defined by this robot because it is the source: the eef
        # site's +x points OUT of the gripper toward the object, and its y is the jaw axis.
        # Identity here, since gripper_end already has that layout. Read it off the geometry, not
        # off eef_offset: in the site frame this gripper's bbox runs x -148.0 to +10.9 mm, i.e.
        # the body extends backwards toward the wrist and the object sits at +x. Independently
        # confirmed by the wrist camera, which sits 90 mm at -x and looks along +x at the gripper.
        # Every target must rotate its site onto this, or the same recorded rot6d aims the two
        # grippers in different directions.
        "eef_quat": (1.0, 0.0, 0.0, 0.0),
        "wrist_cam_pos": (-0.1009, 0.0, 0.0050),   # 90 mm back along -x, looking FRONT (+x)
        "wrist_cam_xyaxes": (0, 1, 0, 0, 0, -1),
        "base_sep": HANDOVER_SEP,
        "base_x": -0.05,
        # Per-side, searched (scripts/findhome.py <robot> <side>), not mirrored from one pose: the
        # correct mirror is robot-specific and copying left->right aimed the arm 0.43 m away.
        "home": {
            "left": (0.111, 0.966, 0.322, 0.543, 0.239, -2.922),
            "right": (0.299, 1.331, 0.463, 0.664, 0.961, -2.716),
        },
        # (closed, open) joint value, shared by both fingers. One pair rather than one value per
        # joint: Panda commands only finger_joint1 (the other follows an <equality>), so a per-joint
        # tuple silently mismatched the joint count.
        "grip_range": (0.0, 0.05),
        "grip_force": 15.0,   # N, small servo gripper. reBot fingers are slide joints, range [0, 0.05]
        # menagerie names each actuator after its joint, so the ros2_control parity check can compare
        # the two sets directly.
        "actuators_match_joints": True,
        # ros2_control / URDF side (source embodiment: needs teleop)
        "urdf": SEEED_URDF,
        "urdf_root": "base_link",
        "urdf_eef_frame": "{side}_gripper_end",
        "urdf_skip_links": (),
        "mesh_src": SEEED_MESHES,
        "mesh_glob": ("*.STL",),
        "mesh_remap": None,
        # One command interface per actuator. The reBot has two real gripper actuators plus an
        # <equality> coupling the fingers, so both are declared; Panda has one actuator driving both
        # fingers, so it declares one. Copying either pattern blindly is wrong.
        "ros2_control_joints": ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
                                "gripper_joint1", "gripper_joint2"),
        "arm_ctrl_joints": ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        "grip_ctrl_joints": ("gripper_joint1", "gripper_joint2"),
    },

    # ---- Trossen ViperX 300s ----------------------------------------------------------
    # The nearest thing to the reBot in menagerie, which is why it is the target: 6 arm DoF, a
    # two-slide parallel jaw coupled by an <equality>, and reach 0.902 m max / 0.855 p95 against
    # the reBot's 0.909 / 0.83. Its jaw is on local y exactly like the reBot's, so the recorded
    # orientations mean nearly the same thing on it, which is what the Panda could not manage:
    # with the eef frames reconciled, the Panda's left arm missed the recorded poses by 146 mm and
    # 76 deg at its best placement.
    # No gripper surgery needed either. `gripper` is already a joint actuator on `left_finger`
    # with an <equality> coupling the pair, so ros2_control can command it by name, unlike
    # menagerie's Panda (tendon) and Robotiq (tendon).
    "vx300s": {
        "mjcf": Path("/home/sid/.cache/robot_descriptions/mujoco_menagerie/"
                     "trossen_vx300s/vx300s.xml"),
        "arm_joints": ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle",
                       "wrist_rotate"),
        "gripper_joints": ("left_finger", "right_finger"),
        "arm_ctrl_joints": ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle",
                            "wrist_rotate"),
        # One actuator; right_finger follows through the <equality>.
        "grip_ctrl_joints": ("left_finger",),
        "ros2_control_joints": ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle",
                                "wrist_rotate", "left_finger"),
        # menagerie names the arm actuators after their joints but calls the gripper one
        # `gripper`, so a name-based joint/actuator check would miss it.
        "actuators_match_joints": False,
        "eef_body": "gripper_link",
        # menagerie's own `pinch` site, which is Trossen's authored grasp point rather than
        # something we inferred. scripts/measure_tcp.py independently lands on (0.1025, 0, 0),
        # agreeing to 2.5 mm, so the two corroborate each other.
        "eef_offset": (0.1000, 0.0000, 0.0000),
        "tcp_tol_mm": 5.0,
        # Identity: this arm already matches the shared convention. Its gripper points along the
        # body's +x and its jaw slides on y, exactly the reBot's layout, which is the main reason
        # it is the target. Checked in the site frame: bbox x -100.0 to +36.5 mm, the same
        # backward-extending shape as the reBot's -148.0 to +10.9.
        "eef_quat": (1.0, 0.0, 0.0, 0.0),
        # (closed, open) on left_finger. Jaw gap 42 mm closed to 114 mm open, so a 66 mm can has
        # 24 mm of clearance per side against the reBot's 17 and the Robotiq's 12.3.
        "grip_range": (0.021, 0.057),
        # A position servo squeezes with kp * (commanded - actual). Gripping the 66 mm can leaves
        # the finger at q=0.0470 against a commanded 0.0210, so the stock kp=300 gives only
        # 300 * 0.026 = 7.6 N and the 20 N forcerange is never reached. Raising grip_force alone
        # changes nothing, measured: 20, 40, 80 and 150 N all give an identical 7.60 N squeeze.
        # kp is the term that matters; grip_force is raised alongside so it stops clipping.
        "grip_force": 60.0,   # N, Dynamixel XM430-class gripper
        "grip_kp": 1500.0,
        # The can slid out of the jaw during the carry. Pad friction is the other half of the
        # holding capacity and the vendor pads ship at the MuJoCo default. (sliding, torsional):
        # torsional is the one that matters here, because the can is held near its rim and rotates
        # out rather than sliding down. Measured over the policy's carry path, slip 80.1 -> 49.3 mm.
        # kp beyond 1500 and friction beyond this buy nothing (48.8 mm at kp 3000, 49.9 at mu 8).
        "pad_friction": (4.0, 0.05),
        "grip_ctrl_n": 1,
        # The reBot's ARRANGEMENT reproduced, not its numbers. Measured on the reBot, its camera
        # sits dead on the approach axis (zero perpendicular offset from the tool point), 11.4 mm
        # behind the rearmost finger edge, looking straight along +x through the open jaw.
        # Copying its raw -0.1009 does not work, because the two robots put the eef body at
        # opposite ends of the gripper:
        #     reBot   gripper_end   bbox x -148.0 .. + 10.9   body at the FAR end
        #     vx300s  gripper_link  bbox x    0.0 .. +136.5   body at the NEAR end
        # so -0.1009 landed 167 mm behind the vx300s's fingers, inside the forearm, and filled 46%
        # of the frame with arm. This arm's fingers span x 66.5..136.4, so the same 11.4 mm standoff
        # puts the camera at x = 55 mm.
        # The gripper still fills 10.3% of the frame against the reBot's 3.2%: this jaw is simply
        # chunkier. x = 0.070 gets to 1.8% if matching the training APPEARANCE matters more than
        # matching the geometry, at the cost of sitting only 30 mm from the tool point.
        # Placed against the GRASP frame, which is the one that matters. Tuning it against the
        # HOME frame was the mistake: at home this arm points nearly level and the object is on a
        # table below, so "too much sky" looked like the problem and a 25 deg down-tilt looked like
        # the fix. It is not. Driving both robots to the SAME commanded grasp pose from episode 0
        # and measuring how much of the wrist frame the can fills:
        #     training video (reBot)        78.4%
        #     reBot in sim                  65.7%   <- the reference
        #     x=55 tilted 25 deg             0.0%   the can is not in frame at all
        #     x=20 on-axis                  69.9%   jaw wedges 9.7%
        #     x=25 on-axis                  77.1%   jaw wedges 7.3%  <- chosen
        # against training's can 78.4% / wedges 8.5%, so both match to under 1.5 points.
        # Moving the camera BACK is bounded: below x=20 the gripper's base casting fills the entire
        # frame (can 0.0%, a grey wall) and the cliff is sharp, 15 mm to 20 mm. With the can absent
        # at the grasp the policy has nothing to trigger a close on, and it held the jaw open at
        # 0.98 through the whole rollout. The sky at home is not a problem: the reBot's own training
        # frames at t=0 are sky, floor and table edge too.
        # 20 was TRIED and it is past the cliff, not at its edge: the pick never happened, the jaw
        # stayed at 0.97 to 1.00 for a whole 1163-tick rollout, which is the exact signature this
        # comment predicts. 25 is the working value; do not go below it without re-testing the pick.
        # The cost of staying at 25 is the RECEIVING view at the handover. The receiving camera has
        # to see the object the giving hand holds, and the standoff sets that: the object sits only
        # ~30 mm from the lens, so a short standoff swings it outside the 35 deg half-FOV. Measured
        # over episode 0's handover frames as the angle from the camera axis to the giving tool
        # point (needs no can placement, so it does not depend on the per-episode can pose):
        #     x=0.025, standoff  -75 mm : azimuth 34.7, elevation 37.5 -> in frame   0% of frames
        #     x=0.020, standoff  -80 mm : azimuth 29.2, elevation 31.7 -> in frame  92%, BLIND
        #     x=0.010, standoff  -90 mm : azimuth 21.9, elevation 24.0 -> in frame 100%, BLIND
        # The reBot sits at -90 mm and reads 21.6 / 22.7. It can, because its tool point is out at
        # the finger tips; the vx300s pinch site is mid-finger, so the same standoff buries the lens
        # in the gripper casting. The two constraints are irreconcilable by standoff alone.
        # This is why the handover fails. Live dumps at the handover, can as a fraction of frame:
        # reBot 9.0% (closes, succeeds), vx300s 0.0% (never closes). Fixing it needs something
        # other than standoff: a lens offset off-axis, a tilt, or a wider fovy, each of which has to
        # be re-tested against the pick because the pick is what the deep standoff breaks.
        # On-axis, no tilt, x_cam along the jaw axis, exactly the reBot's orientation.
        # Per side. Both keep the +20 mm z raise, which is the change that made the object
        # visible at all; only the standoff differs.
        #   left  = the GIVING camera, judged at the grasp with the can at its true launch pose and
        #           the tool at the operator's own grasp height (z=0.841). Rendered can coverage:
        #             (0.025, 0, 0.02) 50.3%   (0.025, 0, 0) 29.7%
        #             (0.050, 0, 0.02)  2.5%   (0.050, 0, 0)  0.0%     training reference 78.3%
        #           x=0.050 is only 50 mm of standoff, which puts the can too close to the lens and
        #           pushes it out of frame at grasp height; it looks like a thin red band at the
        #           bottom edge. That is a grasp-blind camera, not a tight one.
        #   right = the RECEIVING camera, judged from a live handover dump, where x=0.050 measured
        #           10.5% can coverage against 0.0% before and the reBot's 9.0% when it succeeds.
        # Do not collapse these back to one value: 0.025 is blind at the handover and 0.050 is
        # blind at the grasp, and the task needs both.
        # Both sides at 0.050. Two constraints bracket this, and they leave almost no room.
        #
        # Pulling BACK buries the lens in the gripper's own casting. With the can grasped at the
        # tool point, the giving camera renders, as (can / robot body / rest of scene):
        #     x=0.020, 82 mm standoff:   0.7% / 98.8% / 0.6%
        #     x=0.025, 78 mm standoff:   0.0% / 100.0% / 0.0%
        #     x=0.035, 68 mm standoff:   0.0% / 100.0% / 0.0%
        #     x=0.050, 54 mm standoff:  44.4% / 50.1% / 5.5%
        # At 0.025 the frame is entirely the robot looking at itself. The reBot gets away with 90 mm
        # because its tool point sits at the fingertips; the vx300s pinch site is mid-finger, so the
        # same standoff puts the lens behind the casting.
        #
        # Pushing FORWARD used to fail too, which is why these were briefly split with the left at
        # 0.025. That was NOT a standoff problem:
        # the near clipping plane sat at 26.5 mm (see L.CAM_ZNEAR) and the can's near surface sits
        # 20.9 mm from a lens 54 mm behind the tool, so the front of the can was clipped away and
        # what remained was its far wall as a floating band. With the plane at 5.3 mm the same
        # camera renders 44.4%. The right camera never showed it because it only meets the can at
        # the handover, 70 to 150 mm out, which no near plane reaches.
        "wrist_cam_pos": (0.0500, 0.0, 0.0200),
        "wrist_cam_xyaxes": (0, 1, 0, 0, 0, -1),
        # Swept on THREE gates: replay residual on episode 0's recorded poses, whether an arm base
        # ends up inside the task furniture, and ORIENTATION residual on the giving arm during the
        # carry and handover. The third gate is the one that decides it. Position residual barely
        # separates the candidates while rotation separates them by 23x, and the handover depends on
        # holding the object at the recorded ORIENTATION, not just the recorded point:
        #     base_x  0.00, sep 1.00 :  0.50 mm,  0.16 deg mean / 0.28 max   <- chosen
        #     base_x  0.00, sep 0.90 :  4.42 mm,  3.68 deg mean / 6.52 max
        #     base_x  0.00, sep 1.10 :  0.49 mm,  0.05 deg mean / 0.26 max
        # base_x=+0.05 reaches 0.27 deg but buries the right base 12 mm inside the tray.
        # Everything at base_x <= -0.05 exceeds 2 deg, and -0.10 exceeds 30 deg.
        # Held at the reBot's HANDOVER_SEP deliberately, NOT at the kinematic optimum. 1.00 scores
        # far better on the giving arm's orientation (0.03 deg against 3.68) and was tried, but it
        # moves both mounts 50 mm outward relative to the can and tray, which stay at HANDOVER_SEP
        # because they are shared task furniture. The rollout at 1.00 was worse, not better. That
        # test also moved the wrist camera at the same time so it cannot attribute the regression
        # to one change; what is certain is that 0.90 is the configuration with a measured
        # 3/3 pick, grasp and carry, and 1.00 does not have one.
        # If the giving arm's 3.68 deg is worth revisiting, change base_sep ALONE and keep the
        # camera at 0.025, so the next rollout attributes cleanly.
        "base_x": 0.00,
        "base_sep": HANDOVER_SEP,
        # Solved onto the training set's mean episode-start pose, position AND orientation, by
        # scripts/solve_home.py. `observation.state` is the policy's input, so the pose the arm
        # boots into has to be one the policy saw at t=0.
        "home": {
            "left": (-0.1286, -0.4261, 1.0695, 2.6865, 0.5904, 0.1658),
            "right": (-0.2867, -0.1800, 0.9628, 1.9393, 1.0683, 0.4709),
        },
        "urdf": PKG.parent / "robots" / "_vendor" / "vx300s_flat.urdf",
        "urdf_root": "base_link",
        "urdf_eef_frame": "{side}_gripper_link",
        "urdf_skip_links": ("world",),
        # Interbotix's meshes, not menagerie's. gen_urdf keys on the mesh BASENAME, and the two
        # trees disagree: Interbotix calls it base.stl where menagerie calls it vx300s_1_base.stl.
        # The URDF is Interbotix's, so its names have to resolve.
        "mesh_src": PKG.parent / "robots" / "_vendor" / "interbotix" / "interbotix_ros_xsarms"
                    / "interbotix_xsarm_descriptions" / "meshes" / "vx300s_meshes",
        # The .png travels with the STLs: the URDF references interbotix_black.png as a mesh
        # filename, so gen_urdf rewrites it like one and it must be present.
        "mesh_glob": ("*.stl", "*.STL", "*.png"),
        "mesh_remap": None,
    },
    # ---- Unitree G1, the destination embodiment -----------------------------------------
    # INFRASTRUCTURE ONLY at this point: MJCF, URDF, ros2_control and a launch file. The pieces a
    # policy would need are deliberately not done yet, and each is flagged below.
    #
    # Structurally unlike the others: one floating-base humanoid carrying both arms, not two arms
    # bolted to a table, hence `single_body`. Its joints already carry left_/right_, so the
    # per-side names below are the suffixes and L.prefixed() reproduces the model's own spelling.
    # URDF and MJCF come from one upstream tree (unitreerobotics/unitree_ros), so they agree:
    # left_wrist_yaw_link matches to 0.0001 deg in rotation over 300 configurations, with a
    # constant 793 mm position offset that is just the MJCF's floating base sitting at standing
    # pelvis height against the URDF's root at the origin.
    "g1": {
        "single_body": True,
        # Sid's own copy, not the _vendor clone. It is menagerie's G1 with hands and already has
        # what the raw unitreerobotics model lacks: 43/43 POSITION servos rather than bare torque
        # motors, a `stand` keyframe, and no ground plane of its own. Its URDF sits beside it and is
        # MJCF-derived (the *_jointbody links are the converter's signature), so the two agree by
        # construction instead of by luck.
        "mjcf": PKG.parent / "robots" / "unitree_g1_mjcf" / "g1_with_hands.xml",
        # Pelvis pose. From scripts/reach_gate.py, which swept it and verified 16.3 L of table
        # volume shared with both reBots at a 0.20 m standoff. yaw 180 turns the G1 to face the
        # table, which also mirrors the arms: its LEFT arm covers world -y, so the reBot's left arm
        # maps to the G1's right. See scripts/plan_can_poses.py.
        "base_pos": (0.75, 0.0, 0.79),
        "base_quat": (0.0, 0.0, 0.0, 1.0),
        "arm_joints": ('shoulder_pitch_joint', 'shoulder_roll_joint', 'shoulder_yaw_joint', 'elbow_joint', 'wrist_roll_joint', 'wrist_pitch_joint', 'wrist_yaw_joint'),
        "gripper_joints": ('hand_thumb_0_joint', 'hand_thumb_1_joint', 'hand_thumb_2_joint', 'hand_middle_0_joint', 'hand_middle_1_joint', 'hand_index_0_joint', 'hand_index_1_joint'),
        "arm_ctrl_joints": ('shoulder_pitch_joint', 'shoulder_roll_joint', 'shoulder_yaw_joint', 'elbow_joint', 'wrist_roll_joint', 'wrist_pitch_joint', 'wrist_yaw_joint'),
        "grip_ctrl_joints": ('hand_thumb_0_joint', 'hand_thumb_1_joint', 'hand_thumb_2_joint', 'hand_middle_0_joint', 'hand_middle_1_joint', 'hand_index_0_joint', 'hand_index_1_joint'),
        "ros2_control_joints": ('shoulder_pitch_joint', 'shoulder_roll_joint', 'shoulder_yaw_joint', 'elbow_joint', 'wrist_roll_joint', 'wrist_pitch_joint', 'wrist_yaw_joint') + ('hand_thumb_0_joint', 'hand_thumb_1_joint', 'hand_thumb_2_joint', 'hand_middle_0_joint', 'hand_middle_1_joint', 'hand_index_0_joint', 'hand_index_1_joint'),
        "actuators_match_joints": True,
        "eef_body": "wrist_yaw_link",
        # TODO eef_offset and eef_quat are NOT measured. The Dex3 is a three-finger hand, so its
        # grasp centre is the centroid of the pads rather than a jaw midpoint, and scripts/
        # measure_tcp.py assumes two opposing fingers. reach_gate.py has palm offsets derived in
        # the previous project (thumb pad and index+middle centroid at ~7 cm aperture) that are the
        # right starting point. Zero here so nothing silently pretends to be measured.
        "eef_offset": (0.0000, 0.0000, 0.0000),
        "eef_quat": (1.0, 0.0, 0.0, 0.0),
        # TODO the 20-dim action's grip channel is ONE scalar and the Dex3 has seven actuated
        # joints with seven different ranges, so it needs a closing synergy, not a linear map.
        # grip_range is a placeholder so the generators run; do not read a grasp from it.
        "grip_range": (0.0, 1.0),
        # All zeros is the extended, open hand: every joint's open end is 0 (thumb_2 runs
        # [0, 1.745], middle_0 and index_0 run [-1.571, 0]).
        "grip_open": (0.0,) * 7,
        "grip_force": 10.0,
        "grip_ctrl_n": 7,
        # 31 of its 43 actuators are proper position servos; the 12 leg joints are <motor> torque
        # actuators for RL locomotion and come out of the shared default class malformed. See the
        # repair in gen_scene.py.
        "repair_actuators": True,
        # Frozen: the pelvis is welded and a table task uses neither the legs nor the waist. This
        # deletes 12 leg joints and 3 waist joints with their actuators, which removes the 67 mrad
        # hip sag and 15 DOF. Drop this list to get a fully articulated lower body back.
        "freeze_joints": ("hip", "knee", "ankle", "waist"),
        # The legs, waist and neck are not part of the task and are not in arm_joints, so they take
        # their values from the model's own `stand` keyframe rather than defaulting to zero. Zeros
        # means straight legs, which with a welded pelvis left the knee drifting 2.7 rad.
        "base_keyframe": "stand",
        # TODO wrist camera unplaced. The Dex3 has no gripper body to mount on in the way the
        # parallel jaws do; this is the reBot's pose in the shared tool frame and is untested.
        "wrist_cam_pos": (-0.1009, 0.0, 0.0050),
        "wrist_cam_xyaxes": (0, 1, 0, 0, 0, -1),
        # Unused for a single body (the pelvis pose is base_pos), but the registry helpers read
        # them, so they mirror the stance rather than being left absent.
        "base_x": 0.75,
        "base_sep": 0.0,
        # TODO not solved onto the training start pose. Zeros is the model's own rest pose: arms
        # hanging. scripts/solve_home.py is what does this once eef_offset is measured.
        "home": {"left": (0.0,) * 7, "right": (0.0,) * 7},
        "urdf": PKG.parent / "robots" / "unitree_g1_mjcf" / "g1_with_hands.urdf",
        "urdf_root": "pelvis",
        "urdf_eef_frame": "{side}_wrist_yaw_link",
        "urdf_skip_links": (),
        "mesh_src": PKG.parent / "robots" / "unitree_g1_mjcf" / "assets",
        "mesh_glob": ("*.STL", "*.stl"),
        # converted_<link>_<rrggbbaa>.obj -> <link>.STL. The hex is a colour, not an index.
        "mesh_remap": r"converted_(.+)_[0-9a-f]{8}\.obj",
        "mesh_remap_to": "{part}.STL",
        "tcp_tol_mm": 1e9,     # TODO drop once eef_offset is measured
    },
}


def robot_all_joints(key: str) -> tuple[str, ...]:
    r = ROBOTS[key]
    return tuple(r["arm_joints"]) + tuple(r["gripper_joints"])


def robot_prefixed_ros2_joints(key: str) -> list[str]:
    """Prefixed joint names for the <ros2_control> block: one per actuator, both arms."""
    return [prefixed(s, j) for s in SIDES for j in ROBOTS[key]["ros2_control_joints"]]


def start_override_path(key: str) -> Path:
    """Where the launch writes the start-position override for `key`.

    mujoco_ros2_control reads the can's start pose from a file named by the
    `override_start_position_file` hardware parameter, which lives in the URDF, so the path has
    to be fixed at generation time while its contents are written per launch from
    `can_x`/`can_y`/`can_yaw`. Under /tmp because the launch writes it on every run and an
    install tree should stay read-only.

    Why a file and not a service: the installed mujoco_ros2_control exposes only reset_world,
    set_pause and step_simulation. `SetFreeJointState` exists in the upstream source and would let
    the can be moved while running (which is what per-episode randomisation would need), but it is
    not in this build.
    """
    return Path(f"/tmp/zero_{key}_start.xml")


def ft_sensors(key: str) -> list[tuple[str, str]]:
    """(ros2_control sensor name, MJCF body it measures) for every fingertip force/torque pair.

    One per finger, not one per gripper: the useful signal during a grasp is the wrench through
    each finger's own joint, which tells you whether both pads are loaded or the object is
    sitting against one of them. That asymmetry is the failure the TCP bug produced, and a
    per-finger sensor would have shown it at once: +9.5 mm of clearance on one pad, +99 mm on
    the other.

    The naming is a contract with mujoco_ros2_control. It resolves a `<sensor mujoco_type="fts">`
    named X to MJCF sensors `X_force` and `X_torque` (suffixes are hardware params, defaults
    `_force`/`_torque`). Generate both ends from here so they cannot drift.
    """
    out = []
    for side in SIDES:
        for j in ROBOTS[key]["gripper_joints"]:
            out.append((f"{prefixed(side, j)}_ft", prefixed(side, j)))
    return out


def robot_mounts(key: str) -> dict[str, tuple[float, float, float]]:
    r = ROBOTS[key]
    return {
        "left": (r["base_x"], +r["base_sep"] / 2, TABLE_TOP_Z),
        "right": (r["base_x"], -r["base_sep"] / 2, TABLE_TOP_Z),
    }


# ======================================================================================
# THE TASK, identical for every embodiment
# ======================================================================================
# Pick a drinks can off the table, hand it across, place it in the tray. The can is built
# (PICK_PRIMITIVE) rather than taken from an asset; the tray is a robocasa mesh. Sizes are
# measured from mesh vertices, never geom_rbound, which reports a bounding-sphere radius and
# made the flat tray read as 460 mm cubed.
#     can    66 x 66 x 115 mm, 15 g   -> the pick object, a cylinder we define
#     tray   28 x 220 x 396 mm        -> the place target