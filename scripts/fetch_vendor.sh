#!/usr/bin/env bash
# Fetch the upstream descriptions the targets need, and flatten their xacros.
#
#   bash scripts/fetch_vendor.sh
#
# robots/_vendor is gitignored: this is an upstream tree, not ours to version. The apt package
# ros-jazzy-... holds the same files, but cloning needs no root and pins the branch we chose.
#
# Trossen's own ROS 2 description, jazzy branch. menagerie's vx300s MJCF was derived from it, so
# the two agree on forward kinematics to 0.0000 mm over 400 configurations, which is the property
# the Panda never had and the reason this arm replaced it.
#
# The G1 is NOT fetched here. robots/unitree_g1_mjcf/ is checked in, and its URDF is derived from
# its MJCF (the *_jointbody links are the converter's signature), so the pair agrees to 0.0000 mm.
# The upstream unitreerobotics/unitree_ros clone is 1.6 GB, and its MJCF ships bare torque motors
# with no standing keyframe, so it is strictly worse for this scene.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
V="$ROOT/robots/_vendor"
mkdir -p "$V"

if [ -d "$V/interbotix/.git" ]; then
  echo "interbotix already present"
else
  git clone --depth 1 -b jazzy \
    https://github.com/Interbotix/interbotix_ros_manipulators.git "$V/interbotix"
fi

# --- flatten the ViperX xacro into a plain URDF -------------------------------------------------
# Three things have to be corrected on the way out, all of them Interbotix packaging choices:
#   * every link is namespaced `vx300s/`, which would collide with our own per-side prefixing and
#     put a '/' inside a link name;
#   * it carries its own <ros2_control> block, and gen_urdf.py writes ours;
#   * `interbotix_black.png` is referenced as a mesh filename, so it travels with the STLs.
DESC="$V/interbotix/interbotix_ros_xsarms/interbotix_xsarm_descriptions"
OV="$(mktemp -d)"
mkdir -p "$OV/share/ament_index/resource_index/packages"
ln -sfn "$DESC" "$OV/share/interbotix_xsarm_descriptions"
touch "$OV/share/ament_index/resource_index/packages/interbotix_xsarm_descriptions"
AMENT_PREFIX_PATH="$OV:${AMENT_PREFIX_PATH:-}" \
  xacro "$DESC/urdf/vx300s.urdf.xacro" \
    robot_model:=vx300s robot_name:=vx300s hardware_type:=fake > "$V/vx300s_raw.urdf"
python3 - "$V/vx300s_raw.urdf" "$V/vx300s_flat.urdf" <<'PY'
import re, sys, pathlib
t = pathlib.Path(sys.argv[1]).read_text()
t = re.sub(r'<ros2_control\b.*?</ros2_control>', '', t, flags=re.S)
t = t.replace('vx300s/', '')
pathlib.Path(sys.argv[2]).write_text(t)
print(f"flattened -> {sys.argv[2]}")
PY
rm -f "$V/vx300s_raw.urdf"; rm -rf "$OV"

# --- UR5e + Robotiq 2F-85 -----------------------------------------------------------------------
# The UR5e wears a Robotiq 2F-85, which is what a real UR cell is usually fitted with. Two repos:
# the arm's description from Universal Robots, and the gripper's from PickNik's ROS 2 driver, whose
# `robotiq_85_base_link` is the link name gen_urdf's graft expects.
#
# Both need the same corrections on the way out as the ViperX did:
#   * they ship <ros2_control> blocks and gen_urdf.py writes ours;
#   * they root at a `world` link, which would become a second root once grafted into our scene.
if [ -d "$V/ur_description/.git" ]; then
  echo "ur_description already present"
else
  git clone --depth 1 https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git \
    "$V/ur_description"
fi
if [ -d "$V/robotiq/.git" ]; then
  echo "robotiq already present"
else
  git clone --depth 1 https://github.com/PickNikRobotics/ros2_robotiq_gripper.git "$V/robotiq"
fi

OV2="$(mktemp -d)"
mkdir -p "$OV2/share/ament_index/resource_index/packages"
ln -sfn "$V/ur_description" "$OV2/share/ur_description"
ln -sfn "$V/robotiq/robotiq_description" "$OV2/share/robotiq_description"
touch "$OV2/share/ament_index/resource_index/packages/ur_description"
touch "$OV2/share/ament_index/resource_index/packages/robotiq_description"

AMENT_PREFIX_PATH="$OV2:${AMENT_PREFIX_PATH:-}" \
  xacro "$V/ur_description/urdf/ur.urdf.xacro" ur_type:=ur5e name:=ur5e > "$V/ur5e_raw.urdf"
AMENT_PREFIX_PATH="$OV2:${AMENT_PREFIX_PATH:-}" \
  xacro "$V/robotiq/robotiq_description/urdf/robotiq_2f_85_gripper.urdf.xacro" > "$V/rq85_raw.urdf"

python3 - "$V" <<'PY'
import re, sys, pathlib
V = pathlib.Path(sys.argv[1])
for raw, out in (("ur5e_raw.urdf", "ur5e_flat.urdf"), ("rq85_raw.urdf", "robotiq_2f85_flat.urdf")):
    t = (V / raw).read_text()
    t = re.sub(r'<ros2_control\b.*?</ros2_control>', '', t, flags=re.S)
    # Drop the `world` root and whatever fixed joint pins the robot to it. Grafted into our scene
    # the arm is mounted by the generator, so a leftover `world` is a second, disconnected root.
    # Matching the joint by NAME does not work: UR calls it `base_joint` and Robotiq
    # `robotiq_85_base_joint`, neither of which contains "world". Match on the parent instead, then
    # assert a single root survives, because a dangling parent leaves the tree with no root at all
    # and every downstream tool reports something less obvious than the cause.
    t = re.sub(r'<link name="world"\s*/>', '', t)
    t = re.sub(r'<joint\b(?:(?!</joint>).)*?<parent link="world"\s*/>.*?</joint>', '', t, flags=re.S)
    import xml.etree.ElementTree as ET
    root = ET.fromstring(t)
    kids = {j.find("child").get("link") for j in root.findall("joint")}
    roots = [l.get("name") for l in root.findall("link") if l.get("name") not in kids]
    if len(roots) != 1:
        raise SystemExit(f"{out}: expected exactly one root link, got {roots}")
    print(f"  {out}: root link is {roots[0]}")
    (V / out).write_text(t)
    print(f"flattened -> {out}")
PY
rm -f "$V/ur5e_raw.urdf" "$V/rq85_raw.urdf"; rm -rf "$OV2"

echo
echo "vendored into $V:"
ls -1 "$V"
