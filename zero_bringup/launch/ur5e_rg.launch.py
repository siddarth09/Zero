"""Bimanual UR5e wearing the reBot's OWN gripper.

Generated from ur5e.launch.py by substitution; the two differ only in which files they name.
This is the ablation twin of ur5e.launch.py: same arm, same IK, same controllers, same table,
and only the end effector differs. The wrist camera rides on the gripper, so this variant's
wrist view matches the recorded training video (mean |pixel diff| 6.3 against the reBot's 6.4,
gripper 3.2% of frame in both) where the 2F-85 variant scores 52.2.
Rebuild both with:  python3 scripts/build_ur5e.py {robotiq,rebot}
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from pathlib import Path

CONTROLLERS = ['joint_state_broadcaster',
               'left_arm_controller',
               'right_arm_controller',
               'left_gripper_controller',
               'right_gripper_controller',
               'left_gripper_left_ft_broadcaster',
               'left_gripper_right_ft_broadcaster',
               'right_gripper_left_ft_broadcaster',
               'right_gripper_right_ft_broadcaster']
# The F/T broadcasters publish the per-finger pad wrenches on /zero/ft/<sensor>, which is the
# 14-dim observation.force input. Names come from build_ur5e.FT_FINGERS via the generated
# ur5e_controllers.yaml, so this list must match it.
MJCF = '/home/sid/projects25/src/ZERO/zero_description/mjcf/zero_ur5e_rg.xml'
START_OVERRIDE = '/tmp/zero_ur5e_rg_start.xml'
CAN_DEFAULT_XY = (0.34, 0.45)


def _write_start_override(context):
    """Write the start-position override from can_x / can_y / can_yaw.

    The can's pose lives in the MJCF's `home` keyframe, which is baked at generation time. Rather
    than regenerate the model to move it, this reads that keyframe, substitutes the can's
    free-joint block, and writes a `<key qpos=... ctrl=.../>` file, the format
    mujoco_ros2_control's `override_start_position_file` hardware parameter expects. The whole
    qpos vector is copied from the compiled model, so arms, gripper and tray keep exactly the
    pose they were solved for and only the can moves.

    z is not an argument on purpose: the can's resting height is a function of its geometry (base
    on the table), and letting it be set by hand invites a can floating or half-buried. It is
    taken from the keyframe.
    """
    import mujoco
    import numpy as np

    x = float(LaunchConfiguration("can_x").perform(context))
    y = float(LaunchConfiguration("can_y").perform(context))
    yaw = float(LaunchConfiguration("can_yaw").perform(context))

    m = mujoco.MjModel.from_xml_path(MJCF)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid < 0:
        raise RuntimeError(f"no 'home' keyframe in {MJCF}")
    qpos = np.array(m.key_qpos[kid], dtype=float)
    ctrl = np.array(m.key_ctrl[kid], dtype=float)

    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj_root")
    jid = next((j for j in range(m.njnt)
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_bodyid[j] == bid), -1)
    if jid < 0:
        raise RuntimeError("obj_root has no free joint, cannot move the can")
    adr = m.jnt_qposadr[jid]

    # Reject bad placements rather than let the can spawn off the table, where the episode is
    # unrecordable and the cause is invisible from the images.
    tg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    cx, cy = m.geom_pos[tg][0], m.geom_pos[tg][1]
    hx, hy = m.geom_size[tg][0], m.geom_size[tg][1]
    if not (cx - hx <= x <= cx + hx and cy - hy <= y <= cy + hy):
        raise RuntimeError(
            f"can_x={x} can_y={y} is off the table "
            f"(x in [{cx - hx:.3f}, {cx + hx:.3f}], y in [{cy - hy:.3f}, {cy + hy:.3f}])")

    qpos[adr + 0] = x
    qpos[adr + 1] = y
    qpos[adr + 3] = np.cos(yaw / 2.0)      # quat w
    qpos[adr + 4] = 0.0
    qpos[adr + 5] = 0.0
    qpos[adr + 6] = np.sin(yaw / 2.0)      # quat z

    out = Path(START_OVERRIDE)
    # chr(10)/chr(34) instead of escape sequences, and every brace doubled: this text is a .format()
    # template that becomes a Python file, so a backslash escape survives one round of interpretation
    # fewer than it looks like it should, and a single brace is read as a format placeholder (which
    # failed with KeyError: v and left the old file behind).
    nl = chr(10)
    q = chr(34)
    body = [
        "<key",
        "  qpos=" + q + " ".join(f"{v:.6f}" for v in qpos) + q,
        "  qvel=" + q + " ".join("0" for _ in range(m.nv)) + q,
        "  ctrl=" + q + " ".join(f"{v:.6f}" for v in ctrl) + q,
        "/>",
    ]
    out.write_text(nl.join(body) + nl)
    print(f"[zero] can at x={x:.3f} y={y:.3f} yaw={yaw:.3f} -> {out}")
    return []


def _preflight(context):
    """Refuse to start if a mujoco_ros2_control node is already up.

    Two controller_managers on one ROS graph is not a clean failure. Both advertise
    /controller_manager/load_controller, /configure_controller and /list_controllers, so each
    spawner request is answered by whichever responds first: one request lands on the old
    manager and the next on the new one. The output is ~200 lines of contradictions ("Controller
    already loaded, skipping load_controller" from the manager that has them, "no controller with
    this name exists" from the one that does not, a controller logging "configure successful"
    while its spawner reports "Failed to configure controller"), none of which mentions the
    actual problem and all of which point at the controller config.
    """
    import subprocess
    found = subprocess.run(["pgrep", "-f", "mujoco_ros2_control/ros2_control_node"],
                           capture_output=True, text=True).stdout.split()
    if found:
        pids = ' '.join(found)
        raise RuntimeError(
            'a mujoco_ros2_control node is ALREADY RUNNING (pid ' + pids + '). Two '
            'controller_managers on one ROS graph answer each other service calls, so '
            'every spawner fails with a misleading message. Stop it first:  kill ' + pids)
    return []


def generate_launch_description() -> LaunchDescription:
    desc = FindPackageShare("zero_description")
    bringup = FindPackageShare("zero_bringup")
    urdf = PathJoinSubstitution([desc, "urdf", "zero_ur5e_rg.urdf"])
    controllers = PathJoinSubstitution([bringup, "config", "ur5e_rg_controllers.yaml"])
    control_params = PathJoinSubstitution([bringup, "config", "ur5e_rg_control.yaml"])

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description":
                     ParameterValue(Command(["xacro ", urdf]), value_type=str)}],
    )
    ctrl = Node(
        package="mujoco_ros2_control",
        executable="ros2_control_node",
        output="screen",
        parameters=[controllers],
    )
    spawners = [
        Node(package="controller_manager", executable="spawner",
             arguments=[n, "--controller-manager", "/controller_manager"], output="screen")
        for n in CONTROLLERS
    ]
    # SE(3) -> joint IK. Starts alongside the spawners; it holds the measured configuration until a
    # /zero/eef_target arrives, so ordering against the controllers does not matter.
    eef = Node(
        package="zero_control", executable="eef_control", output="screen",
        parameters=[control_params],
    )
    return LaunchDescription([
        DeclareLaunchArgument("can_x", default_value=str(CAN_DEFAULT_XY[0]),
                             description="can position along x, metres, table frame"),
        DeclareLaunchArgument("can_y", default_value=str(CAN_DEFAULT_XY[1]),
                             description="can position along y, metres, table frame"),
        DeclareLaunchArgument("can_yaw", default_value="0.0",
                             description="can yaw, radians"),
        OpaqueFunction(function=_preflight),
        # Must run before the sim: the override file is read once, at hardware init.
        OpaqueFunction(function=_write_start_override),
        rsp, ctrl,
        RegisterEventHandler(OnProcessStart(target_action=ctrl, on_start=spawners + [eef])),
    ])
