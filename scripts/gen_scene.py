"""Build a two-arm MJCF for any registered embodiment, with identical table and cameras.

    MUJOCO_GL=egl python scripts/gen_scene.py rebot
    MUJOCO_GL=egl python scripts/gen_scene.py panda
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np
from PIL import Image

import zero_layout as L

LEG_R = 0.03


def rest_offset(path) -> float:
    """Height of an object's top-body origin above the surface it rests on.

    Compiles the robocasa object on its own and measures the lowest point of its collision
    geometry relative to its top body's frame, from `mesh_vert`, the actual vertices. Never from
    `geom_rbound`, which is a bounding-sphere radius and overestimates badly on a flat tray.

    Computed rather than guessed: the placement z used to be `TABLE_TOP_Z + 0.015`, which left
    the tray 12.9 mm in the air. The tray is welded to the world, so unlike a free-jointed object
    it can never fall and settle; it just hangs there. Each robocasa asset has its own origin
    convention, so a guessed offset is only right by luck.
    """
    child = mujoco.MjSpec.from_file(str(path))
    for coll in (child.meshes, child.textures):
        for asset in coll:
            if asset.file and not asset.file.startswith("/"):
                asset.file = str((path.parent / asset.file).resolve())
    m = child.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    top_z = d.xpos[1][2]                       # body 1 = the child's top body (0 is world)
    lo = np.inf
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            V = m.mesh_vert[m.mesh_vertadr[mid]:
                            m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].reshape(-1, 3)
            W = (d.geom_xmat[g].reshape(3, 3) @ V.T).T + d.geom_xpos[g]
            lo = min(lo, float(W[:, 2].min()))
        else:
            lo = min(lo, float(d.geom_xpos[g][2] - m.geom_size[g][2]))
    assert np.isfinite(lo), f"{path} has no collision geometry to rest on"
    return top_z - lo


def attach_keeping_names(parent, child, prefix, frame):
    """Attach `child` under `frame`, then strip `prefix` back off its bodies and joints.

    A bare attach would keep the child's names, but MuJoCo gives every spec an implicit top-level
    default class called `main`, and models carry assets with generic names (the G1 ships a
    `groundplane` texture), so two un-prefixed specs collide on one or the other. Prefixing avoids
    that; stripping the prefix off bodies and joints afterwards leaves the exact names the upstream
    ROS description uses, so mujoco_ros2_control binds the URDF and the MJCF without a translation
    table. Assets and default classes keep their prefix, which is what the collision needed.

    References are stored as plain strings and do NOT follow a rename, so every one is rewritten by
    hand. Missing the actuator target is silent: the actuator stops resolving and only surfaces much
    later as "no gripper actuator matched".
    """
    parent.attach(child, prefix=prefix, frame=frame)
    if not prefix:
        return
    taken = {b.name for b in parent.bodies} | {j.name for j in parent.joints}
    ren = {el.name: el.name[len(prefix):]
           for el in list(parent.bodies) + list(parent.joints)
           if el.name.startswith(prefix) and el.name[len(prefix):] not in taken}
    for el in list(parent.bodies) + list(parent.joints):
        if el.name in ren:
            el.name = ren[el.name]
    for act in parent.actuators:
        act.target = ren.get(act.target, act.target)
    for eq in parent.equalities:
        eq.name1 = ren.get(eq.name1, eq.name1)
        eq.name2 = ren.get(eq.name2, eq.name2)
    for ex in parent.excludes:
        ex.bodyname1 = ren.get(ex.bodyname1, ex.bodyname1)
        ex.bodyname2 = ren.get(ex.bodyname2, ex.bodyname2)


def add_task(spec: mujoco.MjSpec) -> None:
    """Attach the robocasa pick object and place target, identical for every embodiment.

    Attached with MjSpec.attach and prefixed, same as the arms, so their mesh/material/geom names
    cannot collide with the robot's.

    Both get a free joint. The pick object must be liftable. The tray was initially welded so the
    place target could not be nudged, but that buys determinism at the cost of physics: a welded
    body is infinitely rigid, so brushing it gives an unphysical contact response and the object
    bounces off a tray that a real 2.8 kg one would have shifted. That response is part of what
    transfers, and the two embodiments reach into the tray differently. Read the place target
    from the tray's measured pose, which is already in the observation, rather than bolting it
    down. A free tray also settles instead of hanging in mid-air when its placement is off.
    """
    # 0.5 mm of clearance rather than exact contact, so nothing starts the episode already
    # interpenetrating the table and getting pushed out by the contact solver.
    CLEARANCE = 0.0005

    # The can: a mesh you can see over a cylinder the solver can trust. Visual and collision are
    # split, the way every menagerie robot does it, because neither half alone works here. A bare
    # primitive is numerically ideal and renders as a plain red cylinder, which is not a can. A raw
    # robocasa asset looks right and brings its convex-hull collision with it, which is what made the
    # mesh objects rock on a single narrowphase contact point, tip over, and behave differently at
    # each mass.
    #
    # So the asset's meshes are attached with collision disabled and zero density, purely to be seen,
    # and one analytic cylinder underneath does all the physics and carries all the mass.
    can = L.PICK_PRIMITIVE
    half = can["half_height"]
    # The cylinder hides in the collision group only when a skin is drawn over it. With
    # PICK_SKIN = None it is the only geometry there is, so it must sit in a visible group, otherwise
    # the can is invisible while still perfectly solid.
    skin = getattr(L, "PICK_SKIN", None)
    body = spec.worldbody.add_body(
        name="obj_root",
        pos=[L.PICK_POS[0], L.PICK_POS[1], L.TABLE_TOP_Z + half + CLEARANCE])
    body.add_freejoint()
    g = body.add_geom(
        name="obj_can",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[can["radius"], half, 0.0],
        rgba=list(can["rgba"]),
        friction=list(can["friction"]),
        solref=list(can.get("solref", (0.02, 1.0))),
        group=3 if skin is not None else 2,
    )
    g.mass = L.PICK_MASS

    if skin is not None:
        vis = mujoco.MjSpec.from_file(str(skin))
        for coll in (vis.meshes, vis.textures):
            for asset in coll:
                if asset.file and not asset.file.startswith("/"):
                    asset.file = str((skin.parent / asset.file).resolve())
        # Measure the skin so it can be scaled onto the collider instead of guessed at.
        pm = vis.compile()
        pd = mujoco.MjData(pm)
        mujoco.mj_forward(pm, pd)
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for gi in range(pm.ngeom):
            if pm.geom_type[gi] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = pm.geom_dataid[gi]
            V = pm.mesh_vert[pm.mesh_vertadr[mid]:
                             pm.mesh_vertadr[mid] + pm.mesh_vertnum[mid]].reshape(-1, 3)
            W = (pd.geom_xmat[gi].reshape(3, 3) @ V.T).T + pd.geom_xpos[gi]
            lo, hi = np.minimum(lo, W.min(0)), np.maximum(hi, W.max(0))
        ext = hi - lo
        sxy = (2 * can["radius"]) / max(ext[0], ext[1])
        sz = (2 * half) / ext[2]
        for mesh in vis.meshes:
            mesh.scale = [mesh.scale[0] * sxy, mesh.scale[1] * sxy, mesh.scale[2] * sz]
        for geom in vis.geoms:
            geom.contype = 0
            geom.conaffinity = 0
            # `*_reg_bbox` are robocasa's annotation boxes, not geometry. Park them in group 4 so they stay
            # hidden, or a wireframe box renders around the can.
            geom.group = 4 if geom.name.endswith("reg_bbox") else 2
            geom.density = 0.0        # a visual geom still contributes mass unless told not to
        # Centre the skin on the collider: the asset's body origin is not its centroid.
        centre = (lo + hi) / 2.0 * np.array([sxy, sxy, sz])
        spec.attach(vis, prefix="skin_",
                    frame=body.add_frame(pos=[-centre[0], -centre[1], -centre[2]],
                                         quat=[1, 0, 0, 0]))

    for label, path, pos, free in (
        ("plate", L.PLACE_TARGET, L.PLACE_POS, True),
    ):
        # x,y come from the layout; z is measured so the object rests on the table. The z in
        # PICK_POS/PLACE_POS stays the nominal aim point for the home-pose search.
        pos = (pos[0], pos[1], L.TABLE_TOP_Z + rest_offset(path) + CLEARANCE)
        child = mujoco.MjSpec.from_file(str(path))
        # Absolutise the child's asset paths. A compiled model has one global meshdir, which we point at
        # the robot's assets, while the robocasa object's meshes are relative to its own folder. After
        # `to_xml()` they resolve against the wrong directory and the model fails to load with "Error
        # opening file .../seeed_rebot_devarm/assets/visual/LemonWedge001.obj". In-process compilation
        # happens to work, which makes this easy to miss.
        for coll in (child.meshes, child.textures):
            for asset in coll:
                if asset.file and not asset.file.startswith("/"):
                    asset.file = str((path.parent / asset.file).resolve())
        # robocasa nests the mesh as <worldbody><body><body name="object">, so `object` is two levels
        # down. A free joint is only legal on a direct child of worldbody, so it has to be added to the
        # child's top body before attaching; after attach that body becomes a worldbody child of ours.
        # Adding it to `object` post-attach fails with "free joint can only be used on top level".
        #
        # Geom groups must match the robots' convention or the scene renders wrong. MuJoCo's viewer shows
        # groups 0/1/2 and hides 3+, and menagerie uses 2 for visual and 3 for collision so the collision
        # hulls stay hidden. The robocasa objects ship their collision meshes in group 0, which is
        # visible, so the grey convex hulls get drawn over the textured visual meshes and the object looks
        # like an untextured blob until you press `0` to hide the group (which also hides the table, since
        # it lives there too). `*_reg_bbox` are annotation boxes, not geometry, so park them in 4.
        for geom in child.geoms:
            if geom.name.endswith("reg_bbox"):
                geom.group = 4
            elif geom.contype or geom.conaffinity:
                geom.group = 3
            else:
                geom.group = 2

        # Force the pick object's mass (see L.PICK_MASS). Density scales mass linearly, so measure what
        # the asset's own density gives and rescale every collision geom by the ratio. Done on the spec,
        # before compiling into the scene: a post-compile write to body_mass is silently ignored.
        #
        # Pick object only. Keyed on `free` this also caught the tray, which is free-jointed too, and a
        # 2.8 kg tray became a 0.39 kg one that slides when nudged.
        if label == "obj" and getattr(L, "PICK_MASS", None):
            probe = mujoco.MjSpec.from_file(str(path))
            for coll in (probe.meshes, probe.textures):
                for asset in coll:
                    if asset.file and not asset.file.startswith("/"):
                        asset.file = str((path.parent / asset.file).resolve())
            pm = probe.compile()
            cur = float(sum(pm.body_mass))
            if cur > 1e-9:
                # Every geom, not just the colliding ones: in MuJoCo a visual geom contributes mass exactly
                # like a collision geom unless told otherwise. Filtering on contype left this asset's
                # nested `*_Prop` body unscaled and the can still weighed 1.19 kg. Sum over bodies rather
                # than taking the biggest: the asset splits its mass across a nested body, so `obj_object`
                # alone reads well under the true payload.
                scale = L.PICK_MASS / cur
                for geom in child.geoms:
                    geom.density *= scale

        top = child.worldbody.bodies[0]
        top.name = "root"
        top.pos = list(pos)          # position carried on the body, so the frame stays identity
        if free:
            top.add_freejoint()
        # attach() insists on a frame or site; an identity frame keeps the pose on the body and avoids
        # double-counting it (frames are flattened at compile, so the free joint still ends up on a
        # top-level body).
        spec.attach(child, prefix=f"{label}_",
                    frame=spec.worldbody.add_frame(pos=[0, 0, 0], quat=[1, 0, 0, 0]))


def lookat(eye, target, up=(0.0, 0.0, 1.0)):
    """MJCF `xyaxes` for a camera at `eye` aimed at `target`.

    MuJoCo cameras look down their own -z with +y up: z_cam = normalize(eye-target),
    x_cam = normalize(up x z_cam), y_cam = z_cam x x_cam. Computed rather than hand-written:
    nine numbers with a sign error is a classic silent framing bug.
    """
    eye, target, up = np.array(eye, float), np.array(target, float), np.array(up, float)
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    n = np.linalg.norm(x)
    assert n > 1e-6, "camera axis parallel to `up`"
    x /= n
    return [*x, *np.cross(z, x)]


def build(key: str) -> mujoco.MjSpec:
    r = L.ROBOTS[key]
    mounts = L.robot_mounts(key)
    spec = mujoco.MjSpec()
    spec.modelname = f"zero_{key}"
    spec.compiler.degree = False
    # Scene physics is canonical, not inherited per robot. MjSpec.attach does not bring the child's
    # <option>, so a fresh MjSpec falls back to MuJoCo's defaults (Euler, pyramidal cone), which left
    # Panda's kp=4500 joints oscillating hard enough to pin seven actuators at their force limits
    # while holding a stationary pose.
    #
    # Copying the settings from each arm's source model is also wrong, and more subtly: the reBot
    # ships cone="elliptic" impratio="10" and Panda does not, so the same lemon wedge drifted 0.15 deg
    # per 28 s in one scene and 5.65 deg in the other. The environment must be identical across
    # embodiments or a contact-solver difference shows up as a transfer result, which is the confound
    # this project exists to measure. Pinned here, both scenes give 0.06 mm / 0.15 deg. Elliptic
    # friction with a high impratio is also MuJoCo's own recommendation for grasping, and both source
    # models ask for implicitfast anyway.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0
    # Timestep stays the finest either arm was tuned for, rather than a number invented here.
    spec.option.timestep = min(
        mujoco.MjModel.from_xml_path(str(rb["mjcf"])).opt.timestep for rb in L.ROBOTS.values())

    # Multi-point convex contact. MuJoCo's convex-convex narrowphase returns a single contact point
    # per geom pair, so a mesh object resting on the table balances on one point and rocks forever:
    # the lemon wedge accumulated 24-58 deg of net rotation while sitting still. Timestep, solver
    # iterations, integrator and friction all made no difference (some made it worse), because none
    # of them adds the missing contact points. multiccd generates several, taking the wedge to 0.2 deg
    # over 4 s and 2.8 deg over 30 s. Object pose is part of the recorded observation, so a drifting
    # object is label noise in every demo.
    spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
    # Absolute meshdir: the MJCF is written to zero_description/mjcf/ but the meshes live under
    # robots/<robot>/assets, and a relative path breaks the moment the cwd changes.
    # Absolute asset paths, taken from what the ROBOT'S OWN model declares rather than assumed.
    # A compiled model has one global meshdir, the MJCF is written to zero_description/mjcf/, and a
    # relative path breaks the moment the cwd changes. The subdirectory is not always "assets":
    # the reBot and vx300s say assets/, the G1 says meshes/. Note this is NOT the registry's
    # `mesh_src`, which is where the URDF's meshes come from and legitimately differs (the vx300s
    # URDF is Interbotix's while its MJCF is menagerie's).
    #
    # texturedir as well as meshdir. They are separate compiler paths, and a model that ships a
    # texture (the vx300s carries interbotix_black.png) fails with "Error opening file" if only
    # meshdir is set, because the scene's own texturedir is empty and resolves against the cwd.
    child = mujoco.MjSpec.from_file(str(r["mjcf"]))
    spec.meshdir = str((r["mjcf"].parent / (child.meshdir or "assets")).resolve())
    spec.texturedir = str((r["mjcf"].parent / (child.texturedir or child.meshdir or "assets")).resolve())
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 720

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
    # Directional lights, not spots. Two spots aimed at the table lit the work area but left the
    # grippers in shadow, and the grippers are the one thing a wrist camera must resolve. A spot also
    # falls off with distance and angle, so the same gripper changed brightness as the arm moved,
    # which is appearance variation the policy has to learn around for no reason. Directional lights
    # are parallel rays with no falloff, so illumination no longer depends on where in the workspace
    # the hand happens to be.
    #
    # Two of them from opposite sides: one key, one fill. The fill does not cast shadows, or every
    # object gets two and the table reads as cluttered geometry in the `front` view. Levels were swept
    # and measured: at 0.55/0.35 diffuse plus a 0.45 headlight the totals exceeded 1.0 and 23% of the
    # `front` frame clipped at 255, so the pale table read as flat white. These give mean ~110 with
    # 0.0% clipped on both the scene and the wrist views.
    for dr, dif, shadow in ((( 0.3, -0.3, -1.0), 0.45, True),
                            ((-0.4,  0.3, -1.0), 0.28, False)):
        spec.worldbody.add_light(dir=list(dr), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                                 castshadow=shadow,
                                 diffuse=[dif] * 3, specular=[0.05] * 3, ambient=[0.10] * 3)

    # The headlight travels with whatever camera is rendering, so it fills exactly what that camera
    # sees. That is the direct fix for a gripper that is dark in the wrist view but fine in the scene
    # view. Kept modest: `diffuse` much above this blows out the pale table top and Panda's white
    # meshes, and a blown-out frame is unusable training data.
    spec.visual.headlight.ambient = [0.22, 0.22, 0.22]
    spec.visual.headlight.diffuse = [0.28, 0.28, 0.28]
    spec.visual.headlight.specular = [0.08, 0.08, 0.08]

    cx, cy = L.TABLE_CENTER_XY
    hx, hy, hz = L.TABLE_HALF
    spec.worldbody.add_body(name="table", pos=[cx, cy, L.TABLE_TOP_Z - hz]).add_geom(
        name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz],
        material="table_mat")
    for i, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        lz = (L.TABLE_TOP_Z - 2 * hz) / 2
        spec.worldbody.add_body(
            name=f"leg{i}", pos=[cx + sx * (hx - 0.05), cy + sy * (hy - 0.05), lz]
        ).add_geom(name=f"leg{i}_g", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[LEG_R, lz], material="leg_mat")

    # A humanoid is ONE body carrying both arms, not two arms bolted to a table, so it is attached
    # once and keeps its own names: the G1 already calls its joints left_shoulder_pitch_joint and
    # right_shoulder_pitch_joint, so a per-side prefix would produce left_left_shoulder_pitch_joint
    # and break the by-name binding with its URDF. Everything downstream (eef sites, wrist cameras,
    # fingertip sensors) is written per side against those existing names and needs no change.
    sides = ("single",) if r.get("single_body") else L.SIDES
    for side in sides:
        # Reload per side: reusing one MjSpec for two attaches fails with "incompatible id in exclude
        # array", because these models ship <contact><exclude> pairs and attaching rewrites the child's
        # element ids.
        arm = mujoco.MjSpec.from_file(str(r["mjcf"]))

        # Grafted gripper. A robot declaring `graft_gripper` has its own end effector cut off at
        # `cut_body` and the shared one attached at `host_body` in its place. The Panda wears a
        # Robotiq 2F-85 this way, which is what a real Panda cell is usually fitted with, and it
        # keeps the target's gripper independent of whatever the source arm happens to carry.
        #
        # Attached WITH a prefix and then renamed back. A bare attach would keep the gripper's
        # joint names, but MuJoCo gives every spec an implicit top-level default class called
        # `main`, so two un-prefixed specs collide with "repeated default class name" as soon as
        # the per-side prefix is applied. Prefixing avoids that; stripping the prefix off the
        # bodies and joints afterwards leaves the exact names the official ROS 2 description uses,
        # so mujoco_ros2_control binds the URDF and the MJCF without a translation table. Assets
        # and default classes keep their prefix, which is what the collision needed.
        if r.get("graft_gripper"):
            g = r["graft_gripper"]
            # The arm's own keyframes describe the qpos layout it had before the graft, and one of them is
            # called `home`, which collides with the keyframe built below.
            for k in list(arm.keys):
                arm.delete(k)
            if g.get("cut_body"):
                arm.delete(arm.body(g["cut_body"]))
            grip = mujoco.MjSpec.from_file(str(g["mjcf"]))
            attach_keeping_names(
                arm, grip, g.get("prefix", "rq_"),
                arm.body(g["host_body"]).add_frame(pos=list(g["pos"]), quat=list(g["quat"])))

        # Drop the arm's own lights. Both source models ship an overhead spot intended for viewing that
        # arm alone. Attaching two arms brings two more lights on top of the two tuned here, and their
        # diffuse is 0.7 against our 0.35, so total diffuse hit 2.1 and the `top` camera rendered a white
        # frame. Lighting is a property of the scene, not of a component, and these frames are recorded
        # observations.
        for light in list(arm.lights):
            arm.delete(light)

        # Drop the model's own keyframes. A keyframe is a FLAT qpos vector sized for the model that
        # declared it, so once the arm is attached into a scene with a table and free-jointed
        # objects the sizes no longer match and the compile fails with "Keyframe 'stand' has
        # invalid qpos size, got 50, should be 43". This generator writes its own `home` below, and
        # `base_keyframe` reads the source keyframe from a fresh load of the file, so nothing here
        # is needed.
        for k in list(arm.keys):
            arm.delete(k)

        # Drop the model's own worldbody geometry too. The G1 ships a ground plane in its robot
        # file, and attaching the robot at its pelvis pose carries that plane up to z=0.79, so the
        # legs then "penetrate" a floor floating at hip height by up to 481 mm and the solver blows
        # up with "Nan, Inf or huge value in QACC". The floor belongs to the scene, like the lights.
        for geom in list(arm.worldbody.geoms):
            arm.delete(geom)

        # Marker sites must not render into the camera images. panda_ros2.xml carries a `panda_ee` site
        # in group 0 (visible by default, solid red), which put a red ball in the middle of both wrist
        # views and straight into the recorded observations. Group 4 is hidden by default; a site's group
        # affects rendering only, never its use as a frame.
        for site in arm.sites:
            site.group = 4

        if r.get("single_body"):
            # A floating base would let the whole robot fall over the moment the sim starts, and the
            # study is about the arms, not about balance. The free joint is dropped so the pelvis is
            # welded at the stance reach_gate.py verified, and yaw 180 turns it to face the table.
            for j in list(arm.joints):
                if j.type == mujoco.mjtJoint.mjJNT_FREE:
                    arm.delete(j)
            # Freeze whatever the task does not use. `freeze_joints` lists name substrings; the
            # matching joints and their actuators are deleted, so those links become rigid parts of
            # the body. For a table task with a welded pelvis the legs and waist do nothing but
            # sag: servo-held they still drifted 67 mrad at the hips over 10 s, and every one of
            # them is a DOF the solver spends time on. Deleting is honest here in a way that
            # zeroing a gain is not, since the joint genuinely plays no part.
            for pat in r.get("freeze_joints", ()):
                for j in list(arm.joints):
                    if pat in j.name:
                        for act in list(arm.actuators):
                            if act.target == j.name:
                                arm.delete(act)
                        arm.delete(j)

            # Zero the root body's own offset. The G1's pelvis carries pos=(0, 0, 0.793), its
            # standing height, which the free joint would normally override. Attaching at base_pos
            # ADDS to it, putting the whole robot 793 mm too high, and the URDF root has no such
            # offset so the two descriptions disagree by exactly that. `base_pos` is defined as the
            # pelvis pose in the world, so the intrinsic offset has to go.
            for b_ in arm.worldbody.bodies:
                b_.pos = [0.0, 0.0, 0.0]
            frame = spec.worldbody.add_frame(pos=list(r["base_pos"]), quat=list(r["base_quat"]))
            attach_keeping_names(spec, arm, "g1_", frame)
        else:
            frame = spec.worldbody.add_frame(pos=list(mounts[side]), quat=[1, 0, 0, 0])
            spec.attach(arm, prefix=f"{side}_", frame=frame)

    # Gravity compensation, and it must be set here rather than on the compiled model. MuJoCo counts
    # gravcomp bodies at compile time (`ngravcomp`) and skips the whole force path when that count is
    # zero, so assigning `model.body_gravcomp` at runtime does nothing and reports no error; it
    # measured byte-identical to no compensation.
    #
    # A position servo can only generate holding torque from standing error, so every joint sits
    # gravity_torque/kp away from its command forever. That is the entire residual: with gravity off,
    # Panda's hold error is exactly 0.000000 rad. Left uncorrected, the pose recorded as the action is
    # not the pose the arm reaches, which is label noise in every demo and differs per embodiment.
    # Compensating the arm bodies removes it. The manipulated object is not compensated, so grasping
    # still has to hold real weight.
    # For a single-body embodiment the robot is not per-side, so match on the whole subtree instead
    # of the left_/right_ prefix. Filtering by prefix left the G1's pelvis, waist and torso
    # uncompensated, which is most of its mass.
    # For a single-body embodiment the robot is not per-side, so match every robot body instead of
    # the left_/right_ prefix. Filtering by prefix left the G1's pelvis, waist and torso
    # uncompensated, which is most of its mass, and f"{side}_" with an empty side is just "_",
    # which matches nothing at all.
    task_bodies = {"table", "leg0", "leg1", "leg2", "leg3", "plate_root", "plate_object",
                   "obj_root", "obj_object"}
    if r.get("single_body"):
        for body in spec.bodies:
            if body.name not in task_bodies and not body.name.startswith("obj_"):
                body.gravcomp = 1.0
    else:
        for side in L.SIDES:
            for body in spec.bodies:
                if body.name.startswith(f"{side}_"):
                    body.gravcomp = 1.0

    # Repair inconsistent position servos. A MuJoCo affine actuator produces
    #     force = gainprm[0]*ctrl + biasprm[1]*qpos + biasprm[2]*qvel
    # so a position servo needs gainprm[0] == -biasprm[1] (both the kp) and biasprm[2] <= 0 (damping
    # opposes motion). The G1 model mixes two control styles on purpose: its arms, waist and hands
    # are position servos, while its 12 LEG joints are <motor> torque actuators for RL locomotion.
    # Because those motors sit inside the same default class they inherit the servos' biasprm and
    # come out as gain 1 against a qpos term of -500, with biasprm[2] = +1, i.e. POSITIVE velocity
    # feedback that pumps energy in. Under a welded pelvis that collapsed the knee by 2.3 rad in 5 s.
    #
    # This scene welds the pelvis and the legs play no part in a table task, so they only have to
    # hold still. kd follows the ratio the model's own healthy servos use (40.5 at kp 500).
    #
    # Diagnosed on the COMPILED source, then applied to the spec by name. Reading gainprm off an
    # MjSpec actuator gives MuJoCo's raw defaults (gain 1, bias 0) whenever the value comes from a
    # <default> class, not the resolved value, so checking the spec flagged all 43 actuators and
    # would have set every gain to 1.
    if r.get("repair_actuators"):
        src = mujoco.MjModel.from_xml_path(str(r["mjcf"]))
        broken: dict[str, float] = {}
        for i in range(src.nu):
            if src.actuator_trntype[i] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            gain, kp_bias, kd = src.actuator_gainprm[i][0], -src.actuator_biasprm[i][1], \
                src.actuator_biasprm[i][2]
            if abs(gain - kp_bias) < 1e-6 and kd <= 0.0:
                continue
            # Keyed by the TARGET JOINT, not the actuator name: attaching prefixes actuator names
            # (they come out as g1_left_knee_joint) while attach_keeping_names strips the prefix
            # back off the joints, so only the target is stable across the two. check_parity
            # resolves actuators the same way, and for the same reason.
            jname = mujoco.mj_id2name(src, mujoco.mjtObj.mjOBJ_JOINT,
                                      int(src.actuator_trnid[i, 0]))
            broken[jname] = float(kp_bias if kp_bias > 0 else gain)
        for act in spec.actuators:
            kp = broken.get(act.target)
            if kp is None or kp <= 0:
                continue
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            act.gainprm = [kp] + [0.0] * 9
            act.biasprm = [0.0, -kp, -0.081 * kp] + [0.0] * 7
            jr = spec.joint(act.target).range
            act.ctrlrange = list(jr) if (jr[0] or jr[1]) else [-6.28, 6.28]
        n_fixed = sum(1 for act in spec.actuators if broken.get(act.target, 0) > 0)
        if n_fixed:
            print(f"  repaired {n_fixed} actuator(s) that were not consistent position servos")

    # Cap gripper actuator force, see L.ROBOTS[*]["grip_force"]. Done on the spec so it is compiled
    # in; a post-compile write to actuator_forcerange is ignored like every other one.
    fmax = r.get("grip_force")
    kp = r.get("grip_kp")
    if fmax or kp:
        want = {L.prefixed(side, j) for side in L.SIDES for j in r["gripper_joints"]}
        n_capped = 0
        for act in spec.actuators:
            if act.target in want:
                if fmax:
                    act.forcerange = [-fmax, fmax]
                if kp:
                    # A position servo squeezes with kp * (commanded - actual). When the jaws stop
                    # on the object that error is fixed by geometry, so kp alone sets the grip, and
                    # forcerange only ever clips it. Both terms have to move together: gainprm[0]
                    # is kp and biasprm[1] must be -kp or the servo is no longer a position servo.
                    gain = list(act.gainprm)
                    bias = list(act.biasprm)
                    kv = -bias[2] if len(bias) > 2 else 0.0
                    gain[0] = kp
                    bias[1] = -kp
                    if len(bias) > 2:
                        bias[2] = -kv
                    act.gainprm = gain
                    act.biasprm = bias
                n_capped += 1
        assert n_capped, f"no gripper actuator matched {sorted(want)}"

    mu = r.get("pad_friction")
    if mu:
        # Accepts a scalar (sliding only) or a (sliding, torsional) pair. Torsional matters for a
        # tall object gripped near its rim, which rotates out of the jaw instead of sliding down.
        mu_slide, mu_tors = (mu, None) if np.isscalar(mu) else (mu[0], mu[1])
        n_pads = 0
        for g in spec.geoms:
            if "finger_pad" in (g.name or ""):
                fr = list(g.friction)
                fr[0] = mu_slide
                if mu_tors is not None:
                    fr[1] = mu_tors
                g.friction = fr
                n_pads += 1
        assert n_pads, "pad_friction is set but no finger_pad geom matched"

    add_task(spec)

    # Object poses as ros2_control state interfaces. The cross-embodiment shadow renderer needs the
    # can and tray where they ACTUALLY are, and the sim does not otherwise publish them. Naming is
    # fixed by mujoco_ros2_control: a <sensor mujoco_type="pose"> named X reads MJCF sensors X_pos
    # (framepos) and X_quat (framequat), so these names are a contract with gen_urdf, not a choice.
    # Referenced off an explicit site, not the body. A framepos aimed at a body that was attached
    # with a frame (the tray) reports double its world position, 0.68/-0.90/1.50 for a body actually
    # at 0.34/-0.45/0.75. A site pins the frame unambiguously and both objects then read true.
    for label, body_name in (("can_pose", "obj_root"), ("tray_pose", "plate_root")):
        host = next((b for b in spec.bodies if b.name == body_name), None)
        if host is None:
            continue
        host.add_site(name=f"{label}_site", pos=[0.0, 0.0, 0.0],
                      quat=[1.0, 0.0, 0.0, 0.0], size=[0.005] * 3, group=4)
        for suffix, stype in (("_pos", mujoco.mjtSensor.mjSENS_FRAMEPOS),
                              ("_quat", mujoco.mjtSensor.mjSENS_FRAMEQUAT)):
            sen = spec.add_sensor()
            sen.name = label + suffix
            sen.type = stype
            sen.objtype = mujoco.mjtObj.mjOBJ_SITE
            sen.objname = f"{label}_site"


    # Fingertip force/torque sensors: a site per finger body plus a matching force+torque pair.
    # MuJoCo's force/torque sensors report the wrench transmitted through the sensorised body's own
    # joint, so a site on the finger gives that finger's grasp load, which is what distinguishes
    # "both pads loaded" from "object resting against one pad".
    #
    # Names are a contract: mujoco_ros2_control resolves a <sensor mujoco_type="fts"> named X to MJCF
    # sensors X_force and X_torque. Both ends come from L.ft_sensors() so they cannot drift, and
    # check_parity.py asserts the pairing.
    for sensor_name, joint_name in L.ft_sensors(key):
        jnt = spec.joint(joint_name)
        assert jnt is not None, f"no joint {joint_name!r} to attach an F/T sensor to"
        site_name = f"{sensor_name}_site"
        # The site goes on the body that owns the finger joint, at that body's origin.
        spec.body(jnt.parent.name).add_site(name=site_name, pos=[0.0, 0.0, 0.0],
                                            group=4)     # marker only, never rendered
        for kind, stype in (("force", mujoco.mjtSensor.mjSENS_FORCE),
                            ("torque", mujoco.mjtSensor.mjSENS_TORQUE)):
            sen = spec.add_sensor()
            sen.name = f"{sensor_name}_{kind}"
            sen.type = stype
            sen.objtype = mujoco.mjtObj.mjOBJ_SITE
            sen.objname = site_name

    # eef sites and cameras. Neither menagerie model ships a site, and nothing pose-based (IK, an
    # ee_to_object observation, a reach reward) can work without one.
    for side in L.SIDES:
        body = spec.body(f"{side}_{r['eef_body']}")
        body.add_site(name=f"{side}_eef", pos=list(r["eef_offset"]),
                      quat=list(r.get("eef_quat", (1.0, 0.0, 0.0, 0.0))),
                      size=[0.008] * 3, group=4)
        # fovy is per side, because the two wrist cameras answer different questions: the giving
        # side has to see the object at the pick, the receiving side has to see it at the handover.
        fovy = r.get("wrist_cam_fovy", L.WRIST_CAM_FOVY)
        if isinstance(fovy, dict):
            fovy = fovy.get(side, L.WRIST_CAM_FOVY)
        # pos is per side for the same reason fovy is: the giving camera has to see the object at
        # the grasp and the receiving one has to see it at the handover, and those pull opposite ways.
        campos = r["wrist_cam_pos"]
        if isinstance(campos, dict):
            campos = campos[side]
        body.add_camera(name=f"{side}_wrist", pos=list(campos),
                        fovy=fovy, xyaxes=list(r["wrist_cam_xyaxes"]),
                        resolution=list(L.CAM_RES))
    # Push the near plane in. See L.CAM_ZNEAR: the default clips objects a wrist camera has to
    # see at grasp range, and it fails silently by rendering them away rather than erroring.
    spec.visual.map.znear = L.CAM_ZNEAR

    for name, (eye, fovy) in L.SCENE_CAMS.items():
        spec.worldbody.add_camera(name=name, pos=list(eye), fovy=fovy,
                                  xyaxes=lookat(eye, L.LOOK_AT), resolution=list(L.CAM_RES))
    return spec


def main() -> None:
    key = sys.argv[1] if len(sys.argv) > 1 else "rebot"
    if key not in L.ROBOTS:
        raise SystemExit(f"unknown robot {key!r}; known: {list(L.ROBOTS)}")
    r = L.ROBOTS[key]

    # Round-trip through XML before writing the keyframe. spec.to_xml() does not necessarily emit
    # bodies in the order the in-process compile enumerated them, and a keyframe is a flat qpos vector
    # indexed by that order. Writing one straight from the in-process MjData produced a keyframe whose
    # free-joint blocks were transposed on reload: the can loaded at the tray's pose, 55 mm inside the
    # table, while the tray loaded at the can's. Nothing errors, since the vector is the right length
    # and merely permuted, and it looks like a placement bug in add_task. Re-loading the emitted XML
    # gives a spec whose ordering is the ordering the keyframe will be read back with.
    spec = mujoco.MjSpec.from_string(build(key).to_xml())
    model = spec.compile()
    d = mujoco.MjData(model)

    # Resolve actuators by transmission target, never by joint name. This used to be
    # mj_name2id(mjOBJ_ACTUATOR, f"{side}_{jb}"), which finds an actuator only when it happens to
    # share its joint's name. The reBot's do; the vx300s calls its gripper one `gripper`, so every
    # lookup returned -1, the `if aid >= 0` swallowed it, and the model shipped a home keyframe
    # whose ctrl was all zeros. The sim then drove every joint from the home qpos toward 0 the
    # instant it loaded. check_parity.py resolves actuators the same way.
    act_of_joint: dict[int, int] = {}
    for i in range(model.nu):
        if model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
            act_of_joint[int(model.actuator_trnid[i, 0])] = i

    # Only the arm joints and the COMMANDED gripper joints get written. Writing grip_range[1] to
    # every gripper joint assumes they share a sign, which holds for the reBot's two slides and
    # fails for a mirrored pair: the vx300s runs left_finger [0.021, 0.057] against right_finger
    # [-0.057, -0.021], so writing +0.057 to both puts one finger outside its own range. It then
    # snapped ~0.11 m on load, which is 3x the finger's whole travel.
    # The follower is placed by SETTLING instead: hold every actuator at its commanded value and
    # step, so MuJoCo's own <equality> works out where the coupled joint belongs. That is general
    # over mirrored slides, same-signed slides and four-bar linkages alike.
    # Start from the source model's own keyframe when the registry names one, so joints that are
    # not part of the task still get a sensible pose. The G1's legs, waist and neck are not in
    # arm_joints, and leaving them at zero means straight legs: with the pelvis welded that left
    # the knee drifting 2.7 rad in 5 s. Applied BY JOINT NAME, not by qpos index, because the
    # scene's ordering is not the source model's.
    if r.get("base_keyframe"):
        src = mujoco.MjModel.from_xml_path(str(r["mjcf"]))
        kid = mujoco.mj_name2id(src, mujoco.mjtObj.mjOBJ_KEY, r["base_keyframe"])
        assert kid >= 0, f"{r['mjcf'].name} has no keyframe {r['base_keyframe']!r}"
        for j in range(src.njnt):
            if src.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = mujoco.mj_id2name(src, mujoco.mjtObj.mjOBJ_JOINT, j)
            tgt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if tgt >= 0:
                d.qpos[model.jnt_qposadr[tgt]] = src.key_qpos[kid, src.jnt_qposadr[j]]

    joints = L.robot_all_joints(key)
    commanded = set(r["arm_joints"]) | set(r["grip_ctrl_joints"])
    for side in L.SIDES:
        pose = dict(zip(r["arm_joints"], r["home"][side]))
        # `grip_open` gives a per-joint open pose, for hands whose joints do not share a range.
        # A two-finger jaw is fully described by grip_range, but the G1's Dex3 has seven joints
        # with seven different ranges, so one scalar cannot place them.
        if r.get("grip_open"):
            pose.update(dict(zip(r["grip_ctrl_joints"], r["grip_open"])))
        else:
            for j in r["grip_ctrl_joints"]:
                pose[j] = r["grip_range"][1]                  # start with the jaw open
        for jb, q in pose.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_{jb}")
            assert jid >= 0, f"missing joint {side}_{jb}"
            d.qpos[model.jnt_qposadr[jid]] = q
            aid = act_of_joint.get(jid, -1)
            if aid >= 0:
                d.ctrl[aid] = q
    mujoco.mj_forward(model, d)
    for _ in range(int(0.5 / model.opt.timestep)):
        mujoco.mj_step(model, d)

    # Write a `home` keyframe. Without it mujoco_ros2_control starts the sim at qpos=0 and the arms
    # bolt upright; the generator was only applying the home pose for its own render. qpos here is the
    # full vector (both arms plus the free-jointed object), captured after mj_forward so the object
    # sits where add_task placed it.
    key_ = spec.add_key(name="home")
    key_.qpos = d.qpos.copy().tolist()
    key_.ctrl = [0.0] * model.nu

    # ctrl comes from the SETTLED qpos of each actuator's own joint, so the keyframe commands
    # exactly the configuration it stores. A keyframe whose ctrl disagrees with its qpos is a
    # spring-loaded model that lurches on load.
    n_set = 0
    for jid, aid in act_of_joint.items():
        key_.ctrl[aid] = float(d.qpos[model.jnt_qposadr[jid]])
        n_set += 1
    # A keyframe whose ctrl does not command the qpos it stores is a spring-loaded model, so assert
    # every actuator got a value rather than trusting the loop.
    assert n_set == model.nu, (
        f"home keyframe set {n_set}/{model.nu} actuators. An unset actuator holds 0.0 and "
        f"will drive its joint away from the home pose on load")

    out = L.PKG / "mjcf" / f"zero_{key}.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(spec.to_xml())
    cams = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
    print(f"[{key}] nq={model.nq} nu={model.nu} nbody={model.nbody} ncam={model.ncam} "
          f"neq={model.neq}")
    print(f"  cameras: {cams}")
    for side in L.SIDES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_eef")
        print(f"  {side}_eef @ {np.round(d.site_xpos[sid], 3)}")
    print(f"  wrote {out}")

    # contact sheet, so framing is verified rather than assumed
    tiles = []
    with mujoco.Renderer(model, height=360, width=480) as ren:
        for n in cams:
            ren.update_scene(d, camera=n)
            img = np.array(ren.render())
            img[:24] = (18, 24, 46)
            tiles.append(img)
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    grid = np.zeros((360 * rows, 480 * cols, 3), np.uint8)
    for i, img in enumerate(tiles):
        rr, cc = divmod(i, cols)
        grid[rr * 360:(rr + 1) * 360, cc * 480:(cc + 1) * 480] = img
    png = L.ROOT / "scenes" / f"{key}_cameras.png"
    Image.fromarray(grid).save(png)
    print(f"  wrote {png}")


if __name__ == "__main__":
    main()
