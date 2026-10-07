"""Damped-least-squares SE(3) IK for one arm of a dual-arm ZERO robot, via pinocchio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pinocchio as pin


@dataclass
class IKResult:
    q: np.ndarray          # full-model configuration with this arm's joints updated
    dq: np.ndarray         # the step taken, this arm's joints only
    residual: float        # ‖position error‖ + ‖rotation error‖, metres + radians
    pos_err: float
    rot_err: float
    clamped: bool          # dq hit the per-tick limit -> target is far or ill-conditioned


class ArmIK:
    """One arm. Owns which columns of the full Jacobian belong to it."""

    def __init__(
        self,
        urdf_path: str,
        eef_frame: str,
        joint_names: list[str],
        eef_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        eef_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        damping: float = 0.05,
        gain: float = 1.0,
        dq_max: float = 0.10,
    ) -> None:
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        parent_fid = self.model.getFrameId(eef_frame)
        if parent_fid >= self.model.nframes:
            raise ValueError(f"frame {eef_frame!r} not in {urdf_path}")

        # The MJCF puts the eef site at an offset from the gripper body, and the recorded action is
        # that site's pose. Add a matching offset frame here or IK will servo the wrist instead of the
        # pinch point, a constant few-centimetre bias in every demo.
        if any(eef_offset) or any(abs(v) > 1e-9 for v in np.asarray(eef_quat)[1:]):
            pf = self.model.frames[parent_fid]
            # pin.Frame's `placement` is relative to the parent joint, not to the parent frame, even
            # though a parent frame id is also passed. So the offset must be composed onto the parent
            # frame's own placement; handing over SE3(I, offset) anchors the tool point at
            # (joint origin + offset) instead. This was live: it put the servo point 119.9 mm (reBot) /
            # 107.0 mm (Panda) from the real gripper, so every IK solve converged tidily onto the wrong
            # point and the arm reached past the object. Composing correctly gives 0.003 mm / 0.000 mm
            # against MuJoCo FK, and check_parity.py asserts it for both robots.
            # `eef_quat` rotates the tool frame onto the SOURCE embodiment's axis convention
            # (site -x = approach, site y = jaw). It has to be applied here as well as on the MJCF
            # site: the recorded action is this frame's rotation, so a target whose site axes are
            # laid out differently reads every commanded orientation wrong. On the Panda that was
            # 59 deg of approach-direction error. check_parity.check_eef_frame compares this frame
            # against the MJCF site and is what catches the two drifting apart.
            w, x, y, z = (float(v) for v in eef_quat)
            R_eef = np.array([
                [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
                [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
            ])
            self.fid = self.model.addFrame(
                pin.Frame(f"{eef_frame}_eef", pf.parentJoint, parent_fid,
                          pf.placement * pin.SE3(R_eef,
                                                 np.array(eef_offset, dtype=float)),
                          pin.FrameType.OP_FRAME))
            self.data = self.model.createData()
        else:
            self.fid = parent_fid

        # Column indices of this arm's joints in the full-model velocity vector. The URDF holds both
        # arms plus (on Panda) a mimic finger, so a full-model Jacobian must be masked or one arm's
        # solve will command the other arm's joints.
        self.cols: list[int] = []
        for n in joint_names:
            jid = self.model.getJointId(n)
            if jid >= self.model.njoints:
                raise ValueError(f"joint {n!r} not in {urdf_path}")
            self.cols.append(self.model.joints[jid].idx_v)
        self.cols_arr = np.array(self.cols, dtype=int)
        self.qidx = np.array(
            [self.model.joints[self.model.getJointId(n)].idx_q for n in joint_names], dtype=int)

        self.damping = damping
        self.gain = gain
        self.dq_max = dq_max

    def fk(self, q: np.ndarray) -> pin.SE3:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.fid)
        return self.data.oMf[self.fid].copy()

    def step(self, q: np.ndarray, target: pin.SE3) -> IKResult:
        """One servo step from configuration `q` toward `target`."""
        q = np.asarray(q, dtype=float).copy()
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.fid)
        cur = self.data.oMf[self.fid]

        # Error as a spatial twist in the local frame; log6 of the relative transform is the correct
        # SE(3) error (naively subtracting rotation matrices is not).
        err = pin.log6(cur.actInv(target)).vector
        pos_err = float(np.linalg.norm(err[:3]))
        rot_err = float(np.linalg.norm(err[3:]))

        pin.computeJointJacobians(self.model, self.data, q)
        J_full = pin.getFrameJacobian(self.model, self.data, self.fid, pin.ReferenceFrame.LOCAL)
        J = J_full[:, self.cols_arr]

        lo, hi = self.model.lowerPositionLimit[self.qidx], self.model.upperPositionLimit[self.qidx]
        qa = q[self.qidx]

        # Clamped least squares. Joint limits have to be enforced inside the solve, not by clipping
        # the answer afterwards. Clipping looks harmless and is not: the DLS step is computed
        # assuming every joint is free to move, so once part of it is discarded the motion the arm
        # makes is no longer the motion the Jacobian predicted, and the task error stops being
        # reduced. Seen on the reBot's left arm reaching for the lemon, whose home pose sits
        # 0.013 rad from joint3's limit: the error fell to 57 mm, then climbed to 220 mm and stuck
        # there, further than the 123 mm it started at, with |dq| pinned at dq_max and `clamped` true
        # forever. It reads exactly like an unreachable target.
        #
        # Instead: solve, freeze whichever joints the step would push out of range at their limit,
        # subtract what those frozen joints still contribute, and re-solve for the rest so the
        # remaining DoF take up the slack. A handful of passes is plenty, since each one freezes at
        # least one joint.
        free = np.ones(len(self.cols), dtype=bool)
        dq = np.zeros(len(self.cols))
        for _ in range(len(self.cols)):
            resid = self.gain * err - J[:, ~free] @ dq[~free]
            Jf = J[:, free]
            JJt = Jf @ Jf.T + (self.damping ** 2) * np.eye(6)
            dq[free] = Jf.T @ np.linalg.solve(JJt, resid)
            q_try = qa + dq
            bad = free & ((q_try < lo - 1e-12) | (q_try > hi + 1e-12))
            if not bad.any():
                break
            dq[bad] = np.clip(qa[bad] + dq[bad], lo[bad], hi[bad]) - qa[bad]
            free &= ~bad
            if not free.any():
                break

        # Bound the step size for smoothness near singularities. Scaling a limit-feasible step toward
        # zero stays feasible, since the joint box is convex and q is inside it, so this cannot undo
        # the clamping above.
        clamped = False
        n = float(np.linalg.norm(dq))
        if n > self.dq_max:
            dq *= self.dq_max / n
            clamped = True

        q_out = q.copy()
        q_out[self.qidx] = np.clip(qa + dq, lo, hi)   # guard only; the solve already respects it

        return IKResult(q=q_out, dq=dq, residual=pos_err + rot_err,
                        pos_err=pos_err, rot_err=rot_err, clamped=clamped)

    def solve(self, q0: np.ndarray, target: pin.SE3, iters: int = 50,
              tol: float = 1e-4) -> IKResult:
        """Iterate to convergence: for tests and for seeding, not for the control loop."""
        q = np.asarray(q0, dtype=float).copy()
        res = None
        for _ in range(iters):
            res = self.step(q, target)
            q = res.q
            if res.residual < tol:
                break
        return res
