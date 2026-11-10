"""Apply helios_initial_pose.json to stopped Isaac through its Python server.

Initial states and drive targets are temporary USD session overrides. Press Play
afterward to initialize physics at this pose; no scene files are saved.
"""

import argparse
import json
import math
from pathlib import Path
import socket

# At 0 deg the LU/RL/LL thumbs start 4-8 mm inside their bars (palm anchors are
# 20 mm from the bar axis, radius 15 mm). These angles clear every bar by
# >= 6.7 mm in the solved pose and stay clear of the -18 deg abduction limit.
SUPPORT_THUMB_DEG = {"abd": -15.0, "mcp": -25.0}
SUPPORT_THUMB_PREFIXES = {"LU": "left_thumb_", "RL": "or4r_right_thumb_", "LL": "or4l_left_thumb_"}


def support_thumb_positions():
    """Opened LU/RL/LL thumb joints in radians; other finger joints stay unchanged."""
    return {prefix + joint: math.radians(degrees)
            for prefix in SUPPORT_THUMB_PREFIXES.values()
            for joint, degrees in SUPPORT_THUMB_DEG.items()}


def apply_pose(stage, pose):
    """Author the initial state in one replaceable session layer."""
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    if Path(stage.GetRootLayer().realPath).resolve() != Path(pose["scene"]).resolve():
        raise ValueError("Open the scene used to calculate this initial pose")
    base = stage.GetPrimAtPath(pose["base_prim"])
    if not base or not base.IsA(UsdGeom.Xformable):
        raise ValueError(f"Missing base frame: {pose['base_prim']}")
    position = pose["base_position_world_m"]
    orientation = pose["base_orientation_world_wxyz"]
    thumbs = support_thumb_positions()
    if thumbs.keys() & pose["joint_positions_rad"].keys():
        raise ValueError("The pose already sets support thumb joints; remove them or SUPPORT_THUMB_DEG")
    joints = {**pose["joint_positions_rad"], **thumbs}
    if (len(position) != 3 or len(orientation) != 4 or not joints
            or not all(math.isfinite(v) for v in [*position, *orientation, *joints.values()])
            or not math.isclose(sum(v * v for v in orientation), 1.0, abs_tol=1e-6)):
        raise ValueError("Expected finite joint/base positions and a unit WXYZ quaternion")

    prims = list(stage.Traverse())
    joint_prims = {}
    for name in joints:
        matches = [p for p in prims if p.GetName() == name and p.IsA(UsdPhysics.RevoluteJoint)]
        if len(matches) != 1:
            raise ValueError(f"Expected one revolute joint named {name}, found {len(matches)}")
        joint_prims[name] = matches[0]
    attachments = {}
    cache = UsdGeom.XformCache()
    metres = UsdGeom.GetStageMetersPerUnit(stage)
    for arm in ("LU", "RL", "LL"):
        matches = [p for p in prims if p.GetName() == f"D6Joint_{arm}" and p.IsA(UsdPhysics.Joint)]
        if len(matches) != 1:
            raise ValueError(f"Expected one D6 attachment for {arm}")
        joint = attachments[arm] = UsdPhysics.Joint(matches[0])
        point = Gf.Vec3d(joint.GetLocalPos0Attr().Get())
        bodies = joint.GetBody0Rel().GetTargets()
        if bodies:
            point = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(bodies[0])).Transform(point)
        expected = Gf.Vec3d(*pose["attachments"][arm]["target_world_m"])
        if (point * metres - expected).GetLength() > 1e-5:
            raise ValueError(f"{arm} anchor changed; rerun helios_initial_pose.py")

    desired = Gf.Matrix4d().SetRotate(Gf.Quatd(orientation[0], Gf.Vec3d(*orientation[1:])))
    desired.SetTranslateOnly(Gf.Vec3d(*(v / metres for v in position)))
    xform = UsdGeom.Xformable(base)
    parent = cache.GetLocalToWorldTransform(base.GetParent())
    local = desired if xform.GetResetXformStack() else desired * parent.GetInverse()
    session = stage.GetSessionLayer()
    previous = [p for p in session.subLayerPaths
                if Sdf.Layer.Find(p) and Sdf.Layer.Find(p).GetDisplayName() == "helios_initial_pose"]
    layer = Sdf.Layer.CreateAnonymous("helios_initial_pose")
    session.subLayerPaths.insert(0, layer.identifier)
    try:
        with Usd.EditContext(stage, layer):
            xform.MakeMatrixXform().Set(local)
            for name, prim in joint_prims.items():
                degrees = math.degrees(joints[name])
                state = PhysxSchema.JointStateAPI.Apply(prim, "angular")
                state.CreatePositionAttr(degrees)
                state.CreateVelocityAttr(0.0)
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                if drive:
                    drive.CreateTargetPositionAttr(degrees)
                    drive.CreateTargetVelocityAttr(0.0)
            for prim in prims:
                if prim.GetPath().HasPrefix(base.GetPath()) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid = UsdPhysics.RigidBodyAPI(prim)
                    rigid.CreateVelocityAttr(Gf.Vec3f(0.0))
                    rigid.CreateAngularVelocityAttr(Gf.Vec3f(0.0))
            for joint in attachments.values():
                joint.CreateExcludeFromArticulationAttr(True)
                joint.CreateJointEnabledAttr(True)
            # Legacy scenes may still contain an RU D6. It is never used.
            for prim in prims:
                if prim.GetName() == "D6Joint_RU" and prim.IsA(UsdPhysics.Joint):
                    prim.SetActive(False)
    except Exception:
        session.subLayerPaths.remove(layer.identifier)
        raise
    for identifier in previous:
        session.subLayerPaths.remove(identifier)
    return layer


def apply_in_isaac(pose):
    import builtins
    import omni.timeline
    import omni.usd

    if not omni.timeline.get_timeline_interface().is_stopped():
        raise RuntimeError("Stop Isaac before applying the initial pose")
    helper = getattr(builtins, "helios_grasp", None)
    if helper is not None and not helper.closed:
        raise RuntimeError("Exit the retargeting controller before applying the initial pose")
    apply_pose(omni.usd.get_context().get_stage(), pose)
    thumbs = ", ".join(f"{joint} {degrees:g} deg" for joint, degrees in SUPPORT_THUMB_DEG.items())
    print(f"Initial pose applied for {len(pose['joint_positions_rad'])} joints.")
    print(f"LU/RL/LL thumbs opened clear of the bars ({thumbs}); other fingers unchanged.")
    print("LU/RL/LL attached; RU detached. Joint states take effect on Play.")
    print("Session overrides only; reload the scene to discard them.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-path", type=Path, default=Path("helios_initial_pose.json"))
    parser.add_argument("--isaac-port", type=int, default=8226)
    args = parser.parse_args()
    pose = json.loads(args.pose_path.read_text())
    source = Path(__file__).read_text()
    code = ("_helios_pose = {'__name__': '_helios_pose'}\n"
            f"exec(compile({source!r}, 'helios_apply_initial_pose.py', 'exec'), _helios_pose)\n"
            f"_helios_pose['apply_in_isaac']({pose!r})\n")
    with socket.create_connection(("127.0.0.1", args.isaac_port), timeout=5.0) as connection:
        connection.settimeout(30.0)
        connection.sendall(code.encode())
        connection.shutdown(socket.SHUT_WR)
        with connection.makefile("r", encoding="utf-8") as stream:
            response = json.load(stream)
    if response.get("status") != "ok":
        raise RuntimeError(f"Isaac pose setup failed: {response.get('evalue', response)}")
    print(response.get("output", ""), end="")


if __name__ == "__main__":
    main()
