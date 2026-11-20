"""Isaac side of helios_teleop.py, run through Isaac's Python server.

One session layer holds the helios.yaml actuator gains and the released D6 attachments.
It survives Stop/Play and is removed when the teleop controller exits. Runs in a namespace
that also holds resolve_actuators/apply_actuator (generic helpers in helios_grasp_isaac.py).
"""

import builtins

from pxr import Sdf, Usd, UsdPhysics


def start(actuators):
    import omni.timeline
    import omni.usd

    if not omni.timeline.get_timeline_interface().is_stopped():
        raise RuntimeError("Stop Isaac before starting teleop, then press Play")
    close()
    stage = omni.usd.get_context().get_stage()
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)]
    driven = {p.GetName(): p for p in joints}
    if len(driven) != len(joints):
        raise ValueError("Actuator patterns need unique joint names in the stage")
    values = resolve_actuators(list(driven), actuators)  # noqa: F821 (from helios_grasp_isaac.py)
    layer = Sdf.Layer.CreateAnonymous("helios_teleop")
    stage.GetSessionLayer().subLayerPaths.insert(0, layer.identifier)
    builtins.helios_teleop = (stage, layer)
    with Usd.EditContext(stage, layer):
        for name, params in values.items():
            apply_actuator(driven[name], params)  # noqa: F821
    print(f"Actuator settings applied to {len(values)} joints")


def release(arms):
    """Disable these arms' D6 bar attachments."""
    stage, layer = builtins.helios_teleop
    names = {f"D6Joint_{arm}" for arm in arms}
    with Usd.EditContext(stage, layer):
        for prim in stage.Traverse():
            if prim.GetName() in names and prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)


def close():
    stage, layer = getattr(builtins, "helios_teleop", None) or (None, None)
    builtins.helios_teleop = None
    if layer is not None and layer.identifier in stage.GetSessionLayer().subLayerPaths:
        stage.GetSessionLayer().subLayerPaths.remove(layer.identifier)
