"""Started from the terminal by helios_retargetiing.py --ros --auto-attach.

Enable isaacsim.code_editor.python_server in Isaac first; start while STOPPED,
then press Play. Alternatively, run this file in Isaac's Script Editor.

Requires the ROS 2 Bridge and PhysX. Uses the live scene's D6 joint anchors,
including mesh scale. Keeps all angular settings unchanged (rotation must be
free). RU is enabled once its two anchors are strictly less than 0.05 m apart.
Reads RU localPos0 from your scene; never writes or overrides the target.
With no body0, localPos0 is in world coordinates. Joint changes use a private
temporary layer that is removed on shutdown; the saved asset is not modified.

Publishes /helios/ru_attachment (std_msgs/String JSON): attached, distance_m,
and anchors_m (world-space body0 anchors in RU, LU, RL, LL order).
The IK subscribes automatically; no additional OmniGraph nodes are needed.
Stop, close(), or exit of the launching terminal process removes the helper and
its temporary edits. Restart with --auto-attach after stopping the simulation.
"""

import builtins
import json
import os
import time

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

PALMS = ("or2r_R_Carpals", "or2l_L_Carpals", "or4r_R_Carpals", "or4l_L_Carpals")
ATTACH_DISTANCE = 0.05


def find_attachment_joints(stage):
    joints = [UsdPhysics.Joint(p) for p in stage.Traverse() if p.GetTypeName() == "PhysicsJoint"]
    result = []
    for palm in PALMS:
        matches = [j for j in joints if any(palm in str(p) for p in j.GetBody1Rel().GetTargets())]
        if len(matches) != 1:
            raise ValueError(f"Expected one D6 attachment for {palm}, found {len(matches)}")
        result.append(matches[0])
    return result


def anchor_position(joint, side, transform_point):
    """Resolve localPos against its body, or world when that relationship is empty."""
    body = getattr(joint, f"GetBody{side}Rel")().GetTargets()
    local = np.array(getattr(joint, f"GetLocalPos{side}Attr")().Get(), dtype=float)
    return transform_point(str(body[0]), local) if body else local


class HeliosAttachment:
    def __init__(self, owner_pid=None):
        import carb
        import omni.kit.app
        import omni.timeline
        import omni.usd
        import rclpy
        from isaacsim.core.experimental.prims import XformPrim
        from isaacsim.core.experimental.utils.backend import use_backend
        from std_msgs.msg import String

        self.timeline = omni.timeline.get_timeline_interface()
        if not self.timeline.is_stopped():
            raise RuntimeError("Stop Isaac before running this script, then press Play")
        self.stage = omni.usd.get_context().get_stage()
        self.usd_context = omni.usd.get_context()
        self.owner_pid = owner_pid
        self.closed = False
        self.has_played = False
        self.joints = find_attachment_joints(self.stage)
        self.ru = self.joints[0]
        paths = {str(p) for j in self.joints for rel in (j.GetBody0Rel(), j.GetBody1Rel()) for p in rel.GetTargets()}
        self.prims = {p: XformPrim(p, reset_xform_op_properties=False) for p in paths}
        self.use_backend = use_backend
        settings = carb.settings.get_settings()
        self.backend = "fabric" if settings.get_as_bool("/app/useFabricSceneDelegate") else "usd"
        if self.backend == "usd" and settings.get("/physics/updateToUsd") is False:
            raise RuntimeError("Enable physics updateToUsd or Fabric Scene Delegate to read live hand poses")
        self.metres = UsdGeom.GetStageMetersPerUnit(self.stage)
        if not rclpy.ok():
            rclpy.init(args=[])
        self.node = rclpy.create_node("helios_ru_attachment")
        self.message_type = String
        self.publisher = self.node.create_publisher(String, "/helios/ru_attachment", 1)
        self.last_update = 0.0
        self.layer = Sdf.Layer.CreateAnonymous("helios_auto_attach")
        self.stage.GetSessionLayer().subLayerPaths.insert(0, self.layer.identifier)
        # A switchable external attachment must not become part of the articulation tree.
        with Usd.EditContext(self.stage, self.layer):
            self.ru.CreateExcludeFromArticulationAttr(True)
        self.subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self.update)
        print(f"RU auto-attach ready: {self.ru.GetPath()}, distance < {ATTACH_DISTANCE} m")

    def transform_point(self, path, local):
        # Fabric supplies the current simulated pose even when USD transforms are stale.
        with self.use_backend(self.backend, raise_on_fallback=True):
            p, q = self.prims[path].get_world_poses()
            scale = self.prims[path].get_world_scales()
        p, q, v = p.numpy()[0], q.numpy()[0], local * scale.numpy()[0]
        return p + v + 2.0 * np.cross(q[1:], np.cross(q[1:], v) + q[0] * v)

    def update(self, _event):
        if self.closed:
            return
        if self.owner_pid is not None:
            try:
                os.kill(self.owner_pid, 0)
            except ProcessLookupError:
                self.close()
                return
        if self.usd_context.get_stage() != self.stage:
            self.close()
            return
        if self.timeline.is_stopped():
            if self.has_played:
                self.close()
            return
        if not self.timeline.is_playing():
            return
        self.has_played = True
        if time.monotonic() - self.last_update < 0.02:
            return
        self.last_update = time.monotonic()
        try:
            anchors = np.array([anchor_position(j, 0, self.transform_point) for j in self.joints]) * self.metres
            hand = anchor_position(self.ru, 1, self.transform_point) * self.metres
            distance = float(np.linalg.norm(hand - anchors[0]))
            if not np.isfinite(anchors).all() or not np.isfinite(distance):
                raise ValueError("Nonfinite attachment positions")
            attached = bool(self.ru.GetJointEnabledAttr().Get())
            if not attached and distance < ATTACH_DISTANCE:
                with Usd.EditContext(self.stage, self.layer):
                    self.ru.GetJointEnabledAttr().Set(True)
                attached = True
                print(f"RU attached at {distance * 100:.2f} cm")
            self.publisher.publish(self.message_type(data=json.dumps({
                "attached": attached, "distance_m": distance, "anchors_m": anchors.tolist(),
            })))
        except Exception as error:
            print(f"RU auto-attach stopped: {error}")
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.subscription.unsubscribe()
        self.subscription = None
        self.node.destroy_node()
        layers = self.stage.GetSessionLayer().subLayerPaths
        if self.layer.identifier in layers:
            layers.remove(self.layer.identifier)
        print("RU auto-attach stopped; temporary joint overrides removed")


def start_attachment(owner_pid=None):
    previous = getattr(builtins, "helios_attachment", None)
    if previous is not None:
        previous.close()
    builtins.helios_attachment = HeliosAttachment(owner_pid)
    return builtins.helios_attachment


if __name__ == "__main__":
    if "helios_attachment" in globals():
        helios_attachment.close()
    helios_attachment = start_attachment()
