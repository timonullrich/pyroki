"""Two Helios mesh overlays in RViz; no robot_state_publisher or TF setup needed."""

from copy import deepcopy
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import yourdfpy


class RvizPoses:
    def __init__(self, node, urdf_path, required_names):
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from visualization_msgs.msg import Marker, MarkerArray

        self.node, self.required_names = node, required_names
        self.last_update = 0.0
        model = yourdfpy.URDF.load(str(urdf_path), build_scene_graph=False, load_meshes=False)
        # Measured coupled joints must use their own readings rather than mimic FK.
        for joint in model.robot.joints:
            joint.mimic = None
        model = yourdfpy.URDF(model.robot, load_meshes=False)
        robot = pk.Robot.from_urdf(model)
        self.names = robot.joints.actuated_names
        indices, origins, templates = [], [], []
        for link in model.robot.links:
            for visual in link.visuals:
                mesh = visual.geometry.mesh
                if mesh is None:
                    continue
                marker = Marker()
                marker.header.frame_id = "world"
                marker.id, marker.type = len(templates), Marker.MESH_RESOURCE
                path = Path(mesh.filename.removeprefix("package://"))
                if not path.is_absolute():
                    path = urdf_path.resolve().parents[2] / path
                marker.mesh_resource = path.resolve().as_uri()
                marker.scale.x, marker.scale.y, marker.scale.z = (
                    np.ones(3) if mesh.scale is None else mesh.scale
                ).tolist()
                templates.append(marker)
                indices.append(robot.links.names.index(link.name))
                origins.append(np.eye(4) if visual.origin is None else visual.origin)
        indices = jnp.asarray(indices)
        origins = jaxlie.SE3.from_matrix(jnp.asarray(np.stack(origins)))
        self.poses = jax.jit(lambda cfg, base: (
            base @ jaxlie.SE3(robot.forward_kinematics(cfg)[indices]) @ origins
        ).wxyz_xyz)
        self.poses(jnp.zeros(len(self.names)), jaxlie.SE3.identity()).block_until_ready()
        self.displays = []
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for name, color in (("commanded", (0.1, 0.5, 1.0, 0.45)), ("actual", (1.0, 0.45, 0.05, 0.8))):
            markers = MarkerArray(markers=deepcopy(templates))
            for marker in markers.markers:
                marker.ns = name
                marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
            publisher = node.create_publisher(MarkerArray, f"/helios/{name}_robot", qos)
            self.displays.append((publisher, markers))

    def publish(self, commanded, commanded_base, measured, root_pose):
        """Render the accepted command and measured joints/root in Isaac world."""
        if root_pose is None or time.monotonic() - self.last_update < 0.1:
            return
        actual = dict(zip(measured.name, measured.position))
        if not all(name in actual for name in self.required_names):
            return
        root_pose = np.asarray(root_pose, dtype=float)
        if root_pose.shape != (7,) or not np.isfinite(root_pose).all():
            return
        self.last_update = time.monotonic()
        stamp = self.node.get_clock().now().to_msg()
        for (publisher, markers), joints, base in zip(
            self.displays, (commanded, actual), (commanded_base, jaxlie.SE3(jnp.asarray(root_pose)))
        ):
            # Uncommanded/unreported fingers retain the assumed zero pose.
            cfg = jnp.asarray([joints.get(name, 0.0) for name in self.names])
            poses = np.asarray(self.poses(cfg, base))
            for marker, (w, x, y, z, px, py, pz) in zip(markers.markers, poses.tolist()):
                marker.header.stamp = stamp
                marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = px, py, pz
                marker.pose.orientation.w, marker.pose.orientation.x = w, x
                marker.pose.orientation.y, marker.pose.orientation.z = y, z
            publisher.publish(markers)
