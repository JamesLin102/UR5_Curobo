"""A browser view of what the cameras and the planner see: viser, four layers.

    viewer = CloudViewer(port=8080)          # then open http://localhost:8080
    viewer.show_views(views)                 # camera point clouds + frustums
    viewer.show_voxels(centers, size)        # the planner's map
    viewer.show_scene(spec, bodies, payload) # what the scene says is there
    viewer.box("perception", ...)            # anything else, by primitive

Layers, each a checkbox in the panel:

  cameras     every view's depth back-projected into points, coloured by its
              RGB (by height for a depth-only camera), and a frustum at the
              pose the points were projected from. Views of one scan overlap
              where they agree: a ghost or a shifted copy is a pose error.
  map         the occupied voxels the planner avoids, as of its last ESDF
              refresh, in magenta: one colour, so they stand apart from the
              points of a depth-only camera, which are coloured by height.
  perception  what an example's perception made of the views; the example
              draws it with the primitives below (grasp.viz does).
  reference   the scene: obstacles and keep-out volumes the planner is told
              about, the simulator-only bodies where they were put, the
              payload, the targets.

Simulator-free -- numpy, viser and the shared modules -- so the same viewer
takes views from Isaac Lab or from a real camera. Poses are
[x, y, z, qw, qx, qy, qz] or 4x4, all in one frame (a cell's own, env-local);
the viewer shows one cell at a time.
"""

import math

import numpy as np

from pointcloud import view_cloud
from urdf_frames import matrix_to_quat

LAYERS = ("cameras", "map", "perception", "reference")

GREY = (150, 150, 150)
RED = (220, 60, 60)
ORANGE = (240, 150, 40)
YELLOW = (245, 220, 40)
GREEN = (60, 200, 90)
BLUE = (60, 140, 230)
PURPLE = (160, 90, 220)
MAGENTA = (225, 40, 170)


def height_colors(z, lo=None, hi=None):
    """(N, 3) uint8, blue low to yellow high, for points with no colour of their own."""
    z = np.asarray(z, dtype=np.float64)
    if len(z) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    lo = float(np.min(z)) if lo is None else lo
    hi = float(np.max(z)) if hi is None else hi
    t = np.clip((z - lo) / max(hi - lo, 1e-6), 0.0, 1.0)[:, None]
    low, high = np.array([40, 60, 200.0]), np.array([250, 220, 40.0])
    return (low + t * (high - low)).astype(np.uint8)


def _box_edges(dims):
    """(12, 2, 3) the edges of a box of `dims` centred on its own origin."""
    h = np.asarray(dims, dtype=np.float64) / 2
    c = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * h
    pairs = [(a, b) for a in range(8) for b in range(a + 1, 8)
             if np.count_nonzero(c[a] != c[b]) == 1]
    return np.array([[c[a], c[b]] for a, b in pairs])


def _cylinder_edges(radius, height, sections=24):
    """(3 * sections + 4, 2, 3): top and bottom circles and four sides, axis along z."""
    t = np.linspace(0, 2 * np.pi, sections + 1)
    ring = np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1)
    segs = []
    for z in (-height / 2, height / 2):
        for a, b in zip(ring[:-1], ring[1:]):
            segs.append([[*a, z], [*b, z]])
    for k in range(0, sections, sections // 4):
        segs.append([[*ring[k], -height / 2], [*ring[k], height / 2]])
    return np.array(segs)


def _pose(pose):
    """(wxyz, position) from [x, y, z(, qw, qx, qy, qz)] or a 4x4."""
    p = np.asarray(pose, dtype=np.float64)
    if p.shape == (4, 4):
        return tuple(matrix_to_quat(p[:3, :3])), tuple(p[:3, 3])
    wxyz = tuple(p[3:7]) if len(p) >= 7 else (1.0, 0.0, 0.0, 0.0)
    return wxyz, tuple(p[:3])


class CloudViewer:
    def __init__(self, port=8080, host="127.0.0.1", stride=4, point_size=0.004,
                 title="UR5_curobo"):
        import viser

        self.server = viser.ViserServer(host=host, port=port, label=title, verbose=False)
        # viser moves to the next free port if this one is taken; say the real one.
        self.port = self.server.get_port()
        self.url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{self.port}"
        self.stride, self.point_size = stride, point_size
        self.server.scene.set_up_direction("+z")
        # From in front of the robot and above, at the workspace in front of it.
        self.server.initial_camera.up = (0.0, 0.0, 1.0)
        self.server.initial_camera.position = (1.4, -0.9, 0.9)
        self.server.initial_camera.look_at = (0.4, 0.0, 0.1)
        self.server.scene.add_grid("/grid", width=2.0, height=2.0, cell_size=0.1,
                                   section_size=0.5, plane="xy")
        self.layers = {}
        self.nodes = {name: {} for name in LAYERS}
        for name in LAYERS:
            frame = self.server.scene.add_frame(f"/{name}", show_axes=False)
            box = self.server.gui.add_checkbox(name, True)
            box.on_update(lambda ev, f=frame, b=box: setattr(f, "visible", b.value))
            self.layers[name] = frame
        self.status = self.server.gui.add_markdown("")

    # --- primitives: each replaces the node of the same name in its layer ---------

    def _put(self, layer, name, handle):
        old = self.nodes[layer].pop(name, None)
        if old is not None and old is not handle:
            old.remove()
        self.nodes[layer][name] = handle
        return handle

    def clear(self, layer):
        for handle in self.nodes[layer].values():
            handle.remove()
        self.nodes[layer].clear()

    def points(self, layer, name, pts, colors=None, size=None, shape="rounded"):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        colors = height_colors(pts[:, 2]) if colors is None else np.asarray(colors, dtype=np.uint8)
        return self._put(layer, name, self.server.scene.add_point_cloud(
            f"/{layer}/{name}", pts, colors, point_size=size or self.point_size,
            point_shape=shape))

    def lines(self, layer, name, segments, pose=(0, 0, 0), color=GREY, width=2.0):
        """segments (N, 2, 3), in the frame `pose` puts them in."""
        wxyz, pos = _pose(pose)
        return self._put(layer, name, self.server.scene.add_line_segments(
            f"/{layer}/{name}", np.asarray(segments, dtype=np.float32), color,
            line_width=width, wxyz=wxyz, position=pos))

    def box(self, layer, name, dims, pose, color=GREY, width=2.0):
        """A box's twelve edges."""
        return self.lines(layer, name, _box_edges(dims), pose, color, width)

    def cylinder(self, layer, name, radius, height, pose, color=GREY, width=2.0):
        """A cylinder's rims and four sides, axis along the pose's z."""
        return self.lines(layer, name, _cylinder_edges(radius, height), pose, color, width)

    def axes(self, layer, name, pose, length=0.05):
        wxyz, pos = _pose(pose)
        return self._put(layer, name, self.server.scene.add_frame(
            f"/{layer}/{name}", axes_length=length, axes_radius=length / 20,
            wxyz=wxyz, position=pos))

    def label(self, layer, name, text, position):
        return self._put(layer, name, self.server.scene.add_label(
            f"/{layer}/{name}", text, position=tuple(float(v) for v in position)))

    def set_status(self, text):
        self.status.content = text

    def close(self):
        """Stop the server; its threads otherwise keep the process alive after Ctrl-C."""
        self.server.stop()

    # --- layers -------------------------------------------------------------------

    def show_views(self, views, labels=None):
        """cameras: each view's points and frustum. Replaces what was there."""
        self.clear("cameras")
        total = 0
        for i, view in enumerate(views):
            tag = labels[i] if labels else f"view{i}"
            pts, colors = view_cloud(view, self.stride)
            total += len(pts)
            self.points("cameras", f"{tag}/points", pts, colors)
            h, w = np.asarray(view["depth"]).shape
            K = np.asarray(view["K"])
            wxyz, pos = _pose(view["pose"])
            self._put("cameras", f"{tag}/frustum", self.server.scene.add_camera_frustum(
                f"/cameras/{tag}/frustum", fov=2 * math.atan(h / 2 / K[1, 1]), aspect=w / h,
                scale=0.06, color=(30, 30, 30), wxyz=wxyz, position=pos))
        return total

    def show_voxels(self, centers, size):
        """map: occupied voxel centres, drawn as squares one voxel across."""
        self.clear("map")
        if centers is None or len(centers) == 0:
            return 0
        colors = np.tile(np.array(MAGENTA, dtype=np.uint8), (len(centers), 1))
        self.points("map", "voxels", centers, colors, size=size, shape="square")
        return len(centers)

    def show_scene(self, spec, bodies=None, payload=None):
        """reference: what a scenes.SceneSpec says, with bodies where they stand now.

        bodies {name: pose or None}, simulator-only bodies (None: taken out);
        payload {name: pose}. Missing: the spec's own poses.
        """
        self.clear("reference")
        for name, dims, pose, _ in spec.obstacles:
            self.box("reference", f"obstacle/{name}", dims, pose, GREY)
        for name, dims, pose, _ in spec.keep_out:
            self.box("reference", f"keep_out/{name}", dims, pose, RED)
        for name, dims, pose, _ in spec.unmapped:
            pose = (bodies or {}).get(name, pose)
            if pose is None:
                continue
            if spec.shape(name) == "cylinder":
                self.cylinder("reference", f"body/{name}", dims[0] / 2, dims[2], pose, ORANGE)
            else:
                self.box("reference", f"body/{name}", dims, pose, ORANGE)
        for name, dims, pose, _, _ in spec.payload:
            self.box("reference", f"payload/{name}", dims, (payload or {}).get(name, pose), GREEN)
        for i, t in enumerate(spec.targets):
            self.axes("reference", f"target/{i}", t, 0.04)
