"""cloud_viewer on a live cell: one environment's cameras, map and scene, in a browser.

    viz = CellViz(env, env_id=0, port=8080)     # prints the URL to open
    viz.update()                                # after a step, a scan, a leg ...

update() reads what the cell has now: every camera's current view (or the
views it is given -- a scan's, say), the map from that environment's planner
server (the "voxels" op; nothing with --no-mapping), and the scene with its
simulator-only bodies and payload where they stand. An example adds its own
perception layer on the same viewer (grasp.viz). Call it between steps, from
the thread that steps: it talks to the planner server like the cell does.

Importing this needs nothing from Isaac, so entry points can add_args() before
the app starts -- and must call preload() before it starts too: Kit puts its
own, older websockets on the path, and viser imported after it gets half of
each and fails.
"""


def preload(args):
    """With --viz, import viser (and its websockets) before the Kit app does."""
    if getattr(args, "viz", False):
        import viser  # noqa: F401


def add_args(ap):
    ap.add_argument("--viz", action="store_true",
                    help="serve a point cloud view of one env in the browser (cloud_viewer)")
    ap.add_argument("--viz-env", type=int, default=0, help="which env --viz shows")
    ap.add_argument("--viz-port", type=int, default=8080)
    ap.add_argument("--viz-stride", type=int, default=4,
                    help="every Nth depth pixel each way")


class CellViz:
    def __init__(self, env, env_id=0, port=8080, stride=4, log=print):
        from cloud_viewer import CloudViewer

        self.u = env.unwrapped
        if not 0 <= env_id < self.u.num_envs:
            raise ValueError(f"--viz-env {env_id}: there are {self.u.num_envs} envs")
        self.e = env_id
        self.viewer = CloudViewer(port=port, stride=stride,
                                  title=f"{self.u.cfg.cell.scene} env {env_id}")
        log(f"[viz] env {env_id}: open {self.viewer.url}")

    def update(self, views=None, labels=None, notes=()):
        """Redraw the cameras, map and reference layers. Returns the status lines."""
        cell, e = self.u.cell, self.e
        if views is None:
            labels = sorted(cell.cams)
            views = [cell.camera_view(name, [e])[0] for name in labels]
        n_pts = self.viewer.show_views(views, labels)
        vox = self.u.pool.voxels(e)
        n_vox = self.viewer.show_voxels(*vox) if vox is not None else self.viewer.show_voxels(None, 0)
        obs = cell.observe([e])
        self.viewer.show_scene(
            self.u.scene_spec,
            bodies={name: cell.body_pose[name][e] for name in cell.body_pose},
            payload={name: pose[0].tolist() for name, pose in obs.objects.items()})
        lines = [f"**env {e}**, sim time {float(obs.sim_time):.1f} s",
                 f"cameras: {len(views)} view(s), {n_pts} points" if views else "cameras: none",
                 f"map: {n_vox} voxels" if vox is not None else "map: none (no mapping)"]
        lines += list(notes)
        self.viewer.set_status("  \n".join(lines))
        return lines

    def close(self):
        self.viewer.close()
