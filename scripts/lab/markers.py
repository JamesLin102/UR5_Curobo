"""Goal markers: something to look at in the viewport that the cameras cannot see.

As scene geometry a marker is rendered into depth, fused into the map as an
obstacle sitting on the goal, and then every plan to that goal fails (measured
on the removed Isaac Sim backend: 193 of 199).

    "overlay"  Isaac Sim's debug-draw viewport overlay:
               drawn by a pass render products do not sample. The default.
    "usd"      Isaac Lab's VisualizationMarkers, kept only if a depth check
               shows the cameras do not see them; otherwise "overlay".
    "off"      nothing.

On Isaac Lab 2.3.2 the depth cameras do not see VisualizationMarkers: 5 cm
marker spheres on the goals changed 0 depth pixels in either camera, where a
plain USD sphere of the same size in the same place changed 2019 (and hiding
the slab, as a positive control for the check itself, 113 216). The check
stays, so a version that changes this is caught rather than mapped.

Without a GUI there is no viewport to overlay: "overlay" becomes "off". "usd"
still runs its depth check headless, which is how that check is tested.
"""

import numpy as np
import torch

from .usd_edits import draw_targets


class GoalMarkers:
    MODES = ("overlay", "usd", "off")

    def __init__(self, cell, mode):
        if mode not in self.MODES:
            raise ValueError(f"markers must be one of {self.MODES}, not {mode!r}")
        self.cell = cell
        self.gui = bool(cell.sim.has_gui())
        self.mode = "off" if mode == "overlay" and not self.gui else mode
        self._usd = None

    def targets_world(self):
        """Every scene target, in every environment, in world coordinates."""
        origins = self.cell.origins
        return [list(np.asarray(t[:3]) + origins[e]) + list(t[3:])
                for e in range(self.cell.num_envs) for t in self.cell.spec.targets]

    def draw(self):
        if self.mode == "usd":
            if self._draw_usd():
                self.cell.log("markers: USD markers, invisible to the depth cameras")
            else:
                self.mode = "overlay" if self.gui else "off"
                self.cell.log(f"markers: the depth cameras see the USD markers; "
                              f"using {'the viewport overlay' if self.gui else 'none'} instead")
        if self.mode == "overlay":
            draw_targets(self.targets_world())

    def _draw_usd(self):
        """Show USD markers; False if they reach any depth image."""
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/goals",
            markers={"goal": sim_utils.SphereCfg(
                radius=0.012,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.43, 0.62)))})
        self._usd = VisualizationMarkers(cfg)
        pts = torch.tensor([t[:3] for t in self.targets_world()], dtype=torch.float32)
        self._usd.visualize(translations=pts)
        if not self.cell.cams:
            return True
        seen = self._depth()
        self._usd.set_visibility(False)
        unseen = self._depth()
        leak = max(float(np.abs(a - b).max()) for a, b in zip(seen, unseen))
        if leak > 0.002:
            self._usd = None
            return False
        self._usd.set_visibility(True)
        return True

    def _depth(self, renders=3):
        for _ in range(renders):
            self.cell.sim.render()
            self.cell.scene.update(self.cell.dt)
            self.cell.update_cameras()
        return [cam.data.output["distance_to_image_plane"].cpu().numpy().copy()
                for cam in self.cell.cams.values()]
