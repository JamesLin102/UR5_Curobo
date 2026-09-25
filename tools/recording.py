"""Side-by-side demo videos: the env's own render on the left, cloud_viewer's on the right.

    rec = Recorder(env, viz, out_dir, "pick_place", (800, 600), every=2, log=say)
    rec.start()          # grabs a frame every `every` physics steps, on the cell's tick
    ...                  # drive the env
    rec.stop()
    rec.finish()         # name.mp4 (real time) and name.gif (2x speed, for a README)

Frames follow simulation time, so the video plays at the arm's real speed
however slowly it was recorded. The right view is cloud_viewer.render() from
the viewport camera's own pose and lens (read off USD), so the two line up.
Needs a running Kit app and an env made with render_mode="rgb_array".
"""

import math
import os
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def label(img, text):
    """Write `text` on a black strip in the image's top-left corner."""
    im = Image.fromarray(np.ascontiguousarray(img))
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=22)
    d.rectangle([0, 0, d.textlength(text, font=font) + 20, 36], fill=(0, 0, 0))
    d.text((10, 6), text, fill=(255, 255, 255), font=font)
    return np.asarray(im)


def viewport_camera(path, width, height):
    """(eye, look_at, up, vfov) of the USD camera at `path`, as Kit renders it.

    The pose is cfg.viewer's; the lens is whatever the viewport camera has.
    Kit keeps the horizontal aperture and derives the vertical from the
    render's aspect.
    """
    import omni.usd
    from pxr import UsdGeom

    cam = UsdGeom.Camera(omni.usd.get_context().get_stage().GetPrimAtPath(path))
    focal = cam.GetFocalLengthAttr().Get()
    aperture = cam.GetHorizontalApertureAttr().Get()
    M = np.array(UsdGeom.Xformable(cam.GetPrim()).ComputeLocalToWorldTransform(0)).T
    eye, forward, up = M[:3, 3], -M[:3, 2], M[:3, 1]      # a USD camera looks down -Z, +Y up
    return eye, eye + forward, up, 2 * math.atan(aperture / 2 / focal * height / width)


class Recorder:
    def __init__(self, env, viz, out_dir, name, size, every=2, log=print,
                 right_title="what the cameras and the planner see"):
        self.u = env.unwrapped
        self.viz, self.every, self.log = viz, every, log
        self.w, self.h = size
        self.fps = 60 // every
        self.right_title = right_title
        self.mp4 = os.path.join(out_dir, f"{name}.mp4")
        self.gif = os.path.join(out_dir, f"{name}.gif")
        self.frames = 0
        self.paused = False
        self.view = None
        # crf 30: a point cloud is all fine detail; at the default a minute came
        # out 17 MB, at 30 it is under 5 MB and every point still reads.
        self.writer = imageio.get_writer(self.mp4, fps=self.fps, codec="libx264", quality=None,
                                         macro_block_size=8, ffmpeg_log_level="error",
                                         output_params=["-crf", "30", "-preset", "slow",
                                                        "-movflags", "+faststart"])

    def grab(self):
        if self.paused or self.u.cell.steps % self.every:
            return
        left = np.asarray(self.u.render())[:, :, :3]     # the gym wrapper would want reset() first
        if not left.any():
            return                                     # the render product is still warming up
        if self.view is None:
            self.view = viewport_camera(self.u.cfg.viewer.cam_prim_path, self.w, self.h)
            eye, _, _, vfov = self.view
            self.log(f"viewport camera at {np.round(eye, 3).tolist()}, "
                     f"vertical fov {math.degrees(vfov):.1f} deg")
        eye, look_at, up, vfov = self.view
        self.viz.update()
        right = label(self.viz.viewer.render(eye, look_at, vfov, self.w, self.h, up=up),
                      self.right_title)
        self.writer.append_data(np.concatenate([label(left, "Isaac Lab"), right], axis=1))
        self.frames += 1
        if self.frames % (self.fps * 5) == 0:
            self.log(f"{self.frames / self.fps:.0f} s of video")

    def start(self):
        self.u.cell.on_tick.append(self.grab)

    def stop(self):
        if self.grab in self.u.cell.on_tick:
            self.u.cell.on_tick.remove(self.grab)
        self.writer.close()

    def finish(self):
        """Write the 2x-speed GIF next to the MP4; returns their sizes in MB."""
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", self.mp4,
                        "-vf", f"setpts=0.5*PTS,fps=10,scale={self.w}:-1:flags=lanczos,split[a][b];"
                               "[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer",
                        self.gif], check=True)
        mb = os.path.getsize(self.mp4) / 1e6, os.path.getsize(self.gif) / 1e6
        self.log(f"{self.frames} frames ({self.frames / self.fps:.0f} s): {self.mp4} ({mb[0]:.1f} MB), "
                 f"{self.gif} ({mb[1]:.1f} MB)")
        return mb
