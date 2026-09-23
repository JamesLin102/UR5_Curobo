"""Socket client for planner_server.py.

Imports neither Isaac Sim nor cuRobo -- only numpy and the framing in
proto.py -- so anything can use it: the Isaac Sim side (sim_env.py), and code
that only wants plans, such as a learning loop running beside the simulator.
"""

import socket
import time

import numpy as np

from proto import recv_msg, send_msg
from rig import HOST, PORT


class SceneMismatch(RuntimeError):
    """The server loaded a different scene from the one this side asked for."""


class MappingMismatch(RuntimeError):
    """One side maps and the other does not."""


class Planner:
    """Framed socket client for planner_server.py."""

    def __init__(self, scene, host=HOST, port=PORT, timeout_s=600, log=print,
                 mapping=None):
        """`mapping`: whether this side will send depth frames. When given,
        the handshake refuses a server that disagrees; None skips the check."""
        self.log = log
        deadline = time.time() + timeout_s
        while True:
            try:
                self.sock = socket.create_connection((host, port), timeout=300)
                break
            except OSError:
                if time.time() > deadline:
                    raise RuntimeError(
                        f"planner server not reachable on {host}:{port} - "
                        "start scripts/planner_server.py first"
                    )
                time.sleep(2.0)
        self.log(f"connected to planner on {host}:{port}")
        self._check_scene(scene, mapping)

    def _check_scene(self, scene, mapping=None):
        """Refuse to run against a planner that loaded a different scene, or
        that disagrees about mapping.

        The two processes never exchange geometry, so a scene mismatch would
        show up only as inexplicably wrong plans. A mapping mismatch is worse:
        a sim that maps against a server started with --no-mapping sends
        frames nobody fuses, so the arm drives through what its cameras see;
        a server that maps against a sim that does not keeps planning around
        a map that never updates. Checked on connect rather than on the first
        plan: this is before the stage is built, so the failure is immediate
        and costs nothing.
        """
        msg = {"op": "scene", "scene": scene}
        if mapping is not None:
            msg["mapping"] = bool(mapping)
        send_msg(self.sock, msg)
        header, _ = recv_msg(self.sock)
        theirs = header.get("scene")
        if theirs != scene:
            raise SceneMismatch(
                f"SCENE MISMATCH: the planner is running {theirs!r}, this "
                f"process has {scene!r}. Both sides build their world from the "
                f"scene, so they must match. Restart one of them.")
        # An older server does not say; then there is nothing to compare.
        if mapping is not None and "mapping" in header and header["mapping"] != bool(mapping):
            on, off = ("this process", "the planner") if mapping else ("the planner", "this process")
            raise MappingMismatch(
                f"MAPPING MISMATCH: {on} maps and {off} does not. Start both "
                f"with --no-mapping, or neither.")

    def plan(self, q, target):
        """Header dict; with "traj" (n x dof float32) added when "ok"."""
        send_msg(self.sock, {"op": "plan", "q": list(map(float, q)), "target": target})
        header, payload = recv_msg(self.sock)
        if header.get("ok"):
            header["traj"] = np.frombuffer(payload, dtype=np.float32).reshape(
                header["n"], header["dof"]
            )
        return header

    def ik(self, q, target):
        """Joint angles for one tool pose, or None. See planner_server.handle_ik."""
        send_msg(self.sock, {"op": "ik", "q": list(map(float, q)), "target": target})
        header, payload = recv_msg(self.sock)
        if not header.get("ok"):
            return None
        return np.frombuffer(payload, dtype=np.float32).tolist()

    def map_frame(self, q, depth, K, cam_name):
        send_msg(
            self.sock,
            {
                "op": "map",
                "cam": cam_name,
                "q": list(map(float, q)),
                "h": int(depth.shape[0]),
                "w": int(depth.shape[1]),
                "K": [[float(v) for v in row] for row in K],
            },
            np.ascontiguousarray(depth, dtype=np.float32).tobytes(),
        )
        # Fire-and-forget: the sim must not stall on a round-trip, so there is
        # no reply to return. The server's own frame count is only visible in
        # its log, or through stats().

    def reset_map(self):
        """Empty the server's map; the planner is left with the static scene.

        A no-op on a server started with --no-mapping.
        """
        send_msg(self.sock, {"op": "reset_map"})
        header, _ = recv_msg(self.sock)
        return bool(header.get("ok"))

    def stats(self):
        """What the map holds right now: frames, voxels, per watched volume.

        As of the last ESDF refresh, which is every esdf_every_n_frames fused
        frames -- not necessarily the frame just sent.
        """
        send_msg(self.sock, {"op": "stats"})
        header, _ = recv_msg(self.sock)
        return header
