"""Scene shared by the planner server and the Isaac Sim client.

Deliberately free of cuRobo and Isaac Sim imports so BOTH processes can load it.
Defining the obstacles once here is what keeps the simulated world and the
planner's world from drifting apart.
"""

SIM_DT = 1.0 / 60.0

HOME = [0.0, -2.2, 1.9, -1.383, -1.57, 0.0]

# Selectable robots. Both are 6-DOF: the 2F-85's fingers are rigid collision
# geometry, not actuated joints, so the gripper adds reach and bulk but no DOF.
ROBOTS = {
    "ur5e": {
        "config": "configs/ur5e.yml",
        "urdf": "assets/robot/ur_description/ur5e.urdf",
        "tool_frame": "tool0",
    },
    "ur5e_2f85": {
        "config": "configs/ur5e_robotiq_2f_85.yml",
        "urdf": "assets/robot/ur_description/ur5e_robotiq_2f_85.urdf",
        "tool_frame": "grasp_frame",
    },
}
DEFAULT_ROBOT = "ur5e"

# name, dims (full extents, m), pose [x, y, z, qw, qx, qy, qz], rgb
# Deliberately sparse. An earlier version had a wall at z<=0.70 under a shelf
# at z=0.755, leaving a 5.5 cm corridor -- fine for a planner working from
# exact cuboids, but mapped surfaces are inflated by the ESDF resolution, so
# that corridor closed the moment the map got dense and every plan failed.
# Keep the cell open and let the draggable cube be the only real obstacle.
OBSTACLES = [
    ("table", [2.00, 2.00, 0.10], [0.00, 0.0, -0.06, 1, 0, 0, 0], (0.45, 0.47, 0.50)),
]

# Two targets on opposite sides of the wall, tool pointing down.
TARGETS = [
    [0.45, -0.32, 0.25, 0.0, 1.0, 0.0, 0.0],
    [0.45,  0.32, 0.25, 0.0, 1.0, 0.0, 0.0],
]

# A cube that moves through the cell while the sim runs (see CUBE_SWEEP). The
# planner is never told where it is: the wrist camera has to find it, so a
# change in the planned route when the cube arrives is the proof that the live
# map is driving collision avoidance. It is a visual prim, not a physics body,
# so it can also be dragged by hand in the viewport without fighting PhysX --
# which does mean the arm passes through it rather than hitting it.
DRAG_CUBE = (
    # A low, wide slab standing on the table, not a tall post. The shape was
    # measured, not guessed, because two constraints fight each other:
    #   * the camera only maps up to roughly its own altitude (~0.40 m here),
    #     so anything above that line is invisible -- raising the obstacle from
    #     0.50 m to 0.70 m added exactly zero voxels inside its bounding box;
    #   * with EXACT geometry, an obstacle here only diverts the route above
    #     ~0.45 m, and then only if it is narrow.
    # A 0.35 m slab is entirely inside what the camera can see while still
    # blocking laterally. x=0.30 sits in the middle of the camera's scan band.
    #
    # The size window is narrow, because the MAPPED obstacle is fatter than the
    # real one (ESDF cell size plus the planner's collision activation
    # distance). Measured live:
    #   0.10 x 0.26 x 0.30   no effect
    #   0.11 x 0.31 x 0.33   no effect
    #   0.12 x 0.35 x 0.35   blocked  <-- this one
    #   0.12 x 0.35 x 0.50   no effect (tall post: its top is invisible)
    # Fed exact geometry this size merely detours the route (121 waypoints vs
    # 81 clear); arriving through the camera it blocks outright. No size was
    # found that reliably detours rather than blocks.
    "drag_me", [0.12, 0.35, 0.35], [0.30, 0.0, 0.175, 1, 0, 0, 0], (0.62, 0.20, 0.18)
)

# The cube slides along y on its own, so the test does not depend on dragging it
# by hand. Both ends of the sweep sit inside the band the wrist camera actually
# scans during the cycle -- measured as x 0.30..0.45, y -0.45..+0.30 -- because
# anything outside that is in a blind spot and never reaches the map.
CUBE_SWEEP = {
    # Detour / wait / resume along one pass was measured with an EARLIER cube
    # (0.50 m tall) at x=0.38, not with the DRAG_CUBE above -- the sweep has not
    # been re-measured since the slab replaced it, so treat those thresholds as
    # historical.
    #
    # Note x=0.38 differs from DRAG_CUBE's parked x=0.30: --move-cube therefore
    # carries the cube out of the nominal bounding box that planner_server uses
    # for its INSIDE-CUBE voxel count, so that number only means what it says
    # while the cube is parked (the default, and what the A/B test uses).
    "x": 0.38,
    "y_from": -0.05,
    "y_to": 0.55,
    "period_s": 40.0,
}

# A short sweep at startup so the map has content before the first plan.
# Joint-space poses (rad), kept high and away from the table.
SCAN_POSES = [
    [ 0.00, -2.20,  1.90, -1.383, -1.57, 0.0],
    [-0.55, -1.95,  1.60, -1.300, -1.57, 0.0],
    [ 0.00, -1.80,  1.45, -1.250, -1.57, 0.0],
    [ 0.55, -1.95,  1.60, -1.300, -1.57, 0.0],
    [ 0.00, -2.20,  1.90, -1.383, -1.57, 0.0],
]

# Wrist RealSense D435i, as rendered in Isaac Sim.
CAMERA = {
    "width": 640,
    "height": 480,
    "horizontal_fov_deg": 69.0,   # D435i depth FOV at 4:3
    "near": 0.15,
    "far": 3.0,
    "link": "camera_link",        # optical frame defined in the URDF
}

# Volumetric map covering the cell in front of the robot.
MAPPER = {
    "voxel_size": 0.015,
    "esdf_voxel_size": 0.025,
    "extent": (1.8, 1.8, 1.4),
    "grid_center": (0.35, 0.0, 0.35),
    "depth_min": 0.15,
    "depth_max": 2.5,
    # Blind decay, applied to EVERY voxel on EVERY integrated frame -- so this
    # is per-frame, not a time constant. 1.0 disables it, which is what we want:
    # global decay erodes voxels the camera CANNOT currently see, which just
    # eats geometry (measured: 9456 -> 6946 voxels over 550 frames at 0.99; 0.3
    # wipes the map within a few frames). Clearing belongs to frustum decay.
    "decay_factor": 1.0,
    # Decay for voxels the camera is looking THROUGH -- the honest "I can see
    # that spot and it is empty now" signal, and the only thing that clears an
    # object's old position after it moves.
    # 1.0 disables that too, so the map currently only ever grows. That is fine
    # for the parked-cube A/B test, but a dragged or swept cube leaves its old
    # position occupied forever. 0.97 was the last value that cleared without
    # over-eroding; 0.85 wiped surfaces faster than they could be re-observed.
    "frustum_decay_factor": 1.0,
    # How far from the robot's collision spheres a depth pixel must be to count
    # as scene rather than robot. A wrist camera looks straight at its own
    # gripper, so this needs to be generous.
    "self_mask_margin": 0.12,
    "esdf_every_n_frames": 10,      # recompute the distance field this often
    "minimum_tsdf_weight": 0.01,
}

HOST = "127.0.0.1"
PORT = 5599
