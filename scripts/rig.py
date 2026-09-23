"""The parts of the setup that are not the scene: robots, timing, transport.

Everything here is the same whatever scene is loaded. The scene itself lives in
scripts/scenes/ -- see scenes/base.py for the contract and scenes/pick_place.py
for the shipped scene.

Deliberately free of cuRobo and Isaac Sim imports so BOTH processes can load
it: Isaac Sim 5.1 pins Warp 1.8.2 while cuRobo 0.8 needs Warp >= 1.13, so
anything both sides import has to stay neutral.
"""

SIM_DT = 1.0 / 60.0

# Selectable robots. 6-DOF: the 2F-85's fingers are locked out of the cspace,
# so the gripper adds reach and bulk but no DOF.
#
# tool_frame is the frame plan_pose is given a goal for. Every OTHER frame in
# the config's tool_frames (camera_link, for instance) is still available for
# forward kinematics, but the planner's copy is trimmed to this one -- any
# frame left in tool_frames becomes a frame plan_pose demands a target for.
#
# gripper_joints maps each joint of the gripper linkage to its multiplier of
# one commanded angle. None of them are in the planner's cspace:
# the robot config locks them all, so cuRobo places the fingers (and their
# collision spheres) but does not plan them. The simulator drives them on its
# own channel.
ROBOTS = {
    # UR5 (CB3) with the real wrist stack: FT 300, Robotiq Wrist Camera, 2F-85
    # and a RealSense D435i on its bracket. The whole description is vendored
    # from eugene900805/mir_ur5_humble with the MiR chassis removed -- see
    # assets/robot/ur5_robotiq/PROVENANCE.md. It replaces a model that was
    # assembled here by hand from photographs, whose link positions were wrong.
    # Mesh paths in the URDF are relative to the URDF itself, which is also
    # the asset root cuRobo is given.
    "ur5_robotiq": {
        "config": "configs/ur5_robotiq.yml",
        "urdf": "assets/robot/ur5_robotiq/ur5_robotiq.urdf",
        "assets": "assets/robot/ur5_robotiq",
        "tool_frame": "grasp_frame",
        "gripper_joints": {
            "robotiq_85_left_knuckle_joint": 1.0,
            "robotiq_85_right_knuckle_joint": -1.0,
            "robotiq_85_left_inner_knuckle_joint": 1.0,
            "robotiq_85_right_inner_knuckle_joint": -1.0,
            "robotiq_85_left_finger_tip_joint": -1.0,
            "robotiq_85_right_finger_tip_joint": 1.0,
        },
        "gripper_open": 0.0,
        "gripper_closed": 0.8,
    },
}
DEFAULT_ROBOT = "ur5_robotiq"

# The planner service. Both processes must agree; a stale server holding this
# port is the classic confusing failure (see HANDOVER section 3).
HOST = "127.0.0.1"
PORT = 5599

# The cuRobo build every number in HANDOVER was measured on. cuRobo is an
# editable install of a checkout on main, 42 commits past v0.8.0, and the
# workarounds in HANDOVER section 4 are for bugs in THAT commit -- a pull would
# change the planner underneath them without a word. The checkout sits on a
# local branch `ur5-curobo-pin` with no upstream, so `git pull` refuses, and
# planner_server refuses to start on anything else (--allow-curobo-drift).
CUROBO_COMMIT = "8e734f3ced1df898990bcd92de40abce475907db"
CUROBO_VERSION = "0.8.0.post1.dev42"
