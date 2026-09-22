"""The parts of the setup that are not the scene: robots, timing, transport.

Everything here is the same whatever scene is loaded. The scene itself lives in
scripts/scenes/ -- see scenes/base.py for the contract and scenes/demo_cube.py
for the shipped example.

Deliberately free of cuRobo and Isaac Sim imports so BOTH processes can load
it: Isaac Sim 5.1 pins Warp 1.8.2 while cuRobo 0.8 needs Warp >= 1.13, so
anything both sides import has to stay neutral.
"""

SIM_DT = 1.0 / 60.0

# Selectable robots. Both are 6-DOF: the 2F-85's fingers are rigid collision
# geometry, not actuated joints, so the gripper adds reach and bulk but no DOF.
#
# tool_frame is the frame plan_pose is given a goal for. Every OTHER frame in
# the config's tool_frames (camera_link, for instance) is still available for
# forward kinematics, but the planner's copy is trimmed to this one -- any
# frame left in tool_frames becomes a frame plan_pose demands a target for.
#
# gripper_joint is the one driven joint of the gripper linkage, or None. It is
# NOT in the planner's cspace: the robot config locks it, so cuRobo places the
# fingers (and their collision spheres) but does not plan them. The simulator
# drives it on its own channel, and the URDF's <mimic> tags carry the other
# five joints of the linkage along with it.
ROBOTS = {
    "ur5e": {
        "config": "configs/ur5e.yml",
        "urdf": "assets/robot/ur_description/ur5e.urdf",
        "tool_frame": "tool0",
        "gripper_joint": None,
    },
    "ur5e_2f85": {
        "config": "configs/ur5e_robotiq_2f_85.yml",
        "urdf": "assets/robot/ur_description/ur5e_robotiq_2f_85.urdf",
        "tool_frame": "grasp_frame",
        "gripper_joint": "left_outer_knuckle_joint",
        # 0 rad fully open, 0.8 fully closed. The config locks it at OPEN,
        # which is the widest the gripper gets, so a path planned there stays
        # clear at any opening.
        "gripper_open": 0.0,
        "gripper_closed": 0.8,
    },
}
DEFAULT_ROBOT = "ur5e"

# The planner service. Both processes must agree; a stale server holding this
# port is the classic confusing failure (see HANDOVER section 3).
HOST = "127.0.0.1"
PORT = 5599
