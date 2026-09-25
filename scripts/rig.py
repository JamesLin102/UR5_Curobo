"""The parts of the setup that are not the scene: robots, timing, transport.

Everything here is the same whatever scene is loaded. The scene itself lives in
its example's folder (scripts/<name>/scene.py) -- see scenes/base.py for the
contract and pick_place/scene.py for a worked one.

Deliberately free of cuRobo and Isaac Sim imports so BOTH processes can load
it: Isaac Sim 5.1 pins Warp 1.8.2 while cuRobo 0.8 needs Warp >= 1.13, so
anything both sides import has to stay neutral.
"""

SIM_DT = 1.0 / 60.0

# Selectable robots: everything about the hardware that code elsewhere needs,
# so that a second robot is a second entry here rather than edits scattered
# through the simulator, the planner server and the scenes.
#
#   config/urdf/assets  the cuRobo config, the URDF, and the directory its
#                       mesh paths are relative to
#   tool_frame          the frame plan_pose is given a goal for. Every OTHER
#                       frame in the config's tool_frames (camera_link, for
#                       instance) is still available for forward kinematics,
#                       but the planner's copy is trimmed to this one -- any
#                       frame left in tool_frames becomes a frame plan_pose
#                       demands a target for.
#   solver_iterations   PhysX (position, velocity) iterations for the articulation
#   drive_type          how PhysX applies the drive gains: "acceleration" or "force"
#   arm                 the planned joints and how the simulator drives them,
#                       plus which links' spheres are for the cameras only
#   gripper             the linkage, how it is driven, and the pad geometry
#                       a scene needs to aim a grasp
#
# The gripper's joints are NOT in the planner's cspace: the robot config locks
# them all, so cuRobo places the fingers (and their collision spheres) but
# does not plan them. The simulator drives them on its own channel.
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
        # PhysX solver iterations for the whole articulation, position /
        # velocity, which the simulator sets on the articulation root.
        "solver_iterations": (64, 16),
        # How PhysX reads every drive gain below. "acceleration" scales them
        # by the joint's effective inertia, "force" applies them as torque.
        # Isaac Sim's URDF importer makes acceleration drives, and every gain
        # here was measured on those. The same numbers as FORCE drives are a
        # different robot: on the 2F-85's 14-40 g links they are orders of
        # magnitude stiffer, overpower the 4-bar's pin, and one finger
        # stalls at 0.1 rad while the other shoves the block 24 mm across
        # (measured on the Isaac Lab side before this was a setting).
        "drive_type": "acceleration",
        "arm": {
            "joints": ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"),
            # The importer's own stiffness, kept; the damping it does NOT give.
            #
            # A position drive with no damping is an undamped spring, and it
            # rings around a moving target: during a 12 cm vertical move the
            # joints that were told to hold still buzzed instead, reversing
            # direction 37-47 times, and the two that were moving ran in
            # surges rather than at speed. Measured over that move, as
            # velocity sign flips across all six joints and the velocity
            # ripple on wrist_1:
            #
            #     damping    0 -> 138 flips, ripple 0.48, lag  8 mrad
            #               20 ->  19 flips, ripple 0.13, lag  9 mrad
            #               50 ->   0 flips, ripple 0.19, lag 22 mrad
            #              150 ->   0 flips, ripple 0.20, lag 66 mrad
            #
            # 20 is the knee: the joints that should not move stop moving
            # (|v| max 0.00 rad/s, the remaining flips are noise below 0.005),
            # and it costs almost nothing in tracking. 50 buys only sub-visible
            # noise for four times the lag.
            "drive_stiffness": 625.0,
            "drive_damping": 20.0,
            # Links whose collision spheres exist only so RobotSegmenter can
            # mask them out of the depth image. The base is bolted to the
            # table, so a planner that checks them calls every configuration
            # a collision; planner_server drops them from its own copy.
            "mask_only_links": ("base_link_inertia",),
        },
        "gripper": {
            # Each joint of the linkage, as a multiplier of one commanded
            # angle. The first is the leader.
            "joints": {
                "robotiq_85_left_knuckle_joint": 1.0,
                "robotiq_85_right_knuckle_joint": -1.0,
                "robotiq_85_left_inner_knuckle_joint": 1.0,
                "robotiq_85_right_inner_knuckle_joint": -1.0,
                "robotiq_85_left_finger_tip_joint": -1.0,
                "robotiq_85_right_finger_tip_joint": 1.0,
            },
            "open": 0.0,
            "closed": 0.8,
            # The joints' <limit velocity="..."> in the URDF. Used to budget
            # how many sim steps a full open or close needs; commanding it
            # faster just leaves the joint short of the target when the next
            # plan starts.
            "speed": 2.0,
            # Which loop closure the simulator adds, or None. "robotiq_2f85"
            # pins each inner knuckle to its finger tip -- see
            # lab.usd_edits.close_gripper_linkage for why, and where the pin goes.
            "linkage": "robotiq_2f85",
            # The joints the PIN is responsible for rather than the drive,
            # matched by substring. They are not commanded, and they must not
            # be HELD either: the importer gives every joint a position drive
            # at default_drive_strength, so left alone they are pinned to zero
            # at 1e6 stiffness and spend the whole grasp fighting the loop
            # closure. That fight is what makes the linkage visibly come apart.
            #
            # Measured in free air, closing onto nothing, as the spread across
            # the six joints expressed as a fraction of a full close:
            #
            #     pin only, follower drives left alone : 0.805..0.922, spread 0.118
            #     pin, follower drives zeroed          : 0.997..1.000, spread 0.003
            #
            # The second also reaches its commanded angle, which the first
            # never does.
            "pinned_follower": "inner_knuckle_joint",
            # Drive gains for the gripper joints alone.
            #
            # The importer gives EVERY joint default_drive_strength, which is
            # 1e6 because the arm needs it. A 2F-85 link weighs 14 g, and a 1e6
            # drive on it simply overpowers the 4-bar's loop closure: the links
            # stop agreeing with each other and the linkage visibly comes
            # apart. Measured as the spread across the six joints while
            # gripping a 45 mm block, which is the thing you can see:
            #
            #     1e6 -> 0.201     1e4 -> 0.191     1e2 -> 0.064
            #     1e5 -> 0.197     1e3 -> 0.099
            #
            # Monotonic, and nothing else moved it: solver iterations (64,
            # 255) changed it by 0.000, and driving two joints instead of four
            # by 0.005.
            "drive_stiffness": 1.0e2,
            "drive_damping": 1.0e1,
            # The links that carry the rubber pads, which get the high-friction
            # material. This model has no separate pad link, so it is the tip.
            "pad_links": ("robotiq_85_left_finger_tip_link",
                          "robotiq_85_right_finger_tip_link"),
            # grasp_frame sits at the middle of the finger pads WITH THE
            # GRIPPER OPEN -- 185.3 mm from tool0, measured off
            # left_finger_tip.stl. The 2F-85's fingers swing rather than
            # translate, so closing carries the pads 13.5 mm further out: the
            # pad face goes from tool0 166.3..204.3 to 179.8..217.8. A scene
            # aiming a grasp needs both numbers.
            "pad_half": 0.019,          # half the pad face, 166.3..204.3 mm
            "pad_close_drop": 0.0135,   # how much further out the pads sit once closed
        },
    },
}
DEFAULT_ROBOT = "ur5_robotiq"

# The planner service. Both processes must agree; a stale server holding this
# port is the classic confusing failure: the new server exits with "Address
# already in use" while the client happily connects to the OLD one.
HOST = "127.0.0.1"
PORT = 5599

# The cuRobo build every number in this project was measured on. cuRobo is an
# editable install of a checkout on main, 42 commits past v0.8.0, and the
# upstream bugs planner_server works around are bugs in THAT commit -- a pull would
# change the planner underneath them without a word. The checkout sits on a
# local branch `ur5-curobo-pin` with no upstream, so `git pull` refuses, and
# planner_server refuses to start on anything else (--allow-curobo-drift).
CUROBO_COMMIT = "8e734f3ced1df898990bcd92de40abce475907db"
CUROBO_VERSION = "0.8.0.post1.dev42"
