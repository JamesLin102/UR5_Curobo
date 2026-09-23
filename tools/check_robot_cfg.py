"""Smoke-test a converted robot config under cuRobo 0.8.x: FK -> planner build -> plan_pose."""
import os, sys, time, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/scripts")
from rig import DEFAULT_ROBOT, ROBOTS  # noqa: E402
from planner_server import SEGMENTER_ONLY  # noqa: E402
from curobo.types import ContentPath, JointState, Pose, GoalToolPose
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.scene import Scene, Cuboid
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo._src.robot.loader.util import load_robot_yaml

name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROBOT
spec = ROBOTS[name]

content = ContentPath(
    robot_config_absolute_path=f"{ROOT}/{spec['config']}",
    robot_urdf_absolute_path=f"{ROOT}/{spec['urdf']}",
    robot_asset_absolute_path=f"{ROOT}/{spec['assets']}",
)

kin = Kinematics(KinematicsCfg.from_content_path(content))
print(f"[{name}] joints      : {kin.joint_names}")
q = torch.tensor([[0.0, -2.2, 1.9, -1.383, -1.57, 0.0]], device="cuda")
start = JointState.from_position(q, joint_names=kin.joint_names)
st = kin.compute_kinematics(start)
tool = st.tool_frames[0]
print(f"[{name}] FK {tool:11s}: {st.tool_poses[tool].position.cpu().numpy().round(4)}")
print(f"[{name}] spheres     : {tuple(st.robot_spheres.shape)}")

scene = Scene(cuboid=[Cuboid(name="table", dims=[1.5, 1.5, 0.1], pose=[0, 0, -0.06, 1, 0, 0, 0])])
# Trim tool_frames to the one goal frame. Every frame left in there is a frame
# plan_pose demands a target for, so a config that also exposes camera_link for
# forward kinematics -- which ur5_robotiq does -- fails with
# "Ordered link names [...] not a subset of [...]" unless it is trimmed.
# planner_server.build() does the same thing for the same reason.
import copy
planner_dict = copy.deepcopy(load_robot_yaml(content))
planner_dict["robot_cfg"]["kinematics"]["tool_frames"] = [tool]
# And drop the spheres that exist only for self-masking the depth image: the
# base is bolted to the table, so a planner that checks them calls every
# configuration a collision. planner_server.build() drops the same links.
pk = planner_dict["robot_cfg"]["kinematics"]
pk["collision_link_names"] = [n for n in pk["collision_link_names"] if n not in SEGMENTER_ONLY]
for n in SEGMENTER_ONLY:
    pk["collision_spheres"].pop(n, None)
planner = MotionPlanner(MotionPlannerCfg.create(
    robot=planner_dict, scene_model=scene, use_cuda_graph=True))
planner.warmup()

goal = GoalToolPose.from_poses({tool: Pose.from_list([0.4, 0.2, 0.4, 0, 1, 0, 0])})
t0 = time.time(); res = planner.plan_pose(goal, start); torch.cuda.synchronize()
print(f"[{name}] plan_pose   : success={bool(res.success[0])} "
      f"solve={res.solve_time*1e3:.1f}ms wall={1e3*(time.time()-t0):.1f}ms "
      f"traj={tuple(res.interpolated_trajectory.position.shape)}")
