"""Smoke-test a converted robot config under cuRobo 0.8.x: FK -> planner build -> plan_pose."""
import os, sys, time, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from curobo.types import ContentPath, JointState, Pose, GoalToolPose
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.scene import Scene, Cuboid
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo._src.robot.loader.util import load_robot_yaml

name = sys.argv[1] if len(sys.argv) > 1 else "ur5e"
urdf = sys.argv[2] if len(sys.argv) > 2 else f"{name}.urdf"

content = ContentPath(
    robot_config_absolute_path=f"{ROOT}/configs/{name}.yml",
    robot_urdf_absolute_path=f"{ROOT}/assets/robot/ur_description/{urdf}",
    robot_asset_absolute_path=f"{ROOT}/assets/robot/ur_description",
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
# forward kinematics -- which ur5e_robotiq_2f_85 does -- fails with
# "Ordered link names [...] not a subset of [...]" unless it is trimmed.
# planner_server.build() does the same thing for the same reason.
import copy
planner_dict = copy.deepcopy(load_robot_yaml(content))
planner_dict["robot_cfg"]["kinematics"]["tool_frames"] = [tool]
planner = MotionPlanner(MotionPlannerCfg.create(
    robot=planner_dict, scene_model=scene, use_cuda_graph=True))
planner.warmup()

goal = GoalToolPose.from_poses({tool: Pose.from_list([0.4, 0.2, 0.4, 0, 1, 0, 0])})
t0 = time.time(); res = planner.plan_pose(goal, start); torch.cuda.synchronize()
print(f"[{name}] plan_pose   : success={bool(res.success[0])} "
      f"solve={res.solve_time*1e3:.1f}ms wall={1e3*(time.time()-t0):.1f}ms "
      f"traj={tuple(res.interpolated_trajectory.position.shape)}")
