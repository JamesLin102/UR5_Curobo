import torch, time
from curobo.perception import Mapper, MapperCfg
from curobo.types import CameraObservation, Pose

H, W = 480, 640
cfg = MapperCfg(extent_meters_xyz=(2.0, 2.0, 2.0), voxel_size=0.01,
                esdf_voxel_size=0.05, image_height=H, image_width=W, num_cameras=1)
m = Mapper(cfg)
print("OK  Mapper built, no extra deps. mem:", round(m.memory_usage_mb(), 1), "MB")

# synthetic depth: a flat wall at 1.2 m with a nearer box patch at 0.8 m
depth = torch.full((1, H, W), 1.2, device="cuda")
depth[:, 180:300, 250:400] = 0.8
K = torch.tensor([[[600., 0, W/2], [0, 600., H/2], [0, 0, 1.]]], device="cuda")
rgb = torch.full((1, H, W, 3), 128, dtype=torch.uint8, device="cuda")
obs = CameraObservation(depth_image=depth, rgb_image=rgb, intrinsics=K,
                        pose=Pose.from_list([0., 0., 0.6, 0.5, -0.5, 0.5, -0.5]))
m.integrate(obs); m.compute_esdf(); torch.cuda.synchronize()   # warp JIT warmup
torch.cuda.synchronize(); t0 = time.time()
for _ in range(20):
    m.integrate(obs)
torch.cuda.synchronize()
print(f"OK  integrate (post-warmup): {1e3*(time.time()-t0)/20:.2f} ms/frame @ 640x480")

torch.cuda.synchronize(); t0 = time.time()
vg = m.compute_esdf()
torch.cuda.synchronize()
print(f"OK  compute_esdf (post-warmup): {1e3*(time.time()-t0):.2f} ms -> {type(vg).__name__} dims={vg.dims} voxel={vg.voxel_size}")

from curobo.scene import Scene
scene = Scene(voxel=[vg])
print("OK  ESDF VoxelGrid accepted straight into Scene() for planning")
