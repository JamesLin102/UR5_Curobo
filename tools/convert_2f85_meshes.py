"""Re-export the 2F-85 COLLADA meshes as OBJ.

Isaac Sim's asset converter segfaults (libomniverse_asset_converter ->
tinyxml2) on these particular .dae files. trimesh parses them without trouble,
so convert once and have the URDF reference the .obj copies instead.

Run:
    python tools/convert_2f85_meshes.py
"""

import glob
import os

import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESHES = f"{ROOT}/assets/robot/kinova/kortex_description/grippers/robotiq_2f_85/meshes"

if __name__ == "__main__":
    for src in sorted(glob.glob(f"{MESHES}/*/*.dae")):
        dst = os.path.splitext(src)[0] + ".obj"
        trimesh.load(src, force="mesh").export(dst)
        print(f"  {os.path.relpath(src, MESHES)} -> {os.path.basename(dst)}")
