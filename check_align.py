import glob
import os
import random

import pyvista as pv

mesh_dir = "aligned_meshes"
n_meshes = 10  # number of meshes to overlay

stl_files = sorted(glob.glob(os.path.join(mesh_dir, "*.stl")))

#randomly sample n meshes
sample = random.sample(stl_files, min(n_meshes, len(stl_files)))

plotter = pv.Plotter()
colors = pv.plotting.colors.hexcolors
color_cycle = ["red", "blue", "green", "orange", "purple", "cyan", "yellow", "magenta", "lime", "pink"]

for i, filepath in enumerate(sample):
    mesh = pv.read(filepath)
    plotter.add_mesh(mesh, color=color_cycle[i % len(color_cycle)], opacity=0.3,
                     label=os.path.basename(filepath)[:6])

plotter.add_legend()
plotter.camera_position = "xy"
plotter.show()
