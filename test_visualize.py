import glob
import os

import pyvista as pv


def visualize_femur_grid(mesh_dir="right_femur_downloads", rows=3, cols=3):
    """Display a grid of femur STL meshes for visual comparison."""
    stl_files = sorted(glob.glob(os.path.join(mesh_dir, "*.stl")))

    n = rows * cols
    if len(stl_files) < n:
        print(f"Only {len(stl_files)} meshes found, adjusting grid.")
        n = len(stl_files)

    # Evenly sample meshes across the dataset
    indices = [int(i * len(stl_files) / n) for i in range(n)]
    selected = [stl_files[i] for i in indices]

    plotter = pv.Plotter(shape=(rows, cols), window_size=(1800, 1200))

    for idx, filepath in enumerate(selected):
        row, col = divmod(idx, cols)
        plotter.subplot(row, col)

        mesh = pv.read(filepath)
        label = os.path.basename(filepath).replace("_femurright.stl", "")
        plotter.add_mesh(mesh, color="ivory", show_edges=False, lighting=True)
        plotter.add_text(label, font_size=10, position="upper_left")
        plotter.camera_position = "xy"

    plotter.link_views()
    plotter.show()


if __name__ == "__main__":
    visualize_femur_grid()