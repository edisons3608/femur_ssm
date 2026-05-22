import numpy as np
import pyvista as pv
from gpmm import load_model, reconstruct


def visualize_variance(model_path="gpmm_model.npz"):
    """Color-map the mean shape by per-vertex positional variance."""
    model = load_model(model_path)

    all_alphas = model["all_alphas"]   # (N, n_components, 3)
    mean_alpha = model["mean_alpha"]   # (n_components, 3)

    # Reconstruct every shape
    shapes = np.array([reconstruct(model, a) for a in all_alphas])  # (N, V, 3)

    # Per-vertex variance: sum of coordinate variances → scalar per vertex
    per_vertex_var = np.var(shapes, axis=0).sum(axis=1)  # (V,)

    # Reconstruct the mean shape for display
    mean_shape = reconstruct(model, mean_alpha)  # (V, 3)

    # Build a PyVista point cloud colored by variance
    cloud = pv.PolyData(mean_shape)
    cloud["variance"] = per_vertex_var

    plotter = pv.Plotter()
    plotter.add_mesh(
        cloud,
        scalars="variance",
        cmap="jet",
        clim=[0, 70],
        point_size=5,
        render_points_as_spheres=True,
        scalar_bar_args={"title": "Positional Variance (mm²)"},
    )
    plotter.add_text("Per-Vertex Variance", font_size=14, position="upper_left")
    plotter.camera_position = "xy"
    plotter.show()

visualize_variance()