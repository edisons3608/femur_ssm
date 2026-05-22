import glob
import os

import numpy as np
import open3d as o3d
from scipy.linalg import eigh
from scipy.spatial import cKDTree


def build_kernel_matrix(points, sigma=20.0, length_scale=70.0):
    """Gaussian RBF kernel: K_ij = sigma^2 * exp(-||p_i - p_j||^2 / (2 * l^2))"""
    diff = points[:, None, :] - points[None, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    return sigma ** 2 * np.exp(-dists_sq / (2 * length_scale ** 2))


def find_correspondences(template_points, target_mesh):
    """For each template vertex, find the closest point on the target."""
    tree = cKDTree(np.asarray(target_mesh.vertices))
    _, indices = tree.query(template_points)
    return np.asarray(target_mesh.vertices)[indices]


def run_gpmm(mesh_dir="aligned_meshes", output_path="gpmm_model.npz",
             max_meshes=None, target_vertices=5000,
             sigma=20.0, length_scale=70.0, n_components=50):
    # Load aligned meshes
    stl_files = sorted(glob.glob(os.path.join(mesh_dir, "*.stl")))
    if max_meshes:
        stl_files = stl_files[:max_meshes]
    print(f"Loading {len(stl_files)} aligned meshes...")

    meshes = []
    for f in stl_files:
        m = o3d.io.read_triangle_mesh(f)
        m.compute_vertex_normals()
        meshes.append(m)

    # Sample template as a fixed-size point cloud from the first mesh
    template_pcd = meshes[0].sample_points_poisson_disk(target_vertices)
    template_points = np.asarray(template_pcd.points)
    V = len(template_points)
    print(f"Template: {V} points")

    # Build kernel and extract low-rank basis
    print(f"Building {V}x{V} kernel matrix (sigma={sigma}, l={length_scale})...")
    K = build_kernel_matrix(template_points, sigma, length_scale)

    n_components = min(n_components, V)
    print(f"Eigendecomposition (top {n_components} components)...")
    eigvals, eigvecs = eigh(K, subset_by_index=[V - n_components, V - 1])

    # Sort descending, drop near-zero
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    keep = eigvals > 1e-6
    eigvals, eigvecs = eigvals[keep], eigvecs[:, keep]
    print(f"Kept {len(eigvals)} basis functions.")

    # Basis matrix: Phi = U * sqrt(Lambda), shape (V, n_components)
    Phi = eigvecs * np.sqrt(eigvals)[None, :]

    # Precompute solve matrix: (Phi^T Phi + I)^{-1} Phi^T
    A = Phi.T @ Phi + np.eye(len(eigvals))
    A_inv_PhiT = np.linalg.solve(A, Phi.T)  # (n_components, V)

    # Fit each mesh: find coefficients alpha for each coordinate
    all_alphas = []
    for i, mesh in enumerate(meshes):
        print(f"[{i+1}/{len(meshes)}] Fitting {os.path.basename(stl_files[i])}")
        target_points = find_correspondences(template_points, mesh)
        deformation = target_points - template_points  # (V, 3)

        # alpha[:, c] = (Phi^T Phi + I)^{-1} Phi^T d_c
        alphas = A_inv_PhiT @ deformation  # (n_components, 3)
        all_alphas.append(alphas)

    all_alphas = np.array(all_alphas)  # (N, n_components, 3)
    mean_alpha = all_alphas.mean(axis=0)

    # Save everything needed to reconstruct shapes
    np.savez(output_path,
             template_points=template_points,
             eigvecs=eigvecs,
             eigvals=eigvals,
             all_alphas=all_alphas,
             mean_alpha=mean_alpha)

    print(f"\nGPMM saved to {output_path}")
    print(f"  {len(meshes)} shapes, {len(eigvals)} components, {V} vertices")
    return template_points, eigvecs, eigvals, all_alphas


def load_model(path="gpmm_model.npz"):
    # load the saved GPPM model.
    data = np.load(path)
    return {k: data[k] for k in data.files}


def reconstruct(model, alphas):
    """Reconstruct a shape from coefficients."""
    Phi = model["eigvecs"] * np.sqrt(model["eigvals"])[None, :]
    deformation = Phi @ alphas  # (V, 3)
    return model["template_points"] + deformation


def sample_random(model, n=1, scale=1.0):
    mean = model["mean_alpha"]
    centered = model["all_alphas"] - mean[None, :]
    flat = centered.reshape(len(centered), -1)
    cov = np.cov(flat.T)
    samples = np.random.multivariate_normal(flat.mean(axis=0), cov * scale ** 2, size=n)
    return [mean + s.reshape(mean.shape) for s in samples]


run_gpmm(max_meshes=None)
