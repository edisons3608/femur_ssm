import glob
import os

import numpy as np
import open3d as o3d


def align_all_meshes(mesh_dir="right_femur_downloads", output_dir="aligned_meshes", max_meshes=None):
    """Align all femur meshes to the first mesh using ICP."""
    os.makedirs(output_dir, exist_ok=True)

    stl_files = sorted(glob.glob(os.path.join(mesh_dir, "*.stl")))
    if max_meshes:
        stl_files = stl_files[:max_meshes]
    print(f"Processing {len(stl_files)} meshes.")

    # Use first mesh as reference
    ref_mesh = o3d.io.read_triangle_mesh(stl_files[0])
    ref_mesh.compute_vertex_normals()
    ref_pcd = ref_mesh.sample_points_poisson_disk(5000)
    ref_pcd.estimate_normals()
    o3d.io.write_triangle_mesh(os.path.join(output_dir, os.path.basename(stl_files[0])), ref_mesh)

    for i, filepath in enumerate(stl_files[1:], start=1):
        name = os.path.basename(filepath)
        print(f"[{i}/{len(stl_files)-1}] {name}", end=" ")

        mesh = o3d.io.read_triangle_mesh(filepath)
        mesh.compute_vertex_normals()

        # Sample points for ICP
        src_pcd = mesh.sample_points_poisson_disk(5000)
        src_pcd.estimate_normals()

        # Center both point clouds for better ICP convergence
        src_center = src_pcd.get_center()
        ref_center = ref_pcd.get_center()
        init_transform = np.eye(4)
        init_transform[:3, 3] = ref_center - src_center

        # Run ICP
        result = o3d.pipelines.registration.registration_icp(
            src_pcd, ref_pcd,
            max_correspondence_distance=5.0,
            init=init_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
        )

        # Apply and save
        mesh.transform(result.transformation)
        o3d.io.write_triangle_mesh(os.path.join(output_dir, name), mesh)
        print(f"fitness={result.fitness:.3f}")

    print(f"\nDone. Aligned meshes saved to {output_dir}/")

align_all_meshes(max_meshes=None)
