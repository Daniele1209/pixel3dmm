"""Post-processing script for mesh refinement using AI normal estimation."""
import os
import tyro
import torch
import numpy as np
import trimesh
from pathlib import Path
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt
from typing import List, Tuple, Optional
import logging
import matplotlib.pyplot as plt

from pixel3dmm.env_paths import TRACKING_OUTPUT, PREPROCESSED_DATA
from pixel3dmm.utils.utils_3d import rotation_6d_to_matrix


def load_normals(normals_dir: str, frame_indices: List[int]) -> List[np.ndarray]:
    """Load normal maps (PNG) for given frame indices."""
    normals = []
    for frame_idx in frame_indices:
        normal_path = os.path.join(normals_dir, f"{frame_idx:05d}.png")
        if os.path.exists(normal_path):
            normal_img = np.array(Image.open(normal_path).convert('RGB')).astype(np.float32) / 255.0
            # Convert RGB to [-1, 1] normal vectors
            normal_map = (normal_img * 2.0) - 1.0
            normals.append(normal_map)
        else:
            print(f"Warning: Normal map not found at {normal_path}")
            normals.append(None)
    return normals


class MeshRefiner:
    """Mesh refinement using normal estimation and alpha masks."""
    
    def __init__(self, 
                 max_iterations: int = 10,
                 step_size: float = 0.01,
                 alpha_threshold: float = 0.5,
                 normal_similarity_threshold: float = 0.8,
                 edge_weight_factor: float = 2.0,
                 debug_visualize: bool = False):
        """
        Initialize mesh refiner.
        
        Args:
            max_iterations: Maximum refinement iterations
            step_size: Vertex movement step size
            alpha_threshold: Alpha mask threshold for inside/outside test
            normal_similarity_threshold: Threshold for normal similarity
            edge_weight_factor: Weight factor for edge vertices
            debug_visualize: Whether to visualize debug information
        """
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.alpha_threshold = alpha_threshold
        self.normal_similarity_threshold = normal_similarity_threshold
        self.edge_weight_factor = edge_weight_factor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.debug_visualize = debug_visualize
    
    def compute_signed_distance_field(self, alpha_mask: np.ndarray) -> np.ndarray:
        """
        Compute signed distance field from alpha mask.
        
        Args:
            alpha_mask: Binary alpha mask (H, W)
            
        Returns:
            sdf: Signed distance field (H, W)
        """
        # Distance to boundary
        dist_inside = distance_transform_edt(alpha_mask)
        dist_outside = distance_transform_edt(~alpha_mask)
        
        # Signed distance field (positive inside, negative outside)
        sdf = dist_inside - dist_outside
        
        return sdf
    
    def find_fresnel_vertices(self, mesh: trimesh.Trimesh, 
                            camera_pos: np.ndarray,
                            alpha_mask: np.ndarray,
                            camera_intrinsics: np.ndarray,
                            normal_map: np.ndarray = None,
                            world_to_camera: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find vertices on the fresnel (silhouette) of the mesh from camera view.
        Uses the normal map: a vertex is on the silhouette if the dot product between its normal (from the normal map at its projected location) and the camera viewing direction is close to zero.
        """
        vertices_2d = self._project_vertices(mesh.vertices, camera_pos, camera_intrinsics, alpha_mask.shape, world_to_camera)
        h, w = alpha_mask.shape
        fresnel_indices = []
        dot_products = []
        # Pick 10 random indices for debug printing
        debug_indices = set(np.random.choice(len(vertices_2d), min(10, len(vertices_2d)), replace=False))
        for i, (x, y) in enumerate(vertices_2d):
            if 0 <= x < w and 0 <= y < h:
                if normal_map is not None:
                    # Map (x, y) from alpha mask/image space to normal map space
                    scale_x = normal_map.shape[1] / w
                    scale_y = normal_map.shape[0] / h
                    x_norm = int(x * scale_x)
                    y_norm = int(y * scale_y)
                    # Flip y-axis if needed (OpenGL vs image convention)
                    y_norm_flipped = normal_map.shape[0] - 1 - y_norm
                    # Clamp to valid range
                    x_norm = np.clip(x_norm, 0, normal_map.shape[1] - 1)
                    y_norm_flipped = np.clip(y_norm_flipped, 0, normal_map.shape[0] - 1)
                    normal = normal_map[y_norm_flipped, x_norm]
                    mesh_normal = mesh.vertex_normals[i]
                    view_dir = camera_pos - mesh.vertices[i]
                    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)
                    # --- DEBUG: Normal map sampling ---
                    if i < 5:
                        print(f"DEBUG: Projected (x, y): ({x}, {y}) -> Normal map (x_norm, y_norm_flipped): ({x_norm}, {y_norm_flipped})")
                        print(f"DEBUG: Sampled normal: {normal}")
                    dot = np.dot(normal, view_dir)
                    dot_mesh = np.dot(mesh_normal, view_dir)
                    dot_products.append(dot)
                    # --- DEBUG: Dot products ---
                    if i < 5:
                        print(f"DEBUG: Normal map normal: {normal}")
                        print(f"DEBUG: Mesh normal: {mesh_normal}")
                        print(f"DEBUG: Dot(normal_map, view_dir): {dot}")
                        print(f"DEBUG: Dot(mesh_normal, view_dir): {dot_mesh}")
                    if np.abs(dot) < 0.5:
                        fresnel_indices.append(i)
                else:
                    if self._is_near_boundary(x, y, alpha_mask):
                        fresnel_indices.append(i)
        # --- DEBUG: Silhouette selection ---
        print(f"DEBUG: Number of silhouette/fresnel candidates: {len(fresnel_indices)}")
        if len(fresnel_indices) > 0:
            print(f"DEBUG: Indices of first 10 silhouette candidates: {fresnel_indices[:10]}")
        # After projecting all vertices, print the range of projected 2D coordinates
        xs, ys = zip(*vertices_2d)
        print(f"DEBUG: Projected 2D x range: {min(xs)} to {max(xs)}, y range: {min(ys)} to {max(ys)}")
        if self.debug_visualize:
            # Scatter plot of all projected vertices over the alpha mask
            print(f"DEBUG: Projected verts shape: {np.array(vertices_2d).shape}")
            print(f"DEBUG: First 5 projected verts: {vertices_2d[:5]}")
            plt.figure(figsize=(6,6))
            plt.imshow(alpha_mask, cmap='gray', origin='lower')
            plt.scatter(xs, ys, s=1, c='lime', label='All Projected Verts', alpha=0.5)
            fresnel_xs = [vertices_2d[idx][0] for idx in fresnel_indices]
            fresnel_ys = [vertices_2d[idx][1] for idx in fresnel_indices]
            plt.scatter(fresnel_xs, fresnel_ys, s=5, c='red', label='Fresnel Candidates', alpha=0.7)
            plt.title('Projected Vertices and Fresnel Candidates')
            plt.legend()
            plt.show()
        sdf = self.compute_signed_distance_field(alpha_mask)
        edge_weights = self._compute_edge_weights(vertices_2d, sdf)
        return np.array(fresnel_indices), edge_weights
    
    def _project_vertices(self, vertices: np.ndarray, 
                         camera_pos: np.ndarray, 
                         camera_intrinsics: np.ndarray,
                         image_size: Tuple[int, int],
                         world_to_camera: np.ndarray = None) -> np.ndarray:
        """Project 3D vertices to 2D image coordinates using proper camera projection."""
        vertices_2d = []
        for idx, vertex in enumerate(vertices):
            vertex_2d = self._project_vertex_to_image(vertex, camera_pos, camera_intrinsics, image_size, world_to_camera, debug_idx=idx)
            if vertex_2d is not None:
                vertices_2d.append(vertex_2d)
            else:
                vertices_2d.append((0, 0))  # Fallback
        return np.array(vertices_2d)
    
    def _find_boundary_vertices(self, vertices_2d: np.ndarray, 
                               alpha_mask: np.ndarray) -> np.ndarray:
        """Find vertices that project near the alpha mask boundary."""
        h, w = alpha_mask.shape
        boundary_vertices = []
        
        for i, (x, y) in enumerate(vertices_2d):
            if 0 <= x < w and 0 <= y < h:
                # Check if vertex is near boundary
                if self._is_near_boundary(x, y, alpha_mask):
                    boundary_vertices.append(i)
        
        return np.array(boundary_vertices)
    
    def _is_near_boundary(self, x: int, y: int, alpha_mask: np.ndarray, 
                         radius: int = 3) -> bool:
        """Check if pixel is near alpha mask boundary."""
        h, w = alpha_mask.shape
        y_min, y_max = max(0, y - radius), min(h, y + radius + 1)
        x_min, x_max = max(0, x - radius), min(w, x + radius + 1)
        
        patch = alpha_mask[y_min:y_max, x_min:x_max]
        center_val = alpha_mask[y, x]
        
        # Check if there are different values in the patch
        return np.any(patch != center_val)
    
    def _compute_edge_weights(self, vertices_2d: np.ndarray, 
                            sdf: np.ndarray) -> np.ndarray:
        """Compute weights for edge vertices based on SDF."""
        h, w = sdf.shape
        weights = np.ones(len(vertices_2d))
        
        for i, (x, y) in enumerate(vertices_2d):
            if 0 <= x < w and 0 <= y < h:
                # Weight based on distance to boundary (closer = higher weight)
                dist_to_boundary = abs(sdf[y, x])
                weights[i] = 1.0 + self.edge_weight_factor / (1.0 + dist_to_boundary)
        
        return weights
    
    def compute_vertex_normals(self, mesh: trimesh.Trimesh, 
                             camera_pos: np.ndarray) -> np.ndarray:
        """Compute vertex normals in camera view direction."""
        # Get mesh vertex normals
        vertex_normals = mesh.vertex_normals
        
        # Compute view direction from camera to vertices
        view_directions = mesh.vertices - camera_pos
        view_directions = view_directions / np.linalg.norm(view_directions, axis=1, keepdims=True)
        
        # Project normals to view direction
        projected_normals = np.sum(vertex_normals * view_directions, axis=1, keepdims=True) * view_directions
        
        return projected_normals
    
    def refine_mesh(self, mesh: trimesh.Trimesh,
                   estimated_normals: np.ndarray,
                   alpha_mask: np.ndarray,
                   camera_pos: np.ndarray,
                   camera_intrinsics: np.ndarray,
                   world_to_camera: np.ndarray) -> trimesh.Trimesh:
        """
        Refine mesh using precomputed normal maps and alpha masks.
        """
        print(f"Starting mesh refinement with {len(mesh.vertices)} vertices")
        h, w = alpha_mask.shape
        refined_mesh = mesh.copy()
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}/{self.max_iterations}")
            fresnel_vertices, edge_weights = self.find_fresnel_vertices(
                refined_mesh, camera_pos, alpha_mask, camera_intrinsics, estimated_normals, world_to_camera)
            print(f"  [DEBUG] Iteration {iteration+1}: {len(fresnel_vertices)} fresnel vertices found.")
            if len(fresnel_vertices) == 0:
                print("No fresnel vertices found, stopping refinement")
                if self.debug_visualize:
                    # Optionally show the mesh in 3D
                    refined_mesh.show()
                break
            vertex_normals = self.compute_vertex_normals(refined_mesh, camera_pos)
            moved_vertices = self._move_vertices(
                refined_mesh.vertices,
                fresnel_vertices,
                vertex_normals,
                estimated_normals,
                edge_weights,
                alpha_mask,
                camera_pos,
                camera_intrinsics
            )
            refined_mesh.vertices = moved_vertices
            refined_mesh = self._relax_mesh(refined_mesh)
            vertex_change = np.linalg.norm(moved_vertices - refined_mesh.vertices)
            print(f"Vertex change: {vertex_change:.6f}")
            # if self.debug_visualize:
            #     refined_mesh.show()
            if vertex_change < 1e-6:
                print("Converged, stopping refinement")
                break
        return refined_mesh
    
    def _move_vertices(self, vertices: np.ndarray,
                      fresnel_vertices: np.ndarray,
                      vertex_normals: np.ndarray,
                      estimated_normals: np.ndarray,
                      edge_weights: np.ndarray,
                      alpha_mask: np.ndarray,
                      camera_pos: np.ndarray,
                      camera_intrinsics: np.ndarray) -> np.ndarray:
        """Move vertices based on normal estimates while respecting alpha mask."""
        moved_vertices = vertices.copy()
        
        for vertex_idx in fresnel_vertices:
            # Get current vertex position and normal
            vertex_pos = vertices[vertex_idx]
            vertex_normal = vertex_normals[vertex_idx]
            
            # Project vertex to image space
            vertex_2d = self._project_vertex_to_image(
                vertex_pos, camera_pos, camera_intrinsics, alpha_mask.shape, None)
            
            if vertex_2d is None:
                continue
            
            x, y = vertex_2d
            
            # Get estimated normal at this pixel
            if 0 <= x < estimated_normals.shape[1] and 0 <= y < estimated_normals.shape[0]:
                pixel_normal = estimated_normals[y, x]
                
                # Compute movement direction
                normal_diff = pixel_normal - vertex_normal
                movement_direction = normal_diff / (np.linalg.norm(normal_diff) + 1e-8)
                
                # Apply edge weight
                weight = edge_weights[vertex_idx] if vertex_idx < len(edge_weights) else 1.0
                
                # Compute movement step
                movement_step = movement_direction * self.step_size * weight
                
                # Try to move vertex
                new_pos = vertex_pos + movement_step
                
                # Check if new position is inside alpha mask
                if self._is_inside_alpha_mask(new_pos, alpha_mask, camera_pos, camera_intrinsics):
                    moved_vertices[vertex_idx] = new_pos
        
        return moved_vertices
    
    def _project_vertex_to_image(self, vertex_pos: np.ndarray,
                                camera_pos: np.ndarray,
                                camera_intrinsics: np.ndarray,
                                image_shape: Tuple[int, int],
                                world_to_camera: np.ndarray = None,
                                debug_idx: int = None) -> Optional[Tuple[int, int]]:
        """Project 3D vertex to 2D image coordinates using proper camera projection (OpenGL convention)."""
        vertex_hom = np.append(vertex_pos, 1.0)
        if world_to_camera is not None:
            vertex_cam = world_to_camera @ vertex_hom
        else:
            camera_transform = np.eye(4)
            camera_transform[:3, 3] = -camera_pos
            vertex_cam = camera_transform @ vertex_hom
        # OpenGL convention: camera looks down -Z, so flip Z
        vertex_cam[2] = -vertex_cam[2]
        if debug_idx is not None and debug_idx < 10:
            print(f"  [DEBUG] Vertex {debug_idx}: world pos={vertex_pos}, cam pos={vertex_cam}, Z={vertex_cam[2]}")
        if vertex_cam[2] <= 0:
            return None
        x = vertex_cam[0] / vertex_cam[2]
        y = vertex_cam[1] / vertex_cam[2]
        if debug_idx is not None and debug_idx < 10:
            print(f"  [DEBUG] Vertex {debug_idx}: x={vertex_cam[0]}, y={vertex_cam[1]}, z={vertex_cam[2]}, x'={x}, y'={y}")
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]
        u = fx * x + cx
        v = fy * y + cy
        u = int(u)
        v = int(v)
        if debug_idx is not None and debug_idx < 10:
            print(f"  [DEBUG] Vertex {debug_idx}: u={u}, v={v}")
        return u, v
    
    def _is_inside_alpha_mask(self, vertex_pos: np.ndarray,
                             alpha_mask: np.ndarray,
                             camera_pos: np.ndarray,
                             camera_intrinsics: np.ndarray) -> bool:
        """Check if vertex projects inside alpha mask."""
        vertex_2d = self._project_vertex_to_image(vertex_pos, camera_pos, camera_intrinsics, alpha_mask.shape, None)
        
        if vertex_2d is None:
            return False
        
        x, y = vertex_2d
        h, w = alpha_mask.shape
        
        if 0 <= x < w and 0 <= y < h:
            return alpha_mask[y, x] > self.alpha_threshold
        
        return False
    
    def _relax_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Apply local mesh relaxation to smooth the surface."""
        # Simple Laplacian smoothing
        relaxed_mesh = mesh.copy()
        
        # Get vertex neighbors
        vertex_neighbors = mesh.vertex_neighbors
        
        # Apply Laplacian smoothing
        new_vertices = relaxed_mesh.vertices.copy()
        for i, neighbors in enumerate(vertex_neighbors):
            if len(neighbors) > 0:
                neighbor_positions = relaxed_mesh.vertices[neighbors]
                centroid = np.mean(neighbor_positions, axis=0)
                # Move vertex towards centroid
                new_vertices[i] = relaxed_mesh.vertices[i] * 0.5 + centroid * 0.5
        
        relaxed_mesh.vertices = new_vertices
        return relaxed_mesh


def extract_camera_params_from_ckpt(ckpt, head_centric=True):
    """
    Extract camera parameters from .frame checkpoint using conventions from tracker.py and viz_head_centric_cameras.py.
    Returns a dict with 'extrinsics', 'intrinsics', 'camera_pos', and 'world_to_camera'.
    - world_to_camera: 4x4 matrix (OpenGL, right-handed, camera looks down -Z)
    - camera_pos: camera position in world space (from inverse of world_to_camera)
    - intrinsics: dict with focal_length, principal_point, image_size, intrinsics matrix
    """
    base_rotation = ckpt['camera']['R_base_0'][0]
    base_translation = ckpt['camera']['t_base_0'][0]
    head_rot = rotation_6d_to_matrix(torch.from_numpy(ckpt['flame']['R'])).numpy()[0]
    head_translation = np.squeeze(ckpt['flame']['t'])
    neck_transform = ckpt['joint_transforms'][0, 1, :, :]

    # Compose world-to-camera transform (OpenGL convention)
    if head_centric:
        flame2world = np.eye(4)
        flame2world[:3, :3] = head_rot
        flame2world[:3, 3] = head_translation
        extr_world_to_cam = np.eye(4)
        extr_world_to_cam[:3, :3] = base_rotation
        extr_world_to_cam[:3, 3] = base_translation
        world_to_camera = extr_world_to_cam @ flame2world @ neck_transform
    else:
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = base_rotation
        world_to_camera[:3, 3] = base_translation

    # Camera position in world space (from inverse transform)
    camera_to_world = np.linalg.inv(world_to_camera)
    camera_pos = camera_to_world[:3, 3]

    # Intrinsics
    focal_length = float(ckpt['camera']['fl'][0, 0])
    image_size = 256  # or from config
    # Set principal point to image center for consistency (Unity/pyvista convention)
    principal_point = [image_size / 2, image_size / 2]
    camera_intrinsics = np.eye(3)
    camera_intrinsics[0, 0] = focal_length * image_size
    camera_intrinsics[1, 1] = focal_length * image_size
    camera_intrinsics[0, 2] = principal_point[0]
    camera_intrinsics[1, 2] = principal_point[1]

    return {
        'extrinsics': {
            'rotation': world_to_camera[:3, :3],
            'translation': world_to_camera[:3, 3],
            'world_to_camera': world_to_camera
        },
        'intrinsics': {
            'focal_length': focal_length,
            'principal_point': principal_point,
            'image_size': image_size,
            'intrinsics_matrix': camera_intrinsics
        },
        'camera_pos': camera_pos
    }


def load_alpha_masks(alpha_dir: str, frame_indices: List[int]) -> List[np.ndarray]:
    """Load alpha masks for given frame indices, resizing to 256x256 for alignment and flipping vertically to match projection convention."""
    alpha_masks = []
    for frame_idx in frame_indices:
        alpha_path = os.path.join(alpha_dir, f"{frame_idx:05d}_alpha.png")
        if os.path.exists(alpha_path):
            # Resize to 256x256 to match mesh/camera/normal map
            alpha_mask = np.array(Image.open(alpha_path).convert('L').resize((256, 256))) / 255.0
            alpha_mask = np.flipud(alpha_mask)  # Flip vertically to match projection convention
            alpha_masks.append(alpha_mask > 0.5)  # Binary mask
        else:
            print(f"Warning: Alpha mask not found at {alpha_path}")
            alpha_masks.append(None)
    return alpha_masks


def main(vid_name: str,
         alpha_dir: Optional[str] = None,
         output_dir: str = None,
         max_iterations: int = 10,
         step_size: float = 0.01,
         alpha_threshold: float = 0.5,
         normal_similarity_threshold: float = 0.8,
         edge_weight_factor: float = 2.0,
         start_frame: int = 0,
         end_frame: Optional[int] = None,
         normals_dir: Optional[str] = None,
         debug_visualize: bool = True):
    tracking_dir = f'{TRACKING_OUTPUT}/{vid_name}_nV1_noPho_uv2000.0_n1000.0'
    mesh_dir = os.path.join(tracking_dir, 'mesh')
    checkpoint_dir = os.path.join(tracking_dir, 'checkpoint')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if normals_dir is None:
        normals_dir = f'../preprocessed-pixel3dmm/{vid_name}/p3dmm/normals'
    if alpha_dir is None:
        alpha_dir = f'../postprocessed-pixel3dmm/{vid_name}/alpha_cropped'
    mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith('.ply') and 'canonical' not in f]
    mesh_files.sort()
    ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.frame')]
    ckpt_files.sort()
    if end_frame is None:
        end_frame = len(mesh_files)
    frame_indices = list(range(start_frame, min(end_frame, len(mesh_files))))
    alpha_masks = load_alpha_masks(alpha_dir, frame_indices)
    normals_list = load_normals(normals_dir, frame_indices)
    refiner = MeshRefiner(
        max_iterations=max_iterations,
        step_size=step_size,
        alpha_threshold=alpha_threshold,
        normal_similarity_threshold=normal_similarity_threshold,
        edge_weight_factor=edge_weight_factor,
        debug_visualize=debug_visualize
    )
    for i, frame_idx in enumerate(frame_indices):
        print(f"\nProcessing frame {frame_idx}")
        mesh_path = os.path.join(mesh_dir, mesh_files[frame_idx])
        mesh = trimesh.load(mesh_path, process=False)
        alpha_mask = alpha_masks[i]
        estimated_normals = normals_list[i]
        if alpha_mask is None:
            print(f"Skipping frame {frame_idx} - no alpha mask")
            continue
        if estimated_normals is None:
            print(f"Skipping frame {frame_idx} - no normal map")
            continue
        # --- DEBUG: Print mask and normal map info ---
        print(f"[DEBUG] Alpha mask shape: {alpha_mask.shape}, dtype: {alpha_mask.dtype}, min: {alpha_mask.min()}, max: {alpha_mask.max()}")
        print(f"[DEBUG] Normal map shape: {estimated_normals.shape}, dtype: {estimated_normals.dtype}, min: {estimated_normals.min()}, max: {estimated_normals.max()}")
        # --- DEBUG: Print bounding box of white region in alpha mask ---
        ys, xs = np.where(alpha_mask > 0.5)
        if len(xs) > 0 and len(ys) > 0:
            print(f"[DEBUG] Alpha mask white region: x={xs.min()} to {xs.max()}, y={ys.min()} to {ys.max()}")
        else:
            print("[DEBUG] Alpha mask has no white region!")
        ckpt_path = os.path.join(checkpoint_dir, ckpt_files[frame_idx])
        ckpt = torch.load(ckpt_path, weights_only=False)
        cam_params = extract_camera_params_from_ckpt(ckpt, head_centric=True)
        camera_intrinsics = cam_params['intrinsics']['intrinsics_matrix']
        camera_pos = cam_params['camera_pos']
        world_to_camera = cam_params['extrinsics']['world_to_camera']
        if i == 0:
            print("[DEBUG] Camera intrinsics for first frame:")
            print(camera_intrinsics)
            print("[DEBUG] World-to-camera matrix for first frame:")
            print(world_to_camera)
        # --- DEBUG: Project all mesh vertices and print their bounding box ---
        refiner_for_debug = MeshRefiner(debug_visualize=False)
        projected_verts = refiner_for_debug._project_vertices(mesh.vertices, camera_pos, camera_intrinsics, alpha_mask.shape, world_to_camera)
        xs_proj, ys_proj = projected_verts[:,0], projected_verts[:,1]
        print(f"[DEBUG] Projected verts x: {xs_proj.min()} to {xs_proj.max()}, y: {ys_proj.min()} to {ys_proj.max()}")
        # --- DEBUG: Optionally overlay cropped input image and alpha mask ---
        cropped_img_path = os.path.join(f'../preprocessed-pixel3dmm/{vid_name}/cropped', f'{frame_idx:05d}.jpg')
        if os.path.exists(cropped_img_path):
            from PIL import Image
            cropped_img = np.array(Image.open(cropped_img_path).resize((256, 256)))
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8,4))
            plt.subplot(1,2,1)
            plt.title('Cropped Input Image')
            plt.imshow(cropped_img)
            plt.subplot(1,2,2)
            plt.title('Alpha Mask')
            plt.imshow(alpha_mask, cmap='gray', origin='lower')
            plt.show()
        else:
            print(f"[DEBUG] Cropped input image not found at {cropped_img_path}")
        # --- Continue with refinement ---
        refined_mesh = refiner.refine_mesh(
            mesh, estimated_normals, alpha_mask, camera_pos, camera_intrinsics, world_to_camera)
        output_path = os.path.join(output_dir, f'refined_frame_{frame_idx:05d}.ply')
        refined_mesh.export(str(output_path))
        print(f"Saved refined mesh to {output_path}")


if __name__ == '__main__':
    tyro.cli(main) 