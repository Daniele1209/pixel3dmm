"""Script to visualize camera movement and extract camera parameters."""
import os
import tyro
import mediapy
import torch
import numpy as np
import pyvista as pv
import trimesh
import json
from PIL import Image
from pathlib import Path
from dreifus.matrix import Intrinsics, Pose, CameraCoordinateConvention, PoseType
from dreifus.pyvista import add_camera_frustum, render_from_camera
from pixel3dmm.utils.utils_3d import rotation_6d_to_matrix
from pixel3dmm.env_paths import PREPROCESSED_DATA, TRACKING_OUTPUT


def validate_transform(transform, msg=""):
    """Validate that a transform is a 4x4 matrix with sensible values"""
    if not isinstance(transform, np.ndarray):
        transform = np.array(transform)
    assert transform.shape == (4, 4), f"{msg}: Transform must be 4x4 matrix"
    assert np.allclose(transform[3], [0, 0, 0, 1]), f"{msg}: Last row must be [0, 0, 0, 1]"
    assert np.allclose(np.linalg.det(transform[:3, :3]), 1), f"{msg}: Rotation matrix must have determinant 1"
    return transform.tolist()


def validate_camera_params(params):
    """Validate camera parameters and add computed metadata"""
    # Validate focal length is positive
    assert params["intrinsics"]["focal_length"] > 0, "Focal length must be positive"
    
    # Add normalized focal length
    img_size = params["intrinsics"]["image_size"]
    params["intrinsics"]["normalized_focal_length"] = params["intrinsics"]["focal_length"] / (img_size / 2)
    
    # Set normalized principal point to [0, 0] (center) for Unity
    params["intrinsics"]["normalized_principal_point"] = [0.0, 0.0]
    
    # Validate transforms
    params["extrinsics"]["world_to_camera"] = validate_transform(
        extr_to_matrix(params["extrinsics"]), "World to camera transform")
    params["flame_transform"]["local_to_world"] = validate_transform(
        np.array(params["flame_transform"]["local_to_world"]), "FLAME local_to_world transform")
    
    # Print FOV for Unity import
    fov_y = compute_fov_y(params["intrinsics"]["focal_length"] * img_size, img_size)
    print(f"Unity vertical FOV (degrees): {fov_y:.2f}")
    
    return params


def extr_to_matrix(extr):
    """Convert rotation and translation to 4x4 matrix"""
    mat = np.eye(4)
    mat[:3, :3] = np.array(extr["rotation"])
    mat[:3, 3] = np.array(extr["translation"])
    return mat


def debug_camera_params(camera_data):
    """Print debug information about camera parameters"""
    frames = camera_data["frames"]
    print(f"Total frames: {len(frames)}")
    print(f"Head-centric mode: {camera_data['coordinate_system']['head_centric']}")
    
    # Check if camera positions are varying
    translations = []
    rotations = []
    
    for frame_name, frame_data in frames.items():
        trans = np.array(frame_data["extrinsics"]["translation"])
        rot = np.array(frame_data["extrinsics"]["rotation"])
        translations.append(trans)
        rotations.append(rot)
    
    translations = np.array(translations)
    rotations = np.array(rotations)
    
    print(f"Translation range: {translations.min(axis=0)} to {translations.max(axis=0)}")
    print(f"Translation std: {translations.std(axis=0)}")
    
    # Check rotation variation (using rotation matrix differences)
    rot_diffs = []
    for i in range(1, len(rotations)):
        diff = np.linalg.norm(rotations[i] - rotations[i-1])
        rot_diffs.append(diff)
    
    print(f"Average rotation difference between frames: {np.mean(rot_diffs):.6f}")
    print(f"Max rotation difference: {np.max(rot_diffs):.6f}")
    
    return translations, rotations


def extract_camera_params(ckpt, head_centric=True):
    """Extract camera parameters from checkpoint into a structured dictionary"""
    # Get base camera parameters
    base_rotation = ckpt['camera']['R_base_0'][0]
    base_translation = ckpt['camera']['t_base_0'][0]
    
    # Get FLAME parameters
    head_rot = rotation_6d_to_matrix(torch.from_numpy(ckpt['flame']['R'])).numpy()[0]
    head_translation = np.squeeze(ckpt['flame']['t'])
    neck_transform = ckpt['joint_transforms'][0, 1, :, :]

    # Compose full local-to-world transform for FLAME (head_rot + translation + neck)
    flame_local_to_world = np.eye(4)
    flame_local_to_world[:3, :3] = head_rot
    flame_local_to_world[:3, 3] = head_translation
    flame_local_to_world = flame_local_to_world @ neck_transform

    if head_centric:
        # Apply the same transformations as in the visualization
        # Build the complete world-to-camera transform
        flame2world = np.eye(4)
        flame2world[:3, :3] = head_rot
        flame2world[:3, 3] = head_translation
        
        # Base camera transform
        extr_world_to_cam = np.eye(4)
        extr_world_to_cam[:3, :3] = base_rotation
        extr_world_to_cam[:3, 3] = base_translation
        
        # Apply FLAME and neck transforms
        extr_world_to_cam = extr_world_to_cam @ flame2world @ neck_transform
        
        # Extract rotation and translation from the final transform
        final_rotation = extr_world_to_cam[:3, :3]
        final_translation = extr_world_to_cam[:3, 3]
    else:
        # Use base camera parameters directly
        final_rotation = base_rotation
        final_translation = base_translation

    # Set principal point to image center for Unity compatibility
    image_size = 256  # Hardcoded in original script
    principal_point = [image_size / 2, image_size / 2]

    params = {
        "extrinsics": {
            "rotation": final_rotation.tolist(),
            "translation": final_translation.tolist(),
        },
        "intrinsics": {
            "focal_length": float(ckpt['camera']['fl'][0, 0]),
            "principal_point": principal_point,
            "image_size": image_size
        },
        "flame_transform": {
            "rotation": head_rot.tolist(),
            "translation": head_translation.tolist(),
            "neck_transform": neck_transform.tolist(),
            "local_to_world": flame_local_to_world.tolist()  # Full local-to-world
        },
        "metadata": {
            "head_centric": head_centric,
            "transform_applied": head_centric,
            "description": "Camera extrinsics include FLAME head pose and neck joint transformations when head_centric=True"
        }
    }
    return validate_camera_params(params)

# Utility to compute vertical FOV from focal length and image size (for Unity)
def compute_fov_y(focal_length, image_size):
    # focal_length in pixels, image_size in pixels
    # FOV_y = 2 * arctan((image_size/2) / focal_length)
    import math
    fov_y_rad = 2 * math.atan((image_size / 2) / focal_length)
    fov_y_deg = math.degrees(fov_y_rad)
    return fov_y_deg

def main(vid_name : str,
         HEAD_CENTRIC : bool = True,
         DO_PROJECTION_TEST : bool = True,
         OUTPUT_JSON: bool = True,
         SAVE_BOTH_VERSIONS: bool = False
         ):
    tracking_dir = f'{TRACKING_OUTPUT}/{vid_name}_nV1_noPho_uv2000.0_n1000.0'
    
    # Create camera parameters dictionary
    camera_data = {
        "coordinate_system": {
            "handedness": "right-handed",
            "convention": "OpenGL",
            "axis_orientation": {
                "x": "right",
                "y": "up",
                "z": "backward (out of screen)"
            },
            "transforms": {
                "pose_type": "world_to_camera",
                "rotation_format": "matrix",
                "units": "metric"
            },
            "head_centric": HEAD_CENTRIC,        
            },
        "frames": {}
    }

    meshes = [f for f in os.listdir(f'{tracking_dir}/mesh/') if f.endswith('.ply') and not 'canonical' in f]
    meshes.sort()

    ckpts = [f for f in os.listdir(f'{tracking_dir}/checkpoint/') if f.endswith('.frame')]
    ckpts.sort()

    N_STEPS = len(meshes)

    pl = pv.Plotter()
    vid_frames = []
    
    for i in range(N_STEPS):
        ckpt = torch.load(f'{tracking_dir}/checkpoint/{ckpts[i]}', weights_only=False)
        
        # Extract camera parameters for this frame
        camera_data["frames"][f"frame_{i:05d}"] = extract_camera_params(ckpt, HEAD_CENTRIC)
        
        mesh = trimesh.load(f'{tracking_dir}/mesh/{meshes[i]}', process=False)
        head_rot = rotation_6d_to_matrix(torch.from_numpy(ckpt['flame']['R'])).numpy()[0]

        if not HEAD_CENTRIC:
            # move mesh from FLAME Space into World Space
            mesh.vertices = mesh.vertices @ head_rot.T + (ckpt['flame']['t'])
        else:
            # undo neck rotation
            verts_hom = np.concatenate([mesh.vertices, np.ones_like(mesh.vertices[..., :1])], axis=-1)
            verts_hom = verts_hom @ np.linalg.inv(ckpt['joint_transforms'][0, 1, :, :]).T
            mesh.vertices = verts_hom[..., :3]



        extr_open_gl_world_to_cam = np.eye(4)
        extr_open_gl_world_to_cam[:3, :3] = ckpt['camera']['R_base_0'][0]
        extr_open_gl_world_to_cam[:3, 3] = ckpt['camera']['t_base_0'][0]
        if HEAD_CENTRIC:
            flame2world = np.eye(4)
            flame2world[:3, :3] = head_rot
            flame2world[:3, 3] = np.squeeze(ckpt['flame']['t'])
            #TODO include neck transform as well
            extr_open_gl_world_to_cam = extr_open_gl_world_to_cam @ flame2world @ ckpt['joint_transforms'][0, 1, :, :]




        extr_open_gl_world_to_cam = Pose(extr_open_gl_world_to_cam,
                                         camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL,
                                         pose_type=PoseType.WORLD_2_CAM)

        intr = np.eye(3)
        intr[0, 0] = ckpt['camera']['fl'][0, 0] * 256
        intr[1, 1] = ckpt['camera']['fl'][0, 0] * 256
        intr[:2, 2] = ckpt['camera']['pp'][0] * (256/2+0.5) + 256/2 + 0.5

        intr = Intrinsics(intr)



        pl.add_mesh(mesh, color=[(i/N_STEPS), 0, ((N_STEPS-i)/N_STEPS)])
        add_camera_frustum(pl, extr_open_gl_world_to_cam, intr, color=[(i/N_STEPS), 0, ((N_STEPS-i)/N_STEPS)])

        if DO_PROJECTION_TEST:
            pll = pv.Plotter(off_screen=True, window_size=(256, 256))
            pll.add_mesh(mesh)
            img = render_from_camera(pll, extr_open_gl_world_to_cam, intr)

            gt_img = np.array(Image.open(f'{PREPROCESSED_DATA}/{vid_name}/cropped/{i:05d}.jpg').resize((256, 256)))

            alpha = img[..., 3]

            overlay = (gt_img *0.5 + img[..., :3]*0.5).astype(np.uint8)
            vid_frames.append(overlay)




    # Save camera parameters to JSON if path is provided
    if OUTPUT_JSON:
        output_json_path = Path(f'{tracking_dir}/camera_params.json')
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(camera_data, f, indent=2)
        print(f"Saved camera parameters to {output_json_path}")
        
        # Debug camera parameters
        print("\n=== Camera Parameters Debug ===")
        debug_camera_params(camera_data)
        print("===============================\n")
        
        # Save both versions for comparison if requested
        if SAVE_BOTH_VERSIONS:
            # Create non-head-centric version for comparison
            camera_data_non_head = {
                "coordinate_system": {
                    "handedness": "right-handed",
                    "convention": "OpenGL",
                    "axis_orientation": {
                        "x": "right",
                        "y": "up",
                        "z": "backward (out of screen)"
                    },
                    "transforms": {
                        "pose_type": "world_to_camera",
                        "rotation_format": "matrix",
                        "units": "metric"
                    },
                    "head_centric": False,
                    "description": "Camera parameters in world coordinate system. Base camera parameters without FLAME transformations."
                },
                "frames": {}
            }
            
            for i in range(N_STEPS):
                ckpt = torch.load(f'{tracking_dir}/checkpoint/{ckpts[i]}', weights_only=False)
                camera_data_non_head["frames"][f"frame_{i:05d}"] = extract_camera_params(ckpt, head_centric=False)
            
            output_json_path_non_head = Path(f'{tracking_dir}/camera_params_non_head_centric.json')
            with open(output_json_path_non_head, 'w') as f:
                json.dump(camera_data_non_head, f, indent=2)
            print(f"Saved non-head-centric camera parameters to {output_json_path_non_head}")
            
            print("\n=== Non-Head-Centric Camera Parameters Debug ===")
            debug_camera_params(camera_data_non_head)
            print("================================================\n")

    # Show visualization
    pl.show()

    if DO_PROJECTION_TEST:
        mediapy.write_video(f'{tracking_dir}/projection_test.mp4', images=vid_frames)



if __name__ == '__main__':
    tyro.cli(main)
