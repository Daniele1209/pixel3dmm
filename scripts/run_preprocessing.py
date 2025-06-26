import os
import tyro
import numpy as np
from PIL import Image
import glob

from pixel3dmm import env_paths


def main(video_or_images_path : str):

    if os.path.isdir(video_or_images_path):
        vid_name = video_or_images_path.split('/')[-1]
    else:
        vid_name = video_or_images_path.split('/')[-1][:-4]

    os.system(f'cd {env_paths.CODE_BASE}/scripts/ ; python run_cropping.py --video_or_images_path {video_or_images_path}')

    os.system(f'cd {env_paths.CODE_BASE}/src/pixel3dmm/preprocessing/MICA ; python demo.py -video_name {vid_name}')

    os.system(f'cd {env_paths.CODE_BASE}/scripts/ ; python run_facer_segmentation.py --video_name {vid_name}')

    # --- Alpha mask cropping step ---
    # Define paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    print(f"Corrected project root: {project_root}")
    alpha_mask_dir = os.path.join(project_root, 'postprocessed-pixel3dmm', vid_name, 'alpha_masks')
    alpha_cropped_dir = os.path.join(project_root, 'postprocessed-pixel3dmm', vid_name, 'alpha_cropped')
    print(f"Alpha mask dir: {alpha_mask_dir}")
    print(f"Alpha cropped dir: {alpha_cropped_dir}")
    crop_params_path = f'{env_paths.PREPROCESSED_DATA}/{vid_name}/crop_ymin_ymax_xmin_xmax.npy'

    print(f"Resolved alpha_mask_dir: {alpha_mask_dir}")
    print(f"Resolved alpha_cropped_dir: {alpha_cropped_dir}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in alpha_mask_dir: {os.listdir(alpha_mask_dir) if os.path.exists(alpha_mask_dir) else 'NOT FOUND'}")
    if os.path.exists(alpha_mask_dir) and len(glob.glob(os.path.join(alpha_mask_dir, '*_alpha.png'))):
        print(f"Alpha mask directory found: {alpha_mask_dir}")
        os.makedirs(alpha_cropped_dir, exist_ok=True)
        # Try to load crop params
        if os.path.exists(crop_params_path):
            print(f"Crop params found: {crop_params_path}")
            crop_params = np.load(crop_params_path)
            det_ymin, det_ymax, det_xmin, det_xmax = crop_params
            for mask_path in glob.glob(os.path.join(alpha_mask_dir, '*_alpha.png')):
                print(f"Processing alpha mask: {mask_path}")
                mask = Image.open(mask_path)
                mask_np = np.array(mask)
                # Crop
                cropped = mask_np[int(det_ymin):int(det_ymax), int(det_xmin):int(det_xmax)]
                # Resize
                cropped_img = Image.fromarray(cropped).resize((512, 512), resample=Image.Resampling.NEAREST)
                # Save
                out_name = os.path.basename(mask_path)
                cropped_img.save(os.path.join(alpha_cropped_dir, out_name))
                print(f"Saved cropped alpha mask: {os.path.join(alpha_cropped_dir, out_name)}")
        else:
            print(f"No crop params found at {crop_params_path}, skipping alpha mask cropping.")
    else:
        print(f"No alpha masks found in {alpha_mask_dir}, skipping alpha mask cropping.")
    # --- End alpha mask cropping ---


if __name__ == '__main__':
    tyro.cli(main)