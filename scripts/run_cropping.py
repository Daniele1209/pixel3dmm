import traceback
import os
import sys
import importlib

import mediapy
from PIL import Image
import tyro

import torchvision.transforms as transforms


from pixel3dmm import env_paths
sys.path.append(f'{env_paths.CODE_BASE}/src/pixel3dmm/preprocessing/PIPNet/FaceBoxesV2/')
from pixel3dmm.preprocessing.pipnet_utils import demo_image
from pixel3dmm import env_paths




def run(exp_path, image_dir, start_frame = 0,
        vertical_crop : bool = False,
        static_crop : bool = False,
        max_bbox : bool = False,
        disable_cropping : bool = False,
        ):
    # Break early if directory is empty
    if not os.path.exists(image_dir) or not os.listdir(image_dir):
        print(f"No images found in {image_dir} or directory does not exist")
        return

    experiment_name = exp_path.split('/')[-1][:-3]
    data_name = exp_path.split('/')[-2]
    config_path = '.experiments.{}.{}'.format(data_name, experiment_name)

    my_config = importlib.import_module(config_path, package='pixel3dmm.preprocessing.PIPNet')
    Config = getattr(my_config, 'Config')
    cfg = Config()
    cfg.experiment_name = experiment_name
    cfg.data_name = data_name

    save_dir = os.path.join(f'{env_paths.CODE_BASE}/src/pixel3dmm/preprocessing/PIPNet/snapshots', cfg.data_name, cfg.experiment_name)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose(
        [transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])

    # Get first image file in directory to use as pid
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    if not image_files:
        print("No image files found in directory")
        return
        
    pid = image_files[0]  # Use first image as reference
    print(f"Using {pid} as reference image")

    result = demo_image(image_dir, pid, save_dir, preprocess, cfg, cfg.input_size, cfg.net_stride, cfg.num_nb,
                           cfg.use_gpu,
                            start_frame=start_frame, vertical_crop=vertical_crop, static_crop=static_crop, max_bbox=max_bbox,
               disable_cropping=disable_cropping)
    
    if result is None:
        print("Face detection failed for all images in the directory")
        return


def unpack_images(base_path, video_or_images_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)
    if os.path.isdir(video_or_images_path):
        files = os.listdir(f'{video_or_images_path}')
        files.sort()
        if len(os.listdir(base_path)) == len(files):
            print(f'''
                        <<<<<<<< ALREADY COMPLETED IMAGE CROPPING for {video_or_images_path}, SKIPPING! >>>>>>>>
                        ''')
            return
        for i, file in enumerate(files):
            I = Image.open(f'{video_or_images_path}/{file}')
            I.save(f'{base_path}/{i:05d}.jpg', quality=95)
    elif video_or_images_path.endswith('.jpg') or video_or_images_path.endswith('.jpeg') or video_or_images_path.endswith('.png'):
        Image.open(video_or_images_path).save(f'{base_path}/{0:05d}.jpg', quality=95)
    else:
        frames = mediapy.read_video(f'{video_or_images_path}')
        if len(frames) == len(os.listdir(base_path)):
            return
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(f'{base_path}/{i:05d}.jpg', quality=95)

def main(video_or_images_path : str,
         max_bbox : bool = True, # not used
         disable_cropping : bool = False):
    if os.path.isdir(video_or_images_path):
        video_name = video_or_images_path.split('/')[-1]
    else:
        video_name = video_or_images_path.split('/')[-1][:-4]

    base_path = f'{env_paths.PREPROCESSED_DATA}/{video_name}/rgb/'

    unpack_images(base_path, video_or_images_path)

    if os.path.exists(f'{env_paths.PREPROCESSED_DATA}/{video_name}/cropped/'):
        if len(os.listdir(base_path)) == len(os.listdir(f'{env_paths.PREPROCESSED_DATA}/{video_name}/cropped/')):
            return


    start_frame = -1
    run('experiments/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10.py', base_path, start_frame=start_frame, vertical_crop=False,
        static_crop=True, max_bbox=max_bbox, disable_cropping=disable_cropping)
    # run('experiments/WFLW/pip_32_16_60_r101_l2_l1_10_1_nb10.py', base_path, start_frame=start_frame, vertical_crop=False, static_crop=True)


if __name__ == '__main__':
    tyro.cli(main)

