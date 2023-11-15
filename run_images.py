import os
import torch
import argparse
from PIL import Image
from utils.zero123_utils import init_model, predict_stage1_gradio, zero123_infer
from utils.sam_utils import sam_init, sam_out_nosave
from utils.utils import pred_bbox, image_preprocess_nosave, gen_poses, convert_mesh_format
from elevation_estimate.estimate_wild_imgs import estimate_elev
import tyro
from dataclasses import dataclass
import tempfile
import cloudpathlib
import boto3
import shutil

@dataclass
class Args:
    output_dir : str
    workers : int = 8
    cuda_cnt : int = 8
    half_precision : bool = False
    mesh_resolution : int = 256
    bucket : str = 'gso-renders'
    output_format : str = ".obj"
    ext : str = '018.png'

args = tyro.cli(Args)


def preprocess(predictor, raw_im, lower_contrast=False):
    raw_im.thumbnail([512, 512], Image.Resampling.LANCZOS)
    image_sam = sam_out_nosave(predictor, raw_im.convert("RGB"), pred_bbox(raw_im))
    input_256 = image_preprocess_nosave(image_sam, lower_contrast=lower_contrast, rescale=True)
    torch.cuda.empty_cache()
    return input_256

def stage1_run(model, device, exp_dir,
               input_im, scale, ddim_steps):
    # folder to save the stage 1 images
    stage1_dir = os.path.join(exp_dir, "stage1_8")
    os.makedirs(stage1_dir, exist_ok=True)

    # stage 1: generate 4 views at the same elevation as the input
    output_ims = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(4)), device=device, ddim_steps=ddim_steps, scale=scale)
    
    # stage 2 for the first image
    # infer 4 nearby views for an image to estimate the polar angle of the input
    stage2_steps = 50 # ddim_steps
    zero123_infer(model, exp_dir, indices=[0], device=device, ddim_steps=stage2_steps, scale=scale)
    # estimate the camera pose (elevation) of the input image.
    try:
        polar_angle = estimate_elev(exp_dir)
    except:
        print("Failed to estimate polar angle")
        polar_angle = 90
    print("Estimated polar angle:", polar_angle)
    gen_poses(exp_dir, polar_angle)

    # stage 1: generate another 4 views at a different elevation
    if polar_angle <= 75:
        output_ims_2 = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(4,8)), device=device, ddim_steps=ddim_steps, scale=scale)
    else:
        output_ims_2 = predict_stage1_gradio(model, input_im, save_path=stage1_dir, adjust_set=list(range(8,12)), device=device, ddim_steps=ddim_steps, scale=scale)
    torch.cuda.empty_cache()
    return 90-polar_angle, output_ims+output_ims_2
    
def stage2_run(model, device, exp_dir,
               elev, scale, stage2_steps=50):
    # stage 2 for the remaining 7 images, generate 7*4=28 views
    if 90-elev <= 75:
        zero123_infer(model, exp_dir, indices=list(range(1,8)), device=device, ddim_steps=stage2_steps, scale=scale)
    else:
        zero123_infer(model, exp_dir, indices=list(range(1,4))+list(range(8,12)), device=device, ddim_steps=stage2_steps, scale=scale)

def reconstruct(exp_dir, output_format=".ply", device_idx=0, resolution=256):
    exp_dir = os.path.abspath(exp_dir)
    main_dir_path = os.path.abspath(os.path.dirname("./"))
    os.chdir('reconstruction/')

    bash_script = f'CUDA_VISIBLE_DEVICES={device_idx} python exp_runner_generic_blender_val.py \
                    --specific_dataset_name {exp_dir} \
                    --mode export_mesh \
                    --conf confs/one2345_lod0_val_demo.conf \
                    --resolution {resolution}'
    print(bash_script)
    os.system(bash_script)
    os.chdir(main_dir_path)

    ply_path = os.path.join(exp_dir, f"mesh.ply")
    if output_format == ".ply":
        return ply_path
    if output_format not in [".obj", ".glb"]:
        print("Invalid output format, must be one of .ply, .obj, .glb")
        return ply_path
    return convert_mesh_format(exp_dir, output_format=output_format)


def predict_multiview(shape_dir, gpu_idx , img_path, model_zero123):
    device = f"cuda:{gpu_idx}"

    # initialize the Segment Anything model
    predictor = sam_init(gpu_idx)
    input_raw = Image.open(img_path)

    # preprocess the input image
    input_256 = preprocess(predictor, input_raw)

    # generate multi-view images in two stages with Zero123.
    # first stage: generate N=8 views cover 360 degree of the input shape.
    elev, stage1_imgs = stage1_run(model_zero123, device, shape_dir, input_256, scale=3, ddim_steps=75)
    # second stage: 4 local views for each of the first-stage view, resulting in N*4=32 source view images.
    stage2_run(model_zero123, device, shape_dir, elev, scale=3, stage2_steps=50)

def process_one(img, gpu_idx, model_zero123):

    with tempfile.TemporaryDirectory() as shape_dir:

        ### processing
        print("start processing: ", img)
        cloudpath = cloudpathlib.CloudPath(f's3://gso-renders/{img}')
        id = img.split('/')[-3]
        save_filename = f"{id}.png"
        save_path = os.path.join(shape_dir, save_filename)
        cloudpath.download_to(save_path)

        # predict multiview
        predict_multiview(shape_dir, gpu_idx, args.half_precision, save_path, model_zero123)

        # reconstruct
        mesh_path = reconstruct(shape_dir, output_format=args.output_format, device_idx=gpu_idx, resolution=args.mesh_resolution)

        # copy mesh path to output dir
        obj_path = os.path.join(args.output_dir, f'{id}.obj')
        shutil.copyfile(mesh_path, obj_path)

        print("Saved to:", obj_path)

def worker(queue, count, worker_i):


    # initialize the zero123 model
    models = init_model(device, 'zero123-xl.ckpt', half_precision=args.half_precision)
    model_zero123 = models["turncam"]


    while True:
        item = queue.get()
        if item is None:
            break
        try:
            process_one(item, worker_i % args.cuda_cnt, model_zero123)
        except Exception as e:
            print(e)
        queue.task_done()
        with count.get_lock():
            count.value += 1


if __name__ == "__main__":

    s3 = boto3.resource('s3')

    os.makedirs(args.output_dir, exist_ok=True)

    ## getting file with extensions in s3 bucket
    bucket = s3.Bucket(args.bucket)
    img_lists = []
    for obj in bucket.objects.filter(Prefix=''):
        if obj.key.endswith(args.ext):
            img_lists.append(obj.key)

    print("Number of images: ", len(img_lists))

    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    for worker_i in range(args.workers):
        process = multiprocessing.Process(
            target=worker, args=(queue, count, worker_i)
        )
        process.daemon = True
        process.start()

    for img in img_lists:
        queue.put(img)

    queue.join()

    for _ in range(args.workers):
        queue.put(None)

    print(f'Processed {count.value} models')