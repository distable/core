import os
import shutil


# from basicsr.utils.download_util import load_file_from_url


# def cleanup_models():
#     # This code could probably be more efficient if we used a tuple list or something to store the src/destinations
#     # and then enumerate that, but this works for now. In the future, it'd be nice to just have every "model" scaler
#     # somehow auto-register and just do these things...
#     root_path = script_path
#     src_path = models_path
#     dest_path = os.path.join(models_path, "Stable-diffusion")
#     move_files(src_path, dest_path, ".ckpt")
#     src_path = os.path.join(root_path, "ESRGAN")
#     dest_path = os.path.join(models_path, "ESRGAN")
#     move_files(src_path, dest_path)
#     src_path = os.path.join(root_path, "gfpgan")
#     dest_path = os.path.join(models_path, "GFPGAN")
#     move_files(src_path, dest_path)
#     src_path = os.path.join(root_path, "SwinIR")
#     dest_path = os.path.join(models_path, "SwinIR")
#     move_files(src_path, dest_path)
#     src_path = os.path.join(root_path, "plugin-repos/latent-diffusion/experiments/pretrained_models/")
#     dest_path = os.path.join(models_path, "LDSR")
#     move_files(src_path, dest_path)


def move_files(src_path: str, dest_path: str, ext_filter: str = None):
    try:
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        if os.path.exists(src_path):
            for file in os.listdir(src_path):
                fullpath = os.path.join(src_path, file)
                if os.path.isfile(fullpath):
                    if ext_filter is not None:
                        if ext_filter not in file:
                            continue
                    print(f"Moving {file} from {src_path} to {dest_path}.")
                    try:
                        shutil.move(fullpath, dest_path)
                    except:
                        pass
            if len(os.listdir(src_path)) == 0:
                print(f"Removing empty folder: {src_path}")
                shutil.rmtree(src_path, True)
    except:
        pass