import traceback

import facexlib
import gfpgan

import src_plugins.sd1111_plugin.options
from src_core import devicelib
from src_core.lib import modellib
from src_core.paths import modeldir, root
from src_core.installing import *
from shared import cmd_opts

model_dir = "GFPGAN"
user_path = None
model_path = os.path.join(modeldir, model_dir)
model_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
have_gfpgan = False
loaded_gfpgan_model = None


def gfpgann():
    global loaded_gfpgan_model
    global model_path
    if loaded_gfpgan_model is not None:
        loaded_gfpgan_model.gfpgan.to(devicelib.device_gfpgan)
        return loaded_gfpgan_model

    if gfpgan_constructor is None:
        return None

    models = modellib.load_models(model_path, model_url, user_path, ext_filter="GFPGAN")
    if len(models) == 1 and "http" in models[0]:
        model_file = models[0]
    elif len(models) != 0:
        latest_file = max(models, key=os.path.getctime)
        model_file = latest_file
    else:
        print("Unable to load gfpgan model!")
        return None

    model = gfpgan_constructor(model_path=model_file, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
    loaded_gfpgan_model = model

    return model


def send_model_to(model, device):
    model.gfpgan.to(device)
    model.face_helper.face_det.to(device)
    model.face_helper.face_parse.to(device)




gfpgan_constructor = None


def setup_model(dirname):
    global model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    try:
        from gfpgan import GFPGANer
        from facexlib import detection, parsing
        global user_path
        global have_gfpgan
        global gfpgan_constructor

        load_file_from_url_orig = gfpgan.utils.load_file_from_url
        facex_load_file_from_url_orig = facexlib.detection.load_file_from_url
        facex_load_file_from_url_orig2 = facexlib.parsing.load_file_from_url

        def my_load_file_from_url(**kwargs):
            return load_file_from_url_orig(**dict(kwargs, model_dir=model_path))

        def facex_load_file_from_url(**kwargs):
            return facex_load_file_from_url_orig(**dict(kwargs, save_dir=model_path, model_dir=None))

        def facex_load_file_from_url2(**kwargs):
            return facex_load_file_from_url_orig2(**dict(kwargs, save_dir=model_path, model_dir=None))

        gfpgan.utils.load_file_from_url = my_load_file_from_url
        facexlib.detection.load_file_from_url = facex_load_file_from_url
        facexlib.parsing.load_file_from_url = facex_load_file_from_url2
        user_path = dirname
        have_gfpgan = True
        gfpgan_constructor = GFPGANer

    except Exception:
        print("Error setting up GFPGAN:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)


class GFPGANPlugin:
    def install(self, args):
        gfpgan_package = os.environ.get('GFPGAN_PACKAGE', "git+https://github.com/TencentARC/GFPGAN.git@8d2447a2d918f8eba5a4a01463fd48e45126a379")
        if not is_installed("gfpgan"):
            pipargs(f"install {gfpgan_package}", "gfpgan")

    def setup(self):
        setup_model(cmd_opts.gfpgan_models_path)


    def postprocess(np_image):
        model = gfpgann()
        if model is None:
            return np_image

        send_model_to(model, devicelib.device_gfpgan)

        np_image_bgr = np_image[:, :, ::-1]
        cropped_faces, restored_faces, gfpgan_output_bgr = model.enhance(np_image_bgr, has_aligned=False, only_center_face=False, paste_back=True)
        np_image = gfpgan_output_bgr[:, :, ::-1]

        model.face_helper.clean_all()

        if src_plugins.stable_diffusion_auto2222.options.opts.face_restoration_unload:
            send_model_to(model, devicelib.cpu)

        return np_image

    def copy_models(self):
        src_path = os.path.join(root, "gfpgan")
        dest_path = os.path.join(modeldir, "GFPGAN")
        mvfiles(src_path, dest_path)

