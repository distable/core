import argparse
import os
import pathlib
import sys

import PIL
import scipy
from tqdm import tqdm

from src_core.classes.JobArgs import JobArgs
from src_core.classes.convert import load_pil, load_pilarr, save_jpg, save_npy, save_png
from src_core.classes.Plugin import Plugin
from src_core.installing import gitclone
from src_core.plugins import plugjob
from src_plugins.disco_party.maths import clamp01
from python_color_transfer.color_transfer import ColorTransfer, Regrain
from pathlib import Path

from src_plugins.opticalflow_plugin.__conf__ import half

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import cv2
from PIL import Image, ImageEnhance, ImageOps, ImageStat
from torch.nn import functional as F
# from CLIP import clip
import gc
import os

# NOTE: Turbo mode used to be blocked for anything but 3D mode
# NOTE: VR mode used to be blocked for anything but 3D mode

is_colab = False
root_dir = ""  # TODO colab VM root
root_dir = os.getcwd()
# width_height = [480, 270]  # @param{type: 'raw'}
# width_height = [1024, 1024]  # @param{type: 'raw'}
DEBUG = False


def load_img(img, size, resize_mode='lanczos'):
    if resize_mode == 'lanczos':
        resize_mode = PIL.Image.LANCZOS

    img = PIL.Image.open(img).convert('RGB')
    if img.size != size:
        img = img.resize(size, resize_mode)

    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float()[None, ...].cuda()


def get_flow(frame1, frame2, model, iters=20, half=True):
    # print(frame1.shape, frame2.shape)
    padder = InputPadder(frame1.shape)
    frame1, frame2 = padder.pad(frame1, frame2)
    if half: frame1, frame2 = frame1.half(), frame2.half()
    # print(frame1.shape, frame2.shape)
    flow12 = model(frame1, frame2)
    flow12 = flow12[0][0].permute(1, 2, 0).detach().cpu().numpy()

    return flow12


# region Flow Visualization https://github.com/tomrunia/OpticalFlow_Visualization
# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


# endregion


PT = ColorTransfer()
RG = Regrain()

latent_scale_schedule = [0, 0]  # controls coherency with previous frame in latent space. 0 is a good starting value. 1+ render slower, but may improve image coherency. 100 is a good value if you decide to turn it on.
init_scale_schedule = [0, 0]  # controls coherency with prev frame in pixel space. 0 - off, 1000 - a good starting value if you decide to turn it on.
warp_interp = PIL.Image.LANCZOS

mask_result = False  # imitates inpainting by leaving only inconsistent areas to be diffused
warp_strength = 1  # leave 1 for no change. 1.01 is already a strong value.

# @title Frame correction ------------------------------------------------------------
# @markdown Match frame pixels or latent to other frames to prevent oversaturation and feedback loop artifacts
# @markdown Latent matching
# @markdown Match the range of latent vector towards the 1st frame or a user defined range. Doesn't restrict colors, but may limit contrast.
normalize_latent = 'off'  # @param ['off', 'first_latent', 'user_defined']
# @markdown User defined stats to normalize the latent towards
latent_fixed_mean = 0.  # @param {'type':'raw'}
latent_fixed_std = 0.9  # @param {'type':'raw'}

latent_norm_4d = True  # @param {'type':'boolean'} @markdown Match latent on per-channel basis

# @markdown Video Optical Flow Settings: ------------------------------------------------------------
check_consistency = True  # @param {type: 'boolean'}##@param {type: 'number'} #0 - take next frame, 1 - take prev warped frame

# @title Consistency map mixing
# @markdown You can mix consistency map layers separately\
# @markdown missed_consistency_weight - masks pixels that have missed their expected position in the next frame \
# @markdown overshoot_consistency_weight - masks pixels warped from outside the frame\
# @markdown edges_consistency_weight - masks moving objects' edges\
# @markdown The default values to simulate previous versions' behavior are 1,1,1

missed_consistency_weight = 1  # @param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}
overshoot_consistency_weight = 1  # @param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}
edges_consistency_weight = 1  # @param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}

# Inpaint occluded areas on top of raw frames. 0 - 0% inpainting opacity (no inpainting), 1 - 100% inpainting opacity. Other values blend between raw and inpainted frames.
inpaint_opacity = 0.5
# 0 - off, other values control effect opacity
match_color_strength = 0  # @param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.1'}

# @markdown ###Color matching --------------------------------------------------------------------------------
# @markdown Color match frame towards stylized or raw init frame. Helps prevent images going deep purple. As a drawback, may lock colors to the selected fixed frame. Select stylized_frame with colormatch_offset = 0 to reproduce previous notebooks.
colormatch_frame = 'init_frame_offset'  # @param ['off', 'stylized_frame', 'init_frame', 'stylized_frame_offset', 'init_frame_offset']
# @markdown Color match strength. 1 mimics legacy behavior
color_match_frame_str = 1  # @param {'type':'number'}
# @markdown in offset mode, specifies the offset back from current frame, and 0 means current frame. In non-offset mode specifies the fixed frame number. 0 means the 1st frame.
colormatch_offset = 0  # @param {'type':'number'}
colormatch_method = 'LAB'  # @param ['LAB', 'PDF', 'mean']
colormatch_method_fn = PT.lab_transfer
if colormatch_method == 'LAB':
    colormatch_method_fn = PT.pdf_transfer
if colormatch_method == 'mean':
    colormatch_method_fn = PT.mean_std_transfer
# @markdown Match source frame's texture
colormatch_regrain = False  # @param {'type':'boolean'}

# @title Video mask settings ------------------------------------------------------------
# @markdown Check to enable background masking during render. Not recommended, better use masking when creating the output video for more control and faster testing.
use_background_mask = True  # @param {'type':'boolean'}
# @markdown Check to invert the mask.
invert_mask = False  # @param {'type':'boolean'}
# @markdown Apply mask right before feeding init image to the model. Unchecking will only mask current raw init frame.
apply_mask_after_warp = True  # @param {'type':'boolean'}
# @markdown Choose background source to paste masked stylized image onto: image, color, init video.
background = "init_video"  # @param ['image', 'color', 'init_video']
# @markdown Specify the init image path or color depending on your background source choice.
background_source = 'red'  # @param {'type':'string'}

# @title Video Masking ------------------------------------------------------------
mask_source = 'init_video'  # @param ['init_video','mask_video'] @markdown Generate background mask from your init video or use a video as a mask
extract_background_mask = True  # @param {'type':'boolean'} @markdown Check to rotoscope the video and create a mask from it. If unchecked, the raw monochrome video will be used as a mask.
mask_video_name = ''  # @param {'type':'string'} @markdown Specify path to a mask video for mask_video mode.

# @title Generate optical flow and consistency maps ------------------------------------------------------------
# @markdown Turbo Mode ------------------------------------------------------------
# @markdown (Starts after frame 1,) skips diffusion steps and just uses flow map to warp images for skipped frames.
# @markdown Speeds up rendering by 2x-4x, and may improve image coherence between frames. frame_blend_mode smooths abrupt texture changes across 2 frames.
# @markdown For different settings tuned for Turbo Mode, refer to the original Disco-Turbo Github: https://github.com/zippy731/disco-diffusion-turbo

turbo_mode = False  # @param {type:"boolean"}
turbo_steps = "3"  # @param ["2","3","4","5","6"] {type:"string"}
turbo_preroll = 1  # frames

class flow_job(JobArgs):
    def __init__(self,
                 name=None,
                 nth=1,
                 strength=1.0,
                 padmode='reflect', # [reflect, edge, wrap]
                 padpct=0.2,  # Increase padding if you have a shaky\moving camera footage and are getting black borders.
                 consistency=True,
                 loop=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.nth = nth
        self.consistency = consistency
        self.strength = strength
        self.padmode = padmode
        self.padpct = padpct
        self.loop = loop


class flowinit_job(flow_job):
    def __init__(self,
                 force=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.force = force
    pass


class flowsetup_job(flow_job):
    pass


class consistency_job(flow_job):
    def __init__(self,
                 image=None,  # The image to remain consistent with, usually the previous frame
                 **kwargs):
        super().__init__(**kwargs)
        self.image = image


class ccblend_job(JobArgs):
    def __init__(self,
                 name:str=None, img: str=None, flow: str=None,
                 t: float=0.5,
                 ccblur:int=2,
                 ccstrength:float=1.0,
                 cccolor:float=1.0,
                 reverse: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.img = img or name or flow
        self.flow = flow or name or img
        self.t = t
        self.ccblur = ccblur
        self.ccstrength = ccstrength
        self.cccolor = cccolor
        self.reverse = reverse


class OpticalflowPlugin(Plugin):
    def title(self):
        return "opticalflow"

    def describe(self):
        return ""

    def init(self):
        pass

    def install(self):
        gitclone('https://github.com/princeton-vl/RAFT')
        gitclone('https://github.com/Sxela/WarpFusion', into_dir=self.res())
        gitclone('https://github.com/Sxela/flow_tools', into_dir=self.res())

        sys.path.append(self.repo('RAFT').as_posix())
        sys.path.append(self.repo('RAFT/core').as_posix())

        # TODO
        # !pip install av pims

        # TODO
        # !git clone https://github.com/Sxela/RobustVideoMattingCLI

        # TODO
        # install ffmpeg

    def uninstall(self):
        pass

    def load(self):
        pass

    def unload(self):
        pass

    @plugjob(key='devdef_flow')
    def flow_init(self, j: flowinit_job):
        from core.raft import RAFT

        # path_initframes = j.session.extract_frames(j.vidname, j.init_video_nth, start_frame, end_frame)
        path_flowframes = j.session.extract_frames(j.name, j.nth)

        # TODO This probably warrants its own job or plugin even. Separation of concerns
        # if j.bg_mask:
        #     extract_frames(j.mask_path, f"{path_vidframes_mask()}", extract_nth_frame, start_frame, end_frame)
        #     os.system(f'python "{root_dir}/RobustVideoMattingCLI/rvm_cli.py" --input_path "{path_vidframes_mask()}" --output_alpha "{path_alpha_vid}"')
        #     extract_frames(path_alpha_vid, f"{path_alpha_vidframes}", 1, 0, 999999999)
        # else:
        #     pass  # TODO
        # path_alpha_vid = j.session.res('alpha.mp4')
        # if path_alpha_vid.exists():
        #     path_alpha_vidframes = j.session.extract_frames(path_alpha_vid, j.init_flow_nth, start_frame, end_frame)

        path_raft_half = self.res('WarpFusion/raft/raft_half.jit')
        path_raft_full = self.res('WarpFusion/raft/raft_fp32.jit')
        temp_flo = path_flowframes / '_temp_flo'
        dir_flow21 = path_flowframes / 'flow21'  # _out_flo_fwd
        dir_flow12 = path_flowframes / 'flow12'  # _out_flo_bck
        dir_flowcc21 = path_flowframes / 'flowcc21'
        dir_flowcc12 = path_flowframes / 'flowcc12'

        dir_flowcc21.mkdir(parents=True, exist_ok=True)
        dir_flowcc12.mkdir(parents=True, exist_ok=True)
        dir_flow21.mkdir(parents=True, exist_ok=True)
        dir_flow12.mkdir(parents=True, exist_ok=True)
        cc_path = self.res('flow_tools/check_consistency.py')

        # CREATE FLOW DATA
        # ------------------------------------------------------------
        flows = list(dir_flow21.glob('*.*'))

        if len(flows) == 0 or j.force:
            # Delete existing flow data
            for d in [dir_flow21, dir_flow12, dir_flowcc21, dir_flowcc12]:
                for f in d.glob('*.*'):
                    f.unlink()

            frames = sorted(path_flowframes.glob('*.*'))
            if len(frames) < 2:
                print(f'WARNING!\nCannot create flow maps: Found {len(frames)} frames extracted from your video input.\nPlease check your video path.')

            if len(frames) >= 2:
                parser = argparse.ArgumentParser()
                parser.add_argument('--model', help="restore checkpoint")
                parser.add_argument('--dataset', help="dataset for evaluation")
                parser.add_argument('--small', action='store_true', help='use small model')
                parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
                parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
                args = parser.parse_args()

                if half:
                    raft_model = torch.jit.load(path_raft_half).eval()
                    # raft_model = torch.nn.DataParallel(RAFT(args2))
                else:
                    # raft_model = torch.jit.load(path_raft_full).eval()
                    raft_model = torch.nn.DataParallel(RAFT(args))
                    raft_model.load_state_dict(torch.load(self.res('raft-things.pth')))
                    raft_model = raft_model.module.cuda().eval()

                for f in pathlib.Path(f'{dir_flow21}').glob('*.*'):
                    f.unlink()

                with torch.no_grad():
                    for frame1, frame2 in tqdm(zip(frames[:-1], frames[1:]), total=len(frames) - 1):
                        in_frame1 = Path(frame1)
                        in_frame2 = Path(frame2)
                        out_flow21 = dir_flow21 / in_frame1.stem
                        out_flow12 = dir_flow12 / in_frame1.stem

                        frame1 = load_img(in_frame1, (j.ctx.width, j.ctx.height))
                        frame2 = load_img(in_frame2, (j.ctx.width, j.ctx.height))

                        flow21 = get_flow(frame2, frame1, raft_model, half=half)
                        flow21pil = PIL.Image.fromarray(flow_to_image(flow21))
                        save_npy(out_flow21, flow21)
                        save_png(flow21pil, out_flow21)

                        flow12 = get_flow(frame1, frame2, raft_model, half=half)
                        flow12pil = PIL.Image.fromarray(flow_to_image(flow12))
                        save_npy(out_flow12, flow12)
                        save_png(flow12pil, out_flow12)
                        gc.collect()

                del raft_model
                gc.collect()
                fwd = f"{dir_flow21}/*.npy"
                bwd = f"{dir_flow12}/*.npy"
                os.system(f'python "{cc_path}" --flow_fwd "{fwd}" --flow_bwd "{bwd}" --output "{dir_flowcc21}/" --image_output --output_postfix="" --blur=0. --save_separate_channels --skip_numpy_output')
                os.system(f'python "{cc_path}" --flow_fwd "{bwd}" --flow_bwd "{fwd}" --output "{dir_flowcc12}/" --image_output --output_postfix="" --blur=0. --save_separate_channels --skip_numpy_output')

        frames = sorted(path_flowframes.glob('*.*'))
        if len(frames) == 0:
            sys.exit("ERROR: 0 frames found.\nPlease check your video input path and rerun the video settings cell.")

        flows = list(dir_flow21.glob('*.*'))
        if len(flows) == 0:
            sys.exit("ERROR: 0 flow files found.\nPlease rerun the flow generation cell.")

    @plugjob(key='devdef_flow')
    def flow_setup(self, j: flowsetup_job):
        frame_num = j.session.f
        img = j.ctx.image

        if frame_num == 0 and use_background_mask:
            img = apply_mask(j.session, img, frame_num, background, background_source, invert_mask)

        return img

    @plugjob(key='devdef_flow')
    def flow(self, j: flow_job):
        animpil = j.ctx.image
        flowpath = j.session.res_frame(j.name, 'flow21', ext='npy')

        # if use_background_mask and not apply_mask_after_warp:
        #     # if turbo_mode & (f % int(turbo_steps) != 0):
        #     #   print('disabling mask for turbo step, will be applied during turbo blend')
        #     # else:
        #     print('creating bg mask for frame ', f)
        #     initpil = apply_mask(j.session, initpil, f, background, background_source, invert_mask)
        #     # initpil.save(f'frame2_{f}.jpg')

        warped = warp(animpil, flowpath,
                      padmode=j.padmode,
                      padpct=j.padpct,
                      multiplier=j.strength)

        # warped = warped.resize((side_x,side_y), warp_interp)
        # if use_background_mask and apply_mask_after_warp:
        #     # if turbo_mode & (f % int(turbo_steps) != 0):
        #     #   print('disabling mask for turbo step, will be applied during turbo blend')
        #     #   return warped
        #     print('creating bg mask for frame ', f)
        #     warped = apply_mask(j.session, warped, f, background, background_source, invert_mask)

        return warped

    @plugjob(key='devdef_flow')
    def ccblend(self, j: ccblend_job):
        img1 = j.ctx.image
        img2 = j.session.res_framepil(j.img, ctxsize=True)
        ccpath = get_consistency_path(j.session, j.flow, reverse=j.reverse)

        blended = blend(img1, img2, j.t,
                     ccpath=ccpath, ccstrength=j.ccstrength, ccblur=j.ccblur, cccolor=j.cccolor)

        if j.session.f == 1:
            return blended

        if mask_result:
            imgprev =  j.session.res_framepil(j.session.f - 1, ctxsize=True)

            diffuse_inpaint_mask_blur = 15
            diffuse_inpaint_mask_thresh = 220

            consistency_mask = load_cc(ccpath, blur=j.ccblur)
            consistency_mask = cv2.GaussianBlur(consistency_mask, (diffuse_inpaint_mask_blur, diffuse_inpaint_mask_blur), cv2.BORDER_DEFAULT)
            consistency_mask = np.where(consistency_mask < diffuse_inpaint_mask_thresh / 255., 0, 1.)
            consistency_mask = cv2.GaussianBlur(consistency_mask, (3, 3), cv2.BORDER_DEFAULT)

            # consistency_mask = torchvision.transforms.functional.resize(consistency_mask, image.size)
            print(imgprev.size, consistency_mask.shape, blended.size)
            cc_sz = consistency_mask.shape[1], consistency_mask.shape[0]
            image_masked = np.array(blended) * (1 - consistency_mask) + np.array(imgprev) * consistency_mask

            # image_masked = np.array(image.resize(cc_sz, warp_interp))*(1-consistency_mask) + np.array(init_img_prev.resize(cc_sz, warp_interp))*(consistency_mask)
            image_masked = PIL.Image.fromarray(image_masked.round().astype('uint8'))
            # image = image_masked.resize(image.size, warp_interp)
            image = image_masked

        return blended


@plugjob(key='devdef_flow')
def consistency(self, j: consistency_job):
    if not j.ctx.image: return
    image = j.ctx.image
    f = j.session.f
    ccpath = j.session.res_frame(j.name, f, 'flowcc21')

    return image


def get_consistency_path(session, resname, reverse=False):
    fwd = session.res_frame(resname, 'flowcc21')
    bwd = session.res_frame(resname, 'flowcc12')

    if reverse:
        return bwd
    else:
        return fwd


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def warp_flow(img, flow, mul=1.):
    h, w = flow.shape[:2]
    flow = flow.copy()
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    # print('flow stats', flow.max(), flow.min(), flow.mean())
    # print(flow)
    flow *= mul
    # print('flow stats mul', flow.max(), flow.min(), flow.mean())
    # res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    res = cv2.remap(img, flow, None, cv2.INTER_LANCZOS4)

    return res


def makeEven(_x):
    return _x if (_x % 2 == 0) else _x + 1


def fit(img, maxsize=512):
    maxdim = max(*img.size)
    if maxdim > maxsize:
        # if True:
        ratio = maxsize / maxdim
        x, y = img.size
        size = (makeEven(int(x * ratio)), makeEven(int(y * ratio)))
        img = img.resize(size, warp_interp)
    return img


def warp(framepil, flowpath,
         padmode='reflect', padpct=0.1,
         multiplier=1.0):
    flow21 = np.load(flowpath)
    framearr = load_pilarr(framepil)

    pad = int(max(flow21.shape) * padpct)
    flow21 = np.pad(flow21, pad_width=((pad, pad), (pad, pad), (0, 0)), mode='constant')
    framearr = np.pad(framearr, pad_width=((pad, pad), (pad, pad), (0, 0)), mode=padmode)

    warpedarr21 = warp_flow(framearr, flow21, multiplier)
    warpedarr21 = warpedarr21[pad:warpedarr21.shape[0] - pad, pad:warpedarr21.shape[1] - pad, :]

    return PIL.Image.fromarray(warpedarr21.round().astype('uint8'))


def blend(img1, img2, t=0.5,
          ccpath=None, ccstrength=0.0, ccblur=2, cccolor=0.0,
          padmode='reflect', padpct=0.0)->Image.Image:
    """

    Args:
        img1: The start image.
        img2: The goal image.
        t: How much to blend the two images. 0.0 is all img1, 1.0 is all img2.
        ccpath: The path to the consistency mask.
        ccstrength: How much of the consistency mask to use.
        ccblur: Blur radius to soften the consistency mask. (Softens transition between raw video init and stylized frames in occluded areas)
        cccolor: Match color of inconsistent areas to unoccluded ones, after inconsistent areas were replaced with raw init video or inpainted. 0 to disable, 0 to 1 for strength.
        padmode: The padding mode to use.
        padpct: The padding percentage to use.

    Returns: The blended image.

    """

    img1 = load_pilarr(img1)
    pad = int(max(img1.shape) * padpct)

    img2 = load_pilarr(img2, size=(img1.shape[1] - pad * 2, img1.shape[0] - pad * 2))
    # initarr = np.array(img2.convert('RGB').resize((flow21.shape[1] - pad * 2, flow21.shape[0] - pad * 2), warp_interp))
    t = 1.0

    if ccpath:
        ccweights = load_cc(ccpath, blur=ccblur)
        if cccolor:
            img2 = match_color(img1, img2, blend=cccolor)

        ccweights = ccweights.clip(1 - ccstrength, 1.)
        blended_w = img2 * (1 - t) + t * (img1 * ccweights + img2 * (1 - ccweights))
    else:
        if cccolor:
            img2 = match_color(img1, img2, blend=cccolor)

        blended_w = img2 * (1 - t) + img1 * t

    blended_w = PIL.Image.fromarray(blended_w.round().astype('uint8'))
    return blended_w


def match_color(stylized_img, raw_img, blend=1.0):
    img_arr_ref = cv2.cvtColor(np.array(stylized_img).round().astype('uint8'), cv2.COLOR_RGB2BGR)
    img_arr_in = cv2.cvtColor(np.array(raw_img).round().astype('uint8'), cv2.COLOR_RGB2BGR)
    # img_arr_in = cv2.resize(img_arr_in, (img_arr_ref.shape[1], img_arr_ref.shape[0]), interpolation=cv2.INTER_CUBIC )
    img_arr_col = PT.pdf_transfer(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)
    img_arr_reg = RG.regrain(img_arr_in=img_arr_col, img_arr_col=img_arr_ref)

    blended = img_arr_reg * blend + img_arr_in * (1 - blend)
    blended = cv2.cvtColor(blended.round().astype('uint8'), cv2.COLOR_BGR2RGB)
    return blended


def match_color_var(stylized_img, raw_img, opacity=1., f=PT.pdf_transfer, regrain=False):
    img_arr_ref = cv2.cvtColor(np.array(stylized_img).round().astype('uint8'), cv2.COLOR_RGB2BGR)
    img_arr_in = cv2.cvtColor(np.array(raw_img).round().astype('uint8'), cv2.COLOR_RGB2BGR)
    img_arr_ref = cv2.resize(img_arr_ref, (img_arr_in.shape[1], img_arr_in.shape[0]), interpolation=cv2.INTER_CUBIC)
    img_arr_col = f(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)
    if regrain: img_arr_col = RG.regrain(img_arr_in=img_arr_col, img_arr_col=img_arr_ref)
    img_arr_col = img_arr_col * opacity + img_arr_in * (1 - opacity)
    img_arr_reg = cv2.cvtColor(img_arr_col.round().astype('uint8'), cv2.COLOR_BGR2RGB)

    return img_arr_reg


def load_cc(path: Image.Image | Path | str, blur=2):
    ccpil = PIL.Image.open(path)
    multilayer_weights = np.array(ccpil) / 255
    weights = np.ones_like(multilayer_weights[..., 0])
    weights *= multilayer_weights[..., 0].clip(1 - missed_consistency_weight, 1)
    weights *= multilayer_weights[..., 1].clip(1 - overshoot_consistency_weight, 1)
    weights *= multilayer_weights[..., 2].clip(1 - edges_consistency_weight, 1)

    if blur > 0: weights = scipy.ndimage.gaussian_filter(weights, [blur, blur])
    weights = np.repeat(weights[..., None], 3, axis=2)

    if DEBUG: print('weight min max mean std', weights.shape, weights.min(), weights.max(), weights.mean(), weights.std())
    return weights


def apply_mask(fg: Path | Image.Image, bg: Path | Image.Image, mask: Path | Image.Image, invert=False):
    # Get the size we're working with
    size = (1, 1)
    if isinstance(fg, Image.Image): size = fg.size
    if isinstance(bg, Image.Image): size = bg.size
    if isinstance(mask, Image.Image): size = mask.size

    # Get the images
    fg = load_pil(fg, size)
    bg = load_pil(bg, size)
    mask = load_pil(mask, size).convert('L')
    if invert:
        mask = PIL.ImageOps.invert(mask)

    # Composite everything
    bg.paste(fg, (0, 0), mask)
    return bg


# implemetation taken from https://github.com/lowfuel/progrockdiffusion


def run_consistency(image, frame_num):
    if mask_result and check_consistency and frame_num > 0:
        diffuse_inpaint_mask_blur = 15
        diffuse_inpaint_mask_thresh = 220
        print('imitating inpaint')
        frame1_path = f'{path_init_video_frames}/{frame_num:06}.jpg'
        weights_path = f"{flo_fwd_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
        consistency_mask = load_cc(weights_path, blur=consistency_blur)
        consistency_mask = cv2.GaussianBlur(consistency_mask,
                                            (diffuse_inpaint_mask_blur, diffuse_inpaint_mask_blur), cv2.BORDER_DEFAULT)
        consistency_mask = np.where(consistency_mask < diffuse_inpaint_mask_thresh / 255., 0, 1.)
        consistency_mask = cv2.GaussianBlur(consistency_mask,
                                            (3, 3), cv2.BORDER_DEFAULT)

        # consistency_mask = torchvision.transforms.functional.resize(consistency_mask, image.size)
        init_img_prev = PIL.Image.open(init_image)
        print(init_img_prev.size, consistency_mask.shape, image.size)
        cc_sz = consistency_mask.shape[1], consistency_mask.shape[0]
        image_masked = np.array(image) * (1 - consistency_mask) + np.array(init_img_prev) * (consistency_mask)

        # image_masked = np.array(image.resize(cc_sz, warp_interp))*(1-consistency_mask) + np.array(init_img_prev.resize(cc_sz, warp_interp))*(consistency_mask)
        image_masked = PIL.Image.fromarray(image_masked.round().astype('uint8'))
        # image = image_masked.resize(image.size, warp_interp)
        image = image_masked
    return image


# @title Do the Run
# @markdown Preview max size

display_rate = 9999999
n_batches = 1
first_latent = None
os.chdir(root_dir)

resume_run = False  # @param{type: 'boolean'}
run_to_resume = 'latest'  # @param{type: 'string'}
resume_from_frame = 'latest'  # @param{type: 'string'}
retain_overwritten_frames = False  # @param{type: 'boolean'}


def do_run():
    # if (args.animation_mode == 'Video Input') and (args.midas_weight > 0.0):
    # midas_model, midas_transform, midas_net_w, midas_net_h, midas_resize_mode, midas_normalization = init_midas_depth_model(args.midas_depth_model)


    for i in range(args.n_batches):
        # gc.collect()
        # torch.cuda.empty_cache()
        steps = get_scheduled_arg(frame_num, steps_schedule)
        style_strength = get_scheduled_arg(frame_num, style_strength_schedule)
        skip_steps = int(steps - steps * style_strength)
        cur_t = diffusion.num_timesteps - skip_steps - 1
        total_steps = cur_t

        consistency_mask = None
        if check_consistency and frame_num > 0:
            frame1_path = f'{path_init_video_frames}/{frame_num:06}.jpg'
            if reverse_cc_order:
                weights_path = f"{flo_fwd_folder}/{frame1_path.split('/')[-1]}-21_cc.jpg"
            else:
                weights_path = f"{flo_fwd_folder}/{frame1_path.split('/')[-1]}_12-21_cc.jpg"

            consistency_mask = load_cc(weights_path, blur=consistency_blur)

        #         for k, image in enumerate(sample['pred_xstart']):
        #             # tqdm.write(f'Batch {i}, step {j}, output {k}:')
        #             current_time = datetime.now().strftime('%y%m%d-%H%M%S_%f')
        #             percent = math.ceil(j/total_steps*100)
        #             if args.n_batches > 0:
        #               #if intermediates are saved to the subfolder, don't append a step or percentage to the name
        #               if (cur_t == -1 or cur_t == stop_early-1) and args.intermediates_in_subfolder is True:
        #                 save_num = f'{frame_num:06}' if animation_mode != "None" else i
        #     filename = f'{args.batch_name}({args.batchNum})_{save_num}.png'
        #   else:
        #     #If we're working with percentages, append it
        #     if args.steps_per_checkpoint is not None:
        #       filename = f'{args.batch_name}({args.batchNum})_{i:06}-{percent:02}%.png'
        #     # Or else, iIf we're working with specific steps, append those
        #     else:
        #       filename = f'{args.batch_name}({args.batchNum})_{i:06}-{j:03}.png'

        # image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
        # if frame_num > 0:
        #   print('times per image', o); o+=1
        #   image = PIL.Image.fromarray(match_color_var(first_frame, image, f=PT.lab_transfer))
        #   # image.save(f'/content/{frame_num}_{cur_t}_{o}.jpg')
        #   # image = PIL.Image.fromarray(match_color_var(first_frame, image))

        # #reapply init image on top of

        # if j % args.display_rate == 0 or cur_t == -1 or cur_t == stop_early-1:
        #   image.save('progress.png')
        #   display.clear_output(wait=True)
        #   display.display(display.Image('progress.png'))
        # if args.steps_per_checkpoint is not None:
        #   if j % args.steps_per_checkpoint == 0 and j > 0:
        #     if args.intermediates_in_subfolder is True:
        #       image.save(f'{partialFolder}/{filename}')
        #     else:
        #       image.save(f'{sessionDir}/{filename}')
        # else:
        #   if j in args.intermediate_saves:
        #     if args.intermediates_in_subfolder is True:
        #       image.save(f'{partialFolder}/{filename}')
        #     else:
        #       image.save(f'{sessionDir}/{filename}')
        # if (cur_t == -1) | (cur_t == stop_early-1):
        #   if cur_t == stop_early-1: print('early stopping')
        # if frame_num == 0:
        #   save_settings()
        # if args.animation_mode != "None":
        #   # sys.exit(os.getcwd(), 'cwd')
        #   image.save('prevFrame.png')
        # image.save(f'{sessionDir}/{filename}')
        # if args.animation_mode == 'Video Input':
        #   # If turbo, save a blended image
        #   if turbo_mode and frame_num > 0:
        #     # Mix new image with prevFrameScaled
        #     blend_factor = (1)/int(turbo_steps)
        #     newFrame = cv2.imread('prevFrame.png') # This is already updated..
        #     prev_frame_warped = cv2.imread('prevFrameScaled.png')
        #     blendedImage = cv2.addWeighted(newFrame, blend_factor, prev_frame_warped, (1-blend_factor), 0.0)
        #     cv2.imwrite(f'{sessionDir}/{filename}',blendedImage)
        #   else:
        #     image.save(f'{sessionDir}/{filename}')

        # if frame_num != args.max_frames-1:
        #   display.clear_output()


def fix_pillow_hack():
    # @markdown If you are getting **"AttributeError: module 'PIL.TiffTags' has no attribute 'IFD'"** error,\
    # @markdown just click **"Runtime" - "Restart and Run All"** once per session.
    # hack to get pillow to work w\o restarting
    # if you're running locally, just restart this runtime, no need to edit PIL files.
    global file
    if is_colab:
        filedata = None
        with open('/usr/local/lib/python3.7/dist-packages/PIL/TiffImagePlugin.py', 'r') as file:
            filedata = file.read()
        filedata = filedata.replace('(TiffTags.IFD, "L", "long"),', '#(TiffTags.IFD, "L", "long"),')
        with open('/usr/local/lib/python3.7/dist-packages/PIL/TiffImagePlugin.py', 'w') as file:
            file.write(filedata)
            #             # Mix new image with prevFrameScaled
            #             blend_factor = (1) / int(turbo_steps)
            #             newFrame = cv2.imread('prevFrame.png')  # This is already updated..
            #             prev_frame_warped = cv2.imread('prevFrameScaled.png')
            #             blendedImage = cv2.addWeighted(newFrame, blend_factor, prev_frame_warped, (1 - blend_factor), 0.0)
            #             cv2.imwrite(f'{sessionDir}/{filename}', blendedImage)


@plugjob
def turbo():
    ### Turbo mode - skip some diffusions, use 3d morph for clarity and to save time
    # if turbo_mode:
    #     if frame_num == turbo_preroll:  # start tracking oldframe
    #         next_step_pil.save('oldFrameScaled.png')  # stash for later blending
    #     elif frame_num > turbo_preroll:
    #         # set up 2 warped image sequences, old & new, to blend toward new diff image
    #         old_frame = do_3d_step('oldFrameScaled.png', frame_num, forward_clip=forward_weights_clip_turbo_step)
    #         old_frame.save('oldFrameScaled.png')
    #         if frame_num % int(turbo_steps) != 0:
    #             print('turbo skip this frame: skipping clip diffusion steps')
    #             filename = f'{args.batch_name}({args.batchNum})_{frame_num:06}.png'
    #             blend_factor = ((frame_num % int(turbo_steps)) + 1) / int(turbo_steps)
    #             print('turbo skip this frame: skipping clip diffusion steps and saving blended frame')
    #             newWarpedImg = cv2.imread('prevFrameScaled.png')  # this is already updated..
    #             oldWarpedImg = cv2.imread('oldFrameScaled.png')
    #             blendedImage = cv2.addWeighted(newWarpedImg, blend_factor, oldWarpedImg, 1 - blend_factor, 0.0)
    #             cv2.imwrite(f'{sessionDir}/{filename}', blendedImage)
    #             next_step_pil.save(f'{img_filepath}')  # save it also as prev_frame to feed next iteration
    #             if turbo_frame_skips_steps is not None:
    #                 oldWarpedImg = cv2.imread('prevFrameScaled.png')
    #                 cv2.imwrite(f'oldFrameScaled.png', oldWarpedImg)  # swap in for blending later
    #                 print('clip/diff this frame - generate clip diff image')
    #
    #                 skip_steps = math.floor(steps * turbo_frame_skips_steps)
    #             else: continue
    #         else:
    #             # if not a skip frame, will run diffusion and need to blend.
    #             oldWarpedImg = cv2.imread('prevFrameScaled.png')
    #             cv2.imwrite(f'oldFrameScaled.png', oldWarpedImg)  # swap in for blending later
    #             print('clip/diff this frame - generate clip diff image')


    # Mix new image with prevFrameScaled
    blend_factor = (1) / int(turbo_steps)
    newFrame = cv2.imread('prevFrame.png')  # This is already updated..
    prev_frame_warped = cv2.imread('prevFrameScaled.png')
    blendedImage = cv2.addWeighted(newFrame, blend_factor, prev_frame_warped, (1 - blend_factor), 0.0)
    cv2.imwrite(f'{sessionDir}/{filename}', blendedImage)
