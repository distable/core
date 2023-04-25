import cv2
import numpy as np
import torch

import installing
from classes.convert import load_cv2
from lib.devices import device
from plug_repos.flower.SD_CN_Animation.FloweR.model import FloweR
from src_core.rendering.hud import hud, snap
from src_core.classes.Plugin import Plugin
from src_plugins.flower import flow_viz
from src_plugins.flower.flow_utils import flow_renorm, frames_norm, occl_renorm

# from FloweR.model import FloweR
# from FloweR.utils import flow_viz
# from FloweR import flow_utils


class FlowerFlowPlugin(Plugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w = -1
        self.h = -1
        self.model = None

    def title(self):
        return "flow_flower"

    def describe(self):
        return ""

    def init(self):
        pass

    def install(self):
        pass

    def uninstall(self):
        pass

    def load(self):
        pass

    def unload(self):
        pass

    def ensure_loaded(self, h, w):
        pth = self.res("FloweR_0.1.pth")
        if not pth.exists():
            installing.gdown(pth, "https://drive.google.com/uc?export=download&id=1WhzoVIw6Kdg4EjfK9LaTLqFm5dF-IJ7F")

        if self.w != w or self.h != h:
            self.w = w
            self.h = h
            self.clip_frames = np.zeros((4, h, w, 3), dtype=np.uint8)
            self.model = FloweR(input_size=(h, w))
            self.model.load_state_dict(torch.load(pth))
            # Move the model to the device
            self.model = self.model.to(device)
            print("FlowerR model loaded.")

    def push(self, img):
        img = load_cv2(img)

        self.ensure_loaded(img.shape[0], img.shape[1])

        self.clip_frames = np.roll(self.clip_frames, -1, axis=0)
        self.clip_frames[-1] = img


    def flow(self, img, strength):
        img = load_cv2(img)
        w = img.shape[1]
        h = img.shape[0]

        hud(flower=strength)

        self.ensure_loaded(h, w)

        color_shift = np.zeros((0, 3))
        color_scale = np.zeros((0, 3))

        self.push(img)

        clip_frames_torch = frames_norm(torch.from_numpy(self.clip_frames).to(device, dtype=torch.float32))

        with torch.no_grad():
            pred_data = self.model(clip_frames_torch.unsqueeze(0))[0]

        pred_flow = flow_renorm(pred_data[..., :2]).cpu().numpy()
        pred_occl = occl_renorm(pred_data[..., 2:3]).cpu().numpy().repeat(3, axis=-1)

        pred_flow = pred_flow / (1 + np.linalg.norm(pred_flow, axis=-1, keepdims=True) * 0.05)
        pred_flow = cv2.GaussianBlur(pred_flow, (31, 31), 1, cv2.BORDER_REFLECT_101)

        pred_occl = cv2.GaussianBlur(pred_occl, (21, 21), 2, cv2.BORDER_REFLECT_101)
        pred_occl = (np.abs(pred_occl / 255) ** 1.5) * 255
        pred_occl = np.clip(pred_occl * 25, 0, 255).astype(np.uint8)

        pred_flow = pred_flow * strength

        flow_map = pred_flow.copy()
        flow_map[:, :, 0] += np.arange(w)
        flow_map[:, :, 1] += np.arange(h)[:, np.newaxis]

        warped_frame = cv2.remap(img, flow_map, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)

        # Flow image
        flow_img = flow_viz.flow_to_image(pred_flow)
        frames_img = cv2.hconcat(list(self.clip_frames))
        data_img = cv2.hconcat([flow_img, pred_occl, warped_frame])

        snap('flower_flow', flow_img)
        snap('flower_warped', warped_frame)

        return warped_frame

