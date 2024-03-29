import math

import numpy as np
import torch
from einops import rearrange
from PIL import Image, ImageOps

from classes.convert import load_cv2, pil2cv
from . import py3d_tools as p3d
from .depth import DepthModel

from src_core.classes.JobArgs import JobArgs
from src_core.classes.Plugin import Plugin
from src_core.lib import devices
from src_core.plugins import plugfun, plugfun_img, plugjob

TRANSLATION_SCALE = 1.0 / 200.0


class transform3d_job(JobArgs):
    def __init__(self, x: float = 0,
                 y: float = 0,
                 z: float = 0,
                 rx: float = 0,
                 ry: float = 0,
                 rz: float = 0,
                 fov: float = 90,
                 near: float = 200,
                 far: float = 10000,
                 w_midas: float = 0.3,
                 padding_mode: str = 'border',
                 sampling_mode: str = 'bicubic',
                 flat: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.z = z
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.fov = fov
        self.near = near
        self.far = far
        self.w_midas = w_midas
        self.padding_mode = padding_mode
        self.sampling_mode = sampling_mode
        self.flat = flat
        self.dev = False


class Midas3DPlugin(Plugin):
    def title(self):
        return "midas3d"

    def describe(self):
        return ""

    def load(self):
        self.loaded = True

        print("Loading midas ...")
        self.model = DepthModel(devices.device)
        self.model.download_midas(self.res())
        self.model.download_adabins(self.res())

        self.model.load_midas(self.res())
        self.model.load_adabins(self.res())

    def transform_image_3d(self, prev_img_cv2, depth_tensor, rot_mat, translate, near, far, fov, padding_mode, sampling_mode):
        if not self.loaded:
            self.load()

        # device = devices.device
        device = devices.cpu
        w, h = prev_img_cv2.shape[1], prev_img_cv2.shape[0]

        aspect_ratio = float(w) / float(h)
        persp_cam_old = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov, degrees=True, device=device)
        persp_cam_new = p3d.FoVPerspectiveCameras(near, far, aspect_ratio, fov=fov, degrees=True, R=rot_mat, T=torch.tensor([translate]), device=device)

        # range of [-1,1] is important to torch grid_sample's padding handling
        y, x = torch.meshgrid(torch.linspace(-1., 1., h, dtype=torch.float32, device=device), torch.linspace(-1., 1., w, dtype=torch.float32, device=device))
        z = torch.as_tensor(depth_tensor, dtype=torch.float32, device=device)
        xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

        xyz_old_cam_xy = persp_cam_old.get_full_projection_transform().transform_points(xyz_old_world)[:, 0:2]
        xyz_new_cam_xy = persp_cam_new.get_full_projection_transform().transform_points(xyz_old_world)[:, 0:2]

        offset_xy = xyz_new_cam_xy - xyz_old_cam_xy
        # affine_grid theta param expects a batch of 2D mats. Each is 2x3 to do rotation+translation.
        identity_2d_batch = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=device).unsqueeze(0)
        # coords_2d will have shape (N,H,W,2).. which is also what grid_sample needs.
        coords_2d = torch.nn.functional.affine_grid(identity_2d_batch, [1, 1, h, w], align_corners=False)
        offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h, w, 2)).unsqueeze(0)

        image_tensor = rearrange(torch.from_numpy(prev_img_cv2.astype(np.float32)), 'h w c -> c h w').to(device)
        new_image = torch.nn.functional.grid_sample(
                image_tensor.add(1 / 512 - 0.0001).unsqueeze(0),
                offset_coords_2d,
                mode=sampling_mode,
                padding_mode=padding_mode,
                align_corners=False
        )

        # convert back to cv2 style numpy array
        result = rearrange(
                new_image.squeeze().clamp(0, 255),
                'c h w -> h w c'
        ).cpu().numpy().astype(prev_img_cv2.dtype)
        return result

    def transform_3d(self,
                     imcv,
                     x, y, z,
                     rx, ry, rz,
                     fov, near, far,
                     padding_mode, sampling_mode,
                     w_midas,
                     depth=None):
        imcv = load_cv2(imcv)

        x = x
        y = y
        z = z
        rx = rx
        ry = ry
        rz = rz
        fov = fov
        near = near
        far = far
        w_midas = w_midas
        # vprint(f'transform_3d(translate=({x:.2f}, {y:.2f}, {z:.2f}) rotation=({rx:.2f}, {ry:.2f}, {rz:.2f}) fov={fov:.2f} near={near:.2f} far={far:.2f} w_midas={w_midas:.2f}')

        if depth is None:
            depth = self.model.predict(imcv, w_midas)

        translate_xyz = [-x * TRANSLATION_SCALE, y * TRANSLATION_SCALE, -z * TRANSLATION_SCALE]
        rotate_xyz = [math.radians(rx), math.radians(ry), math.radians(rz)]

        rot_mat = p3d.euler_angles_to_matrix(torch.tensor(rotate_xyz, device=devices.device), "XYZ").unsqueeze(0)
        result = self.transform_image_3d(imcv, depth, rot_mat, translate_xyz, near, far, fov, padding_mode, sampling_mode)
        torch.cuda.empty_cache()

        return result

    @plugfun(plugfun_img)
    def mat3d(self, x: float = 0,
              y: float = 0,
              z: float = 0,
              rx: float = 0,
              ry: float = 0,
              rz: float = 0,
              fov: float = 90,
              near: float = 200,
              far: float = 10000,
              w_midas: float = 0.3,
              padding_mode: str = 'border',
              sampling_mode: str = 'bicubic',
              depth=None,
              flat: bool = False,
              **kwargs):
        from src_core.rendering import renderer
        rv = renderer.rv

        if rv.img is None:
            return

        pil = rv.img

        if flat:
            torch.ones((pil.width, pil.height), device=devices.device)

        if not self.loaded:
            self.load()

        return self.transform_3d(rv.img,
                                 x, y, z,
                                 rx, ry, rz,
                                 fov, near, far,
                                 padding_mode, sampling_mode,
                                 w_midas, depth)

    @plugfun(default_return=(plugfun_img, plugfun_img))
    def get_depth(self, img, w_midas: float = 0.3, **kwargs):
        if not self.loaded:
            self.load()

        from src_core.rendering import renderer
        session = renderer.session

        if img is None:
            img = session.img

        depth_tensor = self.model.predict(np.asarray(img), w_midas)
        pil = self.model.to_pil(depth_tensor)

        # The pil out of the model is inverted (far is white)
        pil = ImageOps.invert(pil)

        return pil2cv(pil), depth_tensor
