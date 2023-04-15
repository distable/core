import argparse
from glob import glob

import numpy as np
import torch
from PIL import Image

from src_core.classes import JobArgs
from src_core.classes.Plugin import Plugin
from conf import plugdef
from lib import devices
from lib.devices import device
from plugins import plugjob
import torch.nn.functional as F

import os

from unimatch.unimatch import UniMatch
from dataloader.depth import augmentation

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class UnimatchPlugin(Plugin):
    def title(self):
        return "unimatch"

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


    def load_unimatch_model(resume, strict_resume=False, feature_channels=128, num_scales=1, upsample_factor=8,
                            num_head=1, ffn_dim_expansion=4, num_transformer_layers=6, reg_refine=False, task='depth'):
        model = UniMatch(feature_channels=feature_channels,
                         num_scales=num_scales,
                         upsample_factor=upsample_factor,
                         num_head=num_head,
                         ffn_dim_expansion=ffn_dim_expansion,
                         num_transformer_layers=num_transformer_layers,
                         reg_refine=reg_refine,
                         task=task)

        loc = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(resume, map_location=loc)
        model.load_state_dict(checkpoint['model'], strict=strict_resume)

        if torch.cuda.is_available():
            model.cuda()

        return model

    @plugjob
    def dummy(self, j:JobArgs):
        pass


    def get_depth(image, model, padding_factor=16, inference_size=None, attn_type='swin',
                  attn_splits_list=[2], prop_radius_list=[-1], num_depth_candidates=64, num_reg_refine=1,
                  min_depth=0.5, max_depth=10, depth_from_argmax=False, pred_bidir_depth=False, output_path='output'):
        # Apply data augmentation
        transform = augmentation.Compose([
            augmentation.ToTensor(),
            augmentation.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        img_ref = transform(image).unsqueeze(0)

        if torch.cuda.is_available():
            img_ref = img_ref.cuda()

        # Estimate depth
        results_dict = inference_depth(model,
                                       img_ref=img_ref,
                                       padding_factor=padding_factor,
                                       inference_size=inference_size,
                                       attn_type=attn_type,
                                       attn_splits_list=attn_splits_list,
                                       prop_radius_list=prop_radius_list,
                                       num_depth_candidates=num_depth_candidates,
                                       num_reg_refine=num_reg_refine,
                                       min_depth=min_depth,
                                       max_depth=max_depth,
                                       depth_from_argmax=depth_from_argmax,
                                       pred_bidir_depth=pred_bidir_depth,
                                       output_path=output_path)

        print(results_dict)

@torch.no_grad()
def inference_depth(model,
                    inference_dir=None,
                    output_path='output',
                    padding_factor=16,
                    inference_size=None,
                    attn_type='swin',
                    attn_splits_list=None,
                    prop_radius_list=None,
                    num_reg_refine=1,
                    num_depth_candidates=64,
                    min_depth=0.5,
                    max_depth=10,
                    depth_from_argmax=False,
                    pred_bidir_depth=False,
                    ):
    model.eval()

    val_transform_list = [augmentation.ToTensor(),
                          augmentation.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                          ]

    val_transform = augmentation.Compose(val_transform_list)

    valid_samples = 0

    fixed_inference_size = inference_size

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # assume scannet dataset file structure
    imgs = sorted(glob(os.path.join(inference_dir, 'color', '*.jpg')) +
                  glob(os.path.join(inference_dir, 'color', '*.png')))
    poses = sorted(glob(os.path.join(inference_dir, 'pose', '*.txt')))

    intrinsics_file = glob(os.path.join(inference_dir, 'intrinsic', '*.txt'))[0]

    assert len(imgs) == len(poses)

    num_samples = len(imgs)

    for i in range(len(imgs) - 1):
        if i % 50 == 0:
            print('=> Predicting %d/%d' % (i, num_samples))

        img_ref = np.array(Image.open(imgs[i]).convert('RGB')).astype(np.float32)
        img_tgt = np.array(Image.open(imgs[i + 1]).convert('RGB')).astype(np.float32)

        intrinsics = np.loadtxt(intrinsics_file).astype(np.float32).reshape((4, 4))[:3, :3]  # [3, 3]

        pose_ref = np.loadtxt(poses[i], delimiter=' ').astype(np.float32).reshape((4, 4))
        pose_tgt = np.loadtxt(poses[i + 1], delimiter=' ').astype(np.float32).reshape((4, 4))
        # relative pose
        pose = np.linalg.inv(pose_tgt) @ pose_ref

        sample = {'img_ref': img_ref,
                  'img_tgt': img_tgt,
                  'intrinsics': intrinsics,
                  'pose': pose,
                  }
        sample = val_transform(sample)

        img_ref = sample['img_ref'].to(device).unsqueeze(0)  # [1, 3, H, W]
        img_tgt = sample['img_tgt'].to(device).unsqueeze(0)  # [1, 3, H, W]
        intrinsics = sample['intrinsics'].to(device).unsqueeze(0)  # [1, 3, 3]
        pose = sample['pose'].to(device).unsqueeze(0)  # [1, 4, 4]

        nearest_size = [int(np.ceil(img_ref.size(-2) / padding_factor)) * padding_factor,
                        int(np.ceil(img_ref.size(-1) / padding_factor)) * padding_factor]

        # resize to nearest size or specified size
        inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

        ori_size = img_ref.shape[-2:]

        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            img_ref = F.interpolate(img_ref, size=inference_size, mode='bilinear',
                                    align_corners=True)
            img_tgt = F.interpolate(img_tgt, size=inference_size, mode='bilinear',
                                    align_corners=True)

        valid_samples += 1

        with torch.no_grad():
            pred_depth = model(img_ref, img_tgt,
                               attn_type=attn_type,
                               attn_splits_list=attn_splits_list,
                               prop_radius_list=prop_radius_list,
                               num_reg_refine=num_reg_refine,
                               intrinsics=intrinsics,
                               pose=pose,
                               min_depth=1. / max_depth,
                               max_depth=1. / min_depth,
                               num_depth_candidates=num_depth_candidates,
                               pred_bidir_depth=pred_bidir_depth,
                               depth_from_argmax=depth_from_argmax,
                               task='depth',
                               )['flow_preds'][-1]  # [1, H, W]

        # remove padding
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            # resize back
            pred_depth = F.interpolate(pred_depth.unsqueeze(1), size=ori_size, mode='bilinear',
                                       align_corners=True).squeeze(1)  # [1, H, W]

        pr_depth = pred_depth[0]

        filename = os.path.join(output_path, os.path.basename(imgs[i])[:-4] + '.png')
        viz_inv_depth = viz_depth_tensor(1. / pr_depth.cpu(),
                                         return_numpy=True)  # [H, W, 3] uint8
        Image.fromarray(viz_inv_depth).save(filename)

        if pred_bidir_depth:
            assert pred_depth.size(0) == 2

            pr_depth_bwd = pred_depth[1]

            filename = os.path.join(output_path, os.path.basename(imgs[i])[:-4] + '_bwd.png')
            viz_inv_depth = viz_depth_tensor(1. / pr_depth_bwd.cpu(),
                                             return_numpy=True)  # [H, W, 3] uint8
            Image.fromarray(viz_inv_depth).save(filename)

    print('Done!')

def viz_depth_tensor(disp, return_numpy=False, colormap='plasma'):
    import matplotlib as mpl
    import matplotlib.cm as cm

    # visualize inverse depth
    assert isinstance(disp, torch.Tensor)

    disp = disp.numpy()
    vmax = np.percentile(disp, 95)
    normalizer = mpl.colors.Normalize(vmin=disp.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=colormap)
    colormapped_im = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(np.uint8)  # [H, W, 3]

    if return_numpy:
        return colormapped_im

    viz = torch.from_numpy(colormapped_im).permute(2, 0, 1)  # [3, H, W]

    return viz
