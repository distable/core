# import os
#
# from PIL import Image, ImageOps, ImageChops
#
# from processing import Processed, StableDiffusionProcessingImg2Img, process_images
# from SDPlugin import opts
# import SDPlugin as SDPlugin
# import processing as processing
#
#
# def img2img(mode: int,
#             prompt: str,
#             negative_prompt: str,
#             prompt_style: str,
#             prompt_style2: str,
#             init_img,
#             init_img_with_mask,
#             init_img_inpaint,
#             init_mask_inpaint,
#             mask_mode,
#             steps: int,
#             sampler_index: int,
#             mask_blur: int,
#             inpainting_fill: int,
#             restore_faces: bool,
#             tiling: bool,
#             n_iter: int,
#             batch_size: int,
#             cfg_scale: float,
#             denoising_strength: float,
#             seed: int,
#             subseed: int,
#             subseed_strength: float,
#             seed_resize_from_h: int,
#             seed_resize_from_w: int,
#             seed_enable_extras: bool,
#             height: int,
#             width: int,
#             resize_mode: int,
#             inpaint_full_res: bool,
#             inpaint_full_res_padding: int,
#             inpainting_mask_invert: int,
#             img2img_batch_input_dir: str,
#             img2img_batch_output_dir: str,
#             *args):
#     is_inpaint = mode == 1
#     is_batch = mode == 2
#
#     if is_inpaint:
#         if mask_mode == 0:
#             image = init_img_with_mask['image']
#             mask = init_img_with_mask['mask']
#             alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
#             mask = ImageChops.lighter(alpha_mask, mask.convert('L')).convert('L')
#             image = image.convert('RGB')
#         else:
#             image = init_img_inpaint
#             mask = init_mask_inpaint
#     else:
#         image = init_img
#         mask = None
#
#     assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'
#
#     p = StableDiffusionProcessingImg2Img(
#         sd_model=SDPlugin.sd_model,
#         outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
#         outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
#         prompt=prompt,
#         negative_prompt=negative_prompt,
#         styles=[prompt_style, prompt_style2],
#         seed=seed,
#         subseed=subseed,
#         subseed_strength=subseed_strength,
#         seed_resize_from_h=seed_resize_from_h,
#         seed_resize_from_w=seed_resize_from_w,
#         seed_enable_extras=seed_enable_extras,
#         sampler_index=sampler_index,
#         batch_size=batch_size,
#         n_iter=n_iter,
#         steps=steps,
#         cfg_scale=cfg_scale,
#         width=width,
#         height=height,
#         restore_faces=restore_faces,
#         tiling=tiling,
#         init_images=[image],
#         mask=mask,
#         mask_blur=mask_blur,
#         inpainting_fill=inpainting_fill,
#         resize_mode=resize_mode,
#         denoising_strength=denoising_strength,
#         inpaint_full_res=inpaint_full_res,
#         inpaint_full_res_padding=inpaint_full_res_padding,
#         inpainting_mask_invert=inpainting_mask_invert,
#     )
#
#     p.script_args = args
#
#     if SDPlugin.cmd_opts.enable_console_prompts:
#         print(f"\nimg2img: {prompt}", file=SDPlugin.progress_print_out)
#
#     p.extra_generation_params["Mask blur"] = mask_blur
#
#     if is_batch:
#         assert not SDPlugin.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"
#
#         processing.fix_seed(p)
#         images1 = [file for file in [os.path.join(img2img_batch_input_dir, x1) for x1 in os.listdir(img2img_batch_input_dir)] if os.path.isfile(file)]
#
#         print(f"Will process {len(images1)} images, creating {p.n_iter * p.batch_size} new images for each.")
#         save_normally = img2img_batch_output_dir == ''
#         p.do_not_save_grid = True
#         p.do_not_save_samples = not save_normally
#         for i, image1 in enumerate(images1):
#             # if state.skipped:
#             #     state.skipped = False
#
#             # if state.interrupted:
#             #     break
#
#             img = Image.open(image1)
#             p.init_images = [img] * p.batch_size
#
#             proc = process_images(p)
#             for n, processed_image in enumerate(proc.images):
#                 filename = os.path.basename(image1)
#
#                 if n > 0:
#                     left, right = os.path.splitext(filename)
#                     filename = f"{left}-{n}{right}"
#
#                 if not save_normally:
#                     processed_image.save(os.path.join(img2img_batch_output_dir, filename))
#
#         processed = Processed(p, [], p.seed, "")
#     else:
#         processed = process_images(p)
#
#     return processed.images
#
#
