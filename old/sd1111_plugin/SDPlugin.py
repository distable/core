import argparse
import csv
import datetime
import html
import os
from datetime import datetime

import numpy as np
import torch
import tqdm
from PIL import Image, PngImagePlugin

import src_core.paths
import src_plugins.sd1111_plugin

import src_plugins.sd1111_plugin.SDState
from src_core import plugins
from src_core.plugins import Plugin, plugjob
from src_plugins.sd1111_plugin import devices, modelsplit, prompt_parser, sd_hijack, sd_hypernetwork, sd_models, SDJob, SDState
from src_plugins.sd1111_plugin.image_embedding import caption_image_overlay, embedding_to_b64, insert_image_data_embed
from src_plugins.sd1111_plugin.sd_hypernetwork import Hypernetwork, hypernetworks, log_statistics, report_statistics, stack_conds
from src_plugins.sd1111_plugin.SD_img2img import SD_img2img
from src_plugins.sd1111_plugin.sd_textinv_dataset import PersonalizedBase
from src_plugins.sd1111_plugin.sd_textinv_learn_schedule import LearnRateScheduler
from src_plugins.sd1111_plugin.sd_txt2img import sd_txt2img
from src_plugins.sd1111_plugin.SDAttention import SDAttention
from src_plugins.sd1111_plugin.SDJob import get_fixed_seed
from src_plugins.sd1111_plugin.SDState import attention


def res(join=""):
    return plugins.get('sd1111_plugin').res(join)


class SDPlugin(Plugin):
    # TODO os env variables for no reason, enjoy
    taming_transformers_commit_hash = os.environ.get('TAMING_TRANSFORMERS_COMMIT_HASH', "24268930bf1dce879235a7fddd0b2355b84d7ea6")
    stable_diffusion_commit_hash = os.environ.get('STABLE_DIFFUSION_COMMIT_HASH', "69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc")
    k_diffusion_commit_hash = os.environ.get('K_DIFFUSION_COMMIT_HASH', "f4e99857772fc3a126ba886aadf795a332774878")

    def title(self):
        return "Stable Diffusion AUTO1111"

    def describe(self):
        return "Stable Diffusion plugin adapted from AUTOMATIC1111's code."

    def init(self):
        sd_hijack.model_hijack.init()
        src_plugins.sd1111_plugin.SDState.instance = self

    def install(self):
        self.gitclone("https://github.com/CompVis/taming-transformers.git", "taming-transformers", SDPlugin.taming_transformers_commit_hash)
        self.gitclone("https://github.com/CompVis/stable-diffusion.git", 'stable_diffusion', SDPlugin.stable_diffusion_commit_hash)
        self.gitclone("https://github.com/crowsonkb/k-diffusion.git", 'k-diffusion', SDPlugin.k_diffusion_commit_hash)

        # TODO install xformers if enabled
        if attention == SDAttention.XFORMERS:
            pass

    def load(self):
        # Interrogate
        # import interrogate
        # interrogator = interrogate.InterrogateModels("interrogate")

        sd_hypernetwork.discover_hypernetworks(res("hypernetworks"))
        sd_models.discover_sdmodels()
        sd_models.load_sdmodel()

        # codeformer.setup_model(cmd_opts.codeformer_models_path)
        # gfpgan.setup_model(cmd_opts.gfpgan_models_path)
        # SDPlugin.face_restorers.append(modules.face_restoration.FaceRestoration())
        # modelloader.load_upscalers()

    @plugjob
    def select_hn(self, name: str):
        sd_hypernetwork.load_hypernetwork(name)

    @plugjob
    def txt2img(self, job:sd_txt2img):
        return run(job)

    @plugjob
    def img2img(self, job:SD_img2img):
        return run(job)


def run(p: SDJob):
    if type(p.prompt) == list:
        assert (len(p.prompt) > 0)
    else:
        assert p.prompt is not None

    devices.torch_gc()

    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)

    sd_hijack.model_hijack.apply_circular(p.tiling)
    sd_hijack.model_hijack.clear_comments()

    # SDPlugin.prompt_styles.apply_styles(p)

    if type(p.prompt) == list:
        p.all_prompts = p.prompt
    else:
        p.all_prompts = p.batch_size * [p.prompt]

    if type(seed) == list:
        p.all_seeds = seed
    else:
        p.all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(p.all_prompts))]

    if type(subseed) == list:
        p.all_subseeds = subseed
    else:
        p.all_subseeds = [int(subseed) + x for x in range(len(p.all_prompts))]

    # if os.path.exists(SDPlugin.embeddings_dir) and not p.do_not_reload_embeddings:
    #     modules.stable_diffusion_auto2222.sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()

    output_images = []

    with torch.no_grad(), SDState.sdmodel.ema_scope():
        with devices.autocast():
            p.init(SDState.sdmodel, p.all_prompts, p.all_seeds, p.all_subseeds)

        # if state.skipped:
        #     state.skipped = False
        # if state.interrupted:
        #     break

        prompts = p.all_prompts
        seeds = p.all_seeds
        subseeds = p.all_subseeds

        if len(prompts) == 0:
            return

        with devices.autocast():
            uc = prompt_parser.get_learned_conditioning(SDState.sdmodel, len(prompts) * [p.promptneg], p.steps)
            c = prompt_parser.get_multicond_learned_conditioning(SDState.sdmodel, prompts, p.steps)

        with devices.autocast():
            samples_ddim = p.sample(conditioning=c, unconditional_conditioning=uc, seeds=seeds, subseeds=subseeds, subseed_strength=p.subseed_strength)

        samples_ddim = samples_ddim.to(devices.dtype_vae)
        x_samples_ddim = decode_first_stage(SDState.sdmodel, samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

        del samples_ddim

        if SDState.lowvram or SDState.medvram:
            modelsplit.send_everything_to_cpu()

        devices.torch_gc()

        # if opts.filter_nsfw:
        #     import modules.safety as safety
        #     x_samples_ddim = modules.safety.censor_batch(x_samples_ddim)

        for i, x_sample in enumerate(x_samples_ddim):
            x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
            x_sample = x_sample.astype(np.uint8)

            image = Image.fromarray(x_sample)

            # if opts.samples_save and not p.do_not_save_samples:
            #     images.save_image(image, p.outpath_samples, "", seeds[i], prompts[i], opts.samples_format, metadata=infotext(n, i), p=p)

            output_images.append(image)

        del x_samples_ddim

        devices.torch_gc()

    devices.torch_gc()
    return output_images

    def train_hypernetwork(hypernetwork_name,
                           learn_rate,
                           batch_size,
                           data_root,
                           log_directory,
                           training_width,
                           training_height,
                           steps,
                           create_image_every,
                           save_hypernetwork_every,
                           template_file,
                           preview_from_txt2img,
                           preview_prompt,
                           preview_negative_prompt,
                           preview_steps,
                           preview_sampler_index,
                           preview_cfg_scale,
                           preview_seed,
                           preview_width, preview_height):
        # images allows training previews to have infotext. Importing it at the top causes a circular import problem.
        from src_plugins.sd1111_plugin import images

        assert hypernetwork_name, 'hypernetwork not selected'

        path = hypernetworks.get(hypernetwork_name, None)
        src_plugins.sd1111_plugin.SDState.hnmodel = Hypernetwork()
        src_plugins.sd1111_plugin.SDState.hnmodel.load(path)

        # SDPlugin.state.textinfo = "Initializing hypernetwork training..."
        # SDPlugin.state.job_count = steps

        filename = os.path.join(src_core.paths.plug_hypernetworks, f'{hypernetwork_name}.pt')

        log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), hypernetwork_name)
        unload = src_plugins.sd1111_plugin.options.opts.unload_models_when_training

        if save_hypernetwork_every > 0:
            hypernetwork_dir = os.path.join(log_directory, "hypernetworks")
            os.makedirs(hypernetwork_dir, exist_ok=True)
        else:
            hypernetwork_dir = None

        if create_image_every > 0:
            images_dir = os.path.join(log_directory, "images")
            os.makedirs(images_dir, exist_ok=True)
        else:
            images_dir = None

        # SDPlugin.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
        with torch.autocast("cuda"):
            ds = PersonalizedBase(data_root=data_root, width=training_width, height=training_height, repeats=src_plugins.sd1111_plugin.options.opts.training_image_repeats_per_epoch, placeholder_token=hypernetwork_name, model=src_plugins.sd1111_plugin.SDState.sdmodel, device=devices.device, template_file=template_file, include_cond=True, batch_size=batch_size)
        if unload:
            src_plugins.sd1111_plugin.SDState.sdmodel.cond_stage_model.to(devices.cpu)
            src_plugins.sd1111_plugin.SDState.sdmodel.first_stage_model.to(devices.cpu)

        hypernetwork = src_plugins.sd1111_plugin.SDState.hnmodel
        weights = hypernetwork.weights()
        for weight in weights:
            weight.requires_grad = True

        size = len(ds.indexes)
        loss_dict = {}
        losses = torch.zeros((size,))
        previous_mean_loss = 0
        print("Mean loss of {} elements".format(size))

        last_saved_file = "<none>"
        last_saved_image = "<none>"
        forced_filename = "<none>"

        ititial_step = hypernetwork.step or 0
        if ititial_step > steps:
            return hypernetwork, filename

        scheduler = LearnRateScheduler(learn_rate, steps, ititial_step)
        # if optimizer == "AdamW": or else Adam / AdamW / SGD, etc...
        optimizer = torch.optim.AdamW(weights, lr=scheduler.learn_rate)

        steps_without_grad = 0

        pbar = tqdm.tqdm(enumerate(ds), total=steps - ititial_step)
        for i, entries in pbar:
            hypernetwork.step = i + ititial_step
            if len(loss_dict) > 0:
                previous_mean_loss = sum(i[-1] for i in loss_dict.values()) / len(loss_dict)

            scheduler.apply(optimizer, hypernetwork.step)
            if scheduler.finished:
                break

            # if SDPlugin.state.interrupted:
            #     break

            with torch.autocast("cuda"):
                c = stack_conds([entry.cond for entry in entries]).to(devices.device)
                # c = torch.vstack([entry.cond for entry in entries]).to(devices.device)
                x = torch.stack([entry.latent for entry in entries]).to(devices.device)
                loss = src_plugins.sd1111_plugin.SDState.sdmodel(x, c)[0]
                del x
                del c

                losses[hypernetwork.step % losses.shape[0]] = loss.item()
                for entry in entries:
                    log_statistics(loss_dict, entry.filename, loss.item())

                optimizer.zero_grad()
                weights[0].grad = None
                loss.backward()

                if weights[0].grad is None:
                    steps_without_grad += 1
                else:
                    steps_without_grad = 0
                assert steps_without_grad < 10, 'no gradient found for the trained weight after backward() for 10 steps in a row; this is a bug; training cannot continue'

                optimizer.step()

            if torch.isnan(losses[hypernetwork.step % losses.shape[0]]):
                raise RuntimeError("Loss diverged.")
            pbar.set_description(f"dataset loss: {previous_mean_loss:.7f}")

            if hypernetwork.step > 0 and hypernetwork_dir is not None and hypernetwork.step % save_hypernetwork_every == 0:
                # Before saving, change name to match current checkpoint.
                hypernetwork.name = f'{hypernetwork_name}-{hypernetwork.step}'
                last_saved_file = os.path.join(hypernetwork_dir, f'{hypernetwork.name}.pt')
                hypernetwork.save(last_saved_file)

            write_loss(log_directory, "hypernetwork_loss.csv", hypernetwork.step, len(ds), {
                "loss"      : f"{previous_mean_loss:.7f}",
                "learn_rate": scheduler.learn_rate
            })

            if hypernetwork.step > 0 and images_dir is not None and hypernetwork.step % create_image_every == 0:
                forced_filename = f'{hypernetwork_name}-{hypernetwork.step}'
                last_saved_image = os.path.join(images_dir, forced_filename)

                optimizer.zero_grad()
                src_plugins.sd1111_plugin.SDState.sdmodel.cond_stage_model.to(devices.device)
                src_plugins.sd1111_plugin.SDState.sdmodel.first_stage_model.to(devices.device)

                p = src_plugins.sd1111_plugin.SDJob_txt.sd_txt2img(
                        sd_model=src_plugins.sd1111_plugin.SDState.sdmodel,
                        do_not_save_grid=True,
                        do_not_save_samples=True,
                )

                if preview_from_txt2img:
                    p.prompt = preview_prompt
                    p.promptneg = preview_negative_prompt
                    p.steps = preview_steps
                    p.sampler_index = preview_sampler_index
                    p.cfg = preview_cfg_scale
                    p.seed = preview_seed
                    p.width = preview_width
                    p.height = preview_height
                else:
                    p.prompt = entries[0].cond_text
                    p.steps = 20

                preview_text = p.prompt

                processed = run(p)
                image = processed.images[0] if len(processed.images) > 0 else None

                if unload:
                    src_plugins.sd1111_plugin.SDState.sdmodel.cond_stage_model.to(devices.cpu)
                    src_plugins.sd1111_plugin.SDState.sdmodel.first_stage_model.to(devices.cpu)

                if image is not None:
                    # SDPlugin.state.current_image = image
                    last_saved_image, last_text_info = images.save_image(image, images_dir, "", p.seed, p.prompt, src_plugins.sd1111_plugin.options.opts.samples_format, processed.infotexts[0], p=p, forced_filename=forced_filename)
                    last_saved_image += f", prompt: {preview_text}"

            # SDPlugin.state.job_no = hypernetwork.step

    #         SDPlugin.state.textinfo = f"""
    # <p>
    # Loss: {previous_mean_loss:.7f}<br/>
    # Step: {hypernetwork.step}<br/>
    # Last prompt: {html.escape(entries[0].cond_text)}<br/>
    # Last saved hypernetwork: {html.escape(last_saved_file)}<br/>
    # Last saved image: {html.escape(last_saved_image)}<br/>
    # </p>
    # """

        report_statistics(loss_dict)
        checkpoint = sd_models.select_checkpoint()

        hypernetwork.sd_checkpoint = checkpoint.hash
        hypernetwork.sd_checkpoint_name = checkpoint.model_name
        # Before saving for the last time, change name back to the base name (as opposed to the save_hypernetwork_every step-suffixed naming convention).
        hypernetwork.name = hypernetwork_name
        filename = os.path.join(src_core.paths.plug_hypernetworks, f'{hypernetwork.name}.pt')
        hypernetwork.save(filename)

        return hypernetwork, filename


def write_loss(log_directory, filename, step, epoch_len, values):
    if src_plugins.sd1111_plugin.options.opts.training_write_csv_every == 0:
        return

    if step % src_plugins.sd1111_plugin.options.opts.training_write_csv_every != 0:
        return

    write_csv_header = False if os.path.exists(os.path.join(log_directory, filename)) else True

    with open(os.path.join(log_directory, filename), "a+", newline='') as fout:
        csv_writer = csv.DictWriter(fout, fieldnames=["step", "epoch", "epoch_step", *(values.keys())])

        if write_csv_header:
            csv_writer.writeheader()

        epoch = step // epoch_len
        epoch_step = step - epoch * epoch_len

        csv_writer.writerow({
            "step"      : step + 1,
            "epoch"     : epoch + 1,
            "epoch_step": epoch_step + 1,
            **values,
        })


def train_embedding(embedding_name, learn_rate, batch_size, data_root, log_directory, training_width, training_height, steps, create_image_every, save_embedding_every, template_file, save_image_with_stored_embedding, preview_from_txt2img, preview_prompt, preview_negative_prompt, preview_steps, preview_sampler_index, preview_cfg_scale, preview_seed, preview_width, preview_height):
    from src_plugins.sd1111_plugin.SDPlugin import SDPlugin
    assert embedding_name, 'embedding not selected'

    SDPlugin.state.textinfo = "Initializing textual inversion training..."
    SDPlugin.state.job_count = steps

    filename = os.path.join(src_core.paths.plug_embeddings, f'{embedding_name}.pt')

    log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), embedding_name)

    if save_embedding_every > 0:
        embedding_dir = os.path.join(log_directory, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)
    else:
        embedding_dir = None

    if create_image_every > 0:
        images_dir = os.path.join(log_directory, "images")
        os.makedirs(images_dir, exist_ok=True)
    else:
        images_dir = None

    if create_image_every > 0 and save_image_with_stored_embedding:
        images_embeds_dir = os.path.join(log_directory, "image_embeddings")
        os.makedirs(images_embeds_dir, exist_ok=True)
    else:
        images_embeds_dir = None

    cond_model = SDPlugin.sdmodel.cond_stage_model

    SDPlugin.state.textinfo = f"Preparing dataset from {html.escape(data_root)}..."
    with torch.autocast("cuda"):
        ds = src_plugins.textual_inversion.dataset.PersonalizedBase(data_root=data_root, width=training_width, height=training_height, repeats=src_plugins.sd1111_plugin.options.opts.training_image_repeats_per_epoch, placeholder_token=embedding_name, model=SDPlugin.sdmodel, device=devices.device, template_file=template_file, batch_size=batch_size)

    from src_plugins.sd1111_plugin import sd_hijack
    hijack = sd_hijack.model_hijack

    embedding = hijack.embedding_db.word_embeddings[embedding_name]
    embedding.vec.requires_grad = True

    losses = torch.zeros((32,))

    last_saved_file = "<none>"
    last_saved_image = "<none>"
    embedding_yet_to_be_embedded = False

    ititial_step = embedding.progress_i or 0
    if ititial_step > steps:
        return embedding, filename

    scheduler = LearnRateScheduler(learn_rate, steps, ititial_step)
    optimizer = torch.optim.AdamW([embedding.vec], lr=scheduler.learn_rate)

    pbar = tqdm.tqdm(enumerate(ds), total=steps - ititial_step)
    for i, entries in pbar:
        embedding.progress_i = i + ititial_step

        scheduler.apply(optimizer, embedding.progress_i)
        if scheduler.finished:
            break

        if SDPlugin.state.interrupted:
            break

        with torch.autocast("cuda"):
            c = cond_model([entry.cond_text for entry in entries])
            x = torch.stack([entry.latent for entry in entries]).to(devices.device)
            loss = SDPlugin.sdmodel(x, c)[0]
            del x

            losses[embedding.progress_i % losses.shape[0]] = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_num = embedding.progress_i // len(ds)
        epoch_step = embedding.progress_i - (epoch_num * len(ds)) + 1

        pbar.set_description(f"[Epoch {epoch_num}: {epoch_step}/{len(ds)}]loss: {losses.mean():.7f}")

        if embedding.progress_i > 0 and embedding_dir is not None and embedding.progress_i % save_embedding_every == 0:
            last_saved_file = os.path.join(embedding_dir, f'{embedding_name}-{embedding.progress_i}.pt')
            embedding.save(last_saved_file)
            embedding_yet_to_be_embedded = True

        write_loss(log_directory, "textual_inversion_loss.csv", embedding.progress_i, len(ds), {
            "loss"      : f"{losses.mean():.7f}",
            "learn_rate": scheduler.learn_rate
        })

        if embedding.progress_i > 0 and images_dir is not None and embedding.progress_i % create_image_every == 0:
            last_saved_image = os.path.join(images_dir, f'{embedding_name}-{embedding.progress_i}.png')

            from src_plugins.sd1111_plugin.sd_txt2img import sd_txt2img
            p = sd_txt2img(
                    sd_model=SDPlugin.sdmodel,
                    do_not_save_grid=True,
                    do_not_save_samples=True,
                    do_not_reload_embeddings=True,
            )

            if preview_from_txt2img:
                p.prompt = preview_prompt
                p.promptneg = preview_negative_prompt
                p.steps = preview_steps
                p.sampler_index = preview_sampler_index
                p.cfg = preview_cfg_scale
                p.seed = preview_seed
                p.width = preview_width
                p.height = preview_height
            else:
                p.prompt = entries[0].cond_text
                p.steps = 20
                p.width = training_width
                p.height = training_height

            preview_text = p.prompt

            processed = run(p)
            image = processed.images[0]

            SDPlugin.state.current_image = image

            if save_image_with_stored_embedding and os.path.exists(last_saved_file) and embedding_yet_to_be_embedded:
                last_saved_image_chunks = os.path.join(images_embeds_dir, f'{embedding_name}-{embedding.progress_i}.png')

                info = PngImagePlugin.PngInfo()
                data = torch.load(last_saved_file)
                info.add_text("sd-ti-embedding", embedding_to_b64(data))

                title = "<{}>".format(data.get('name', '???'))

                try:
                    vectorSize = list(data['string_to_param'].values())[0].shape[0]
                except Exception as e:
                    vectorSize = '?'

                checkpoint = sd_models.select_checkpoint()
                footer_left = checkpoint.model_name
                footer_mid = '[{}]'.format(checkpoint.hash)
                footer_right = '{}v {}s'.format(vectorSize, embedding.progress_i)

                captioned_image = caption_image_overlay(image, title, footer_left, footer_mid, footer_right)
                captioned_image = insert_image_data_embed(captioned_image, data)

                captioned_image.save(last_saved_image_chunks, "PNG", pnginfo=info)
                embedding_yet_to_be_embedded = False

            image.save(last_saved_image)

            last_saved_image += f", prompt: {preview_text}"

        SDPlugin.state.job_no = embedding.progress_i

        SDPlugin.state.textinfo = f"""
<p>
Loss: {losses.mean():.7f}<br/>
Step: {embedding.progress_i}<br/>
Last prompt: {html.escape(entries[0].cond_text)}<br/>
Last saved embedding: {html.escape(last_saved_file)}<br/>
Last saved image: {html.escape(last_saved_image)}<br/>
</p>
"""

    checkpoint = sd_models.select_checkpoint()

    embedding.sd_checkpoint = checkpoint.hash
    embedding.sd_checkpoint_name = checkpoint.model_name
    embedding.cached_checksum = None
    embedding.save(filename)

    return embedding, filename


def decode_first_stage(model, x):
    with devices.autocast(disable=x.dtype == devices.dtype_vae):
        x = model.decode_first_stage(x)

    return x