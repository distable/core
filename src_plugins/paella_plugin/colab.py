import os
import time
import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO
from Paella.modules import DenoiseUNet
import open_clip
from open_clip import tokenizer
from rudalle import get_vae
from einops import rearrange

from src_core.lib import devices

cpu = torch.device("cpu")
# device = cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
clip_device = cpu

model = None
vqmodel = None
clip_model = None

def showmask(mask):
    plt.axis("off")
    plt.imshow(torch.cat([
        torch.cat([i for i in mask[0:1].cpu()], dim=-1),
    ], dim=-2).cpu())
    plt.show()

def saveimages(imgs, name, **kwargs):
    name = name.replace(" ", "_").replace(".", "")
    path = os.path.join("outputs", name + ".jpg")
    while os.path.exists(path):
        base, ext = path.split(".")
        num = base.split("_")[-1]
        if num.isdigit():
            num = int(num) + 1
            base = "_".join(base.split("_")[:-1])
        else:
            num = 0
        path = base + "_" + str(num) + "." + ext
    torchvision.utils.save_image(imgs, path, **kwargs)


def log(t, eps=1e-20):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)

def sample(model, c, x=None, mask=None, T=12, size=(32, 32), starting_t=0, temp_range=[1.0, 1.0], typical_filtering=True, typical_mass=0.2, typical_min_tokens=1, classifier_free_scale=-1, renoise_steps=11, renoise_mode='start'):
    with torch.inference_mode():
        r_range = torch.linspace(0, 1, T+1)[:-1][:, None].expand(-1, c.size(0)).to(c.device)
        temperatures = torch.linspace(temp_range[0], temp_range[1], T)
        preds = []
        if x is None:
            x = torch.randint(0, model.num_labels, size=(c.size(0), *size), device=c.device)
        elif mask is not None:
            noise = torch.randint(0, model.num_labels, size=(c.size(0), *size), device=c.device)
            x = noise * mask + (1-mask) * x
        init_x = x.clone()
        for i in tqdm(range(starting_t, T)):
            if renoise_mode == 'prev':
                prev_x = x.clone()
            r, temp = r_range[i], temperatures[i]
            logits = model(x, c, r)
            if classifier_free_scale >= 0:
                logits_uncond = model(x, torch.zeros_like(c), r)
                logits = torch.lerp(logits_uncond, logits, classifier_free_scale)
            x = logits
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.size(1))
            if typical_filtering:
                x_flat_norm = torch.nn.functional.log_softmax(x_flat, dim=-1)
                x_flat_norm_p = torch.exp(x_flat_norm)
                entropy = -(x_flat_norm * x_flat_norm_p).nansum(-1, keepdim=True)

                c_flat_shifted = torch.abs((-x_flat_norm) - entropy)
                c_flat_sorted, x_flat_indices = torch.sort(c_flat_shifted, descending=False)
                x_flat_cumsum = x_flat.gather(-1, x_flat_indices).softmax(dim=-1).cumsum(dim=-1)

                last_ind = (x_flat_cumsum < typical_mass).sum(dim=-1)
                sorted_indices_to_remove = c_flat_sorted > c_flat_sorted.gather(1, last_ind.view(-1, 1))
                if typical_min_tokens > 1:
                    sorted_indices_to_remove[..., :typical_min_tokens] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, x_flat_indices, sorted_indices_to_remove)
                x_flat = x_flat.masked_fill(indices_to_remove, -float("Inf"))
            # x_flat = torch.multinomial(x_flat.div(temp).softmax(-1), num_samples=1)[:, 0]
            x_flat = gumbel_sample(x_flat, temperature=temp)
            x = x_flat.view(x.size(0), *x.shape[2:])
            if mask is not None:
                x = x * mask + (1-mask) * init_x
            if i < renoise_steps:
                if renoise_mode == 'start':
                    x, _ = model.add_noise(x, r_range[i+1], random_x=init_x)
                elif renoise_mode == 'prev':
                    x, _ = model.add_noise(x, r_range[i+1], random_x=prev_x)
                else: # 'rand'
                    x, _ = model.add_noise(x, r_range[i+1])
            preds.append(x.detach())
    return preds


def encode(x):
    return vqmodel.model.encode((2 * x - 1))[-1][-1]


def decode(img_seq, shape=(32, 32)):
    img_seq = img_seq.view(img_seq.shape[0], -1)
    b, n = img_seq.shape
    one_hot_indices = torch.nn.functional.one_hot(img_seq, num_classes=vqmodel.num_tokens).float()
    z = (one_hot_indices @ vqmodel.model.quantize.embed.weight)
    z = rearrange(z, 'b (h w) c -> b c h w', h=shape[0], w=shape[1])
    img = vqmodel.model.decode(z)
    img = (img.clamp(-1., 1.) + 1) * 0.5
    return img

def torchvis_to_pil(tensor, grid=True):
    # copied from torchvision.utils.save_image
    if grid:
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = make_grid(tensor).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        return Image.fromarray(ndarr)
    else:
        for i,img in enumerate(tensor):
            ndarr = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            return Image.fromarray(ndarr)

def load(ckpt_path):
    global model, vqmodel, clip_model
    global clip_preprocess, preprocess

    devices.set(devices.get_optimal_device(), 'full')

    os.makedirs("outputs", exist_ok=True)

    vqmodel = get_vae().to(device)
    vqmodel.eval().requires_grad_(False)

    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
    clip_model = clip_model.to(clip_device).eval().requires_grad_(False)

    clip_preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        # torchvision.transforms.CenterCrop(256),
        torchvision.transforms.ToTensor(),
    ])

    state_dict = torch.load(str(ckpt_path), map_location=device)
    # state_dict = torch.load("./models/f8_img_40000.pt", map_location=device)
    model = DenoiseUNet(num_labels=8192).to(device)
    model.load_state_dict(state_dict)
    model.eval().requires_grad_()
    print()


def txt2img(prompt:str):
    batch_size = 1
    latent_shape = (4, 4)
    tokenized_text = tokenizer.tokenize([prompt] * batch_size).to(device)
    with torch.inference_mode():
        with devices.autocast():
            clip_embeddings = clip_model.encode_text(tokenized_text.to(clip_device))
            clip_embeddings = clip_embeddings.to(device)

            s = time.time()
            sampled = sample(model, clip_embeddings, T=12, size=latent_shape, starting_t=0, temp_range=[1.0, 1.0],
                             typical_filtering=True, typical_mass=0.2, typical_min_tokens=1, classifier_free_scale=5, renoise_steps=11,
                             renoise_mode="start")
            print(time.time() - s)
        sampled = decode(sampled[-1], latent_shape)

    return torchvis_to_pil(sampled)


def txt2img_interpolate(prompt1:str, prompt2:str, t:float):
    mode = "interpolation"
    # text = "surreal painting of a yellow tulip. artstation"
    # text2 = "surreal painting of a red tulip. artstation"
    text_encoded = tokenizer.tokenize([prompt1]).to(device)
    text2_encoded = tokenizer.tokenize([prompt2]).to(device)
    with torch.inference_mode():
        with devices.autocast():
            clip_embeddings = clip_model.encode_text(text_encoded).float()
            clip_embeddings2 = clip_model.encode_text(text2_encoded).float()

            l = torch.linspace(0, 1, 10).to(device)
            embeddings = []
            for i in l:
                lerp = torch.lerp(clip_embeddings, clip_embeddings2, i)
                embeddings.append(lerp)
            embeddings = torch.cat(embeddings)

            s = time.time()
            sampled = sample(model, embeddings, T=12, size=(32, 32), starting_t=0, temp_range=[1.0, 1.0],
                             typical_filtering=True, typical_mass=0.2, typical_min_tokens=1, classifier_free_scale=4, renoise_steps=11)
            print(time.time() - s)
        sampled = decode(sampled[-1])

    return sampled
    # showimages(sampled)
    # saveimages(sampled, mode + "_" + text + "_" + text2, nrow=len(sampled))

def txt2img_interpolate2(prompt1, prompt2):
    text = "High quality front portrait photo of a tiger."
    text2 = "High quality front portrait photo of a dog."
    text_encoded = tokenizer.tokenize([text]).to(device)
    text2_encoded = tokenizer.tokenize([text2]).to(device)
    with torch.inference_mode():
        with devices.autocast():
            clip_embeddings = clip_model.encode_text(text_encoded).float()
            clip_embeddings2 = clip_model.encode_text(text2_encoded).float()

            l = torch.linspace(0, 1, 10).to(device)
            s = time.time()
            outputs = []
            for i in l:
                # lerp = torch.lerp(clip_embeddings, clip_embeddings2, i)
                low, high = clip_embeddings, clip_embeddings2
                low_norm = low / torch.norm(low, dim=1, keepdim=True)
                high_norm = high / torch.norm(high, dim=1, keepdim=True)
                omega = torch.acos((low_norm * high_norm).sum(1)).unsqueeze(1)
                so = torch.sin(omega)
                lerp = (torch.sin((1.0 - i) * omega) / so) * low + (torch.sin(i * omega) / so) * high
                with torch.random.fork_rng():
                    torch.random.manual_seed(32)
                    sampled = sample(model, lerp, T=12, size=(32, 32), starting_t=0, temp_range=[1.0, 1.0],
                                     typical_filtering=True, typical_mass=0.2, typical_min_tokens=1, classifier_free_scale=5, renoise_steps=11)
                    outputs.append(sampled[-1])
            print(time.time() - s)
        sampled = torch.cat(outputs)
        sampled = decode(sampled)

    return sampled

def txt2img_multiprompt(prompts:list[str]):
    batch_size = 4
    latent_shape = (32, 100)
    conditions = [
        ["an oil painting of a lighthouse standing on a hill", 20],
        ["an oil painting of a majestic boat sailing on the water during a storm. front view", 60],
        ["an oil painting of a majestic castle standing by the water", 100],
    ]
    clip_embedding = torch.zeros(batch_size, 1024, *latent_shape).to(device)
    last_pos = 0
    for text, pos in conditions:
        tokenized_text = tokenizer.tokenize([text] * batch_size).to(device)
        part_clip_embedding = clip_model.encode_text(tokenized_text).float()[:, :, None, None]
        print(f"{last_pos}:{pos}={text}")
        clip_embedding[:, :, :, last_pos:pos] = part_clip_embedding
        last_pos = pos
    with torch.inference_mode():
        with devices.autocast():
            sampled = sample(model, clip_embedding, T=12, size=latent_shape, starting_t=0, temp_range=[1.0, 1.0],
                             typical_filtering=True, typical_mass=0.2, typical_min_tokens=1, classifier_free_scale=5, renoise_steps=11,
                             renoise_mode="start")
        sampled = decode(sampled[-1], latent_shape)

    return sampled
    # showimages(sampled)
    # saveimages(sampled, mode + "_" + ":".join(list(map(lambda x: x[0], conditions))), nrow=batch_size)

def txt2img_multiprompt2(prompts:list[str]):
    batch_size = 4
    latent_shape = (32, 32)
    text_a = "a cute portrait of a dog"
    text_b = "a cute portrait of a cat"
    mode = "vertical"
    # mode = "horizontal"
    text = tokenizer.tokenize([text_a, text_b] * batch_size).to(device)

    with torch.inference_mode():
        with devices.autocast():
            clip_embeddings = clip_model.encode_text(text).float()[:, :, None, None].expand(-1, -1, latent_shape[0], latent_shape[1])
            if mode == 'vertical':
                interp_mask = torch.linspace(0, 1, latent_shape[0], device=device)[None, None, :, None].expand(batch_size, 1, -1, latent_shape[1])
            else:
                interp_mask = torch.linspace(0, 1, latent_shape[1], device=device)[None, None, None, :].expand(batch_size, 1, latent_shape[0], -1)
            # LERP
            clip_embeddings = clip_embeddings[0::2] * (1 - interp_mask) + clip_embeddings[1::2] * interp_mask
            # # SLERP
            # low, high = clip_embeddings[0::2], clip_embeddings[1::2]
            # low_norm = low/torch.norm(low, dim=1, keepdim=True)
            # high_norm = high/torch.norm(high, dim=1, keepdim=True)
            # omega = torch.acos((low_norm*high_norm).sum(1)).unsqueeze(1)
            # so = torch.sin(omega)
            # clip_embeddings = (torch.sin((1.0-interp_mask)*omega)/so)*low + (torch.sin(interp_mask*omega)/so) * high

            sampled = sample(model, clip_embeddings, T=12, size=latent_shape, starting_t=0, temp_range=[1.0, 1.0],
                             typical_filtering=True, typical_mass=0.2, typical_min_tokens=1, classifier_free_scale=5, renoise_steps=11,
                             renoise_mode="start")
        sampled = decode(sampled[-1], latent_shape)

    return sampled

def img2img_inpaint():
    mode = "inpainting"
    text = "a delicious spanish paella"
    tokenized_text = tokenizer.tokenize([text] * images.shape[0]).to(device)
    with torch.inference_mode():
        with devices.autocast():
            # clip_embeddings = clip_model.encode_image(clip_preprocess(images)).float() # clip_embeddings = clip_model.encode_text(text).float()
            clip_embeddings = clip_model.encode_text(tokenized_text).float()
            encoded_tokens = encode(images)
            latent_shape = encoded_tokens.shape[1:]
            mask = torch.zeros_like(encoded_tokens)
            mask[:,5:28,5:28] = 1
            sampled = sample(model, clip_embeddings, x=encoded_tokens, mask=mask, T=12, size=latent_shape, starting_t=0, temp_range=[1.0, 1.0],
                   typical_filtering=True, typical_mass=0.2, typical_min_tokens=1, classifier_free_scale=6, renoise_steps=11)
        sampled = decode(sampled[-1], latent_shape)

    showimages(images[0:1], height=10, width=10)
    showmask(mask[0:1])
    showimages(sampled, height=16, width=16)
    saveimages(torch.cat([images[0:1], sampled]), mode + "_" + text, nrow=images.shape[0]+1)

def img2img_outpaint():
    mode = "outpainting"
    size = (40, 64)
    top_left = (0, 16)
    text = "black & white photograph of a rocket from the bottom."
    tokenized_text = tokenizer.tokenize([text] * images.shape[0]).to(device)
    with torch.inference_mode():
        with devices.autocast():
            # clip_embeddings = clip_model.encode_image(clip_preprocess(images)).float()
            clip_embeddings = clip_model.encode_text(tokenized_text).float()
            encoded_tokens = encode(images)
            canvas = torch.zeros((images.shape[0], *size), dtype=torch.long).to(device)
            canvas[:, top_left[0]:top_left[0] + encoded_tokens.shape[1], top_left[1]:top_left[1] + encoded_tokens.shape[2]] = encoded_tokens
            mask = torch.ones_like(canvas)
            mask[:, top_left[0]:top_left[0] + encoded_tokens.shape[1], top_left[1]:top_left[1] + encoded_tokens.shape[2]] = 0
            sampled = sample(model, clip_embeddings, x=canvas, mask=mask, T=12, size=size, starting_t=0, temp_range=[1.0, 1.0],
                             typical_filtering=True, typical_mass=0.2, typical_min_tokens=1, classifier_free_scale=4, renoise_steps=11)
        sampled = decode(sampled[-1], size)

    showimages(images[0:1], height=10, width=10)
    showmask(mask[0:1])
    showimages(sampled, height=16, width=16)
    saveimages(sampled, mode + "_" + text, nrow=images.shape[0])

def img2img_structuralmorph():
    mode = "morphing"
    max_steps = 24
    init_step = 8

    text = "A fox posing for a photo. stock photo. highly detailed. 4k"

    with torch.inference_mode():
        with devices.autocast():
            # images = preprocess(Image.open("data/city sketch.png")).unsqueeze(0).expand(4, -1, -1, -1).to(device)[:, :3]
            latent_image = encode(images)
            latent_shape = latent_image.shape[-2:]
            r = torch.ones(latent_image.size(0), device=device) * (init_step / max_steps)
            noised_latent_image, _ = model.add_noise(latent_image, r)

            tokenized_text = tokenizer.tokenize([text] * images.size(0)).to(device)
            clip_embeddings = clip_model.encode_text(tokenized_text).float()

            sampled = sample(model, clip_embeddings, x=noised_latent_image, T=max_steps, size=latent_shape, starting_t=init_step, temp_range=[1.0, 1.0],
                             typical_filtering=True, typical_mass=0.2, typical_min_tokens=1, classifier_free_scale=6, renoise_steps=max_steps - 1,
                             renoise_mode="prev")
        sampled = decode(sampled[-1], latent_shape)
    showimages(sampled)
    showimages(images)
    saveimages(torch.cat([images[0:1], sampled]), mode + "_" + text, nrow=images.shape[0] + 1)

def img2img_variation(images):
    clip_preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    latent_shape = (32, 32)
    with torch.inference_mode():
        with devices.autocast():
            clip_embeddings = clip_model.encode_image(clip_preprocess(images)).float()  # clip_embeddings = clip_model.encode_text(text).float()
            sampled = sample(model, clip_embeddings, T=12, size=latent_shape, starting_t=0, temp_range=[1.0, 1.0],
                             typical_filtering=True, typical_mass=0.2, typical_min_tokens=1, classifier_free_scale=5, renoise_steps=11)
        sampled = decode(sampled[-1], latent_shape)

    showimages(images)
    showimages(sampled)