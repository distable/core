# discore

**stable-core** is a backend server for AI art with plugins and GUIs to use, the ultimate swiss-army knife to media synthesis. Every use case is covered, from ordinary artists to intense creative coders.

**AI art and models, GUIs, animation, prompt engineering, audio-reactivity, iterating, experimenting, rendering, math functions, multi-modality,** everything is covered by stable-core and it all chains together. The amazing entourage effect of our components encourages developers implement all their best ideas and models as a stable-core plugin. 

**under heavy development**

## 🧬 Components

1. **Backend:** The server/client architecture means clients implement UIs or bridge other apps like blender nodes, kdenlive clips, effects, etc.
2. **Sessions:** your outputs are organized and clustered into sessions. A new timestamped session starts when a client connects, and you can re-open old sessions to do more work in them. Could be a prompt engineering session, an animation, a batch of variants, a history for an outpainting project, etc. 
3. **Jobs:** Generate/transform some data with input/output. 
4. **Plugins:** they implement models, packages, techniques, features, handle all installation and cloning in a well-defined manner. They plug into the system by providing jobs, which are simply functions. Create a new plugin generated from a template and you can instantly get to work.
5. **Cloud Deploy:** transparently switch between local & cloud computing with [runpod](https://www.runpod.io/) or [vast.ai](https://vast.ai/).

## 🚀 Installation

NOTE: currently there may be unexpected errors and computers explosion

1. Download and extract into a directory.

2. Setup a minimal `user_conf.py`:

```py
ip = '0.0.0.0'
port = 5000

# Plugins to install
install = ['distable/sd1111_plugin']

# Plugins to load on startup
startup = ['sd1111_plugin']
```

   * Check https://github.com/orgs/distable/repositories for plugins
   * Check the wiki for some fancy user_conf.py examples.

3. Launch run.sh (Linux) or run.bat (Windows)

```log
❯ ./run.sh
Python: 3.10.6 (main, Aug  3 2022, 17:39:45) [GCC 12.1.1 20220730]
Revision: <none>

[core] 1. Downloading plugins
Fetching updates for sd1111_plugin...
Checking out commit for sd1111_plugin with hash: /home/nuck/stable-core/src_plugins...

[core] 2. Initializing plugins
[core] Found 5 jobs:
[core]  - txt2img
[core]  - imagine
[core]  - dream
[core]  - sd1111.img2img
[core]  - sd1111.txt2img

[core] 3. Installing plugins...
[plugin]   - sd1111

[core] 4. Loading plugins...
[plugin]   - sd1111
[core] All ready!
[server] Starting ...
[session] New session: 001_2022-10-29_23-07-55
```

   The server is launched on `127.0.0.0:5000`, you may connect with a GUI or use the shell.


## 🍻 Usage


### 1. Interactive Shell

An [interactive shell](https://github.com/distable/core/wiki#shell) is available out of the box, type `help` to see commands.

```
> Enter commands here or use a client ...

> txt2img p="Woaaa! Kawaii monster by salvador dali"
p=Woaaa! Kawaii monster by salvador dali
100%|███████████████████████████████████████████████| 22/22 [00:12<00:00,  1.71it/s]

```

### 2. Graphical User Interface (GUI)

Some GUI clients are available to connect to the core and use it.

* ImGUI

### 3. Bridges

Bridge are another type of client which allow using the core inside an existing GUI, like Photoshop or Blender.

### 4. Code

You can use the core for creative coding.

```
core.init()

p = core.prompt("A <scale> <glow> galaxy painted by <artist>")

# Create the init image
core.job('txt2img', prompt=p, cfg=7.75, steps=8, sampler='euler-a')

for i in range(1000):
    core.job('img2img', chg=0.65)
    core.job("mat2d", zoom=0.015)
```

## ⚗ Plugins

This is a preview of the end-game plugin ecosystem, how it will look and feel. We encourage community members to contribute and maintain some of these plugins themselves, or make new ones.

* **[AUTO1111 StableDiffusion](https://github.com/distable/sd1111_plugin):** txt2img, img2img, ported from AUTO1111's webui with all optimizations 
* **[HuggingFace Diffusers](https://github.com/distable/sdhug_plugin):** txt2img, img2img
* **VQGAN+CLIP / PyTTI:** txt2img, img2img
* **DiscoDiffusion:** txt2img, img2img
* **StableHorde:** txt2img, serve
* **CLIP Interrogate:** img2txt
* **Dreambooth**: train_ckpt
* **StyleGAN:** train_ckpt, img2img
* **[2D Transforms](https://github.com/distable/math2d_plugin):** simple 2D transforms like translate, rotate, and scale.
* **[3D Transforms](https://github.com/stablecore-ai/math3d_plugin):** 3D transforms using virtual depth like rotating a sphere OR predicted depth from AdaBins+MiDaS. Could implement depth guidance to try and keep the depth more stable.
* **Guidance:** these plugins implement guidance losses into other generation plugins.
   * **CLIP Guidance:** guidance using CLIP models.
   * **Lpips Guidance:** guidance using lpips. _(used in Disco Diffusion to reduce flickering)_
   * **Convolution Guidance:** guidance using convolutions. _(edge_weight in PyTTI)_
* **Audio Analysis:** turn audio inputs into numbers for audio-reactivity, using FFT and stuff like that. Can maybe use Magenta.
* **Palette Match:** img2img, adjust an image's palette to match an input image.
* **Flow Warp:** img2img, displace an image using estimated flow between 2 input images.
* **[Wildcards](distable/wildcard_plugin):** a plugin to add RNG into your prompts.
* **Whisper:** audio2txt
* **MetaPlugin:** a plugin to string other plugins together, either with job macros or straight-up python. Could be done without a plugin but this allows all clients to automatically support these features.
* **Deforum:** txt2img, technically just a macro of other plugins (sorry)
* **LucidSonicDreams:** txt2img, hopefully we can make it adapt to any model



Upscalers:
  * **RealSR:** img2img, on Linux this is easily installed thru AUR with `realsr-ncnn-vulkan`
  * **BasicSR:** img2img, port
  * **LDSR:** img2img
  * **CodeFormer:** img2img, port
  * **GFPGAN:** img2img, port

Clients:
   * ImGUI
   * Gradio
   * Blender
