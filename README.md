# stable-core

**stable-core** is a backend server for AI art with plugins and UIs, your swiss-army knife and package manager to media synthesis. We cover every use case, from ordinary artists to intense creative coders.

**AI art and models, GUIs, animation, prompt engineering, audio-reactivity, iterating, experimenting, rendering, math functions, multi-modality,** everything is covered by stable-core and it all chains together. The amazing entourage effect of our components encourages developers implement all their best ideas and models as a stable-core plugin. 

Like demoscene, we dogfood the hell out of it.

# Components

1. **Backend:** The server/client architecture means clients implement UIs or bridge other apps like blender nodes, kdenlive clips, effects, etc.
2. **Sessions:** your outputs are organized and clustered into sessions. A new timestamped session starts when a client connects, and you can re-open old sessions to do more work in them. Could be a prompt engineering session, an animation, a batch of variants, a history for an outpainting project, etc. 
3. **Jobs:** Generate/transform some data with input/output. 
4. **Plugins:** they implement models, packages, techniques, features, handle all installation and cloning in a well-defined manner. They plug into the system by providing jobs, which are simply functions. Create a new plugin generated from a template and you can instantly get to work.
5. **Cloud Deploy:** Instantly render on runpod, vast.ai in just a few clicks. Paste in your SSH information to copy your configuration and your installation will automatically be installed and local jobs are deferred to the instance. The StableHorde is also worth supporting.

## Installation

NOTE: currently there may be unexpected errors

1. clone the repository somewhere on your drive.

2. Setup your `user_config.py` file in the root of the directory, see an example below

```py
ip = '0.0.0.0'
port = 5000

# Plugins to install
plugins = ['stablecore-ai/sd1111_plugin']

# Plugins to load on startup
startup = ['sd1111_plugin']
```

3. Launch run-server.sh (Linux) or backend.bat (Windows)

```log
❯ ./run.sh

[session] New session: 001_2022-10-26_12-00-42
Python: 3.10.6 (main, Aug  3 2022, 17:39:45) [GCC 12.1.1 20220730]
Revision: <none>

[core] (0.00s) Initializing plugins
[core] (3.40s) Installing plugins...
[plugin] (0.00s)   - sd1111_plugin
[core] (0.01s) Loading plugins...
[plugin] (0.00s)   - sd1111_plugin
LatentDiffusion: Running in eps-prediction mode
DiffusionWrapper has 859.52 M params.
making attention of type 'vanilla' with 512 in_channels
Working with z of shape (1, 4, 32, 32) = 4096 dimensions.
making attention of type 'vanilla' with 512 in_channels
Loading weights [7460a6fa] from /home/nuck/stable-core/plug-res/sd1111_plugin/sd-v1-4.ckpt
Global Step: 470000
Applying cross attention optimization (Doggettx)
[core] (10.38s) All ready!
[server] (0.00s) Starting ...

> Enter commands here or use a client ...

> txt2img prompt="Woaaa! Kawaii monster by salvador dali"
prompt=Woaaa! Kawaii monster by salvador dali
[session]  run sd1111_plugin <src_plugins.sd1111_plugin.sd_txt2img.sd_txt2img object at 0x7f7963df5a50>
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 22/22 [00:12<00:00,  1.71it/s]

```

## Plugin

This list is a preview of our end-game plugin ecosystem, how it will look and feel. We encourage community members to contribute and maintain some of these plugins themselves.

* **[AUTO1111 StableDiffusion](https://github.com/stablecore-ai/sd1111_plugin):** txt2img, img2img, ported from AUTO1111's webui with all optimizations 
* **Diffusers StableDiffusion:** txt2img, img2img
* **VQGAN+CLIP / PyTTI:** txt2img, img2img
* **DiscoDiffusion:** txt2img, img2img
* **StableHorde:** txt2img, serve
* **CLIP Interrogate:** img2txt
* **Dreambooth**: train_ckpt
* **StyleGAN:** train_ckpt, img2img
* **2D Transforms:** simple 2D transforms like translate, rotate, and scale.
* **3D Transforms:** 3D transforms using virtual depth like rotating a sphere OR predicted depth from AdaBins+MiDaS. Could implement depth guidance to try and keep the depth more stable.
* **Guidance:** these plugins implement guidance losses into other generation plugins.
   * **CLIP Guidance:** guidance using CLIP models.
   * **Lpips Guidance:** guidance using lpips. _(used in Disco Diffusion to reduce flickering)_
   * **Convolution Guidance:** guidance using convolutions. _(edge_weight in PyTTI)_
* **Audio Analysis:** turn audio inputs into numbers for audio-reactivity, using FFT and stuff like that. Can maybe use Magenta.
* **Palette Match:** img2img, adjust an image's palette to match an input image.
* **Flow Warp:** img2img, displace an image using estimated flow between 2 input images.
* **Prompt Wildcards:** txt2txt


Upscalers:
  * **RealSR:** img2img, on Linux this is easily installed thru AUR with `realsr-ncnn-vulkan`
  * **BasicSR:** img2img, port
  * **LDSR:** img2img
  * **CodeFormer:** img2img, port
  * **GFPGAN:** img2img, port
* **Deforum:** txt2img, technically just a macro of other plugins (sorry)
* **LucidSonicDreams:** txt2img, hopefully we can make it adapt to any model
* **MetaPlugin:** a plugin to string other plugins together, either with job macros or straight-up python. Could be done without a plugin but this allows all clients to automatically support these features.
* **Wildcards:** a plugin to add RNG into your prompts.
* **Whisper:** audio2txt

Clients:
   * ImGUI
   * Gradio
   * Blender

# Current roadmap

If I am alone working on this, this will be my roadmap.

1. ~Core backend components (server, jobs, plugins) to a usable state.~
2. ~Run the StableDiffusionPlugin txt2img job from CLI~
3. Get the session management working to a tee.
4. Write a UI to manage sessions and do my work in.
5. Plugins
   - Port upscalers from AUTO1111 so we can see the workflow in action.
   - Port my [disco-party](https://github.com/oxysoft/disco-party/) math plugin with dedicated graphing calculator for audio-reactivity
