from src_core.classes.Plugin import Plugin

class KandinskyPlugin(Plugin):
    def title(self):
        return "kandinsky"

    def describe(self):
        return ""

    def init(self):
        pass

    def install(self):
        pass

    def uninstall(self):
        pass

    def load(self):
        from kandinsky2 import get_kandinsky2
        self.model = get_kandinsky2('cuda', task_type='img2img', model_version='2.0', use_flash_attention=False)

    def unload(self):
        pass

    def txt2img(self, prompt="", cfg=4, steps=25, h=768, w=768, sampler='p_sampler', prior_cf_scale=4, prior_steps='5', **kwargs):
        return self.model.generate_text2img(
                prompt,
                num_steps=steps,
                batch_size=1,
                guidance_scale=cfg,
                h=h, w=w,
                sampler=sampler,
                prior_cf_scale=prior_cf_scale,
                prior_steps=prior_steps,
        )

    def img2img(self, img, prompt="", chg=0.8, cfg=10, steps=25, **kwargs):
        return self.model.generate_img2img(
                prompt,
                img,
                strength=chg,
                num_steps=steps,
                denoised_type='dynamic_threshold',
                dynamic_threshold_v=99.5,
                sampler='ddim_sampler',
                ddim_eta=0.05,
                guidance_scale=cfg
        )
