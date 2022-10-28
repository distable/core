import json

import gradio

from src_core.printlib import printerr


def options_section(name, dic):
    for k, v in dic.items():
        v.section = name

    return dic


class OptionInfo:
    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None, show_on_main_page=False):
        self.default = default
        self.label = label
        self.component = component
        self.component_args = component_args
        self.onchange = onchange
        self.section = None
        self.show_on_main_page = show_on_main_page


class Options:
    def __init__(self, template):
        self.__dict__['data'] = None
        self.__dict__['data'] = {k: v.default for k, v in self.template.items()}
        self.template = template
        self.typemap = {int: float}

    def __setattr__(self, key, value):
        if self.__dict__['data'] is not None:
            if key in self.__dict__['data']:
                self.__dict__['data'][key] = value

        return super(Options, self).__setattr__(key, value)

    def __getattr__(self, item):
        if self.__dict__['data'] is not None:
            if item in self.__dict__['data']:
                return self.__dict__['data'][item]

        if item in self.template:
            return self.template[item].default

        return super(Options, self).__getattribute__(item)

    def save(self, filename):
        with open(filename, "w", encoding="utf8") as file:
            json.dump(self.__dict__['data'], file)

    def same_type(self, x, y):
        if x is None or y is None:
            return True

        type_x = self.typemap.get(type(x), type(x))
        type_y = self.typemap.get(type(y), type(y))

        return type_x == type_y

    def load(self, filename):
        with open(filename, "r", encoding="utf8") as file:
            self.__dict__['data'] = json.load(file)

        bad_settings = 0
        for k, v in self.__dict__['data'].items():
            info = self.template.get(k, None)
            if info is not None and not self.same_type(info.default, v):
                printerr(f"Warning: bad setting value: {k}: {v} ({type(v).__name__}; expected {type(info.default).__name__})")
                bad_settings += 1

        if bad_settings > 0:
            printerr(f"The program is likely to not work with bad settings.\nSettings file: {filename}\nEither fix the file, or delete it and restart.")

    def onchange(self, key, func):
        item = self.template.get(key)
        item.onchange = func

    def dumpjson(self):
        d = {k: self.__dict__['data'].get(k, self.template.get(k).default) for k in self.template.keys()}
        return json.dumps(d)


# hide_dirs = {"visible": not cargs.hide_ui_dir_config}

options_templates = {}

options_templates.update(options_section(('saving-images', "Saving images/grids"), {
    "samples_save"                       : OptionInfo(True, "Always save all generated images"),
    "samples_format"                     : OptionInfo('png', 'File format for images'),
    "samples_filename_pattern"           : OptionInfo("", "Images filename pattern"),

    "grid_save"                          : OptionInfo(True, "Always save all generated image grids"),
    "grid_format"                        : OptionInfo('png', 'File format for grids'),
    "grid_extended_filename"             : OptionInfo(False, "Add extended info (seed, prompt) to filename when saving grid"),
    "grid_only_if_multiple"              : OptionInfo(True, "Do not save grids consisting of one picture"),
    "n_rows"                             : OptionInfo(-1, "Grid row count; use -1 for autodetect and 0 for it to be same as batch size", gradio.Slider, {"minimum": -1, "maximum": 16, "step": 1}),

    "enable_pnginfo"                     : OptionInfo(True, "Save text information about generation parameters as chunks to png files"),
    "save_txt"                           : OptionInfo(False, "Create a text file next to every image with generation parameters."),
    "save_images_before_face_restoration": OptionInfo(False, "Save a copy of image before doing face restoration."),
    "jpeg_quality"                       : OptionInfo(80, "Quality for saved jpeg images", gradio.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
    "export_for_4chan"                   : OptionInfo(True, "If PNG image is larger than 4MB or any dimension is larger than 4000, downscale and save copy as JPG"),

    "use_original_name_batch"            : OptionInfo(False, "Use original name for output filename during batch process in extras tab"),
    "save_selected_only"                 : OptionInfo(True, "When using 'Save' button, only save a single selected image"),
    "do_not_add_watermark"               : OptionInfo(False, "Do not add watermark to images"),
}))

# options_templates.update(options_section(('saving-paths', "Paths for saving"), {
#     "outdir_samples"        : OptionInfo("", "Output directory for images; if empty, defaults to three directories below", component_args=hide_dirs),
#     "outdir_txt2img_samples": OptionInfo("outputs/txt2img-images", 'Output directory for txt2img images', component_args=hide_dirs),
#     "outdir_img2img_samples": OptionInfo("outputs/img2img-images", 'Output directory for img2img images', component_args=hide_dirs),
#     "outdir_extras_samples" : OptionInfo("outputs/extras-images", 'Output directory for images from extras tab', component_args=hide_dirs),
#     "outdir_grids"          : OptionInfo("", "Output directory for grids; if empty, defaults to two directories below", component_args=hide_dirs),
#     "outdir_txt2img_grids"  : OptionInfo("outputs/txt2img-grids", 'Output directory for txt2img grids', component_args=hide_dirs),
#     "outdir_img2img_grids"  : OptionInfo("outputs/img2img-grids", 'Output directory for img2img grids', component_args=hide_dirs),
#     "outdir_save"           : OptionInfo("log/images", "Directory for saving images using the Save button", component_args=hide_dirs),
# }))

options_templates.update(options_section(('saving-to-dirs', "Saving to a directory"), {
    "save_to_dirs"                : OptionInfo(False, "Save images to a subdirectory"),
    "grid_save_to_dirs"           : OptionInfo(False, "Save grids to a subdirectory"),
    "use_save_to_dirs_for_ui"     : OptionInfo(False, "When using \"Save\" button, save images to a subdirectory"),
    "directories_filename_pattern": OptionInfo("", "Directory name pattern"),
    "directories_max_prompt_words": OptionInfo(8, "Max prompt words for [prompt_words] pattern", gradio.Slider, {"minimum": 1, "maximum": 20, "step": 1}),
}))

# options_templates.update(options_section(('upscaling', "Upscaling"), {
#     "ESRGAN_tile"              : OptionInfo(192, "Tile size for ESRGAN upscalers. 0 = no tiling.", gradio.Slider, {"minimum": 0, "maximum": 512, "step": 16}),
#     "ESRGAN_tile_overlap"      : OptionInfo(8, "Tile overlap, in pixels for ESRGAN upscalers. Low values = visible seam.", gradio.Slider, {"minimum": 0, "maximum": 48, "step": 1}),
#     "realesrgan_enabled_models": OptionInfo(["R-ESRGAN x4+", "R-ESRGAN x4+ Anime6B"], "Select which Real-ESRGAN models to show in the web UI. (Requires restart)", gradio.CheckboxGroup, lambda: {"choices": realesrgan_models_names()}),
#     "SWIN_tile"                : OptionInfo(192, "Tile size for all SwinIR.", gradio.Slider, {"minimum": 16, "maximum": 512, "step": 16}),
#     "SWIN_tile_overlap"        : OptionInfo(8, "Tile overlap, in pixels for SwinIR. Low values = visible seam.", gradio.Slider, {"minimum": 0, "maximum": 48, "step": 1}),
#     "ldsr_steps"               : OptionInfo(100, "LDSR processing steps. Lower = faster", gradio.Slider, {"minimum": 1, "maximum": 200, "step": 1}),
#     "upscaler_for_img2img"     : OptionInfo(None, "Upscaler for img2img", gradio.Dropdown, lambda: {"choices": [x.name for x in sd_upscalers]}),
# }))

# options_templates.update(options_section(('face-restoration', "Face restoration"), {
#     "face_restoration_model" : OptionInfo(None, "Face restoration model", gradio.Radio, lambda: {"choices": [x.name() for x in face_restorers]}),
#     "code_former_weight"     : OptionInfo(0.5, "CodeFormer weight parameter; 0 = maximum effect; 1 = minimum effect", gradio.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
#     "face_restoration_unload": OptionInfo(False, "Move face restoration model from VRAM into RAM after processing"),
# }))

options_templates.update(options_section(('system', "System"), {
    "memmon_poll_rate"  : OptionInfo(8, "VRAM usage polls per second during generation. Set to 0 to disable.", gradio.Slider, {"minimum": 0, "maximum": 40, "step": 1}),
    "samples_log_stdout": OptionInfo(False, "Always print all generation info to standard output"),
    "multiple_tqdm"     : OptionInfo(True, "Add a second progress bar to the console that shows progress for an entire job."),
}))

options_templates.update(options_section(('training', "Training"), {
    "unload_models_when_training": OptionInfo(False, "Unload VAE and CLIP from VRAM when training"),
}))

options_templates.update(options_section(('interrogate', "Interrogate Options"), {
    "interrogate_keep_models_in_memory"    : OptionInfo(False, "Interrogate: keep models in VRAM"),
    "interrogate_use_builtin_artists"      : OptionInfo(True, "Interrogate: use artists from artists.csv"),
    "interrogate_clip_num_beams"           : OptionInfo(1, "Interrogate: num_beams for BLIP", gradio.Slider, {"minimum": 1, "maximum": 16, "step": 1}),
    "interrogate_clip_min_length"          : OptionInfo(24, "Interrogate: minimum description length (excluding artists, etc..)", gradio.Slider, {"minimum": 1, "maximum": 128, "step": 1}),
    "interrogate_clip_max_length"          : OptionInfo(48, "Interrogate: maximum description length", gradio.Slider, {"minimum": 1, "maximum": 256, "step": 1}),
    "interrogate_deepbooru_score_threshold": OptionInfo(0.5, "Interrogate: deepbooru score threshold", gradio.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "deepbooru_sort_alpha"                 : OptionInfo(True, "Interrogate: deepbooru sort alphabetically"),
}))

options_templates.update(options_section(('ui', "User interface"), {
    "show_progressbar"                  : OptionInfo(True, "Show progressbar"),
    "show_progress_every_n_steps"       : OptionInfo(0, "Show image creation progress every N sampling steps. Set 0 to disable.", gradio.Slider, {"minimum": 0, "maximum": 32, "step": 1}),
    "return_grid"                       : OptionInfo(True, "Show grid in results for web"),
    "do_not_show_images"                : OptionInfo(False, "Do not show any images in results for web"),
    "add_model_hash_to_info"            : OptionInfo(True, "Add model hash to generation information"),
    "add_model_name_to_info"            : OptionInfo(False, "Add model name to generation information"),
    "font"                              : OptionInfo("", "Font for image grids that have text"),
    "js_modal_lightbox"                 : OptionInfo(True, "Enable full page image viewer"),
    "js_modal_lightbox_initially_zoomed": OptionInfo(True, "Show images zoomed in by default in full page image viewer"),
    "show_progress_in_title"            : OptionInfo(True, "Show generation progress in window title."),
}))

# opts = Options(options_templates)
