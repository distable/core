# import math
# import os
# import time
# from src_core.lib.corelib import to_dict
# from src_core.classes.JobArgs import JobArgs
# from src_core.classes.Plugin import Plugin
# from src_core.plugins import plugjob
#
# os.environ.setdefault("DEMOSYS_SETTINGS_MODULE", "src_plugins.glsl_plugin.demosys_settings")
# from demosys.conf import default, settings
# from src_plugins.glsl_plugin import demosys_settings
# import demosys
#
# settings.update(**to_dict(default))
# settings.update(**to_dict(demosys_settings))
#
# demosys.setup(
#         PROJECT='demosys.project.default.Project',
#         TIMELINE='demosys.timeline.single.Timeline'
# )
#
# from demosys.management.base import RunCommand
# from demosys import context, geometry
# from demosys.effects import effect
# from demosys.resources.meta import ProgramDescription, TextureDescription
#
# from demosys.project.base import BaseProject
# from demosys.resources.meta import ProgramDescription, TextureDescription
#
#
# class Project(BaseProject):
#     resources = [
#         ProgramDescription(label='cube_textured', path="cube_textured.glsl"),
#         TextureDescription(label='wood', path="wood.png", mipmap=True),
#     ]
#
#     def create_resources(self):
#         # Override the method adding additional resources
#
#         # Create some shared fbo
#         size = (256, 256)
#         self.shared_framebuffer = self.ctx.framebuffer(
#                 color_attachments=self.ctx.texture(size, 4)
#                 # depth_attachement=self.ctx.depth_texture(size)
#         )
#
#         return self.resources
#
#     def create_effect_instances(self):
#         # Create and register instances of an effect class we loaded from the effect packages
#         self.create_effect('cube1', 'CubeEffect')
#
#         # Using full path to class
#         self.create_effect('cube2', 'myproject.efect_package1.CubeEffect')
#
#         # Passing variables to initializer
#         self.create_effect('cube3', 'CubeEffect', texture=self.get_texture('wood'))
#
#         # Assign resources manually
#         cube = self.create_effect('cube1', 'CubeEffect')
#         cube.program = self.get_program('cube_textured')
#         cube.texture = self.get_texture('wood')
#         cube.fbo = self.shared_framebuffer
#
#
# class Canvas:
#     def __init__(self):
#         self.create_window()
#         from demosys.utils.module_loading import import_string
#         self.project = import_string(settings.PROJECT)('')
#         self.init()
#         self.effect = Effect()
#
#     def create_window(self):
#         self.window = context.create_window()
#
#     def destroy_window(self):
#         self.window.terminate()
#
#
#     def init(self):
#         from demosys.effects.registry import Effect
#         from demosys.scene import camera
#
#         win = self.window
#
#         # Inject attributes into the base Effect class
#         setattr(Effect, '_window', win)
#         setattr(Effect, '_ctx', win.ctx)
#         setattr(Effect, '_project', self.project)
#
#         # Set up the default system camera
#         win.sys_camera = camera.SystemCamera(aspect=win.aspect_ratio, fov=60.0, near=1, far=1000)
#         setattr(Effect, '_sys_camera', win.sys_camera)
#
#         # print("Loading started at", time.time())
#         self.project.load()
#
#         # Initialize timer
#         from demosys.utils.module_loading import import_string
#         timer_cls = import_string(settings.TIMER)
#         win.timer = timer_cls()
#         win.timer.start()
#
#     def loop(self):
#         window = self.window
#
#         # Main loop
#         frame_time = 60.0 / 1000.0
#         time_start = time.time()
#         prev_time = window.timer.get_time()
#
#         while not window.should_close():
#             current_time = window.timer.get_time()
#
#             window.use()
#             window.clear()
#             self.effect.draw(current_time, frame_time, window.fbo)
#             window.swap_buffers()
#
#             frame_time = current_time - prev_time
#             prev_time = current_time
#
#         duration_timer = window.timer.stop()
#         duration = time.time() - time_start
#
#         if duration > 0:
#             fps = round(window.frames / duration, 2)
#             print("Duration: {}s rendering {} frames at {} fps".format(duration, window.frames, fps))
#             print("Timeline duration:", duration_timer)
#
#
# class Effect(effect.Effect):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.vao = geometry.quad_fs()
#         self.program = self.get_program("shader")
#         self.texture1 = self.get_texture("WarpspeedTexture")
#         self.texture2 = self.get_texture("WarpspeedTexture2")
#         self.zangle = 0.0
#
#         self.effect_packages = []
#         self.resources = [
#             ProgramDescription(label="shader", path="warpspeed/shader.glsl"),
#             TextureDescription(label="WarpspeedTexture", path="warpspeed/WarpspeedTexture.jpg"),
#             TextureDescription(label="WarpspeedTexture2", path="warpspeed/WarpspeedTexture2.jpg"),
#         ]
#
#     def draw(self, time, frametime, target):
#         self.texture1.use(location=0)
#         self.program["tex"].value = 0
#         self.texture2.use(location=1)
#         self.program["tex2"].value = 1
#
#         self.program["scroll"].value = time * 1.5
#         self.program["intensity"].value = 1.0 + ((math.sin(time / 2)) / 2)
#         self.program["zangle"].value = time
#         self.program["speedlayer_alpha"].value = (math.sin(time) + 1.0) / 2
#         self.program["accelerate"].value = 0.5
#
#         self.vao.render(self.program)
#
#
# class glsl_job(JobArgs):
#     def __init__(self, name, props, img2, **kwargs):
#         JobArgs.__init__(self, **kwargs)
#         self.name = name
#         self.props = props
#         self.img2 = img2
#
#
# class GLSLPlugin(Plugin):
#     def title(self):
#         return "glsl"
#
#     def describe(self):
#         return ""
#
#     def load(self):
#         # self.canvas = Canvas()
#         pass
#
#     # def render_glsl(self, name, props, image, image2=None):
#     @plugjob
#     def glsl(self, j: glsl_job):
#         pass
