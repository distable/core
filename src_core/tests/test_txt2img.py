from src_core import core

if __name__ == "__main__":
    core.init()
    for i in range(20):
        core.run('txt2img', prompt="Beautiful contorted airbrushed landscape by salvador dali. 70s prog rock album cover, artistic, monochromatic, water reflections, green, poster print, acrylic, metallic, chrome, caustics, reflections, 90s cgi")
