
# Contributions

The backend is made with flask and flask-sockio, it was fast and easy to use but could change in the future.

Contribution points for anyone who'd like to help.

- **Plugins:** We already 'have' a bunch of plugins courtesy of AUTOMATIC1111 (mainly upscalers), the code still needs to be ported for each of them, however it's best to wait for the plugin and job API to solidify more as I'm working on the SD plugin. Then we can try to implement new ones. In the meantime...
   - We need a **Plugin Shell Script** (written in python) for the following features...
   - **Discovery:** Figure out how to host plugins on github and automatically collect them for listing. I'm pretty sure a bunch of other projects do it, it has to be possible somehow, maybe check with `https://vimawesome.com/` how they do it or if it's all manual.
   - **Creation:** Create a new plugin, ready to work on it and push to a repository. This is a directory with __init__ and a class extending `Plugin`, `stable_diffusion` is currently the best example we have. The directory will be used as its identifier for client/server communication so it should be all lowercase, and a valid module name so no dashes.
   - **Update:** Update an existing plugin with git pull on it.
- **UI:** we don't have a UI yet
- **Authentication:** session system to connect with a passwords, ssh, etc. no sharing without this obviously.

## Coding Standards

- **KISS:** We abid KISS, must be able to read and understood whole thing in under an hour. Always consider more than one approach, pick the simplest. Adding more code to a function is always fine, but adding additional class, function, or even fields/properties introduces top-level complexity and should always lead to some API reflection.
- **Documentation:** There is a severe lack of quality documentation in the world of programming, see `launch.py` for a good demo of proper documentation. Long methods are fine, but add big header comments with titles. Any code where it isn't immediately obvious _why we do it_ or _why that way exactly_, should be commented.
- **Stability:** Don't use exceptions for simple stuff. Fail gracefully with an error message, recognize that the error state is possible as part of the API, and return a default value or ability to handle the error. The core must be able to remain up 24/7 for weeks at a time. We should try to keep the backend core running when maxing out VRAM, either by preventing or by running each plugin on separate web-core process so the backend can keep running even if a plugin results in OOM.  
- **Orthogonality:** Huge emphasis on local responsabilities, e.g. don't do any saving or logging directly as part of a job. Instead, report the progress and output data, and let the specifics be handled externally by some other specialized component. Avoid passing some huge bags of options, and if there's a lot of default values put them in a nested option class for the related job, or architecture the code such as to be able to post-process the values and apply defaults inside the plugin itself.
- **Unit Testing:** not planned currently but test suites could certainly be useful eventually, especially on individual plugins that might change a lot like StableDiffusionPlugin.

**Formatting**
- 4 spaces indent
- Prefer Pathlib Path over filename strings
