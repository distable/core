# Installations
# ----------------------------------------

from src_core.installing import *

gitclone("https://github.com/dome272/Paella")

path = Path(__file__).resolve().parent / "requirements.txt"
pipreqs(path.as_posix())
