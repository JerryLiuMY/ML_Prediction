import os
from pathlib import Path


# directories
DESKTOP_PATH = str(Path(os.getcwd()).parent.absolute())
DATA_PATH = os.path.join(DESKTOP_PATH, "shared")

