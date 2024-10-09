"""
For detecting the presence of devices, notebook environments, and so on.
"""
import torch
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import platform
IS_NOT_LINUX = platform.system() != "Linux"  # Can be used as a check to see if you're running on an HPC if you dev on Windows/Mac.
