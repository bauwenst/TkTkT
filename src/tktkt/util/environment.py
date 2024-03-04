"""
For detecting the presence of devices, notebook environments, and so on.
"""
import torch

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
