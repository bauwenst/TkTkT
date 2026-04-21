"""
For detecting the presence of devices, notebook environments, and so on.
"""
from .strings import anySubstringIn


def is_linux() -> bool:
    import platform
    return platform.system() == "Linux"


def is_cluster() -> bool:
    """
    Uses heuristics to determine if you are working on an HPC cluster.
    """
    import os

    # Test 1: if the OS is not Linux, it is never an HPC.
    if not is_linux():
        return False

    # Test 2: environment variables that reveal the HPC's scheduler.
    if any(var in os.environ for var in [
        "SLURM_JOB_ID",  # Slurm
        "PBS_JOBID",  # PBS/Torque
        "LSB_JOBID",  # LSF
        "SGE_JOB_ID",  # Sun Grid Engine
    ]):
        return True

    # Test 3: if you have access to a super expensive NVIDIA GPU, you're definitely not on a personal computer.
    try:
        if anySubstringIn(["A100", "H100", "B200"], get_gpu_name()):
            return True
    except:  # No GPU, but could be a compute node.
        pass

    # Test 4: cluster file systems are probably reliable indicators too.
    if any(os.path.exists(p) for p in ["/scratch", "/lustre", "/gpfs"]):
        return True

    # Else, you cannot know. It could be a multicore Linux laptop.
    # has_many_cores = os.cpu_count() > 32
    return False


def is_transformers_installed() -> bool:
    try:
        import transformers
        return True
    except ImportError:
        return False


def has_gpu() -> bool:  # Requires PyTorch.
    import torch
    return torch.cuda.is_available()


def get_gpu_name() -> str:
    if not has_gpu():
        raise RuntimeError("No GPU name exists because there is no GPU.")
    else:
        import torch
        return torch.cuda.get_device_name()


def get_torch_device():
    import torch
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
