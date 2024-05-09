from tst.preamble import *  # sets fiject path too

from tktkt.models.neural.canine_finetuning import MBR


if __name__ == "__main__":
    task = MBR()
    task.train()