import random

import pytest
import torch


@pytest.fixture(autouse=True)
def _torch_fix_seed(seed: int = 42) -> None:
    # Python random
    random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True)
