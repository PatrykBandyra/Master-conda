import random
from collections import deque
from typing import List, Tuple, NewType

from torch.types import Tensor

Experience = NewType('Experience', Tuple[Tensor, Tensor, Tensor, Tensor, bool])


class ExperienceBuffer:
    def __init__(self, max_len: int, seed=None):
        self.memory = deque([], maxlen=max_len)

        if seed is not None:
            random.seed(seed)

    def append(self, experience: Experience) -> None:
        self.memory.append(experience)

    def sample(self, sample_size: int) -> List[Experience]:
        return random.sample(self.memory, sample_size)

    def __len__(self) -> int:
        return len(self.memory)
