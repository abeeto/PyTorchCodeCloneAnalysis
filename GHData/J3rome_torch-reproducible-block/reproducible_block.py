import random

import torch
import numpy as np


# TODO : PYTHONHASHSEED for reproductible dictionaries order -- https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED
class Reproducible_Block:
    """
    Python, Numpy and PyTorch random states manager statement
    """
    # Class attribute
    reference_state = None

    def __init__(self, block_seed=None, reset_state_after=False):
        assert block_seed is not None, \
            "Block seed must be passed to the decorator/with clause Ex: @Reproducible_Block(block_seed=42)"

        assert Reproducible_Block.reference_state is not None, \
            "Seed & Reference state must be set prior to the creation of a Reproducible Block : Reproducible_Block.set_seed(42)"

        self.block_seed = block_seed
        self.reset_state_after = reset_state_after
        self.initial_state = None

    # With clause handling
    def __enter__(self):
        if self.reset_state_after:
            self.initial_state = Reproducible_Block._get_random_state()

        Reproducible_Block.reset_to_reference_state()

        # Modify the random state by performing a serie of random operations.
        # This is done to create unique random state paths using the same initial state for different code blocks
        # If we were to set the same state before every operation, the same operation at different stage of the model
        # would always have the same result (Ex : Weight initialisation)
        _modify_random_state(self.block_seed)

    # With clause handling
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.reset_state_after:
            Reproducible_Block._set_random_state(self.initial_state)

    # Decorator handling
    def __call__(self, fct, *args):
        def wrapped_fct(*args):
            with self:
                return fct(*args)
        return wrapped_fct


    @classmethod
    def set_seed(cls, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        cls._set_reference_state()

    @classmethod
    def _set_reference_state(cls):
        cls.reference_state = cls._get_random_state()

    @classmethod
    def reset_to_reference_state(cls):
        assert cls.reference_state is not None, \
            "Reference random state must be set before reseting state"

        cls._set_random_state(cls.reference_state)

    @classmethod
    def _get_random_state(cls):
        state = {
            'py': random.getstate(),
            'np': np.random.get_state(),
            'torch': torch.random.get_rng_state()
        }

        if torch.cuda.is_available():
            state['torch_cuda'] = torch.cuda.get_rng_state()

        return state

    @classmethod
    def _set_random_state(cls, state):
        random.setstate(state['py'])
        np.random.set_state(state['np'])
        torch.random.set_rng_state(state['torch'])

        if torch.cuda.is_available() and 'torch_cuda' in state:
            torch.cuda.set_rng_state(state['torch_cuda'])


def _modify_random_state(modify_seed):
    # Modify the random state by performing a serie of random operations.
    for i in range(modify_seed):
        random.randint(1, 10)
        np.random.randint(1, 10)

        # TODO : Check if both torch & torch.cuda random state are impacted by this. Might need to do operation on cuda tensor
        torch.randint(1, 10, (1,))
