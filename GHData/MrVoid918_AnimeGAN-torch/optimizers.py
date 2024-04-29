import torch.optim as optim
from optim.omd import OptimisticAdam
from optim.extragradient import ExtraAdam
from adabelief_pytorch import AdaBelief

"https://howto.lintel.in/python-__new__-magic-method-explained/"


class GANOptimizer(optim.Optimizer):

    def __new__(cls, optim_type, *args, **kwargs):
        OPTIM_TYPE_MAP = {
            'SGD': optim.SGD,
            'RMS': optim.RMSprop,
            'ADAM': optim.Adam,
            'ADAMW': optim.AdamW,
            'OADAM': OptimisticAdam,
            'XADAM': ExtraAdam,
            'ADAB': AdaBelief
        }

        if optim_type not in OPTIM_TYPE_MAP:
            raise ValueError(f"Bad Optim Type {optim_type}")

        sub_optim = OPTIM_TYPE_MAP[optim_type]
        instance = super(GANOptimizer, cls).__new__(sub_optim)
        instance.__init__(*args, **kwargs)
        return instance
