
class Config:

    ENV: str = None
    EPISODE: int = None
    MAX_STEP: int = None
    BUFFER_SIZE: int = None
    BATCH_SIZE: int = None

    state_dim: int = None
    state_high = None
    state_low = None
    seed = None

    output = 'out'

    action_dim: int = None
    action_high = None
    action_low = None
    action_lim = None

    learning_rate: float = None
    learning_rate_actor: float = None

    gamma: float = None
    tau: float = None
    epsilon: float = None
    eps_decay = None
    epsilon_min: float = None

    use_cuda: bool = True

    checkpoint: bool = False
    checkpoint_interval: int = None

    use_matplotlib: bool = False

    record: bool = False
    record_ep_interval: int = None
