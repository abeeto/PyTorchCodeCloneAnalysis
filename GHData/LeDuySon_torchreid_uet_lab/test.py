import torchreid
from torchreid.data.datasets import UET_REID

torchreid.data.register_image_dataset('uet_dataset', UET_REID)
datamanager = torchreid.data.ImageDataManager(
            root='reid-data',
            sources='uet_dataset'
            )

