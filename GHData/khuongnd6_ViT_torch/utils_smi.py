
import nvidia_smi

class NVIDIA_SMI:
    def __init__(self, device_id=0):
        nvidia_smi.nvmlInit()
        self.device_id = device_id
        self.nvml_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.device_id)
    
    @property
    def info(self):
        return self.get_info()
    
    def get_info(self):
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(self.nvml_handle)
        r = {
            'total': info.total / 1024**3,
            'used': info.used / 1024**3,
            'free': info.free / 1024**3,
            'usage': info.used / info.total,
        }
        return r
    
    def get_vram_used(self):
        _info = self.get_info()
        return _info['used']
    
    @classmethod
    def dispose(cls):
        nvidia_smi.nvmlShutdown()
