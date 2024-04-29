import torch


class Network (object):
    def __init__(self, model):
        self.model = model

    def print_infreezed_layers(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print("\t", name)

    def unfreeze_all_layers(self):
        pass

    def get_model(self):
        return self.model

    def load_model_from_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location='cpu')
        self.model = checkpoint['arch']
        self.model.classifier = checkpoint['classifier']
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer = checkpoint['optimizer']
        epochs = checkpoint['epochs']
        scheduler = checkpoint['scheduler']
        return optimizer, epochs, scheduler

    def save_model_checkpoint(
            self,
            filename,
            input_size,
            output_size,
            epochs,
            batch_size,
            arch,
            classifier,
            sched,
            optimizer,
            model):
        checkpoint = {
            'input_size': input_size,
            'output_size': output_size,
            'epochs': epochs,
            'batch_size': batch_size,
            'arch': arch,
            'classifier': classifier,
            'scheduler': sched,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.state_dict()
        }
        self.model = model
        torch.save(checkpoint, filename)
        return filename
