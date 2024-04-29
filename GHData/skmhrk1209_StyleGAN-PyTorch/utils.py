import torch
import tensorboardX as tbx


class Dict(dict):

    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]


class SummaryWriter(tbx.SummaryWriter):

    def add_images(self, main_tag, tag_images_dict, global_step=None, walltime=None, dataformats="NCHW"):

        for tag, images in tag_images_dict.items():

            self.file_writer.add_summary(
                summary=tbx.summary.image(
                    tag=f"{main_tag}/{tag}",
                    tensor=images,
                    dataformats=dataformats
                ),
                global_step=global_step,
                walltime=walltime
            )
