import cv2 as cv
import numpy as np
import torch
import pyaudio

p = pyaudio.PyAudio()


# base streaming functions. data must be batched
def image_out(img, name='out'):

    img_tensor = img.cpu().detach().numpy()

    for img in img_tensor:

        cv.imshow(name, img.transpose(2, 1, 0))
        cv.waitKey(1)



def audio_out(data, stream):

    out = data.cpu().detach().numpy().reshape(data.shape[1], data.shape[0] * data.shape[2])
    stream.write(out.tobytes())



def streamer(_args, key, name='out'):

    output_streamer = None

    if _args[key] == 'video':

        output_streamer = lambda img: image_out(img, name=name)
    
    else:

        stream = p.open(format=pyaudio.paFloat32, channels=_args['info'][0]['num_channels'], rate=_args['info'][0]['sample_rate'], output=True)
        output_streamer = lambda aud: audio_out(aud, stream)

    return output_streamer



# Parallelized helper function for output display
# img_tensor is a torch tensor with shape [C, H, W]
def parallel_image_out(q, name='out'):

    img_tensor = q.get().cpu().detach().numpy()

    cv.imshow(name, img_tensor.transpose(2, 1, 0))
    cv.waitKey(1)



# audio_tensor is a torch tensor with shape [C, L]
# where L is the framerate
def parallel_audio_out(q, stream):

    sample = q.get().cpu().numpy().tobytes()
    stream.write(sample)


# TODO: add input stream?
# Right now, the key value points to 'output' or 'input'
# But since we only show the output for the training loop 
# and only training is parallelized, we can set the key as const 'output'
# But it would be nice to get input streams as well.

def parallel_streamer(_args, key, q, name="out"):

    output_streamer = None

    # Img case
    if _args['output'] == 'video':

        output_streamer = lambda : parallel_image_out(q, name=name)

    # Audio case
    else:

        stream = p.open(format=pyaudio.paFloat32, channels=_args['info'][0]['num_channels'], rate=_args['info'][0]['sample_rate'], output=True)
        output_streamer = lambda : parallel_audio_out(q, stream)

    return output_streamer



# Parallelized versions of above
def parallel_display(_args, tq, oq):

    tq_streamer, oq_streamer = None, None

    if _args['display_truth']:

        tq_streamer = parallel_streamer(_args, 'input', tq, name='truth')

    else:

        tq_streamer = lambda : tq.get()
    
    if _args['display_out']:

        oq_streamer = parallel_streamer(_args, 'output', oq, name='out')
    
    else:

        oq_streamer = lambda : oq.get()

    while True:

        tq_streamer()
        oq_streamer()

