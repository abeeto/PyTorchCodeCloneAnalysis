from argparse import ArgumentParser, ArgumentTypeError
parser = ArgumentParser()

def str2bool(v):
   if isinstance(v, bool): return v 
   if v.lower() in {"yes", "true", "t", "y", "1"}: return True 
   elif v.lower() in {"no", "false", "f", "n", "0"}: return False
   else:
      raise ArgumentTypeError("Boolean value expected")
##############################
# From the other repo #
# parser = argparse.ArgumentParser(description='Style Transfer')
parser.add_argument('--fps', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--video-name', type=str, default='avi\output.avi', metavar='N',
                    help='video-name')
parser.add_argument('--mode', type=str, default='video', metavar='N',
        help='video mode: video,video_style,concat')
parser.add_argument('--save-directory', type=str, default='trained_models',
                    help='learnt models are saving here')
parser.add_argument('--imgs-path', type=str, default='',
                    help='images path')

parser.add_argument('--model-name', type=str, default='',
                    help='model name')
parser.add_argument('--video-source', type=str, 
                    help="Name of source video. Should be in the videos/ folder")
parser.add_argument("--use-audio", type=str2bool, default=False, help="On the existing_video mode, you can use this flag to activate audio or not. This might slow the process.")
##############################
opt = parser.parse_args()


if opt.mode == "existing_video" and opt.video_source is None:
   raise ValueError("If using existing_video mode, need to give the video-source. If you want to test your video stream, use video_stream mode.")
if opt.mode != "existing_video" and opt.use_audio:
   raise ValueError(f"use_audio flag is not available on the {opt.mode} mode. Use the existing_video mode instead")