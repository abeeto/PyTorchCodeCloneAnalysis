import pyaudio
import os
import wave
import subprocess
def create_wav(path, output_path):
    """
    Creates and saves a wav file from the video in path, and returns the path
    Throws AssertionErro if there is no video in path
    """
    assert os.path.exists(path), f"There is no video in {path}"
    print(f"Creating wav from {path}")
    cmd = f"ffmpeg -i {path} -ab 160k -ac 2 -ar 44100 -vn {output_path}"
    subprocess.call(cmd)#, shell= True)
    return output_path
    
class AudioPlay: #Inspired on https://stackoverflow.com/questions/18721780/play-a-part-of-a-wav-file-in-python
    def __init__(self, filename):
        self.wav = wave.open(filename, "rb")
        self.pyaudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(format=self.pyaudio.get_format_from_width(self.wav.getsampwidth()),
                                        channels=self.wav.getnchannels(),
                                        rate=self.wav.getframerate(),
                                        output=True)
        self.fr = self.wav.getframerate()
        print(f"wav framerate = {self.fr}")
    def play(self, start, end):
        length = end - start
        unwanted_frames = int(start * self.fr)
        print(f"unwanted frames = {unwanted_frames/ self.fr}")
        self.wav.setpos(unwanted_frames)
        wanted_frames = int(length * self.fr)
        frames = self.wav.readframes(wanted_frames)
        self.stream.write(frames)
    def close(self):
        self.stream.close()
        self.pyaudio.terminate()
        self.wav.close()
class VideoPlay:
    def __init__(self, video_path, audio_path, nframes, spf):
        if not os.path.exists(audio_path):
            print(f"{audio_path} already exists, so not creating it")
            create_wav(video_path, audio_path) #wav path for the audio of the video
        self.nframes = nframes
        self.audio_obj = AudioPlay(audio_path)
        self.spf = spf#seconds per frame
        self.duration = self.nframes * self.spf
    def play_audio_for_frame(self, i):
        assert 0 <= i < self.nframes, f"index {i} should be in the interval [0, {self.nframes})"
        start = i * self.spf 
        end = min(self.duration, start + self.spf)
        print(f"duration = {self.duration}")
        self.audio_obj.play(start, end)
    def close(self):
        self.audio_obj.close()