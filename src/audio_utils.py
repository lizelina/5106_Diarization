import random
import time

import pyaudio
import wave
from pydub import AudioSegment
from pyannote.audio import Audio
import subprocess
import numpy as np
import os
import shutil

SAMPLE_RATE = 16000
FILEPATH = []


def record_audio(seconds, filename):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def convert_to_wav(file, output_file=None, sr=16000):
    if not output_file:
        output_file = replace_extension_with_wav(file)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file,
        "-f",
        "wav",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        output_file
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        print(f"Conversion successful. Saved as {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
    return output_file


def replace_extension_with_wav(file_name):
    # 使用 os.path.splitext 分离文件名和扩展名
    base_name, ext = os.path.splitext(file_name)
    base_name = base_name + '_' + str(random.randint(0, 100))
    # 返回新的文件名，替换为 .wav
    return f"{base_name}.wav"


def transform_mp3_to_wav(fname, output_fname=None):
    sound = None
    file_extension = fname.split('.')[-1].lower()  # Get file extension and handle case insensitivity

    # Load the input file based on its extension
    if file_extension == 'mp3':
        sound = AudioSegment.from_mp3(fname)
    elif file_extension == 'wav':
        sound = AudioSegment.from_wav(fname)
    elif file_extension == 'flac':
        sound = AudioSegment.from_file(fname, format="flac")
    else:
        raise ValueError("Unknown file type")

    # Convert sound to mono and set frame rate to 16000Hz
    sound = sound.set_channels(1)  # mono
    sound = sound.set_frame_rate(16000)  # 16000Hz

    # If output_fname is provided, use it; otherwise create a temporary file
    if output_fname:
        this_file_name = output_fname
    else:
        # If no output filename is given, use the same directory and name with a .wav extension
        base_name = os.path.splitext(os.path.basename(fname))[0]
        this_file_name = os.path.join(os.path.dirname(fname), f"{base_name}.wav")

    # Export the processed sound as a wav file
    sound.export(this_file_name, format="wav")

    return this_file_name


def crop_wav(fname, segments=None):
    if not segments:
        segments = audio_cutter(fname)
    directory = "./data/temp_audios/"
    filenames = [directory + '/' + str(i) + '.wav' for i in range(len(segments))]
    if fname.split('.')[-1] == 'mp3':
        sound = AudioSegment.from_mp3(fname)
    elif fname.split('.')[-1] == 'wav':
        sound = AudioSegment.from_wav(fname)
    elif fname.split('.')[-1] == 'flac':
        sound = AudioSegment.from_file(fname, format="flac")
    else:
        raise ValueError("Unknown file type")
    sound.set_channels(1)
    sound.set_frame_rate(SAMPLE_RATE)
    for i in range(len(segments)):
        # Extract the first frames (60000 equals 60 seconds, 1000 per second)
        excerpt = sound[segments[i][0] * 1000: segments[i][1] * 1000]
        # write to disk
        excerpt.export(filenames[i], format="wav")
    return filenames

def crop_wav222(fname, segments=None):
    if not segments:
        segments = audio_cutter(fname)
    directory = "../data/error/"
    filenames = [directory + '/' + str(i) + '.wav' for i in range(len(segments))]
    if fname.split('.')[-1] == 'mp3':
        sound = AudioSegment.from_mp3(fname)
    elif fname.split('.')[-1] == 'wav':
        sound = AudioSegment.from_wav(fname)
    elif fname.split('.')[-1] == 'flac':
        sound = AudioSegment.from_file(fname, format="flac")
    else:
        raise ValueError("Unknown file type")
    sound.set_channels(1)
    sound.set_frame_rate(SAMPLE_RATE)
    for i in range(len(segments)):
        # Extract the first frames (60000 equals 60 seconds, 1000 per second)
        excerpt = sound[segments[i][0] * 1000: segments[i][1] * 1000]
        # write to disk
        excerpt.export(filenames[i], format="wav")
    return filenames


def calculate_duration(wav_file):
    audio = AudioSegment.from_wav(wav_file)
    channels = audio.channels
    frames = audio.frame_count()
    framerate = audio.frame_rate
    duration = len(audio) / 1000.0  # 单位为秒
    print(f"Channels:{channels}")
    print(f"Frames: {frames}")
    print(f"Framerate: {framerate}")
    print(f"Duration: {duration} seconds")

    return duration


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # Launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI to be installed.
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def audio_cutter(filename, length=360):
    duration = round(calculate_duration(filename), 2)
    segments = []
    if duration < length:
        return [[0, duration]]
    num = int(duration // length)
    remainder = duration % length
    # audio length is n x 6 minutes (n * 360 seconds)
    if remainder == 0:
        for i in range(num):
            start = i * length
            end = start + length
            segments.append([start, end])
    # if the remainder length is less than 3 minutes, merge with the previous segment
    elif remainder > 0 and remainder <= 180:
        for i in range(num):
            start = i * length
            if i == num - 1:
                end = duration
                segments.append([start, end])
            else:
                end = start + length
                segments.append([start, end])
    # If the remainder length is greater than 3 minutes, treat it as the last segment
    else:
        for i in range(num + 1):
            start = i * length
            end = min(start + length, duration)
            segments.append([start, end])
    return segments


def create_audio_files():
    directory = "./data/temp_audios"

    # 判断目录是否存在，如果不存在则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")


def create_transcripts_files():
    directory = "./data/transcripts"

    # 判断目录是否存在，如果不存在则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")


# 相对于上一级目录的文件夹结构
AUDIOS_DIR = "temp_audios/"
TRANSCRIPTS_DIR = "transcripts/"
ALIGNED_TRANSCRIPTS_DIR = "transcripts_aligned/"
MERGED_TRANSCRIPTS_DIR = "merged_transcript/"
DIARIZATION_DIR = "diarization/"

DIRS = [AUDIOS_DIR, TRANSCRIPTS_DIR, ALIGNED_TRANSCRIPTS_DIR,
        MERGED_TRANSCRIPTS_DIR, DIARIZATION_DIR]


def create_directory():
    # 获取当前文件的上一级目录的绝对路径
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

    # 遍历目录列表，在上一级目录的 data 文件夹下创建每个子目录
    for directory in DIRS:
        path = os.path.join(root, directory)
        # 判断目录是否存在，如果不存在则创建
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory '{path}' created.")
        else:
            print(f"Directory '{path}' already exists.")


def delete_files_in_directory(folder_path):
    # 删除文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除子目录及其内容
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def list_files(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, file))]

    # 示例用法


if __name__ == '__main__':
    # transform_mp3_to_wav("../data/roundtable.mp3", "../data/roundtable.wav")
    # calculate_duration("../6MinuteEnglish_new.wav")
    # res = load_audio("../data/6MinuteEnglish.mp3")
    # print(res)

    # res = audio_cutter("../data/6MinuteEnglish_new.wav")
    # print(res)
    # crop_wav("../data/6MinuteEnglish_new.wav", "../data/test/t1.wav")

    # crop_wav("../data/6MinuteEnglish_new.wav")

    # directory = "../data/temp_audios"
    # delete_files_in_directory(directory)
    #
    # crop_wav("../data/roundtable.wav")
    # crop_wav("../data/roundtable.mp3")
    # create_directory()

    # name = replace_extension_with_wav("../data/roundtable.mp3")
    # print(name)
    # segments = [[64.138, 84.804]]
    # crop_wav222('../data/VAD_R8009/temp_audios/2.wav', segments)
    create_directory()