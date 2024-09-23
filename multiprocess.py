import multiprocessing as mp
from pyannote.audio import Audio
from pyannote.core import Segment
import time
from typing import List
from src.pyannote_model import DiarizationPipeline
from loguru import logger

audio = Audio()
path = 'data/6MinuteEnglish_new.wav'
segemnt = {'start':0, 'end':30}
SAMPLE_RATE = 16000
def segment_embedding(segment):
    start = segment["start"]
    # Whisper overshoots the end timestamp in the last segment
    end = segment["end"]
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(path, clip)
    print(waveform.shape)





# 假设你已有的函数，如 DiarizationPipeline 和 load_audio

def process_file(file_path: str, use_auth_token: str, model_name: str, sample_rate: int):
    start = time.time()
    diarize_model = DiarizationPipeline(use_auth_token=use_auth_token, device="cpu", model_name=model_name)
    diarize_segments = diarize_model(file_path)
    end = time.time()
    print(f"Processed {file_path} in {end - start} seconds")
    return diarize_segments


def process_files_in_parallel(file_paths: List[str], num_workers: int, use_auth_token: str, model_name: str):
    # 创建进程池
    with mp.Pool(processes=num_workers) as pool:
        # 将任务分配给多个进程
        results = pool.starmap(process_file,
                               [(file_path, use_auth_token, model_name, SAMPLE_RATE) for file_path in file_paths])

    return results




if __name__ == '__main__':
    # 假设这些是你的文件路径
    file_paths = [
        '../data/temp_audios/0_modified.wav',
        '../data/temp_audios/1.wav',
        '../data/temp_audios/2.wav',
    ]

    # 设置你希望使用的进程数（通常设置为CPU核心数）
    num_workers = 4  # 自动获取CPU核数

    # 并行处理文件
    use_auth_token = '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'
    model_name = "pyannote/speaker-diarization-3.1"
    start = time.time()
    results = process_files_in_parallel(file_paths, num_workers, use_auth_token, model_name)

    diarize_model = DiarizationPipeline(use_auth_token='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', device="cpu")

    end = time.time()
    logger.info(f"diarization process in {end - start} seconds")