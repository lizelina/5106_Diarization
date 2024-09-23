from typing import Optional, Union, List
import multiprocessing as mp

from pyannote.audio import Pipeline
import pandas as pd
import numpy as np
import torch
from src.audio_utils import load_audio, SAMPLE_RATE
from loguru import logger
import time
import os
from src.audio_utils import list_files
from src.segment_merger import merge_segments

class DiarizationPipeline:
    def __init__(
        self,
        model_name="pyannote/speaker-diarization-3.1",
        use_auth_token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.model = Pipeline.from_pretrained(model_name, use_auth_token=use_auth_token).to(device)

    def __call__(self, audio: Union[str, np.ndarray], num_speakers=None, min_speakers=None, max_speakers=None):
        file_number = ""
        if isinstance(audio, str):
            # 提取文件名部分
            file_name = audio.split('/')[-1]
            # 去掉文件扩展名，提取数字部分
            file_number = file_name.split('.')[0]
            audio = load_audio(audio)

        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }
        segments = self.model(audio_data, num_speakers = num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: round(x.start, 2))
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: round(x.end, 2))
        path = "./data/diarization/"
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory '{path}' created.")
        path = path + file_number + ".csv"

        return diarize_df.to_csv(path)
def apply_diarization_model(pyannote_token, fname):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                        use_auth_token=pyannote_token)

    # apply the pipeline to an audio file
    diarization = pipeline(fname)

    with open('.audio.rttm', 'w') as rttm:
        diarization.write_rttm(rttm)

    return diarization


def diarization_inference(fname, pyannote_token):
    diarization = apply_diarization_model(pyannote_token, fname)
    dia_df = diarization_to_dataframe(diarization)

    return dia_df


def diarization_to_dataframe(diarization):
    seg_info_list = []
    for speech_turn, track, speaker in diarization.itertracks(yield_label=True):
        this_seg_info = {'start': np.round(speech_turn.start, 2),
                         'end': np.round(speech_turn.end, 2),
                         'speaker': speaker}
        this_df = pd.DataFrame.from_dict({track: this_seg_info},
                                         orient='index')

        seg_info_list.append(this_df)

    all_seg_infos_df = pd.concat(seg_info_list, axis=0)
    all_seg_infos_df = all_seg_infos_df.reset_index()

    return all_seg_infos_df


def process_file(file_path: str, use_auth_token: str, model_name: str):
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
                               [(file_path, use_auth_token, model_name) for file_path in file_paths])

    return results


if __name__ == '__main__':
    diarize_model = DiarizationPipeline(use_auth_token='HuggingFace Token', device="cpu")
    start = time.time()
    logger.info(f'start diarization')

    # add min/max number of speakers if known
    # diarize_segments = diarize_model("../data/temp_audios/t.wav")
    # 假设这些是你的文件路径
    file_paths = list_files("../data/temp_audios/")

    # 设置你希望使用的进程数（通常设置为CPU核心数）
    num_workers = 6 # 自动获取CPU核数


    # 并行处理文件
    use_auth_token = 'HuggingFace Token'
    model_name = "pyannote/speaker-diarization-3.1"

    results = process_files_in_parallel(file_paths, num_workers, use_auth_token, model_name)

    logger.info(f'end time{time.time() - start}')



    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    # result = whisperx.assign_word_speakers(diarize_segments, result)
    # print(diarize_segments)
    # merge_segments(diarize_segments)

    # print(result["segments"])  # segments are now assigned speaker IDs
    # pd.DataFrame(result["segments"]).to_csv("diarized_segments.csv", index=False)
