import os
import pandas as pd
import numpy as np
import torch
import time
from pathlib import Path
from loguru import logger
from multiprocessing import Pool
from typing import Optional, Union, List
from pyannote.audio import Pipeline
import subprocess

SAMPLE_RATE = 16000


class SpeakerDiarizationPipeline:
    def __init__(self, num_workers=4, use_auth_token=None, device="cpu"):
        """
        初始化 SpeakerDiarizationPipeline 类
        :param num_workers: 并行处理的进程数量
        :param use_auth_token: Hugging Face 的身份验证 token
        :param device: 设备 (cpu 或 cuda)
        """
        self.num_workers = num_workers
        self.use_auth_token = use_auth_token
        self.device = torch.device(device) if isinstance(device, str) else device
        self.project_dir = Path("./data")
        self.temp_audios_dir = self.project_dir / "temp_audios"
        self.diarization_dir = self.project_dir / "diarization"
        self.transcripts_aligned_dir = self.project_dir / "transcripts_aligned"
        self.merged_transcript_dir = self.project_dir / "merged_transcript"
        self.model_name = "pyannote/speaker-diarization-3.1"
        self._check_directories()

    def _check_directories(self):
        """检查必要的目录，创建缺少的目录或报错"""
        # 检查 temp_audios 目录是否存在，不存在则报错
        if not self.temp_audios_dir.exists():
            raise FileNotFoundError(f"Required directory {self.temp_audios_dir} not found. Please ensure it exists.")

        # 检查 diarization 和 merged_transcript 目录，如果不存在则创建
        for directory in [self.diarization_dir, self.merged_transcript_dir]:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directory '{directory}' created.")

        # 检查 transcripts_aligned 目录是否存在，不存在则抛出错误
        if not self.transcripts_aligned_dir.exists():
            raise FileNotFoundError(
                f"Required directory {self.transcripts_aligned_dir} not found. Please ensure it exists.")

    def diarize_file(self, file_path: str):
        """执行单个音频文件的分割任务"""
        file_number = Path(file_path).stem
        logger.info(f"Processing file: {file_path}")

        diarization_model = Pipeline.from_pretrained(self.model_name, use_auth_token=self.use_auth_token).to(
            self.device)
        audio = self.load_audio(file_path)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }
        segments = diarization_model(audio_data)
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: round(x.start, 2))
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: round(x.end, 2))

        output_path = self.diarization_dir / f"{file_number}.csv"
        diarize_df.to_csv(output_path, index=False)
        return diarize_df

    # def assign_transcripts_to_speakers(self):
    #     """将转录文件与分割后的发言者对齐"""
    #     filenames = sorted(self.list_files(self.diarization_dir))
    #     for filename in filenames:
    #         diarize_df = pd.read_csv(filename)
    #         csv_name = Path(filename).name
    #         transcript_path = self.transcripts_aligned_dir / csv_name
    #
    #         if not transcript_path.exists():
    #             raise FileNotFoundError(f"Aligned transcript file {transcript_path} not found.")
    #
    #         transcript_segments = pd.read_csv(transcript_path).to_dict(orient='records')
    #         res = self._assign_word_speakers(diarize_df, transcript_segments)
    #
    #         merged_transcript_path = self.merged_transcript_dir / csv_name
    #         pd.DataFrame(res).to_csv(merged_transcript_path, index=False)

    def _assign_word_speakers(self, diarize_df, transcript_segments, fill_nearest=False):
        """将发言者分配给转录文件中的片段"""
        for seg in transcript_segments:
            # 分配发言者给片段
            diarize_df['intersection'] = np.minimum(diarize_df['end'], seg['end']) - np.maximum(diarize_df['start'],
                                                                                                seg['start'])
            diarize_df['union'] = np.maximum(diarize_df['end'], seg['end']) - np.minimum(diarize_df['start'],
                                                                                         seg['start'])

            # 根据发言者分配的逻辑
            if not fill_nearest:
                dia_tmp = diarize_df[diarize_df['intersection'] > 0]
            else:
                dia_tmp = diarize_df
            if len(dia_tmp) > 0:
                speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
                seg["speaker"] = speaker

            # 为每个词分配发言者
            if 'words' in seg:
                for word in seg['words']:
                    if 'start' in word:
                        diarize_df['intersection'] = np.minimum(diarize_df['end'], word['end']) - np.maximum(
                            diarize_df['start'], word['start'])
                        diarize_df['union'] = np.maximum(diarize_df['end'], word['end']) - np.minimum(
                            diarize_df['start'], word['start'])
                        if not fill_nearest:
                            dia_tmp = diarize_df[diarize_df['intersection'] > 0]
                        else:
                            dia_tmp = diarize_df
                        if len(dia_tmp) > 0:
                            speaker = \
                            dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
                            word["speaker"] = speaker

        return transcript_segments

    # def process_files_in_parallel(self):
    #     """并行处理 temp_audios 目录下的多个文件的分割任务"""
    #     file_list = self.list_files(self.temp_audios_dir)  # 从 temp_audios 目录中获取文件
    #     with Pool(processes=self.num_workers) as pool:
    #         pool.starmap(self.diarize_file, [(file_path,) for file_path in file_list])

    def process_and_assign_speakers_in_parallel(self, fill_nearest=False):
        """并行处理 temp_audios 目录下的多个文件的分割任务，并为转录片段分配发言者"""
        file_list = self.list_files(self.temp_audios_dir)  # 从 temp_audios 目录中获取文件

        # 并行处理文件
        with Pool(processes=self.num_workers) as pool:
            pool.starmap(self.process_file, [(file_path, fill_nearest) for file_path in file_list])

    def process_file(self, file_path, fill_nearest):
        """处理单个文件，执行发言者分配"""
        diarize_df = self.diarize_file(file_path)  # 获取分割后的发言者数据
        transcript_segments = self.load_transcript_segments(file_path)  # 加载相应的转录片段
        updated_segments = self._assign_word_speakers(diarize_df, transcript_segments,
                                                      fill_nearest=fill_nearest)  # 分配发言者

        # 保存更新后的转录片段
        output_path = self.merged_transcript_dir / f"{Path(file_path).stem}.csv"
        pd.DataFrame(updated_segments).to_csv(output_path, index=False)

    def load_transcript_segments(self, file_path):
        """加载转录片段的辅助方法"""
        # 获取相应的转录文件名
        transcript_file = self.transcripts_aligned_dir / f"{Path(file_path).stem}.csv"
        if transcript_file.exists():
            transcript_segments = pd.read_csv(transcript_file).to_dict(orient='records')
            return transcript_segments
        else:
            raise FileNotFoundError(f"Transcript file {transcript_file} not found.")

    def load_audio(self, file: str, sr: int = SAMPLE_RATE):
        """加载音频文件"""
        try:
            cmd = [
                "ffmpeg",
                "-nostdin",
                "-threads",
                "0",
                "-i", file,
                "-f", "s16le",
                "-ac", "1",
                "-acodec", "pcm_s16le",
                "-ar", str(sr),
                "-",
            ]
            out = subprocess.run(cmd, capture_output=True, check=True).stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    def list_files(self, directory: Path):
        """列出指定目录下的所有文件"""
        return [os.path.join(directory, file) for file in os.listdir(directory) if
                os.path.isfile(os.path.join(directory, file))]
