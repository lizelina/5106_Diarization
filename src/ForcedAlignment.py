import os
import pandas as pd
import whisperx
import time
from pathlib import Path
from loguru import logger
from multiprocessing import Pool

class WhisperXAligner:
    def __init__(self, device="cpu", num_workers=4):
        """
        初始化对齐器类
        :param device: 设备 (cpu 或 cuda)
        :param num_workers: 并行处理的进程数量
        """
        self.device = device
        self.num_workers = num_workers
        self.project_dir = Path("./data")  # 默认的项目根目录
        self.temp_audios_dir = self.project_dir / "temp_audios"
        self.transcripts_dir = self.project_dir / "transcripts"
        self.aligned_transcripts_dir = self.project_dir / "transcripts_aligned"
        self.alignment_model_dir = Path("./models/alignment")  # 默认的对齐模型目录
        self._check_directories()

    def _check_directories(self):
        """检查所需的目录是否存在"""
        # 检查 temp_audios 和 transcripts 目录是否存在
        if not self.temp_audios_dir.exists() or not self.transcripts_dir.exists():
            raise FileNotFoundError(f"Required directories {self.temp_audios_dir} or {self.transcripts_dir} not found in project directory.")

        # 如果 transcripts_aligned 目录不存在，创建该目录
        if not self.aligned_transcripts_dir.exists():
            self.aligned_transcripts_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory '{self.aligned_transcripts_dir}' created.")

    def list_files(self, directory):
        """列出指定目录下的所有文件"""
        return [os.path.join(directory, file) for file in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, file))]

    def align_transcripts(self, filename):
        """对齐音频与转录文件"""
        file_number = Path(filename).stem

        # 查找对应的转录文件
        transcript_path = self.transcripts_dir / f"{file_number}.csv"
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript file {transcript_path} not found for audio {filename}.")

        # 从转录文件中获取语言信息
        df = pd.read_csv(transcript_path)
        language = df['language'].iloc[0]  # 从第一行获取语言信息
        logger.info(f"Using language: {language} for file {filename}")

        # 加载音频和对齐模型
        audio = whisperx.load_audio(filename)
        model_a, metadata = whisperx.load_align_model(language_code=language, device=self.device, model_dir=str(self.alignment_model_dir))

        df_dict = df.to_dict(orient="records")

        start = time.time()
        aligned_result = whisperx.align(df_dict, model_a, metadata, audio, self.device, return_char_alignments=False)
        end = time.time()
        logger.info(f"Processed {filename} in {end - start:.2f} seconds")

        # 保存对齐后的转录文件
        aligned_path = self.aligned_transcripts_dir / f"{file_number}.csv"
        pd.DataFrame(aligned_result["segments"]).to_csv(aligned_path)

    def align_in_parallel(self):
        """并行处理 temp_audios 目录下的所有音频文件的对齐操作"""
        file_list = self.list_files(self.temp_audios_dir)  # 获取 temp_audios 下的所有音频文件
        with Pool(processes=self.num_workers) as pool:
            pool.starmap(self.align_transcripts, [(filename,) for filename in file_list])
