import os
import subprocess
import whisperx
from pathlib import Path
from loguru import logger
from pydub import AudioSegment
import time
import random
import torchaudio
from pyannote.audio import Pipeline
import pandas as pd
import numpy as np



class AudioTranscriberPipeline:
    SAMPLE_RATE = 16000
    AUDIOS_DIR = Path("./data/temp_audios/")
    TRANSCRIPTS_DIR = Path("./data/transcripts/")
    ALIGNED_TRANSCRIPTS_DIR = Path("./data/transcripts_aligned/")
    MERGED_TRANSCRIPTS_DIR = Path("./data/merged_transcript/")
    DIARIZATION_DIR = Path("./data/diarization/")
    MODEL_DIR = Path("./models/fast_whisper/")

    def __init__(self, model_type='base', device="cpu", batch_size=4, compute_type="int8", segment_length=360,
                 user_token=None):
        self.model_type = model_type
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.segment_length = segment_length  # 默认按6分钟（360秒）裁剪音频
        self.user_token = user_token  # Added for VAD
        self._create_directories()

    def _create_directories(self):
        """创建项目中需要的所有目录"""
        dirs = [
            self.AUDIOS_DIR, self.TRANSCRIPTS_DIR, self.ALIGNED_TRANSCRIPTS_DIR,
            self.MERGED_TRANSCRIPTS_DIR, self.DIARIZATION_DIR
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory '{directory}' created or already exists.")

    def apply_vad(self, filename):
        """Apply VAD on the audio file using the user_token"""
        waveform, sample_rate = torchaudio.load(filename)
        audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}
        vad = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                       use_auth_token=self.user_token)
        res = vad(audio_in_memory)
        return res

    def voice_segments_to_dataframe(self, vad_result):
        """Convert VAD results into a DataFrame"""
        seg_info_list = []
        for speech_turn, track, speaker in vad_result.itertracks(yield_label=True):
            this_seg_info = {'start': np.round(speech_turn.start, 2),
                             'end': np.round(speech_turn.end, 2),
                             'track': track}
            this_df = pd.DataFrame.from_dict({track: this_seg_info}, orient='index')
            seg_info_list.append(this_df)
        all_seg_infos_df = pd.concat(seg_info_list, axis=0).reset_index()
        return all_seg_infos_df

    def merge_intervals(self, df, threshold=0.4):
        """Merge intervals that are closer than a given threshold"""
        merged_rows = []
        current_start = df.iloc[0]['start']
        current_end = df.iloc[0]['end']
        current_track = df.iloc[0]['track']

        for i in range(1, len(df)):
            next_start = df.iloc[i]['start']
            next_end = df.iloc[i]['end']
            next_track = df.iloc[i]['track']

            if next_start - current_end <= threshold:
                current_end = max(current_end, next_end)
                current_track += ',' + next_track
            else:
                merged_rows.append({'start': current_start, 'end': current_end, 'track': current_track})
                current_start = next_start
                current_end = next_end
                current_track = next_track

        merged_rows.append({'start': current_start, 'end': current_end, 'track': current_track})
        return pd.DataFrame(merged_rows)

    def vad_cut(self, fname, threshold=0.4):
        """使用VAD结果剪切音频并根据阈值合并区间"""
        duration = self.calculate_duration(fname)
        segments = []
        if duration < self.segment_length:
            segments.append([0, duration])
        else:
            dia_df = self.voice_segments_to_dataframe(self.apply_vad(fname))
            df = self.merge_intervals(dia_df, threshold)

            previous_point = 0

            # 遍历DataFrame的每一行
            for i in range(1, len(df)):
                current_start = df.iloc[i]['start']
                current_end = df.iloc[i]['end']

                # 如果当前区间长度超过segment_length，则进行切割
                if current_end - previous_point > self.segment_length:
                    previous_end = df.iloc[i - 1]['end']
                    mid = round((previous_end + current_start) / 2, 2)
                    segments.append([previous_point, mid])
                    previous_point = mid

                # 处理最后一个区间
                if i == len(df) - 1:
                    final_end = df.iloc[i]['end']
                    if final_end - previous_point < self.segment_length // 2:
                        segments[-1][1] = final_end
                    else:
                        segments.append([previous_point, final_end])

        return segments

    def segments_cutting_times(self, segments):
        times = [0]
        for i in range(1, len(segments)):
            times.append(segments[i][0])
        return times

    def convert_audio_to_wav(self, file):
        """将输入音频文件转换为 WAV 格式"""
        output_file = self.replace_extension_with_wav(file)
        cmd = [
            "ffmpeg", "-nostdin", "-threads", "0", "-i", file, "-f", "wav", "-ac", "1",
            "-acodec", "pcm_s16le", "-ar", str(self.SAMPLE_RATE), output_file
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            logger.info(f"Conversion successful. Saved as {output_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during conversion: {e}")
        return output_file

    def replace_extension_with_wav(self, file_name):
        """将文件扩展名替换为 .wav"""
        base_name, ext = os.path.splitext(file_name)
        base_name = base_name + '_' + str(random.randint(0, 100))
        return f"{base_name}.wav"

    def segment_audio(self, fname, use_vad=False):
        """Segment WAV file based on VAD or equal length"""
        if use_vad and self.user_token:
            segments = self.vad_cut(fname)  # Using VAD to cut the audio
        else:
            segments = self.audio_cutter(fname)

        times = self.segments_cutting_times(segments)

        filenames = [self.AUDIOS_DIR / f"{i}.wav" for i in range(len(segments))]
        sound = AudioSegment.from_wav(fname)
        sound.set_channels(1)
        sound.set_frame_rate(self.SAMPLE_RATE)

        for i, segment in enumerate(segments):
            excerpt = sound[segment[0] * 1000: segment[1] * 1000]
            excerpt.export(filenames[i], format="wav")
        logger.info(f"Audio cropped into {len(filenames)} segments.")
        return times

    def audio_cutter(self, filename):
        """按给定的 segment_length 裁剪音频"""
        duration = round(self.calculate_duration(filename), 2)
        segments = []
        length = self.segment_length  # 使用初始化时提供的分段长度
        half_length = self.segment_length // 2
        if duration < length:
            return [[0, duration]]
        num = int(duration // length)
        remainder = duration % length
        if remainder == 0:
            for i in range(num):
                start = i * length
                end = start + length
                segments.append([start, end])
        elif remainder > 0 and remainder <= half_length:
            for i in range(num):
                start = i * length
                if i == num - 1:
                    end = duration
                    segments.append([start, end])
                else:
                    end = start + length
                    segments.append([start, end])
        else:
            for i in range(num + 1):
                start = i * length
                end = min(start + length, duration)
                segments.append([start, end])
        return segments

    def calculate_duration(self, wav_file):
        """计算音频文件的时长"""
        audio = AudioSegment.from_wav(wav_file)
        channels = audio.channels
        frames = audio.frame_count()
        framerate = audio.frame_rate
        duration = len(audio) / 1000.0  # 单位为秒

        logger.info(f"Channels: {channels}")
        logger.info(f"Frames: {frames}")
        logger.info(f"Framerate: {framerate}")
        logger.info(f"Duration: {duration:.2f} seconds")

        return duration

    def transcribe_audio_files(self, directory):
        """对目录中的音频文件进行转录"""
        files = [f for f in directory.iterdir() if f.suffix == '.wav']
        model = whisperx.load_model(self.model_type, self.device, compute_type=self.compute_type,
                                    download_root=str(self.MODEL_DIR))

        results = {}
        for filename in files:
            file_number = filename.stem
            logger.info(f"Start transcribing audio-{file_number}.wav")
            start = time.time()

            # 加载音频并转录
            audio = whisperx.load_audio(str(filename))
            result = model.transcribe(audio, batch_size=self.batch_size)

            # 添加语言信息并保存
            language = result['language']
            output_file = self.TRANSCRIPTS_DIR / f"{file_number}.csv"
            logger.info(f"Transcription of {filename} finished in {time.time() - start:.2f} seconds.")

            df = pd.DataFrame(result["segments"]).reset_index()
            df['language'] = language
            df.to_csv(output_file, index=False)

            results[file_number] = result
        return results

    def process_and_transcribe(self, input_file, use_vad=False):
        """一键处理音频文件，包括转换、裁剪、转录"""
        # 转换音频为 WAV
        wav_file = self.convert_audio_to_wav(input_file)

        # 裁剪音频，默认按 segment_length 或 VAD 切分
        times = self.segment_audio(wav_file, use_vad)

        # 转录裁剪后的音频
        self.transcribe_audio_files(self.AUDIOS_DIR)

        return times

