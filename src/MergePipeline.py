import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from pyannote.audio import Audio
from pyannote.core import Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import random
from datetime import datetime
from loguru import logger


class AudioInfo:
    def __init__(self, filename, audio_name, transcript_name, number, original_df, modified_df, speaker_dict,
                 speaker_match_dict=None):
        self.filename = filename
        self.audio_name = audio_name
        self.transcript_name = transcript_name
        self.number = number
        self.original_df = original_df
        self.modified_df = modified_df
        self.speaker_dict = speaker_dict
        self.speaker_match_dict = speaker_match_dict


class MergeTranscriptionsPipeline:
    def __init__(self, segment_length=360, device="cpu", use_vad=False):
        self.segment_length = segment_length
        self.device = torch.device(device) if isinstance(device, str) else device
        self.embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device=self.device
        )
        self.project_dir = Path("./data")
        self.evaluation_reference_dir = self.project_dir / "evaluation_reference"
        self.use_vad = use_vad  # 新增 use_vad 参数
        self._check_directories()

    def _check_directories(self):
        """检查所需目录是否存在"""
        required_dirs = [self.project_dir / "diarization", self.project_dir / "transcripts",
                         self.project_dir / "merged_transcript"]
        for directory in required_dirs:
            if not directory.exists():
                raise FileNotFoundError(f"Required directory {directory} not found.")

        # 确保 evaluation_reference 目录存在
        if not self.evaluation_reference_dir.exists():
            self.evaluation_reference_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory '{self.evaluation_reference_dir}' created.")

    def is_near_multiple_of_segment_length(self, num, tolerance=0.1):
        """检查数字是否接近 segment_length 的倍数"""
        mod = num % self.segment_length
        return abs(mod) < tolerance or abs(mod - self.segment_length) < tolerance

    def is_near_cutting_points(self, num, points, tolerance=0.05):
        """判断某个数是否接近某些切割点"""
        for point in points:
            if abs(num - point) <= tolerance:
                return True
        return False

    def generate_embeddings(self, audio_info: AudioInfo):
        audio = audio_info.audio_name
        speaker_dict = audio_info.speaker_dict
        speaker_embeddings = {}

        for speaker, segments in speaker_dict.items():
            embeddings = []
            count = 0
            for segment in segments:
                if segment[0] > 2 and segment[0] <= 4:
                    number = self._random_sample_segments(segment[1], audio, embeddings)
                    count += number
                elif segment[0] > 4:
                    number = self._random_sample_segments(segment[1], audio, embeddings)
                    count += number
                if count >= 10:
                    break
            if count == 0:
                for segment in segments:
                    if segment[0] > 1:
                        embedding = self._tuple_embedding(segment[1], audio)
                        embeddings.append(embedding)
            if count == 0:
                for segment in segments:
                    embedding = self._tuple_embedding(segment[1], audio)
                    embeddings.append(embedding)
            speaker_embeddings[speaker] = self._aggregate_embeddings(embeddings)

        logger.info(f'Processed audio info: {audio}, number of speakers: {len(speaker_embeddings)}')
        return speaker_embeddings

    def _random_sample_segments(self, segment, audio, embeddings):
        start, end = segment[0], segment[1]
        total_duration = end - start
        num_samples = min(max(total_duration // 2, 2), 10)

        for _ in range(int(num_samples)):
            random_start = random.uniform(start, end - 2)
            random_end = random_start + 2
            embedding = self._tuple_embedding((random_start, random_end), audio)
            embeddings.append(embedding)
            if len(embeddings) >= num_samples:
                break
        return num_samples

    def _tuple_embedding(self, segment, audio_path):
        """从音频段生成嵌入"""
        audio = Audio()
        start, end = segment
        clip = Segment(start, end)
        waveform, sample_rate = audio.crop(audio_path, clip)
        return self.embedding_model(waveform[None])

    def _aggregate_embeddings(self, embeddings):
        """将嵌入合并为一个矩阵并求平均"""
        embeddings_matrix = np.vstack(embeddings)
        aggregated_embedding = np.mean(embeddings_matrix, axis=0)
        return aggregated_embedding

    def process_df(self, filename):
        """处理 CSV 文件并生成 AudioInfo 对象"""
        temp = filename.split('/')[-1]
        num = int(temp.split('.')[0])
        time_offset = num * self.segment_length  # 使用 segment_length 代替固定值

        audio_path = './data/temp_audios/'
        audio_name = audio_path + str(num) + ".wav"

        transcript_path = './data/temp_audios/'
        transcript_name = transcript_path + str(num) + ".csv"

        df = pd.read_csv(filename)
        df_copy = df.copy()

        # Add time to 'start' and 'end' columns
        df_copy['start'] += time_offset
        df_copy['end'] += time_offset

        # Extract unique speakers
        speakers = df_copy['speaker'].unique()

        # Initialize dictionary to store speaker information
        speaker_dict = {speaker: [] for speaker in speakers}

        # Populate dictionary with segment information
        for _, row in df.iterrows():
            duration = row['end'] - row['start']
            speaker_dict[row['speaker']].append((duration, (row['start'], row['end'])))

        # Sort in reversed order
        for k, v in speaker_dict.items():
            speaker_dict[k] = sorted(v, key=lambda x: x[0], reverse=True)

        return AudioInfo(filename, audio_name, transcript_name, number=num, original_df=df, modified_df=df_copy,
                         speaker_dict=speaker_dict)

    def compare_speakers(self, audio_infos):
        """比较不同 AudioInfo 对象中的发言者"""
        pre_audio = audio_infos[0]

        embeddings_dict = self.generate_embeddings(pre_audio)
        speakers = embeddings_dict.keys()
        csv_name = pre_audio.filename
        csv_key = csv_name.split('/')[-1]

        speakers_dict = {csv_key: {speaker: speaker for speaker in speakers}}

        for i in range(1, len(audio_infos)):
            new_audio = audio_infos[i]
            new_csv = new_audio.filename
            csv_key = new_csv.split('/')[-1]

            new_speaker_embeddings_dict = self.generate_embeddings(new_audio)
            speaker_matching = {}
            found_speakers = []

            for speaker, speaker_embeddings in new_speaker_embeddings_dict.items():
                add_to_base_dict = True
                maxval = 0
                most_matched = ''
                for base_speaker, base_embedding in embeddings_dict.items():
                    if speaker in found_speakers:
                        continue
                    logger.info(
                        f'Comparing speaker {base_speaker} with new audio {new_audio.filename} speaker {speaker}')
                    val = self.compare_embeddings(base_embedding, speaker_embeddings)
                    if val > maxval:
                        maxval = val
                        most_matched = base_speaker

                if maxval > 0.25:
                    found_speakers.append(speaker)
                    speaker_matching[speaker] = most_matched
                    add_to_base_dict = False

                if add_to_base_dict:
                    new_speaker_name = self.add_to_base_embeddings(embeddings_dict, speaker_embeddings)
                    speaker_matching[speaker] = new_speaker_name

            speakers_dict[csv_key] = speaker_matching

        return speakers_dict

    def compare_embeddings(self, embedding1, embedding2):
        """比较两个嵌入的余弦相似度"""
        dot_product = np.dot(embedding1, embedding2.T)
        norm_vector1 = np.linalg.norm(embedding1)
        norm_vector2 = np.linalg.norm(embedding2)
        cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
        return cosine_similarity

    def add_to_base_embeddings(self, embeddings_dict, embeddings):
        """将新的发言者嵌入添加到基础字典"""
        num = len(embeddings_dict)
        new_speaker_name = f"SPEAKER_{str(num).zfill(2)}"
        embeddings_dict[new_speaker_name] = embeddings
        return new_speaker_name

    def list_files(self, directory):
        """列出指定目录下的所有文件"""
        return [os.path.join(directory, file) for file in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, file))]

    def merge_all_transcriptions(self, times=None):
        """合并所有转录，如果 use_vad 为 True 且提供了 times 则调用 VAD 合并方式"""
        if self.use_vad and times is not None:
            return self.merge_VAD_transcriptions(times)
        else:
            diarization_files = self.list_files('./data/diarization/')
            diarization_files.sort()

            audio_infos = [self.process_df(file) for file in diarization_files]
            base_speakers = self.compare_speakers(audio_infos)

            transcript_files = self.list_files('./data/merged_transcript/')
            merged_csv = []
            for transcript_csv in transcript_files:
                num = int(transcript_csv.split('/')[-1].split('.')[0])
                time_offset = num * self.segment_length

                df = pd.read_csv(transcript_csv)
                df['start'] += time_offset
                df['end'] += time_offset

                csv_key = transcript_csv.split('/')[-1]
                if csv_key in base_speakers:
                    speaker_matching_dict = base_speakers[csv_key]
                    df['speaker'] = df['speaker'].replace(speaker_matching_dict)
                    merged_csv.append(df)

            current_datetime = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            merged_df = pd.concat(merged_csv)
            merged_df = merged_df.drop(columns=[col for col in merged_df.columns if 'Unnamed' in col or col == 'words'])

            merged_rows = []
            current_row = merged_df.iloc[0].copy()
            for i in range(1, len(merged_df)):
                next_row = merged_df.iloc[i]
                if (current_row['speaker'] == next_row['speaker']) and (next_row['start'] - current_row['end'] < 0.1):
                    current_row['text'] += ' ' + next_row['text']
                    current_row['end'] = next_row['end']
                elif (next_row['start'] - current_row['end'] < 0.1) and self.is_near_multiple_of_segment_length(
                        next_row['start']) and self.is_near_multiple_of_segment_length(current_row['end']):
                    current_row['text'] += ' ' + next_row['text']
                    current_row['end'] = next_row['end']
                else:
                    merged_rows.append(current_row)
                    current_row = next_row.copy()

            merged_rows.append(current_row)

            final_df = pd.DataFrame(merged_rows).reset_index(drop=True)
            final_res = './data/final_transcript_' + str(current_datetime) + '.csv'
            final_df.to_csv(final_res)

            return final_df

    def merge_VAD_transcriptions(self, times):
        """基于 VAD 和提供的 times 合并所有转录"""
        diarization_files = self.list_files('./data/diarization/')
        diarization_files.sort()

        audio_infos = [self.process_df(file) for file in diarization_files]
        base_speakers = self.compare_speakers(audio_infos)

        transcripts_list = self.list_files('./data/merged_transcript/')
        merged_csv = []
        for transcript_csv in transcripts_list:
            temp = transcript_csv.split('/')[-1]
            num = int(temp.split('.')[0])
            time = times[num]

            df = pd.read_csv(transcript_csv)

            # 增加时间偏移
            df['start'] = (df['start'] + time).round(2)
            df['end'] = (df['end'] + time).round(2)

            csv_key = transcript_csv.split('/')[-1]
            if csv_key in base_speakers:
                speaker_matching_dict = base_speakers[csv_key]
                df['speaker'] = df['speaker'].replace(speaker_matching_dict)
                merged_csv.append(df)

        current_datetime = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        merged_df = pd.concat(merged_csv)
        merged_df = merged_df.drop(columns=[col for col in merged_df.columns if 'Unnamed' in col or col == 'words'])

        merged_rows = []
        current_row = merged_df.iloc[0].copy()
        for i in range(1, len(merged_df)):
            next_row = merged_df.iloc[i]
            if (current_row['speaker'] == next_row['speaker']) and (next_row['start'] - current_row['end'] < 0.1):
                current_row['text'] += ' ' + next_row['text']
                current_row['end'] = next_row['end']
            elif ((next_row['start'] - current_row['end'] < 0.1) and self.is_near_cutting_points(next_row['start'],
                                                                                                 times)
                  and self.is_near_cutting_points(current_row['end'], times)):
                current_row['text'] += ' ' + next_row['text']
                current_row['end'] = next_row['end']
            else:
                merged_rows.append(current_row)
                current_row = next_row.copy()
        merged_rows.append(current_row)

        final_df = pd.DataFrame(merged_rows).reset_index(drop=True)
        final_res = './data/final_transcript_VAD_' + str(current_datetime) + '.csv'
        final_df.to_csv(final_res)

        return final_df

    def merge_all_transcriptions_for_evaluation(self, times=None):
        """合并所有转录用于评估，如果 use_vad 为 True 且提供了 times 则调用 VAD 合并方式"""
        if self.use_vad and times is not None:
            return self._merge_VAD_transcriptions_for_evaluation(times)
        else:
            return self._merge_for_evaluation_standard()

    def _merge_VAD_transcriptions_for_evaluation(self, times):
        """基于 VAD 和提供的 times 合并所有转录用于评估"""
        diarization_files = self.list_files('./data/diarization/')
        diarization_files.sort()

        audio_infos = [self.process_df(file) for file in diarization_files]
        base_speakers = self.compare_speakers(audio_infos)

        transcripts_list = self.list_files('./data/diarization/')
        merged_csv = []
        for transcript_csv in transcripts_list:
            temp = transcript_csv.split('/')[-1]
            num = int(temp.split('.')[0])
            time = times[num]

            df = pd.read_csv(transcript_csv)

            # 增加时间偏移
            df['start'] = (df['start'] + time).round(2)
            df['end'] = (df['end'] + time).round(2)

            csv_key = transcript_csv.split('/')[-1]
            if csv_key in base_speakers:
                speaker_matching_dict = base_speakers[csv_key]
                df['speaker'] = df['speaker'].replace(speaker_matching_dict)
                merged_csv.append(df)

        current_datetime = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        merged_df = pd.concat(merged_csv)
        merged_df = merged_df.drop(columns=[col for col in merged_df.columns if 'Unnamed' in col or col == 'words'])

        merged_rows = []
        current_row = merged_df.iloc[0].copy()
        for i in range(1, len(merged_df)):
            next_row = merged_df.iloc[i]
            if (current_row['speaker'] == next_row['speaker']) and (next_row['start'] - current_row['end'] < 0.1):
                current_row['end'] = next_row['end']
            elif (next_row['start'] - current_row['end'] < 0.1) and self.is_near_cutting_points(next_row['start'],
                                                                                                times) and self.is_near_cutting_points(
                    current_row['end'], times):
                current_row['end'] = next_row['end']
            else:
                merged_rows.append(current_row)
                current_row = next_row.copy()
        merged_rows.append(current_row)

        final_df = pd.DataFrame(merged_rows).reset_index(drop=True)
        final_res = './data/evaluation_reference/' + 'evaluated_diarization_VAD_' + str(current_datetime) + '.csv'
        final_df.to_csv(final_res)

        return final_df

    def _merge_for_evaluation_standard(self):
        """合并所有转录用于评估"""
        diarization_files = self.list_files('./data/diarization/')
        diarization_files.sort()

        audio_infos = [self.process_df(file) for file in diarization_files]
        base_speakers = self.compare_speakers(audio_infos)

        transcripts_list = self.list_files('./data/diarization/')
        merged_csv = []

        for transcript_csv in transcripts_list:
            temp = transcript_csv.split('/')[-1]
            num = int(temp.split('.')[0])
            time_offset = num * self.segment_length

            df = pd.read_csv(transcript_csv)

            # Add time to 'start' and 'end' columns
            df['start'] = (df['start'] + time_offset).round(2)
            df['end'] = (df['end'] + time_offset).round(2)

            csv_key = transcript_csv.split('/')[-1]
            if csv_key in base_speakers:
                speaker_matching_dict = base_speakers[csv_key]
                df['speaker'] = df['speaker'].replace(speaker_matching_dict)
                merged_csv.append(df)

        # 获取当前日期和时间
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        merged_df = pd.concat(merged_csv)
        merged_df = merged_df.drop(columns=[col for col in merged_df.columns if 'Unnamed' in col or col == 'words'])

        # 初始化一个列表用于存储合并后的行
        merged_rows = []
        current_row = merged_df.iloc[0].copy()

        for i in range(1, len(merged_df)):
            next_row = merged_df.iloc[i]

            # 检查当前行和下一行是否需要合并
            if (current_row['speaker'] == next_row['speaker']) and (next_row['start'] - current_row['end'] < 0.1):
                # 合并行并更新当前行的结束时间
                current_row['end'] = next_row['end']
            elif (next_row['start'] - current_row['end'] < 0.1) and self.is_near_multiple_of_segment_length(
                    next_row['start']) and self.is_near_multiple_of_segment_length(current_row['end']):
                # 合并行更新当前行的结束时间
                current_row['end'] = next_row['end']
            else:
                # 如果不需要合并，将当前行添加到结果中，并更新为下一行
                merged_rows.append(current_row)
                current_row = next_row.copy()

        # 添加最后一行
        merged_rows.append(current_row)

        # 创建一个新的 DataFrame
        merged_df = pd.DataFrame(merged_rows)
        # 重置索引，确保索引从0开始且没有多余的列
        merged_df = merged_df.reset_index(drop=True)

        # 保存合并后的结果
        final_res = self.evaluation_reference_dir / f"evaluated_diarization_{current_datetime}.csv"
        merged_df.to_csv(final_res)
        logger.info(f"Saved merged evaluation transcription to {final_res}")

        return merged_df
