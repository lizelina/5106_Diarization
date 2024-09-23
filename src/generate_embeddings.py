from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import torch
from pyannote.audio import Audio
from pyannote.core import Segment
from src.audio_utils import calculate_duration
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

import numpy as np
import random
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from scipy.spatial.distance import cdist


# embedding_model = PretrainedSpeakerEmbedding(
#     "pyannote/embedding",
#     device=torch.device("cpu"),
# use_auth_token='HuggingFace Token')

embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cpu") )



audio = Audio(sample_rate=16000, mono="downmix")




def segment_embedding(segment, filepath, duration):
    audio = Audio()
    start = segment["start"]

    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(filepath, clip)
      # Ensure waveform is single-channel by selecting the first channel if necessary
      # Select the first channel
    return embedding_model(waveform[None]).shape

def tuple_embedding(tuple, filepath):
    audio = Audio()
    start = tuple[0]
    end = tuple[1]
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(filepath, clip)
    return embedding_model(waveform[None])

def generate_embeddings(segments, filepath):
    embeddings = np.zeros(shape=(len(segments), 192))
    duration = calculate_duration(filepath)
    for i, segment in enumerate(segments):
        embeddings[i] = segment_embedding(segment, filepath, duration)
        print(embeddings[i].shape)
    # 函数将 NaN 值替换为零，并可以将无穷大值替换为有限的数值
    embeddings = np.nan_to_num(embeddings)
    return embeddings



def speaker_verification():
    # extract embedding for a speaker speaking between t=3s and t=6s
    speaker1 = Segment(36,40)
    waveform1, sample_rate = audio.crop("../data/temp_audios/0_modified.wav", speaker1)
    embedding1 = embedding_model(waveform1[None])
    print(embedding1.shape)

    # extract embedding for a speaker speaking between t=7s and t=12s
    speaker2 = Segment(18,22)
    waveform2, sample_rate = audio.crop("../data/temp_audios/1.wav", speaker2)
    embedding2 = embedding_model(waveform2[None])
    print(embedding2.shape)

    # compare embeddings using "cosine" distance
    distance = cdist(embedding1, embedding2, metric="cosine")

    dot_product = np.dot(embedding1, embedding2.T)

    # 计算向量的范数
    norm_vector1 = np.linalg.norm(embedding1)
    norm_vector2 = np.linalg.norm(embedding2)

    # 计算余弦相似度
    cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
    print(cosine_similarity)
    print(type(distance))
    print(distance[0][0])



def generate_embeddings_from_AudioInfo(audioInfo):
    # audio = "../data/temp_audios/0_modified.wav"
    audio = audioInfo.audio_name
    speaker_dict = audioInfo.speaker_dict
    speaker_embeddings = {}
    for speaker, segments in speaker_dict.items():
        embeddings = []
        count = 0
        for segment in segments:
            if segment[0] > 3 and segment[0] <= 6 and count < 3:
                new_segment = middle_3_seconds(segment[1])
                embedding = tuple_embedding(new_segment, audio)
                embeddings.append(embedding)
                count += 1
            elif segment[0] > 6 and count < 3:
                new_segments = extract_segments(segment[1])
                for seg in new_segments:
                    embedding = tuple_embedding(seg, audio)
                    embeddings.append(embedding)
                    count += 1
                    if count > 3:
                        break
        if count == 0:
            for segment in segments[:3]:
                embedding = tuple_embedding(segment[1], audio)
                embeddings.append(embedding)
        speaker_embeddings[speaker] = embeddings
    print(f'Processed audio info: {audio}, number of speakers: {len(speaker_embeddings)}' )
    return speaker_embeddings

def generate_embeddings_from_AudioInfo_enhanced(audioInfo):
    # audio = "../data/temp_audios/0_modified.wav"
    audio = audioInfo.audio_name
    speaker_dict = audioInfo.speaker_dict
    speaker_embeddings = {}
    for speaker, segments in speaker_dict.items():
        embeddings = []
        count = 0
        for segment in segments:
            if segment[0] > 2 and segment[0] <= 4:
                number = random_sample_segments(segment[1], audio, embeddings)
                count += number
            elif segment[0] > 4:
                number = random_sample_segments(segment[1], audio, embeddings)
                count += number
            if count >= 10:
                break
        if count == 0:
            for segment in segments:
                if segment > 1:
                    embedding = tuple_embedding(segment[1], audio)
                    embeddings.append(embedding)
        if count == 0:
            for segment in segments:
                embedding = tuple_embedding(segment[1], audio)
                embeddings.append(embedding)
        speaker_embeddings[speaker] = aggregate_embeddings(embeddings)
    print(f'Processed audio info: {audio}, number of speakers: {len(speaker_embeddings)}' )
    return speaker_embeddings

def random_sample_segments(segment, audio, embeddings):
    start, end = segment[0], segment[1]
    total_duration = end - start
    #  Adaptive number
    num_samples = min(max(total_duration//2, 2), 10)

    for _ in range(int(num_samples)):
        random_start = random.uniform(start, end - 2)
        random_end = random_start + 2
        embedding = tuple_embedding((random_start, random_end), audio)
        embeddings.append(embedding)
        if len(embeddings) >= num_samples:
            break
    return num_samples

def aggregate_embeddings(embeddings):
    # 将这些嵌入合并为一个矩阵，形状为 (n, 1, 192)
    embeddings_matrix = np.vstack(embeddings)

    # 对这些嵌入求平均值，axis=0 表示对第0维（嵌入的数量）进行平均
    aggregated_embedding = np.mean(embeddings_matrix, axis=0)
    return aggregated_embedding

def middle_3_seconds(segment):
    start_time = segment[0]
    end_time = segment[1]
    # 计算总时长
    total_duration = end_time - start_time

    # 计算中间4秒的起始时间
    middle_start = start_time + (total_duration - 3) / 2

    # 计算中间4秒的结束时间
    middle_end = middle_start + 3

    # 保留两位小数
    middle_start_rounded = round(middle_start, 2)
    middle_end_rounded = round(middle_end, 2)
    return (middle_start_rounded, middle_end_rounded)

def extract_segments(segment, segment_duration=3):
    start = segment[0]
    end = segment[1]
    total_duration = end - start
    num_segments = total_duration // segment_duration  # 计算可提取的段数
    segments = []

    if num_segments >= 2:  # 确保至少有两个4秒段
        middle_start = start + (total_duration - num_segments * segment_duration) / 2
        for i in range(int(num_segments)):
            segment_start = middle_start + i * segment_duration
            segment_end = segment_start + segment_duration
            segments.append((round(segment_start, 2), round(segment_end, 2)))

    return segments

def compare_embeddings(speaker1_embeddings, speaker2_embeddings):
    count = 0
    value = 0
    for embedding1 in speaker1_embeddings:
        for embedding2 in speaker2_embeddings:
            val = calculate_cosine_simiarity(embedding1, embedding2)
            count += 1
            value += val
    avg_cos = value / count
    print(avg_cos)
    return  avg_cos


def calculate_cosine_simiarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2.T)

    # 计算向量的范数
    norm_vector1 = np.linalg.norm(embedding1)
    norm_vector2 = np.linalg.norm(embedding2)

    # 计算余弦相似度
    cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
    return cosine_similarity



if __name__ == '__main__':
    # print(extract_segments(9.84, 22.27))
    speaker_verification()