import torchaudio
from pyannote.audio import Pipeline
import pandas as pd
import numpy as np
from loguru import logger
import time


def apply_vad(filename, user_token):
    waveform, sample_rate = torchaudio.load(filename)

    audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}
    vad = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                   use_auth_token=user_token)
    res = vad(audio_in_memory)

    return res

def voice_segments_to_dataframe(vad):
    seg_info_list = []
    for speech_turn, track, speaker in vad.itertracks(yield_label=True):
        this_seg_info = {'start': np.round(speech_turn.start, 2),
                         'end': np.round(speech_turn.end, 2),
                         'track': track}
        this_df = pd.DataFrame.from_dict({track: this_seg_info},
                                         orient='index')

        seg_info_list.append(this_df)

    all_seg_infos_df = pd.concat(seg_info_list, axis=0)
    all_seg_infos_df = all_seg_infos_df.reset_index()

    return all_seg_infos_df

    # start = time.time()
    # logger.info('Start voice activity detection')
    # res = vad_inference('../6MinuteEnglish.wav', user_token='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')
    # logger.info(f'End voice activity detection:{time.time() - start}')
    # res.to_csv('../vad.csv')
    # print(res)


def split_segments(segments, max_duration=60):
    result = []

    for segment in segments:
        start = segment['start']
        end = segment['end']
        track = segment['track']

        # 如果当前片段时长超过最大限制，则进行切割
        while (end - start) > max_duration:
            new_end = start + max_duration
            result.append({'start': start, 'end': new_end, 'track': track})
            start = new_end  # 更新start，开始下一个子片段

        # 处理最后一段（或者时长不超过max_duration的片段）
        result.append({'start': start, 'end': end, 'track': track})

    return result

def vad_inference(fname, user_token):
    diarization = apply_vad(fname, user_token)
    dia_df = voice_segments_to_dataframe(diarization)

    return dia_df
def merge_intervals(df, threshold=0.4):
    # 确保输入 DataFrame 包含所需的列
    if not all(col in df.columns for col in ['start', 'end', 'track']):
        raise ValueError("DataFrame must contain 'start', 'end', and 'track' columns")

    # 初始化合并后的结果列表
    merged_rows = []

    # 当前的合并区间
    current_start = df.iloc[0]['start']
    current_end = df.iloc[0]['end']
    current_track = df.iloc[0]['track']

    # 遍历 DataFrame 的每一行
    for i in range(1, len(df)):
        next_start = df.iloc[i]['start']
        next_end = df.iloc[i]['end']
        next_track = df.iloc[i]['track']

        # 检查间隔是否小于阈值
        if next_start - current_end <= threshold:
            # 如果间隔小于阈值，则扩展当前区间
            current_end = max(current_end, next_end)
            current_track += ',' + next_track
        else:
            # 如果间隔大于阈值，保存当前合并区间，并更新为新的区间
            merged_rows.append({'start': current_start, 'end': current_end, 'track': current_track})
            current_start = next_start
            current_end = next_end
            current_track = next_track

    # 添加最后一个区间
    merged_rows.append({'start': current_start, 'end': current_end, 'track': current_track})

    # 转换为 DataFrame
    merged_df = pd.DataFrame(merged_rows)

    return merged_df

def vad_cut(fname, user_token, threshold = 0.4, length = 360):
    dia_df = vad_inference(fname, user_token)
    df = merge_intervals(dia_df, threshold)
    # 确保输入 DataFrame 包含所需的列
    if not all(col in df.columns for col in ['start', 'end']):
        raise ValueError("DataFrame must contain 'start', 'end' columns")

    # 初始化合并后的结果列表
    segments = []

    previous_point = 0

    # 遍历 DataFrame 的每一行
    for i in range(1, len(df)):
        current_start = df.iloc[i]['start']
        current_end = df.iloc[i]['end']

        # 检查间隔是否小于阈值
        if current_end - previous_point > length:
            # 如果间隔小于阈值，则扩展当前区间
            previous_end = df.iloc[i - 1]['end']
            mid = round((previous_end + current_start)/2, 2)
            segments.append([previous_point, mid])
            previous_point = mid

        if i == len(df) - 1:
            final_end = df.iloc[i]['end']
            if final_end - previous_point < length//2:
                segments[-1][1] = final_end
            else:
                segments.append([previous_point, final_end])

    return segments

def segments_cutting_times(segments):
    times = [0]
    for i in range(1, len(segments)):
        times.append(segments[i][0])
    return times





if __name__ == '__main__':
    segments = [
        {'start': 1.19, 'end': 5.86, 'track': 'A'},
        {'start': 7.89, 'end': 13.28, 'track': 'B'},
        {'start': 13.68, 'end': 18.93, 'track': 'C'},
        {'start': 19.33, 'end': 24.58, 'track': 'D'},
        {'start': 24.99, 'end': 48.87, 'track': 'E'},
        {'start': 49.26, 'end': 65.21, 'track': 'F'},
        {'start': 65.49, 'end': 89.22, 'track': 'G'},
        {'start': 89.77, 'end': 96.39, 'track': 'H'},
        {'start': 96.61, 'end': 98.93, 'track': 'I'},
        {'start': 99.39, 'end': 104.43, 'track': 'J'},
        {'start': 104.65, 'end': 122.18, 'track': 'K'},
        {'start': 122.55, 'end': 147.04, 'track': 'L'},
        {'start': 147.86, 'end': 168.23, 'track': 'M'},
        {'start': 168.78, 'end': 188.95, 'track': 'N'},
        {'start': 189.10, 'end': 197.57, 'track': 'O'},
        {'start': 198.54, 'end': 232.96, 'track': 'P'},
        {'start': 233.83, 'end': 246.87, 'track': 'Q'},
        {'start': 247.18, 'end': 257.48, 'track': 'R'},
        {'start': 257.82, 'end': 280.44, 'track': 'S'},
        {'start': 280.67, 'end': 293.47, 'track': 'T'},
        {'start': 294.02, 'end': 322.07, 'track': 'U'},
        {'start': 322.52, 'end': 325.98, 'track': 'V'},
        {'start': 326.20, 'end': 338.18, 'track': 'W'},
        {'start': 339.21, 'end': 355.23, 'track': 'X'},
        {'start': 355.88, 'end': 366.41, 'track': 'Y'},
        {'start': 367.55, 'end': 372.18, 'track': 'Z'},
    ]
    # res = split_segments(segments)
    # print(res)
    # res = vad_inference("../data/roundtable.wav", user_token='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')
    # merged_df = merge_intervals(res)
    # s = vad_cut(merged_df, 400)
    # print(s)
    # res = vad_inference("../6Minute_short.wav", user_token='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')
    # print(res)

    segments = [[0, 2],[2, 3], [4, 6]]
    cutting_times = segments_cutting_times(segments)
    print(cutting_times)


