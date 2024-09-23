import pandas as pd
import os
from src.segment_merger import merge_segments
from src.generate_embeddings import speaker_verification, generate_embeddings_from_AudioInfo, compare_embeddings, generate_embeddings_from_AudioInfo_enhanced
from datetime import datetime

class AudioInfo:
    def __init__(self, filename, audio_name, transcript_name, number, original_df, modified_df, speaker_dict, speaker_match_dict=None):
        self.filename = filename
        self.audio_name = audio_name
        self.transcript_name = transcript_name
        self.number = number
        self.original_df = original_df
        self.modified_df = modified_df
        self.speaker_dict = speaker_dict


def list_files(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, file))]

CHUNK_LENGTH = 360

def process_df(filename, chunk_length = 360):
    temp = filename.split('/')[-1]
    num = int(temp.split('.')[0])
    time = num * chunk_length  # 6 minutes

    audio_path = './data/temp_audios/'
    audio_name = audio_path + str(num) + ".wav"

    transcript_path = './data/temp_audios/'
    transcript_name = transcript_path + str(num) + ".csv"

    df = pd.read_csv(filename)
    df_copy = df.copy()

    # Add time to 'start' and 'end' columns

    df_copy['start'] += time
    df_copy['end'] += time

    # Extract unique speakers
    speakers = df_copy['speaker'].unique()

    # Initialize dictionary to store speaker information
    speaker_dict = {speaker: [] for speaker in speakers}

    # Populate dictionary with segment information
    for _, row in df.iterrows():
        duration = row['end'] - row['start']
        speaker_dict[row['speaker']].append((duration, (row['start'], row['end'])))

    # sort in reversed order
    for k, v in speaker_dict.items():
        speaker_dict[k] = sorted(v, key=lambda x: x[0], reverse=True)

    return AudioInfo(filename, audio_name, transcript_name, number=num, original_df=df, modified_df=df_copy, speaker_dict=speaker_dict)


def compare_speakers(fileinfos):
    pre_audio = fileinfos[0]
    # Store the first speaker_dict as base dict
    embeddings_dict = generate_embeddings_from_AudioInfo(pre_audio)
    speakers = embeddings_dict.keys()
    csv_name = pre_audio.filename
    csv_key = csv_name.split('/')[-1]

    # Map speakers to corresponding csv file
    speakers_dict = {csv_key: {speaker: speaker for speaker in speakers}}

    for i in range(1, len(fileinfos)):
        new_audio = fileinfos[i]
        new_csv = new_audio.filename

        csv_key = new_csv.split('/')[-1]

        new_speaker_embeddings_dict = generate_embeddings_from_AudioInfo(new_audio)
        speaker_matching = {}
        found_speakers = []

        for speaker, speaker_embeddings in new_speaker_embeddings_dict.items():
            add_to_base_dict = True
            maxval = 0
            most_matched = ''
            for base_speaker, base_embedding in embeddings_dict.items():
                if speaker in found_speakers:
                    continue
                print(f'compare speaker {base_speaker} with new audio {new_audio.filename} speaker {speaker}')
                val = compare_embeddings(base_embedding, speaker_embeddings)
                # Keep the speaker that have the highest embedding cosine similarity
                if val > maxval:
                    maxval = val
                    most_matched = base_speaker
            # Threshld is 0.25 from speechbrain source code _MS810@.wav
            if maxval > 0.25:
                found_speakers.append(speaker)
                speaker_matching[speaker] = most_matched
                add_to_base_dict = False

            # less than threshold, means current speaker is a new speaker
            # add its embeddings to base embedding dict
            if add_to_base_dict:
                new_speaker_name = add_to_base_embeddings(embeddings_dict, speaker_embeddings)
                speaker_matching[speaker] = new_speaker_name

        # Store the matched speaker info to the dictionary
        speakers_dict[csv_key] = speaker_matching

    return speakers_dict

def compare_speakers_enhanced(fileinfos):
    pre_audio = fileinfos[0]
    # Store the first speaker_dict as base dict
    embeddings_dict = generate_embeddings_from_AudioInfo_enhanced(pre_audio)
    speakers = embeddings_dict.keys()
    csv_name = pre_audio.filename
    csv_key = csv_name.split('/')[-1]

    # Map speakers to corresponding csv file
    speakers_dict = {csv_key: {speaker: speaker for speaker in speakers}}

    for i in range(1, len(fileinfos)):
        new_audio = fileinfos[i]
        new_csv = new_audio.filename

        csv_key = new_csv.split('/')[-1]

        new_speaker_embeddings_dict = generate_embeddings_from_AudioInfo_enhanced(new_audio)
        speaker_matching = {}
        found_speakers = []

        for speaker, speaker_embeddings in new_speaker_embeddings_dict.items():
            add_to_base_dict = True
            maxval = 0
            most_matched = ''
            for base_speaker, base_embedding in embeddings_dict.items():
                if speaker in found_speakers:
                    continue
                print(f'compare speaker {base_speaker} with new audio {new_audio.filename} speaker {speaker}')
                val = compare_embeddings(base_embedding, speaker_embeddings)
                # Keep the speaker that have the highest embedding cosine similarity
                if val > maxval:
                    maxval = val
                    most_matched = base_speaker
            # Threshld is 0.25 from speechbrain source code
            if maxval > 0.25:
                found_speakers.append(speaker)
                speaker_matching[speaker] = most_matched
                add_to_base_dict = False

            # less than threshold, means current speaker is a new speaker
            # add its embeddings to base embedding dict
            if add_to_base_dict:
                new_speaker_name = add_to_base_embeddings(embeddings_dict, speaker_embeddings)
                speaker_matching[speaker] = new_speaker_name

        # Store the matched speaker info to the dictionary
        speakers_dict[csv_key] = speaker_matching

    return speakers_dict
def merge_all_transcriptions():
    ls = list_files('./data/diarization/')
    ls.sort()
    audioinfos = []
    for filename in ls:
        af = process_df(filename)
        audioinfos.append(af)

    base_speakers = compare_speakers(audioinfos)
    transcripts_list = list_files('./data/merged_transcript/')

    merged_csv = []
    for transcript_csv in transcripts_list:
        temp = transcript_csv.split('/')[-1]
        print(temp)
        num = int(temp.split('.')[0])
        print(num)
        time = num * CHUNK_LENGTH

        df = pd.read_csv(transcript_csv)

        # Add time to 'start' and 'end' columns
        df['start'] += time
        df['end'] += time

        # # Extract unique speakers
        # speakers = df['speaker'].unique()
        csv_key = transcript_csv.split('/')[-1]

        if csv_key in base_speakers:
            speaker_matching_dict = base_speakers[csv_key]
            df['speaker'] = df['speaker'].replace(speaker_matching_dict)
            merged_csv.append(df)

    # 获取当前日期和时间
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H%M%S")
    merged_df = pd.concat(merged_csv)
    merged_df = merged_df.drop(columns=[col for col in merged_df.columns if 'Unnamed' in col or col == 'words'])

    # 初始化一个列表用于存储合并后的行
    merged_rows = []

    # 初始化一个临时存储器，用于存储待合并的行
    current_row = merged_df.iloc[0].copy()
    for i in range(1, len(merged_df)):
        next_row = merged_df.iloc[i]

        # 检查当前行和下一行是否需要合并
        if (current_row['speaker'] == next_row['speaker']) and (next_row['start'] - current_row['end'] < 0.1):
            # 合并文本并更新当前行的结束时间
            current_row['text'] += ' ' + next_row['text']
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

    final_res = './data/' + 'final_transcript_' + str(formatted_datetime) + '.csv'
    merged_df.to_csv(final_res)
    return merged_df


def merge_all_transcriptions_for_evaluation():
    ls = list_files('./data/diarization/')
    ls.sort()
    audioinfos = []
    for filename in ls:
        af = process_df(filename)
        audioinfos.append(af)

    base_speakers = compare_speakers(audioinfos)
    transcripts_list = list_files('./data/diarization/')

    merged_csv = []
    for transcript_csv in transcripts_list:
        temp = transcript_csv.split('/')[-1]
        print(temp)
        num = int(temp.split('.')[0])
        print(num)
        time = num * CHUNK_LENGTH

        df = pd.read_csv(transcript_csv)

        # Add time to 'start' and 'end' columns
        df['start'] = (df['start'] + time).round(2)
        df['end'] = (df['end'] + time).round(2)

        # # Extract unique speakers
        # speakers = df['speaker'].unique()
        csv_key = transcript_csv.split('/')[-1]

        if csv_key in base_speakers:
            speaker_matching_dict = base_speakers[csv_key]
            df['speaker'] = df['speaker'].replace(speaker_matching_dict)
            merged_csv.append(df)

    # 获取当前日期和时间
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H%M%S")
    merged_df = pd.concat(merged_csv)
    merged_df = merged_df.drop(columns=[col for col in merged_df.columns if 'Unnamed' in col or col == 'words'])

    # 初始化一个列表用于存储合并后的行
    merged_rows = []

    # 初始化一个临时存储器，用于存储待合并的行
    current_row = merged_df.iloc[0].copy()
    for i in range(1, len(merged_df)):
        next_row = merged_df.iloc[i]

        # 检查当前行和下一行是否需要合并
        if (current_row['speaker'] == next_row['speaker']) and (next_row['start'] - current_row['end'] < 0.1):
            # 合并行并更新当前行的结束时间
            current_row['end'] = next_row['end']
        elif (next_row['start'] - current_row['end'] < 0.1) and is_near_multiple_of_360(next_row['start']) and is_near_multiple_of_360(current_row['end']):
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

    final_res = './data/evaluation_reference/' + 'evaluated_diarization_' + str(formatted_datetime) + '.csv'
    merged_df.to_csv(final_res)
    return merged_df
def merge_VAD_transcriptions_for_evaluation(times):
    ls = list_files('./data/diarization/')
    ls.sort()
    audioinfos = []
    for filename in ls:
        af = process_df(filename)
        audioinfos.append(af)

    base_speakers = compare_speakers(audioinfos)
    transcripts_list = list_files('./data/diarization/')

    merged_csv = []
    for transcript_csv in transcripts_list:
        temp = transcript_csv.split('/')[-1]
        print(temp)
        num = int(temp.split('.')[0])
        print(num)
        time = times[num]

        df = pd.read_csv(transcript_csv)

        # Add time to 'start' and 'end' columns
        df['start'] = (df['start'] + time).round(2)
        df['end'] = (df['end'] + time).round(2)

        # # Extract unique speakers
        # speakers = df['speaker'].unique()
        csv_key = transcript_csv.split('/')[-1]

        if csv_key in base_speakers:
            speaker_matching_dict = base_speakers[csv_key]
            df['speaker'] = df['speaker'].replace(speaker_matching_dict)
            merged_csv.append(df)

    # 获取当前日期和时间
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H%M%S")
    merged_df = pd.concat(merged_csv)
    merged_df = merged_df.drop(columns=[col for col in merged_df.columns if 'Unnamed' in col or col == 'words'])

    # 初始化一个列表用于存储合并后的行
    merged_rows = []

    # 初始化一个临时存储器，用于存储待合并的行
    current_row = merged_df.iloc[0].copy()
    for i in range(1, len(merged_df)):
        next_row = merged_df.iloc[i]

        # 检查当前行和下一行是否需要合并
        if (current_row['speaker'] == next_row['speaker']) and (next_row['start'] - current_row['end'] < 0.1):
            # 合并行并更新当前行的结束时间
            current_row['end'] = next_row['end']
        elif (next_row['start'] - current_row['end'] < 0.1) and is_near_cutting_points(next_row['start'], times) and is_near_cutting_points(current_row['end'], times):
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

    final_res = './data/evaluation_reference/' + 'evaluated_diarization_VAD' + str(formatted_datetime) + '.csv'
    merged_df.to_csv(final_res)
    return merged_df

def is_near_multiple_of_360(num, tolerance=0.1):
    mod = num % 360
    # 检查是否接近0或接近360
    return abs(mod) < tolerance or abs(mod - 360) < tolerance

def is_near_cutting_points(num, points, tolerance=0.05):
    for point in points:
        if abs(num - point) <= tolerance:
            return True
    return False

def add_to_base_embeddings(embeddings_dict, embeddings):
    num = len(embeddings_dict)
    new_speaker_name = ''
    if num < 10:
        new_speaker_name = "SPEAKER_" + '0' + str(num)
    else:
        new_speaker_name = "SPEAKER" + str(num)
    embeddings_dict[new_speaker_name] = embeddings
    return new_speaker_name



if __name__ == '__main__':
    merge_all_transcriptions_for_evaluation()

    # # 读取 CSV 文件并删除包含 'Unnamed' 的列
    # df = pd.read_csv('../data/final_transcript_2024-09-03_000558.csv')
    # df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')
    #
    # # 重置索引
    # df = df.reset_index(drop=False)
    #
    # # 初始化一个列表用于存储合并后的行
    # merged_rows = []
    #
    # # 初始化一个临时存储器，用于存储待合并的行
    # current_row = df.iloc[0].copy()
    #
    # for i in range(1, len(df)):
    #     next_row = df.iloc[i]
    #
    #     # 检查当前行和下一行是否需要合并
    #     if (current_row['speaker'] == next_row['speaker']) and (next_row['start'] - current_row['end'] < 0.1):
    #         # 合并文本并更新当前行的结束时间
    #         current_row['text'] += ' ' + next_row['text']
    #         current_row['end'] = next_row['end']
    #     else:
    #         # 如果不需要合并，将当前行添加到结果中，并更新为下一行
    #         merged_rows.append(current_row)
    #         current_row = next_row.copy()
    #
    # # 添加最后一行
    # merged_rows.append(current_row)
    #
    # # 创建一个新的 DataFrame
    # merged_df = pd.DataFrame(merged_rows)
    #
    # # 重置索引，确保索引从0开始且没有多余的列
    # merged_df = merged_df.reset_index(drop=True)
    #
    # # 保存合并后的 DataFrame 为新的 CSV 文件
    # final_res = '../data/final_transcript.csv'
    # merged_df.to_csv(final_res, index=False)
    #
    # print(f"CSV file saved to: {final_res}")
    #
    # # ls = list_files('../data/diarization/')
    # # ls.sort()
    # # audioinfos = []
    # # for filename in ls:
    # #     af = process_df(filename)
    # #     audioinfos.append(af)
    # #
    # # base_speakers = compare_speakers(audioinfos)
    # # files = list_files(path)
    # # files.sort()
    #
    # # fileinfo = process_df(file1)
    # #
    # # print(fileinfo.speaker_dict)
    # # print(fileinfo.audio_name)
    # #
    # # res = generate_embeddings_from_AudioInfo(fileinfo)



    # speakers = ["s1", "s2", "s3", "s4", "s5", "s6"]
    #
    # speakers_dict = {0: {speaker: speaker for speaker in speakers}}
    # print(speakers_dict)
