import numpy as np
import pandas as pd
import os
def assign_word_speakers(diarize_df, transcript_segments, fill_nearest=False):
    for seg in transcript_segments:
        # assign speaker to segment (if any)
        diarize_df['intersection'] = np.minimum(diarize_df['end'], seg['end']) - np.maximum(diarize_df['start'],
                                                                                            seg['start'])
        diarize_df['union'] = np.maximum(diarize_df['end'], seg['end']) - np.minimum(diarize_df['start'], seg['start'])
        # remove no hit, otherwise we look for closest (even negative intersection...)
        if not fill_nearest:
            dia_tmp = diarize_df[diarize_df['intersection'] > 0]
        else:
            dia_tmp = diarize_df
        if len(dia_tmp) > 0:
            # sum over speakers
            speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
            seg["speaker"] = speaker

        # assign speaker to words
        if 'words' in seg:
            for word in seg['words']:
                if 'start' in word:
                    diarize_df['intersection'] = np.minimum(diarize_df['end'], word['end']) - np.maximum(
                        diarize_df['start'], word['start'])
                    diarize_df['union'] = np.maximum(diarize_df['end'], word['end']) - np.minimum(diarize_df['start'],
                                                                                                  word['start'])
                    # remove no hit
                    if not fill_nearest:
                        dia_tmp = diarize_df[diarize_df['intersection'] > 0]
                    else:
                        dia_tmp = diarize_df
                    if len(dia_tmp) > 0:
                        # sum over speakers
                        speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
                        word["speaker"] = speaker

    return transcript_segments

def list_files(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, file))]

def assign_transrcipts_to_speakers():
    path1 = './data/diarization/'
    path2 = './data/transcripts_aligned/'
    filenames = list_files(path1)
    filenames.sort()
    for filename in filenames:
        diarize_df = pd.read_csv(filename)
        csv_name = filename.split('/')[-1]
        csvpath = path2 + csv_name
        transcript_segments = pd.read_csv(csvpath).to_dict(orient='records')
        res = assign_word_speakers(diarize_df, transcript_segments, True)
        path3 = './data/merged_transcript/'
        if not os.path.exists(path3):
            os.makedirs(path3)

        pd.DataFrame(res).to_csv(path3 + csv_name)



def get_project_path():
    project_root =  os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # 固定的相对路径
    # 创建相对路径，与 'src' 目录同层级
    diarization_path = os.path.join(project_root, 'data', 'diarization')
    transcripts_path = os.path.join(project_root, 'data', 'transcripts_aligned')
    output_path = os.path.join(project_root, 'data', 'merged_transcript')

    # 确保目录存在，如果不存在则创建
    os.makedirs(diarization_path, exist_ok=True)
    os.makedirs(transcripts_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    print(diarization_path)
    print(transcripts_path)
    print(output_path)
    return


if __name__ == "__main__":
    # assign_transrcipts_to_speakers()
    get_project_path()
