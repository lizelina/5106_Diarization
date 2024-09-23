import os
import sys

import pandas as pd
from textgrid import TextGrid
from pyannote.metrics import diarization
from pyannote.core import Segment, Annotation
import re
import jiwer
from src.audio_utils import list_files
from hanziconv import HanziConv


def save_textGrid_to_df(filename):
    # 读取 TextGrid 文件
    tg = TextGrid.fromFile(filename)

    # 遍历每一个层级 (tier)
    segments = []
    for tier in tg.tiers:
        print(f"Tier name: {tier.name}")

        # 遍历每个 interval（时间段）
        for interval in tier.intervals:
            if interval.mark.strip():  # 过滤掉空标注
                segments.append(((interval.minTime, interval.maxTime), tier.name, interval.mark))

    # 对 segments 列表原地排序
    segments.sort(key=lambda x: x[0][0])

    # 打印 segments 列表
    segment_dict = []
    for segment in segments:
        segment_dict.append({
            'start': segment[0][0],
            'end': segment[0][1],
            'speaker': segment[1],
            'text': segment[2]
        })

    # 将 segments 列表转换为 DataFrame
    df = pd.DataFrame(segment_dict)
    print(df)

    # 生成同名的 CSV 文件
    base, _ = os.path.splitext(filename)
    csv_filename = base + ".csv"

    # 保存为 CSV 文件
    df.to_csv(csv_filename, index=False)
    print(f"Saved CSV to {csv_filename}")


def construct_ref(filename):
    df = pd.read_csv(filename)
    ref_dict = df.to_dict(orient='records')
    ref = Annotation()
    for row in ref_dict:
        ref[Segment(row['start'], row['end'])] = row['speaker']
    # print(ref)
    return ref


def evaluate(ref, hyp):
    metric = diarization.DiarizationErrorRate()
    der = metric(ref, hyp)
    metric = diarization.JaccardErrorRate()
    jac = metric(ref, hyp)
    metric = diarization.DiarizationCoverage()
    coverage = metric(ref, hyp)
    metric = diarization.DiarizationPurity()
    purity = metric(ref, hyp)
    metric = diarization.DiarizationErrorRate(collar=0.2)
    der2 = metric(ref, hyp)
    print(f'der: {der}')
    print(f'jac: {jac}')
    print(f'coverage: {coverage}')
    print(f'purity: {purity}')
    print(f'der2: {der2}')


def process_text_with_regex(text):
    # 使用正则表达式替换标点符号为空格
    text_no_punctuation = re.sub(r'[^\w\s]', '', text)

    # 将小写字母转换为大写字母
    text_upper = text_no_punctuation.upper()

    # 去除所有空格
    text_no_spaces = text_upper.replace(' ', '')

    return text_no_spaces


# 示例文本
# text = "这不是快到了教师节了吗，那个看看都给孩子就是老师送点什么呀，给他们班主任嗯。"
#
# # 调用方法
# processed_text = process_text(text)
#
# # 输出结果
# print(processed_text)


def construct_transcripts(fname):
    df = pd.read_csv(fname)
    transcripts = ''.join(df['text'])
    processed_text = process_text_with_regex(transcripts)

    # 繁体字转简体字
    simpilified = HanziConv.toSimplified(processed_text)

    return simpilified
    # return processed_text
def construct_hyps(path):
    files = list_files(path)
    transcripts = []
    for file in files:
        text = construct_transcripts(file)
        transcripts.append(text)
    return ''.join(transcripts)
    # text = construct_transcripts(path)

def evaluate_cer(ref, hyp):
    cer = jiwer.cer(ref, hyp)
    print(f"Character Error Rate: {cer}")





# save_textGrid_to_df('../data/evaluation_reference/R8009_M8020.TextGrid')
# ref = construct_ref("../data/evaluation_reference/R8009_M8020.csv")
# hyp = construct_ref("../data/evaluation_reference/evaluated_diarization_2024-09-07_204755.csv")
# hyp2 = construct_ref("../data/evaluation_reference/R8009_M8020_MS810@.csv")





##### R8003
# print("##### R8003  #####")
# ref = construct_ref("../data/evaluation_reference/R8003_M8001.csv")
# hyp1 = construct_ref("../data/evaluation_reference/R8003_M8001_MS801.csv")
# hyp2 = construct_ref("../data/evaluation_reference/evaluated_diarization_2024-09-15_203801.csv")
# # # with vad
# # print(ref)
# hyp3 = construct_ref("../data/evaluation_reference/evaluated_diarization_VAD2024-09-12_201344.csv")
# evaluate(ref, hyp1)
# evaluate(ref, hyp2)
# evaluate(ref, hyp3)

text = construct_transcripts("../data/evaluation_reference/R8003_M8001.csv")
# #一次性转录
# print(text)
trans1 = construct_transcripts("../data/evaluation_reference/transcripts_R8003_M8001.csv")
transcripts = construct_hyps('../data/R8003/transcripts/')
transcripts2 = construct_hyps('../data/VAD_R8003/transcripts/')
print(text)
print(trans1)
print(transcripts)
print(transcripts2)
# Character Error Rate: 0.3876410004904365
# Character Error Rate: 0.42275625306522807
# Character Error Rate: 0.3019127023050515
evaluate_cer(text, trans1)
evaluate_cer(text, transcripts)
evaluate_cer(text, transcripts2)


#### R8007
# print("##### R8007  #####")
# ref = construct_ref("../data/evaluation_reference/R8007_M8011.csv")
# hyp1 = construct_ref("../data/evaluation_reference/R8007_M8011_MS806.csv")
# # print(ref)
# hyp2 = construct_ref("../data/evaluation_reference/evaluated_diarization_2024-09-15_201505.csv")
# #
# # # # with vad
# hyp3 = construct_ref("../data/evaluation_reference/evaluated_diarization_VAD2024-09-13_004853.csv")
# evaluate(ref, hyp1)
# evaluate(ref, hyp2)
# evaluate(ref, hyp3)

text = construct_transcripts("../data/evaluation_reference/R8007_M8011.csv")
print(text)
trans1 = construct_transcripts("../data/evaluation_reference/transcripts_R8007_M8011.csv")
print(trans1)
transcripts = construct_hyps('../data/R8007/transcripts/')
print(transcripts)
transcripts2 = construct_hyps('../data/VAD_R8007/transcripts/')
print(transcripts2)
# Character Error Rate: 0.3250820690267057
# Character Error Rate: 0.3265903646526484
# Character Error Rate: 0.3262354715641913
evaluate_cer(text, trans1)
evaluate_cer(text, transcripts)
evaluate_cer(text, transcripts2)

#### R8009
# print("##### R8009  #####")
# ref = construct_ref("../data/evaluation_reference/R8009_M8020.csv")
# hyp1 = construct_ref("../data/evaluation_reference/R8009_M8020_MS810@.csv")
# hyp2 = construct_ref("../data/evaluation_reference/evaluated_diarization_2024-09-15_204502.csv")
# # # with vad
# # print(ref)
# hyp3 = construct_ref("../data/evaluation_reference/evaluated_diarization_VAD2024-09-13_130842.csv")
# evaluate(ref, hyp1)
# evaluate(ref, hyp2)
# evaluate(ref, hyp3)

text = construct_transcripts("../data/evaluation_reference/R8009_M8020.csv")
print(text)
trans1 = construct_transcripts("../data/evaluation_reference/transcripts_R8009_M8020.csv")
print(trans1)
transcripts = construct_hyps('../data/transcripts/')
print(transcripts)
transcripts2 = construct_hyps('../data/VAD_R8009/transcripts/')
print(transcripts2)
# Character Error Rate: 0.17073170731707318
# Character Error Rate: 0.16862101313320826
# Character Error Rate: 0.1772983114446529
evaluate_cer(text, trans1)
evaluate_cer(text, transcripts)
evaluate_cer(text, transcripts2)


# trans1 = construct_transcripts("../data/evaluation_reference/transcripts_R8007_M8011.csv")
# transcripts = construct_hyps('../data/R8007/transcripts/')
#
#
# evaluate_cer(text, trans1)
# evaluate_cer(text, transcripts)

