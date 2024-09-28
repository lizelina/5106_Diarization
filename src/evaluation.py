import os
import sys

import pandas as pd
from textgrid import TextGrid
from pyannote.metrics import diarization
from pyannote.core import Segment, Annotation
import re
import jiwer
from hanziconv import HanziConv


def list_files(directory):
    """列出指定目录下的所有文件"""
    return [os.path.join(directory, file) for file in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, file))]


def save_textGrid_to_df(filename):
    """将TextGrid文件转为csv文件"""
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
    """从CSV文件构建参考Annotation对象，用于计算说话人识别 DER"""
    df = pd.read_csv(filename)
    ref_dict = df.to_dict(orient='records')
    ref = Annotation()
    for row in ref_dict:
        ref[Segment(row['start'], row['end'])] = row['speaker']
    # print(ref)
    return ref


def evaluate_der(ref, hyp):
    """评估参考和假设之间的Diarization Error Rate (DER) 及其他错误率指标"""
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
    """处理文本字符串，去除标点符号，将其转换为大写，并去除所有空格"""
    # 使用正则表达式替换标点符号为空格
    text_no_punctuation = re.sub(r'[^\w\s]', '', text)

    # 将小写字母转换为大写字母
    text_upper = text_no_punctuation.upper()

    # 去除所有空格
    text_no_spaces = text_upper.replace(' ', '')

    return text_no_spaces


def construct_transcripts(fname):
    """通过读取CSV文件构建简化的转录文本"""
    df = pd.read_csv(fname)
    transcripts = ''.join(df['text'])
    # 去除标点符号和空格
    processed_text = process_text_with_regex(transcripts)

    # 繁体字转简体字
    simpilified = HanziConv.toSimplified(processed_text)

    return simpilified


def construct_hyps(path):
    """读取transcripts文件夹下所有转录结果，合并成转录文本"""
    files = list_files(path)
    transcripts = []
    for file in files:
        text = construct_transcripts(file)
        transcripts.append(text)
    return ''.join(transcripts)


def evaluate_cer(ref, hyp):
    """评估参考文本和假设文本之间的字符错误率（CER）"""
    cer = jiwer.cer(ref, hyp)
    print(f"Character Error Rate: {cer}")


# save_textGrid_to_df('../data/evaluation_reference/R8009_M8020.TextGrid')
# ref = construct_ref("../data/evaluation_reference/R8009_M8020.csv")
# hyp = construct_ref("../data/evaluation_reference/evaluated_diarization_2024-09-07_204755.csv")
# hyp2 = construct_ref("../data/evaluation_reference/R8009_M8020_MS810@.csv")


if __name__ == '__main__':

    # 示例代码
    # 1. 将Alimeeting benchmark提供的TextGrid文件转成csv "R8003_M8001.csv"
    save_textGrid_to_df("../data/evaluation_reference/sample/R8003_M8001.TextGrid")

    # 2. 计算说话人识别的错误率 Diarization Error Rate (DER)
    # 获取reference说话人识别的结果
    ref = construct_ref("../data/evaluation_reference/sample/R8003_M8001.csv")
    # 获取pipeline说话人识别的结果 - hyp
    hyp = construct_ref("../data/evaluation_reference/sample/evaluated_diarization_2024-09-15_203801.csv")
    evaluate_der(ref, hyp)

    # 3. 计算转录文本的错误率 Character Error Rate (CER)
    # 获取reference 转录的内容
    text = construct_transcripts("../data/evaluation_reference/sample/R8003_M8001.csv")
    print(text)
    # hyp_text = construct_hyps('../data/transcripts/')
    # 获取pipeline转录的文字 从../transcripts/路径获取
    hyp_text = construct_hyps("../data/evaluation_reference/sample/transcripts/")
    print(hyp_text)
    evaluate_cer(text, hyp_text)


