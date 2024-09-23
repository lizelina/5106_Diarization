
def merge_segments(segments):
    # 初始化变量
    merged_segments = []
    current_segment = segments[0]

    # 遍历 segments 数据
    for i in range(1, len(segments)):
        segment = segments[i]

        if segment["speaker"] == current_segment["speaker"]:
            # 如果 speaker 相同，更新结束时间
            current_segment["end"] = segment["end"]
        else:
            # 如果 speaker 不同，将当前段落添加到结果中，并开始新的段落
            merged_segments.append(current_segment)
            current_segment = segment

    # 添加最后一个段落
    merged_segments.append(current_segment)

    # 打印合并后的 segments
    for segment in merged_segments:
        print(f"{segment['start']}\t{segment['end']}\t{segment['speaker']}")

    return merged_segments