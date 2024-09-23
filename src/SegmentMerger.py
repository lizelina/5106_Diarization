from collections import defaultdict


class SegmentMerger:
    def __init__(self, max_gap=2.0):
        self.max_gap = max_gap

    @staticmethod
    def key_with_max_value(d):
        """
        返回字典中值最大的键。

        参数:
        d (dict): 一个字典，其值可以比较大小。

        返回:
        max_key: 值最大的键。如果字典为空，返回 None。
        """
        if not d:
            return None
        return max(d, key=d.get)

    def merge_segments(self, segments):
        final_segments = []
        curr_merged = segments[0]
        duration = curr_merged['end'] - curr_merged['start']
        sort_dict = defaultdict(float)

        sort_dict[curr_merged['speaker']] = duration
        if duration > self.max_gap:
            final_segments.append(curr_merged)
            duration = 0

        for segment in segments[1:]:
            time_gap = segment["end"] - segment["start"]

            if time_gap < self.max_gap:
                duration += time_gap
                sort_dict[segment['speaker']] += time_gap

                if duration < self.max_gap:
                    curr_merged["end"] = segment["end"]
                else:
                    curr_merged['start'] = segment['start']
                    speaker = self.key_with_max_value(sort_dict)
                    curr_merged['speaker'] = speaker
                    final_segments.append(curr_merged)
                    sort_dict.clear()
                    curr_merged = segment
                    duration = time_gap
                    sort_dict[segment['speaker']] = duration
            else:
                final_segments.append(curr_merged)
                curr_merged = segment
                duration = time_gap
                sort_dict.clear()
                sort_dict[segment['speaker']] = duration

        # 添加最后一个合并的段落
        if curr_merged not in final_segments:
            final_segments.append(curr_merged)

        return final_segments

if __name__ == '__main__':
    import pandas as pd

    # 从CSV文件中读取数据
    df = pd.read_csv('../speaker2.csv')

    # 将数据转换为列表，每一行是一个字典
    segments2 = df.to_dict(orient='records')
    finalSegments = SegmentMerger().merge_segments(segments2)
    for segment in finalSegments:
        # 打印合并后的 segments
        print(f"{segment['start']}\t{segment['end']}\t{segment['speaker']}")