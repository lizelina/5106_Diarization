from src.TranscriptionPipeline import *
from src.ForcedAlignmentPipeline import *
from src.SpeakerDiarizationPipeline import *
from src.MergePipeline import *
from src.clean_temporary_files import *

if __name__ == '__main__':
    # 1. 初始化转录Pipeline, 对语音使用Voice Activity Detection(VAD)识别后切割
    # 使切割后语音不超过360秒
    transcriber = AudioTranscriberPipeline(model_type='base', device='cpu', batch_size=4, compute_type='int8',
                                           segment_length=360, user_token='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')
    input_file = "./sample.wav"
    # 返回VAD识别后音频的切割时间点，合并转录结果时需要
    cut_times = transcriber.process_and_transcribe(input_file, use_vad=True)
    print(cut_times)

    # 2. 使用多进程执行Forced Alignment,将转录结果和音频文件对齐
    # 初始化对齐器，只需指定设备和并行进程数量（可选）
    aligner = WhisperXAligner(device="cpu", num_workers=4)
    # 将temp_audios下的语音和transcripts的下的转录结果一一对齐
    aligner.align_in_parallel()

    # 3. 初始化说话人识别Pipeline,指定设备和并行进程数量，指定pyannote.audio需要token
    pipeline = SpeakerDiarizationPipeline(num_workers=4, use_auth_token='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',
                                          device="cpu")
    # 多进程执行说话人识别，并将说话人识别的结果 和 对齐后的转录文本 合并
    pipeline.process_and_assign_speakers_in_parallel()

    # 4. 合并所有音频对应的转录结果 需指定是否使用了VAD进行切割
    mergedPipeline = MergeTranscriptionsPipeline(use_vad=True)
    # 传入具体切割的时间段
    mergedPipeline.merge_all_transcriptions(cut_times)

    # 5. (Optional) Evaluation用，合并多个音频说话人识别的结果
    mergedPipeline.merge_all_transcriptions_for_evaluation(cut_times)

    # 6. (Optional) 删除所有中间文件，保留最终转录文件以及evaluation文件
    delete_all_temp_files()


