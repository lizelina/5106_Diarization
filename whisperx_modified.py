from src.TranscriptionPipeline import *
from src.ForcedAlignmentPipeline import *
from src.SpeakerDiarizationPipeline import *
from src.MergePipeline import *
from src.clean_temporary_files import *

if __name__ == '__main__':
    # 1. 初始化转录Pipeline, 使用默认360秒切割语音，可使用segment_length指定切割长度
    transcriber = AudioTranscriberPipeline(model_type='base', device='cpu', batch_size=4, compute_type='int8')
    input_file = "./sample.wav"
    transcriber.process_and_transcribe(input_file)

    # 2. 使用多进程执行Forced Alignment, 将转录结果和音频文件对齐
    # 初始化对齐器，只需指定设备和并行进程数量（可选）
    aligner = WhisperXAligner(device="cpu", num_workers=4)
    # 并行对齐 temp_audios 目录下的所有音频文件
    aligner.align_in_parallel()

    # 3. 初始化说话人识别Pipeline,指定设备和并行进程数量，指定pyannote.audio需要token
    pipeline = SpeakerDiarizationPipeline(num_workers=4, use_auth_token='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',
                                          device="cpu")
    # 多进程执行说话人识别，并将说话人识别的结果 和 对齐后的转录文本 合并
    pipeline.process_and_assign_speakers_in_parallel()

    # 4. 合并所有音频对应的转录结果 需指定是否使用了VAD进行切割 默认False
    mergedPipeline = MergeTranscriptionsPipeline()
    mergedPipeline.merge_all_transcriptions()

    # 5. (Optional) Evaluation用，合并多个音频说话人识别的结果
    mergedPipeline.merge_all_transcriptions_for_evaluation()

    # 6. (Optional) 删除所有中间文件，保留最终转录文件以及evaluation文件
    delete_all_temp_files()
