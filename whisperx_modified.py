from src.audio_utils import *
from src.transcribe import *
from src.pyannote_model import *
from src.merge_csv import *
from src.assign_word_speakers import *
from src.WhisperX_Modified import *
from src.ForcedAlignment import *
from src.SpeakerDiarizationPipeline import *

if __name__ == '__main__':
    #
    # # # 1. Create temporary directories
    # # create_directory()
    # #
    # # # 2. Convert audio file to wav
    name = convert_to_wav("data/录音 (6).m4a")
    # # # calculate_duration("./data/RMM.wav")
    # #
    # # # Split audio to 6 minute chunks
    crop_wav(name)
    # #
    # # # 3. Perform transcription for audio chunks
    model_type = 'medium'
    device = "cpu"
    batch_size = 4
    compute_type = "int8"
    result = whisperX_transcribe_files_in_directory("./data/temp_audios/", model_type=model_type,
                                                    batch_size=batch_size, compute_type=compute_type)
    print(result)
    #
    # # 初始化处理器，使用默认 6 分钟切分
    # transcriber = AudioTranscriber(model_type='base', device='cpu', batch_size=4, compute_type='int8')
    #
    # input_file = "./0.wav"
    # transcriber.process_and_transcribe(input_file)


    # 4. Perform transcripts alignment with multiprocessing
    # 初始化对齐器，只需指定设备和并行进程数量（可选）
    # aligner = WhisperXAligner(device="cpu", num_workers=4)
    #
    # # 并行对齐 temp_audios 目录下的所有音频文件
    # aligner.align_in_parallel()
    #
    # # 5. Perform diarization with multiprocessing
    # start = time.time()
    # logger.info(f'start diarization')
    # # 设置你希望使用的进程数（通常设置为CPU核心数）
    # num_workers = 6  # 自动获取CPU核数
    # # 并行处理文件
    # use_auth_token = 'HuggingFace Token'
    # model_name = "pyannote/speaker-diarization-3.1"
    # results = process_files_in_parallel(filelist, num_workers, use_auth_token, model_name)
    # logger.info(f'end time{time.time() - start}')
    # # # 6. Assign transcripts to speakers
    # assign_transrcipts_to_speakers()
    # #

    # 初始化分割管道
    pipeline = SpeakerDiarizationPipeline(num_workers=4, use_auth_token='HuggingFace Token',
                                          device="cpu")

    # 并行处理 temp_audios 目录下的音频文件
    pipeline.process_files_in_parallel()

    # 将转录文件与发言者对齐
    pipeline.assign_transcripts_to_speakers()

    # # 7. Merge all transcripts with diarization files
    merge_all_transcriptions()
    #
    # # 8. Merge the diarization documents for evaluation
    # merge_all_transcriptions_for_evaluation()
    #


