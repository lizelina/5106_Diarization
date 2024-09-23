from src.audio_utils import *
from src.transcribe import *
from src.pyannote_model import *
from src.merge_csv import *
from src.assign_word_speakers import *
from src.voice_activity_detection import *

if __name__ == '__main__':

     # 1. Create temporary directories
    create_directory()

    # 2. Split audio to 6 minute chunks
    name = convert_to_wav("data/roundtable.wav")
    segments = vad_cut(name,  user_token='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', length = 400)
    # Obtain VAD cutting points
    res = segments_cutting_times(segments)

    crop_wav(name, segments)

    #3. Perform transcription for audio chunks
    model_type = 'medium'
    device = "cpu"
    batch_size = 4
    compute_type = "int8"
    start_time = time.time()
    logger.info('start transcription')
    result = whisperX_transcribe_files_in_directory("./data/temp_audios/", model_type=model_type,
                                                    batch_size=batch_size, compute_type=compute_type)
    logger.info(f'transcription end{time.time() - start_time}')

    # 4. Perform transcripts alignment with multiprocessing
    num_works = 4 # reduce is memory is low
    filelist = list_files('./data/temp_audios/')
    result ={'language':'zh'}
    start_time = time.time()
    logger.info('start alignment')
    process_alignment_in_parallel(filelist, result, num_workers=num_works, device=device)
    logger.info(f'end alignment{time.time() - start_time}')

    # 5. Perform diarization with multiprocessing
    start = time.time()
    logger.info(f'start diarization')
    # add min/max number of speakers if known
    file_paths = list_files("./data/temp_audios/")
    # 设置你希望使用的进程数（通常设置为CPU核心数）
    num_workers = 6  # 自动获取CPU核数
    # 并行处理文件
    use_auth_token = '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'
    model_name = "pyannote/speaker-diarization-3.1"
    results = process_files_in_parallel(file_paths, num_workers, use_auth_token, model_name)
    logger.info(f'end diarization time{time.time() - start}')

    # 6. Assign transcripts to speakers
    start = time.time()
    logger.info(f'start assign word speakers')
    assign_transrcipts_to_speakers()
    logger.info(f'end assign word speakers{time.time() - start}')


    # 7. Merge all transcripts with diarization files
    start = time.time()
    logger.info(f'start merge csv')
    merge_all_transcriptions()
    logger.info(f'end merge csv{time.time() - start}')

    # 8. Merge the diarization documents for evaluation
    start = time.time()
    logger.info(f'start evaluation')
    result = merge_VAD_transcriptions_for_evaluation(res)
    logger.info(f'end evaluation{time.time() - start}')



