from pydub import AudioSegment
import tempfile
import pandas as pd
from loguru import logger
import time
import torchaudio
from pyannote.audio import Pipeline
from pyannote.core import notebook, Segment
from src.generate_embeddings import generate_embeddings
from src.clustering import cluster_known_number


def transform_mp3_to_wav(mp3_fname, output_fname=None):
    sound = AudioSegment.from_mp3(mp3_fname)  # load source

    if output_fname:
        this_temp_file_name = output_fname
        sound.export(output_fname, format="wav")

    else:
        # create temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        this_temp_file_name = temp_file.name

        sound.export(this_temp_file_name, format="wav")

    return this_temp_file_name

def crop_wav(fname, output_fname, start_frame=0, n_frames=60000):
    sound = AudioSegment.from_wav(fname)

    sound = sound.set_channels(1)  # mono
    sound = sound.set_frame_rate(16000)  # 16000Hz

    # Extract the first frames (60000 equals 60 seconds)
    excerpt = sound[start_frame:(start_frame + n_frames)]

    # write to disk
    excerpt.export(output_fname, format="wav")



# 12 min to run a 6 minute audio
if __name__ == '__main__':
    # start_time = time.time()
    # logger.info(f'start time {start_time}')
    # dia = pyannote_inference_df('6Minute_short.wav', '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')
    # logger.info(f'end time {time.time() - start_time}')
    # dia.to_csv('speaker3.csv')
    # res = wav_to_transcript('6Minute_short.wav', 'base', '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')
    # print(res)
    # res.to_csv('transcript.csv')


    # waveform, sample_rate = torchaudio.load('6Minute_short.wav')
    #
    # print(f"{type(waveform)=}")
    # print(f"{waveform.shape=}")
    # print(f"{waveform.dtype=}")
    # print(f"{sample_rate=}")
    #
    # audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}
    # vad = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token='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')
    # logger.info('start')
    # start = time.time()
    # print(vad(audio_in_memory))
    # logger.info(f'end{time.time() - start}')
    df = pd.read_csv('./vad.csv')
    segments = df.to_dict(orient='records')
    print(segments)
    logger.info("start")
    start_time = time.time()
    embeddings = generate_embeddings(segments, "data/6MinuteEnglish_new.wav")
    print(embeddings.shape)
    logger.info(f"end embedding generation{time.time()-start_time}")
    start_time = time.time()
    logger.info("start clustering")
    labels = cluster_known_number(2, embeddings)
    logger.info(f"end clustering{time.time()-start_time}")

    for i in range(len(segments)):
        segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)
    df = pd.DataFrame(segments)
    print(df)
    df.to_csv('segments.csv', index=False)




