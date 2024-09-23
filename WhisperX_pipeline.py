import whisperx
import pandas as pd
from loguru import logger
import time

device = "cpu"
# audio_file = "./6Minute_short.csv"
audio_file = "data/VAD_R8009/temp_audios/2.wav"
batch_size = 4 # reduce if low on GPU mem
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("medium", device, compute_type=compute_type, download_root='./models/fast_whisper/')

# save model to local path (optional)
model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)
start = time.time()
print("1. start transcription")
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print("end transcription", time.time() - start)

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
print("2. start transcription")
start2 = time.time()
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device, model_dir="./models/alignment/")
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
print("end alignment", time.time() - start)

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
start3 = time.time()
logger.info(f'start diarization')
diarize_model = whisperx.DiarizationPipeline(use_auth_token='HuggingFace Token', device=device)

# add min/max number of speakers if known
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
diarize_segments = diarize_model(audio)
logger.info(f'end time{time.time() - start}')
pd.DataFrame(diarize_segments).to_csv("whisperx_6Minute_short_diarized.csv", index=False)
result = whisperx.assign_word_speakers(diarize_segments, result)



