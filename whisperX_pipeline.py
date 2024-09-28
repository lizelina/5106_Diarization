import whisperx
import pandas as pd
from loguru import logger
import time

device = "cpu"
audio_file = "./sample.wav"
batch_size = 4  # reduce if low on GPU mem
compute_type = "int8"  # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
# save model to local path (optional)
model_dir = './models/fast_whisper/'
model = whisperx.load_model("base", device, compute_type=compute_type, download_root=model_dir)
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

start = time.time()
print("1. start transcription")
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print("end transcription", time.time() - start)

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
print("2. start alignment")
start2 = time.time()
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device,
                                              model_dir="./models/alignment/")
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
print("end alignment", time.time() - start2)
# results after alignment
# print(result["segments"])
# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
start3 = time.time()
logger.info(f'start diarization')
diarize_model = whisperx.DiarizationPipeline(use_auth_token='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', device=device)

# add min/max number of speakers if known
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
diarize_segments = diarize_model(audio)
logger.info(f'end time{time.time() - start3}')
result = whisperx.assign_word_speakers(diarize_segments, result)
# segments are now assigned speaker IDs
pd.DataFrame(result["segments"]).to_csv("whisperx_transcription.csv", index=False)

