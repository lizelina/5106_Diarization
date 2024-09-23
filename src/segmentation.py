from pyannote.audio import Model
import numpy as np
import time
model = Model.from_pretrained("pyannote/segmentation", use_auth_token="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")
print(model.specifications)

from pyannote.audio import Inference
inference = Inference(model, duration=5.0, step=2.5)
output = inference("../6Minute_short.wav")
# print(output.data)
BATCH_AXIS = 0
TIME_AXIS = 1
SPEAKER_AXIS = 2
to_vad = lambda o: np.max(o, axis=SPEAKER_AXIS, keepdims=True)
to_vad(output)

start = time.time()
vad = Inference("pyannote/segmentation", pre_aggregation_hook=to_vad)
vad_prob = vad("../data/roundtable.wav")
from pyannote.audio.utils.signal import Binarize
binarize = Binarize(onset=0.5)
speech = binarize(vad_prob)
print(f"Toatal time: {time.time() - start}")
print(speech)


# to_scd = lambda probability: np.max(
#     np.abs(np.diff(probability, n=1, axis=TIME_AXIS)),
#     axis=SPEAKER_AXIS, keepdims=True)
# scd = Inference("pyannote/segmentation", pre_aggregation_hook=to_scd)
# scd_prob = scd("../6Minute_short.wav")
# from pyannote.audio.utils.signal import Peak
# peak = Peak(alpha=0.05)
# res = peak(scd_prob).crop(speech.get_timeline())
# print(res)

