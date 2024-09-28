
# CPU-Based Pipelines for Long-form Audio Transcription and Diarization

This repository presents three pipelines for transcribing long-form audio and performing speaker diarization using CPU resources.

### Pipelines Overview

1. **WhisperX**:  
   The original pipeline utilizing batch transcription with Whisper and forced alignment. It splits audio into fixed-length segments and aligns the transcriptions with audio files.

2. **WhisperX_Modified**:  
   An optimized version of WhisperX that incorporates multiprocessing for enhanced speed. This version also splits the audio into equal-length segments.

3. **VAD_WhisperX**:  
   This enhanced pipeline uses Voice Activity Detection (VAD) for more intelligent audio splitting, combined with multiprocessing to improve efficiency.

### How to Use

The code below shows how to use the different pipelines for long-form audio transcription and speaker diarization.

```python
from src.TranscriptionPipeline import *
from src.ForcedAlignmentPipeline import *
from src.SpeakerDiarizationPipeline import *
from src.MergePipeline import *
from src.clean_temporary_files import *

if __name__ == '__main__':
    # 1. Initialize transcription pipeline using VAD for splitting, ensuring each segment is under 360 seconds
    transcriber = AudioTranscriberPipeline(model_type='base', device='cpu', batch_size=4, compute_type='int8',
                                           segment_length=360, user_token='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')
    input_file = "./sample.wav"
    # Perform transcription with VAD-based splitting and return the cut times
    cut_times = transcriber.process_and_transcribe(input_file, use_vad=True)
    print(cut_times)

    # 2. Perform forced alignment using multiprocessing to align transcriptions with audio files
    aligner = WhisperXAligner(device="cpu", num_workers=4)
    # Align audio files in 'temp_audios' with corresponding transcriptions in 'transcripts'
    aligner.align_in_parallel()

    # 3. Initialize speaker diarization pipeline, specifying the device and authentication token
    pipeline = SpeakerDiarizationPipeline(num_workers=4, use_auth_token='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', device="cpu")
    # Perform speaker diarization in parallel and merge the results with aligned transcriptions
    pipeline.process_and_assign_speakers_in_parallel()

    # 4. Merge all transcriptions, specifying whether VAD was used for splitting
    mergedPipeline = MergeTranscriptionsPipeline(use_vad=True)
    mergedPipeline.merge_all_transcriptions(cut_times)

    # 5. (Optional) For evaluation purposes, merge multiple speaker diarization results
    mergedPipeline.merge_all_transcriptions_for_evaluation(cut_times)

    # 6. (Optional) Clean up temporary files, keeping only the final transcriptions and evaluation results
    delete_all_temp_files()
```

### Parameters for Each Pipeline

#### 1. `AudioTranscriberPipeline`
- **model_type**: Specifies the Whisper model to use. Options include 'base', 'small', 'medium', etc.
- **device**: The device to run the transcription (e.g., 'cpu' or 'cuda').
- **batch_size**: Batch size for processing the audio chunks.
- **compute_type**: The precision for computation (e.g., 'int8', 'float16').
- **segment_length**: The maximum length (in seconds) of each audio segment after splitting.
- **user_token**: Authentication token required for certain model downloads or API accesses.
- **use_vad**: Whether to use VAD for intelligent audio splitting (True/False).

#### 2. `WhisperXAligner`
- **device**: Specifies the device for alignment (e.g., 'cpu', 'cuda').
- **num_workers**: Number of parallel processes to use for forced alignment.

#### 3. `SpeakerDiarizationPipeline`
- **num_workers**: Number of parallel processes to use for speaker diarization.
- **use_auth_token**: Authentication token for accessing `pyannote.audio` models.
- **device**: The device to run the diarization (e.g., 'cpu' or 'cuda').

#### 4. `MergeTranscriptionsPipeline`
- **use_vad**: Indicates whether VAD was used for splitting audio during transcription.

#### 5. `delete_all_temp_files`
- This function removes all temporary files generated during processing, retaining only the final transcriptions and evaluation results.

### Evaluation

This repository includes scripts for evaluating diarization and transcription results.

#### 1. **Diarization Error Rate (DER) Evaluation**
You can evaluate the Diarization Error Rate (DER) by comparing the diarization results (hypothesis) with the reference annotations (ground truth) from the dataset.

```python
from pyannote.core import Segment, Annotation
from pyannote.metrics import diarization

# Example: Evaluating DER
ref = construct_ref("../data/evaluation_reference/sample/R8003_M8001.csv")  # Reference file
hyp = construct_ref("../data/evaluation_reference/sample/evaluated_diarization_2024-09-15_203801.csv")  # Hypothesis file
evaluate_der(ref, hyp)  # Compute DER and other metrics
```

#### 2. **Character Error Rate (CER) Evaluation**
For transcription, you can compute the Character Error Rate (CER) between the reference transcription and the system output.

```python
import jiwer

# Example: Evaluating CER
ref_text = construct_transcripts("../data/evaluation_reference/sample/R8003_M8001.csv")  # Reference transcription
hyp_text = construct_hyps("../data/evaluation_reference/sample/transcripts/")  # Hypothesis transcription
evaluate_cer(ref_text, hyp_text)  # Compute CER
```

### Additional Evaluation Metrics
- **Diarization Error Rate (DER)**: Measures speaker diarization accuracy.
- **Jaccard Error Rate (JER)**: Measures overlap between reference and hypothesis segments.
- **Character Error Rate (CER)**: Measures transcription accuracy at the character level.
  
You can customize the evaluation to suit your needs by adapting the provided functions.

### Example Usage for Evaluation
```python
# Convert a TextGrid file to CSV for evaluation
save_textGrid_to_df("../data/evaluation_reference/sample/R8003_M8001.TextGrid")

# Evaluate diarization error rate
ref = construct_ref("../data/evaluation_reference/sample/R8003_M8001.csv")
hyp = construct_ref("../data/evaluation_reference/sample/evaluated_diarization_2024-09-15_203801.csv")
evaluate_der(ref, hyp)

# Evaluate character error rate
ref_text = construct_transcripts("../data/evaluation_reference/sample/R8003_M8001.csv")
hyp_text = construct_hyps("../data/evaluation_reference/sample/transcripts/")
evaluate_cer(ref_text, hyp_text)
