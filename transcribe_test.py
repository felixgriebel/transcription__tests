
from pathlib import Path
from typing import Callable, List, Optional
import numpy as np
import pandas as pd
import whisper
import torch
import os
import azure.cognitiveservices.speech as speechsdk
import json
import ast, re 
from faster_whisper import WhisperModel
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperProcessor, WhisperForConditionalGeneration
import transformers
import time
import soundfile as sf
import torch.nn.functional as F
import librosa


from opencc import OpenCC        
cc = OpenCC("t2s")


import cn2an, unicodedata, re
# regex: keep digits, letters, spaces and CJK Unified Ideographs

FILLERS = "嗯啊呃呀呀唉诶哦噢".split()

def normalise(txt: str) -> str:
    txt = cn2an.transform(txt, "an2cn")          # digits -> 中文
    txt = unicodedata.normalize("NFKC", txt)      # full-/half-width
    txt = txt.lower()
    txt = re.sub(r"[^\w\s\u4e00-\u9fff]", "", txt)
    txt = txt.replace(" ", "")
    for f in FILLERS:                             # optional
        txt = txt.replace(f, "")
    return txt










re_keep = re.compile(r"[^\w\s\u4E00-\u9FFF]", flags=re.UNICODE)


def clean(text: str) -> str:
    """Convert to simplified and drop punctuation."""
    if pd.isna(text):
        return ""
    txt = str(text)
    if cc:
        txt = cc.convert(txt)
    return re_keep.sub("", txt).strip()
























#Whisper
model = None#whisper.load_model("large-v3")



#FasterWhisper

model2 = None#WhisperModel("large-v3", device="cuda", compute_type="float16")

model3 = None#WhisperModel("minzhi42/Belle-whisper-large-v3-zh-ct2", device="cuda", compute_type="float16")


name1 = "BELLE-2/Belle-distilwhisper-large-v2-zh"
proc1 = AutoProcessor.from_pretrained(name1)
modelbelle1 = AutoModelForSpeechSeq2Seq.from_pretrained(
    name1,
    torch_dtype=torch.float16,   # << half precision
    low_cpu_mem_usage=True
).to("cuda")
belle_fast1 = pipeline(
    "automatic-speech-recognition",
    model=modelbelle1,
    tokenizer=proc1.tokenizer,
    feature_extractor=proc1.feature_extractor,
    device=0,                   # make sure it stays on GPU
)






name2 = "BELLE-2/Belle-whisper-large-v3-zh"
proc2 = AutoProcessor.from_pretrained(name2)
modelbelle2 =AutoModelForSpeechSeq2Seq.from_pretrained(name2,torch_dtype=torch.float16,low_cpu_mem_usage=True).to("cuda")
belle_fast2 = pipeline("automatic-speech-recognition",model=modelbelle2,tokenizer=proc2.tokenizer,feature_extractor=proc2.feature_extractor,device=0)






MODEL_ID   = "BELLE-2/Belle-whisper-large-v3-zh"   # ← your checkpoint

proc4  = WhisperProcessor.from_pretrained(MODEL_ID)
model4 = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to("cuda")







#! WHISPER Baseline
def asr_workflow_1(audio_path: str) -> str:
    """Baseline transcription without any prompt."""
    result = model.transcribe(
        audio_path,
        temperature=0.0,
        task="transcribe",
        language="zh",
        condition_on_previous_text=False
    )
    return clean(result.get("text", ""))


#! WHISPER with generic "this is learner"-prompt
def asr_workflow_2(audio_path: str) -> str:
    result = model.transcribe(
        audio_path,
        temperature=0.0,
        task="transcribe",
        language="zh",
        initial_prompt=(
            "音频可能来自非母语者，因此他们可能会出现发音或语法错误:"
        ),
        condition_on_previous_text=False
    )
    return clean(result.get("text", ""))


def asr_workflow_3(audio_path: str):
    speech_config = speechsdk.SpeechConfig(
    subscription="BH3EglxujJ0fVtEOK65fgEO23Ap05hZjmGIc5dk9gfEKgqO1ToYwJQQJ99BFACqBBLyXJ3w3AAAEACOGaFVl",
    region="southeastasia"
    )
    speech_config.speech_recognition_language = "zh-CN"
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)

    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config,
        audio_config=audio_config
    )
    result = recognizer.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return clean(result.text)
    else:
        return ""


def asr_workflow_4(
    audio_path: str,
    user_history: Optional[List[str]] | None = None,
    assistant_history: Optional[List[str]] | None = None,
):
    user_history = user_history or []
    assistant_history = assistant_history or []
    if user_history and assistant_history:
        user_history = ast.literal_eval(re.sub(r'""', '"', user_history[0]))
        assistant_history = ast.literal_eval(re.sub(r'""', '"', assistant_history[0]))


    lines = []
    for i in range(max(len(user_history), len(assistant_history))):
        if i < len(user_history):
            lines.append(f"张：{user_history[i].strip()}")
        if i < len(assistant_history):
            lines.append(f"李：{assistant_history[i].strip()}")
    lines.append(f"张：")
    prompt_text = "\n".join(lines)


    result = model.transcribe(
        audio_path,
        temperature=0.0,
        task="transcribe",
        language="zh",
        initial_prompt=prompt_text,
        condition_on_previous_text=False
    )
    return clean(result.get("text", ""))



def asr_workflow_5(
    audio_path: str
):
    segments, _ = model2.transcribe(audio_path, beam_size=5,language="zh", task = "transcribe", temperature=0.0,condition_on_previous_text=False)

    text = "".join(seg.text for seg in segments).strip()
    return clean(text)





def asr_workflow_7(audio_path: str):
    transcription = belle_fast1(
        audio_path,
        generate_kwargs={
            "language": "zh",
            "task": "transcribe",
            "condition_on_prev_tokens": False,
            # any other decoding knobs here
        },
    )
    return clean(transcription["text"])






def asr_workflow_8(audio_path: str):
    transcription = belle_fast2(
        audio_path,
        generate_kwargs={
            "language": "zh",
            "task": "transcribe",
            "condition_on_prev_tokens": False,
            "temperature":0.0
        },
    )
    print(transcription)
    return clean(transcription["text"])





def asr_workflow_9(
    audio_path: str
):
    segments, _ = model3.transcribe(audio_path, language="zh", word_timestamps=True)

    words = [(w.word, w.probability) for seg in segments for w in seg.words]
    cleaned_pairs = [(clean(t), c) for t, c in words if len(clean(t))>0]
    text = "".join(w for w, _ in cleaned_pairs).strip()
    
    return clean(text)




def asr_workflow_17(
    audio_path: str
):
    segments, _ = model3.transcribe(audio_path, language="zh", word_timestamps=True)

    words = [(w.word, w.probability) for seg in segments for w in seg.words]
    cleaned_pairs = [(clean(t), c) for t, c in words if len(clean(t))>0]
    text = "".join(w for w, _ in cleaned_pairs).strip()
    
    return [s for t, s in cleaned_pairs]







def asr_workflow_18(
    audio_path: str
):
    segments, _ = model3.transcribe(
        audio_path,
        language="zh",
        task="transcribe",
        beam_size=5,
        best_of=5,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,
        repetition_penalty=1.15,
        compression_ratio_threshold=2.8,
        no_repeat_ngram_size=3,
        word_timestamps=True

    )
    words = [(w.word, w.probability) for seg in segments for w in seg.words]
    cleaned_pairs = [(clean(t), c) for t, c in words if len(clean(t))>0]
    text = "".join(w for w, _ in cleaned_pairs).strip()
    
    return clean(text)







def asr_workflow_19(
    audio_path: str
):
    segments, _ = model3.transcribe(
        audio_path,
        language="zh",
        task="transcribe",
        beam_size=5,
        best_of=5,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,
        repetition_penalty=1.15,
        compression_ratio_threshold=2.8,
        no_repeat_ngram_size=3,
        word_timestamps=True
    )
    words = [(w.word, w.probability) for seg in segments for w in seg.words]
    cleaned_pairs = [(clean(t), c) for t, c in words if len(clean(t))>0]
    text = "".join(w for w, _ in cleaned_pairs).strip()
    
    return [s for t, s in cleaned_pairs]















#!======================================================================================================================
#!======================================================================================================================
#!======================================================================================================================
#!======================================================================================================================
#!======================================================================================================================













import copy

gen_cfg = copy.deepcopy(model4.generation_config)   # universal
gen_cfg.output_scores           = True
gen_cfg.return_dict_in_generate = True




def asr_workflow_11(audio_path: str) -> str:
    audio, sr = librosa.load(audio_path, sr=16_000)          # ← force 16 kHz

    inp = proc4(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
        return_attention_mask=True,
    )
    feat = inp.input_features.to(model4.device)      # (1, 80, ≤3000)
    am   = inp.attention_mask.to(model4.device)

    with torch.inference_mode():
        out = model4.generate(
            input_features = feat,
            attention_mask = am,
            generation_config = gen_cfg,
            language = "zh",
            task     = "transcribe",
            temperature = 0.0,
        )

    toks   = out.sequences[0].cpu()                   # full token ids
    # out.scores is a list (len = seq_len – start_tokens) of (1, vocab)
    scores = torch.stack(out.scores).squeeze(1).cpu() # (L-K, vocab)
    K      = toks.size(0) - scores.size(0)            # start tokens (≈2)
    probs  = torch.softmax(scores, dim=-1)
    conf   = probs[torch.arange(scores.size(0)), toks[K:]].tolist()

    token_text = proc4.decode(toks, skip_special_tokens=True)   # str
    token_conf = list(zip(token_text, conf))  # [(token, confidence), …]

    cleaned_pairs = [(clean(t), c) for t, c in token_conf if len(clean(t))>0]
    return "".join(t for t, _ in cleaned_pairs)





def asr_workflow_12(audio_path: str) -> str:
    audio, sr = librosa.load(audio_path, sr=16_000)          # ← force 16 kHz

    inp = proc4(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
        return_attention_mask=True,
    )
    feat = inp.input_features.to(model4.device)      # (1, 80, ≤3000)
    am   = inp.attention_mask.to(model4.device)

    with torch.inference_mode():
        out = model4.generate(
            input_features = feat,
            attention_mask = am,
            generation_config = gen_cfg,
            language = "zh",
            task     = "transcribe",
            temperature = 0.0,
            num_beams=5,
            early_stopping=True,
        )

    toks   = out.sequences[0].cpu()                   # full token ids
    # out.scores is a list (len = seq_len – start_tokens) of (1, vocab)
    scores = torch.stack(out.scores).squeeze(1).cpu() # (L-K, vocab)
    K      = toks.size(0) - scores.size(0)            # start tokens (≈2)
    probs  = torch.softmax(scores, dim=-1)
    conf   = probs[torch.arange(scores.size(0)), toks[K:]].tolist()

    token_text = proc4.decode(toks, skip_special_tokens=True)   # str
    token_conf = list(zip(token_text, conf))  # [(token, confidence), …]

    cleaned_pairs = [(clean(t), c) for t, c in token_conf if len(clean(t))>0]
    return "".join(t for t, _ in cleaned_pairs)



def asr_workflow_13(audio_path: str) -> str:
    audio, sr = librosa.load(audio_path, sr=16_000)          # ← force 16 kHz

    inp = proc4(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
        return_attention_mask=True,
    )
    feat = inp.input_features.to(model4.device)      # (1, 80, ≤3000)
    am   = inp.attention_mask.to(model4.device)
    forced_decoder_ids = proc4.get_decoder_prompt_ids(language="zh", task="transcribe")
    with torch.inference_mode():
        out = model4.generate(
            input_features = feat,
            attention_mask = am,
            generation_config = gen_cfg,
            temperature = 0.0,
            num_beams=5,
            early_stopping=True,
            forced_decoder_ids=forced_decoder_ids
        )

    toks   = out.sequences[0].cpu()                   # full token ids
    # out.scores is a list (len = seq_len – start_tokens) of (1, vocab)
    scores = torch.stack(out.scores).squeeze(1).cpu() # (L-K, vocab)
    K      = toks.size(0) - scores.size(0)            # start tokens (≈2)
    probs  = torch.softmax(scores, dim=-1)
    conf   = probs[torch.arange(scores.size(0)), toks[K:]].tolist()

    token_text = proc4.decode(toks, skip_special_tokens=True)   # str
    token_conf = list(zip(token_text, conf))  # [(token, confidence), …]

    cleaned_pairs = [(clean(t), c) for t, c in token_conf if len(clean(t))>0]
    return "".join(t for t, _ in cleaned_pairs)










# ! ===================================================================================================================
# ! ===================================================================================================================
# ! ===================================================================================================================
# ! ===================================================================================================================




# ── asr_workflow_beams.py ──────────────────────────────────────────────────
# from faster_whisper.tokenizer import Tokenizer


# def asr_workflow_aa(
#     audio_path: str,
#     beam_size: int = 5,
#     n_best:   int = 5,
# ) -> list[str]:
#     """
#     Faster-Whisper N-best without confidences.
#     Uses model3 (Belle CT2) and returns the top `n_best` hypotheses.
#     """

#     # 1) audio → log-Mel
#     wav, sr = sf.read(audio_path)
#     if wav.ndim == 2:
#         wav = wav.mean(axis=1)
#     mel = model3.feature_extractor(wav, sr)[..., :-1]          # (80, T)
#     mel = mel[np.newaxis, ...]                                 # add batch dim

#     # 2) encode once (gives the StorageView that generate wants)
#     enc_out = model3.encode(mel)

#     # 3) build prompt tokens
#     tok = Tokenizer(model3.hf_tokenizer,
#                     model3.model.is_multilingual,
#                     task="transcribe",
#                     language="zh")
#     prompt = [tok.sot, tok.language, tok.task]

#     # 4) beam search with N hypotheses
#     res = model3.model.generate(
#         enc_out,
#         [prompt],                     # batch of size 1
#         beam_size      = beam_size,
#         num_hypotheses = n_best,
#         return_scores  = False,
#     )[0]                              # unpack batch

#     # 5) decode & clean
#     hyps = []
#     for ids in res.sequences_ids:
#         txt = tok.decode(ids)
#         txt     = clean(txt.replace("▁", ""))
#         hyps.append(clean(txt))

#     print(hyps)


#     return hyps[0]











# def asr_workflow_bb(
#     audio_path: str,
#     user_history: Optional[List[str]] | None = None,
#     assistant_history: Optional[List[str]] | None = None
# ) -> str:
#     user_history = user_history or []
#     assistant_history = assistant_history or []
#     if user_history and assistant_history:
#         user_history = ast.literal_eval(re.sub(r'""', '"', user_history[0]))
#         assistant_history = ast.literal_eval(re.sub(r'""', '"', assistant_history[0]))


#     lines = []
#     for i in range(max(len(user_history), len(assistant_history))):
#         if i < len(user_history):
#             lines.append(f"张：{user_history[i].strip()}")
#         if i < len(assistant_history):
#             lines.append(f"李：{assistant_history[i].strip()}")
#     lines.append(f"张：")
#     prompt_text = "\n".join(lines)

#     segments, _ = model3.transcribe(
#         audio_path,
#         language="zh",
#         task="transcribe",
#         beam_size=5,
#         temperature=0.0,
#         initial_prompt=prompt_text,
#         condition_on_previous_text=False,
#         word_timestamps=True,          # gives us per-word probability
#     )

#     # Collect words and their probabilities
#     word_conf = [(w.word, w.probability)
#                  for seg in segments for w in seg.words
#                  if clean(w.word)]                # drop fillers/punct

#     transcript = "".join(w for w, _ in word_conf).strip()
#     print(word_conf)
#     print(clean(transcript))
#     return clean(transcript)











# ── asr_workflow_xx.py ─────────────────────────────────────────────────────────
# * multiple hypotheses and multiple confidence scores


from typing import List, Dict, Tuple


def _token_confidence(
    seq_ids: torch.Tensor,
    encoder_feats: torch.Tensor,
    enc_mask: torch.Tensor,
) -> List[Tuple[str, float]]:

    # Teacher-force the decoder on all tokens *except the last*
    inp_ids  = seq_ids[:-1].unsqueeze(0).to(model4.device)   # (1, L-1)
    with torch.inference_mode():
        out = model4(
            input_features = encoder_feats,
            attention_mask = enc_mask,
            decoder_input_ids = inp_ids,
            use_cache = False,
        )
    # logits: (1, L-1, vocab) → probabilities
    probs = F.softmax(out.logits.squeeze(0), dim=-1)         # (L-1, vocab)

    next_ids = seq_ids[1:]                                   # tokens to predict
    conf     = probs[torch.arange(probs.size(0)), next_ids].cpu().tolist()

    toks_txt = proc4.decode(seq_ids, skip_special_tokens=True)
    return list(zip(toks_txt, conf))


def asr_workflow_xx(
    audio_path: str,
    beam_size: int = 5,
    n_best: int = 5,
    lang: str = "zh",
) -> List[Dict]:
    """
    Transcribe `audio_path` and return an N-best list.
    Each item contains:
        {
          "text": str,                   # hypothesis text
          "log_prob": float,             # sequence-level log P
          "posterior": float,            # P(hyp | N-best) after soft-max
          "token_conf": [(token, p), …]  # per-token confidence
        }
    """
    # 1) audio → log-Mel features (16 kHz to match Whisper training)
    wav, sr = librosa.load(audio_path, sr=16_000)
    inp = proc4(
        wav,
        sampling_rate=sr,
        return_tensors="pt",
        return_attention_mask=True,
    )
    feats = inp.input_features.to(model4.device)             # (1, 80, T)
    amask = inp.attention_mask.to(model4.device)

    # 2) generate N-best with scores
    with torch.inference_mode():
        out = model4.generate(
            input_features = feats,
            attention_mask = amask,
            generation_config = gen_cfg,
            language = lang,
            task     = "transcribe",
            temperature = 0.0,
            num_beams  = beam_size,
            num_return_sequences = n_best,
            output_scores = True,
            return_dict_in_generate = True,
        )

    seqs   = out.sequences.cpu()                 # (n_best, L)
    scores = out.sequences_scores.cpu()          # log P (length-norm by default)
    post   = F.softmax(scores, dim=0).tolist()   # posterior over N-best

    results = []
    for i in range(n_best):
        token_conf = _token_confidence(
            seqs[i], encoder_feats=feats, enc_mask=amask
        )

        cleaned_pairs = [(clean(t), c) for t, c in token_conf if len(clean(t))]
        text          = "".join(t for t, _ in cleaned_pairs)

        results.append(
            {
                "rank":      i + 1,
                "text":      text,
                "log_prob":  float(scores[i]),
                "posterior": float(post[i]),
                "token_conf": cleaned_pairs,   # [(token, conf ∈ 0-1), …]
            }
        )
    #print("hehehehe")
    #print(results)
    comparisons = [p["text"] for p in results]
    if len(set(comparisons))!=1:
        print("UNEQUAL==============================================")
        print(comparisons)


    return results[0]["text"]










#! check the following
# from itertools import zip_longest

# def asr_workflow_9(
#     audio_path: str,
#     user_history: Optional[List[str]] | None = None,
#     assistant_history: Optional[List[str]] | None = None,
# ) -> str:

#     user_history = user_history or []
#     assistant_history = assistant_history or []
#     if user_history and assistant_history:
#         user_history = ast.literal_eval(re.sub(r'""', '"', user_history[0]))
#         assistant_history = ast.literal_eval(re.sub(r'""', '"', assistant_history[0]))


#     lines = []
#     for i in range(max(len(user_history), len(assistant_history))):
#         if i < len(user_history):
#             lines.append(f"张：{user_history[i].strip()}")
#         if i < len(assistant_history):
#             lines.append(f"李：{assistant_history[i].strip()}")
#     lines.append(f"张：")
#     prompt_text = "\n".join(lines)
#     #print(prompt_text)

#     # 2) Convert prompt → token IDs  (HF way)

#     tok = proc2.tokenizer
#     tok.truncation_side = "left"                      # keep the tail
#     ids = tok(
#         prompt_text,
#         max_length=300,                       # keep ≤449 tokens
#         truncation=True,
#         add_special_tokens=False
#     ).input_ids
#     prompt_ids = torch.tensor(ids, dtype=torch.long, device=modelbelle2.device)
#     forced_decoder_ids = proc2.get_decoder_prompt_ids(language="zh", task="transcribe")
    
#     remaining = max(1, modelbelle2.config.max_target_positions - len(prompt_ids)-len(forced_decoder_ids) -3)

#     result = belle_fast2(
#         audio_path,
#         generate_kwargs={
#             "forced_decoder_ids": forced_decoder_ids,
#             "prompt_ids": prompt_ids,
#             "max_new_tokens": remaining
#         },
#     )
#     return clean(result["text"])







WORKFLOWS: List[Callable[..., str]] = [asr_workflow_xx]#asr_workflow_1,asr_workflow_2,asr_workflow_3,asr_workflow_4
#asr_workflow_1,asr_workflow_2,asr_workflow_4,asr_workflow_5,asr_workflow_7,asr_workflow_8
def batch_transcribe(csv_path: Path, audio_dir: Path, workflows: List[Callable[..., str]], out_path: Path, sample_size: Optional[int],timeing = False):
    df_raw = pd.read_csv(csv_path)
    in_cols  = {"filename", "transcription", "user_history", "assistant_history"}
    out_cols = ["filename", "transcription"]
    # Keep only relevant input columns; silently drop everything else
    df = df_raw.filter(items=in_cols, axis=1)
    # Verify the mandatory columns exist
    if not {"filename", "transcription"}.issubset(df.columns):
        raise ValueError("Input CSV must contain 'filename' and 'transcription' columns.")

    df["filename"] = df["filename"].astype(str)
    has_history_cols = {"user_history", "assistant_history"}.issubset(df.columns)

    # Optional even-spacing subsample
    if sample_size is not None and 0 < sample_size < len(df):
        sel = np.linspace(0, len(df) - 1, num=sample_size, dtype=int)
        df = df.iloc[sel].reset_index(drop=True)
        print(f"Processing evenly spaced subset of {sample_size} rows")

    # Prepare transcript columns for every workflow
    for w_idx in range(1, len(workflows) + 1):
        col_name = f"transcript_{w_idx}"
        df[col_name] = ""
        out_cols.append(col_name)  # ensure they survive to the output

    # ! =========================================================================================================
    # Main loop
    for row_idx, row in df.iterrows():
        audio_path = audio_dir / row["filename"]


        # Extract history (as lists) if present
        if has_history_cols:
            usr_raw = row.get("user_history", np.nan)
            asst_raw = row.get("assistant_history", np.nan)
            usr_hist: List[str] = [] if pd.isna(usr_raw) else [str(usr_raw)]
            asst_hist: List[str] = [] if pd.isna(asst_raw) else [str(asst_raw)]
        else:
            usr_hist = asst_hist = []

        # Execute each workflow
        for w_idx, pipeline in enumerate(workflows, start=1):

            if (pipeline is asr_workflow_4) :#or (pipeline is asr_workflow_9):
                start = time.perf_counter()

                txt = pipeline(
                    str(audio_path),
                    user_history=usr_hist,
                    assistant_history=asst_hist,
                )
                elapsed = time.perf_counter() - start
                if timeing:
                    txt= elapsed
            else:
                start = time.perf_counter()

                txt = pipeline(str(audio_path))

                elapsed = time.perf_counter() - start
                if timeing:
                    txt= elapsed

            df.at[row_idx, f"transcript_{w_idx}"] = txt

        print(f"Done row {row_idx} ({row['filename']})")

    # ---------------------------------------------------------------------
    df_out = df[out_cols]

    df_out["transcription"].apply(normalise)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"\n✓ Results written to {out_path.resolve()}")
    return df_out


if __name__ == "__main__":
    batch_transcribe(
        csv_path=Path("C:/Users/felix/PROGRAMMING/THESIS/data/data/LATIC/orig_transcript.csv"),
        audio_dir=Path("C:/Users/felix/PROGRAMMING/THESIS/data/data/LATIC/AUDIO/"),
        workflows=WORKFLOWS,
        out_path=Path("output_xx_LATIC.csv"),
        sample_size=100,  # Set None to process the entire CSV  
        )
    
    

    
