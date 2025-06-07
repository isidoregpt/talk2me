import os
import time
import json
import tempfile
import base64
import requests
import wave
import threading
from datetime import datetime
from collections import deque

import streamlit as st
import openai
import numpy as np
from anthropic import Client as AnthropicClient
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, VideoProcessorBase, WebRtcMode
import av
from PIL import Image

# ---------------------- Helper Classes ----------------------
class ContinuousAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = deque(maxlen=48000 * 10)
        self.speech = []
        self.speaking = False
        self.silence_counter = 0
        self.silence_thresh = 48000 * 2
        self.vol_thresh = 0.01
        self.min_speech = 48000 * 1

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        arr = frame.to_ndarray()
        for sample in arr.flatten():
            self.buffer.append(sample)
        vol = np.sqrt(np.mean(arr**2))
        if vol > self.vol_thresh:
            if not self.speaking:
                self.speaking = True
                self.speech = list(self.buffer)
            else:
                self.speech.extend(arr.flatten())
            self.silence_counter = 0
        else:
            if self.speaking:
                self.silence_counter += arr.size
                self.speech.extend(arr.flatten())
                if self.silence_counter > self.silence_thresh and len(self.speech) > self.min_speech:
                    st.session_state.pending_speech = np.array(self.speech)
                    self.speaking = False
                    self.speech = []
                    self.silence_counter = 0
        return frame

    def get_audio(self):
        if len(self.speech) > 0:
            return np.array(self.speech)
        return None

class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.frames = []
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.frames.append(frame.to_ndarray())
        return frame
    def get_audio(self):
        if not self.frames:
            return None
        return np.concatenate(self.frames, axis=0)

class VideoRecorder(VideoProcessorBase):
    def __init__(self):
        self.frame = None
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        arr = frame.to_ndarray(format="bgr24")
        self.frame = Image.fromarray(arr[:, :, ::-1])
        return frame
    def get_image(self):
        return self.frame

# ---------------------- Helper Functions ----------------------
def save_wav(data: np.ndarray, rate=48000, channels=1) -> str:
    tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tf.name, 'wb') as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        if data.dtype != np.int16:
            data = (data * 32767).astype(np.int16)
        w.writeframes(data.tobytes())
    return tf.name


def transcribe(data_bytes: bytes, key: str) -> str:
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {key}"}
    resp = requests.post(url, headers=headers,
                         files={"file": ("audio.wav", data_bytes, "audio/wav")},
                         data={"model": "whisper-1"})
    if resp.ok:
        return resp.json().get("text", "")
    else:
        raise RuntimeError(f"Whisper error {resp.status_code}: {resp.text}")


def generate_tts(text: str, voice: str, key: str) -> str:
    client = openai.OpenAI(api_key=key)
    resp = client.audio.speech.create(model="tts-1", voice=voice, input=text)
    tf = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    resp.stream_to_file(tf.name)
    return tf.name


def get_openai_reply(msgs: list, model: str, key: str) -> str:
    client = openai.OpenAI(api_key=key)
    resp = client.chat.completions.create(model=model, messages=msgs, temperature=0.7)
    return resp.choices[0].message.content


def encode_img(img: Image.Image) -> str:
    buf = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    img.save(buf.name, format='JPEG', quality=85)
    b64 = base64.b64encode(open(buf.name, 'rb').read()).decode()
    os.unlink(buf.name)
    return b64


def get_anthropic_reply(msgs: list, model: str, key: str, img: Image.Image=None) -> str:
    client = AnthropicClient(api_key=key)
    anthro_msgs = []
    system = "You are a helpful assistant."
    for m in msgs:
        if m['role']=='system': system = m['content']
        else: anthro_msgs.append({"role": m['role'], "content": m['content']})
    if img and anthro_msgs and anthro_msgs[-1]['role']=='user':
        anthro_msgs[-1]['content'] = [
            {"type":"text","text": anthro_msgs[-1]['content']},
            {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":encode_img(img)}}
        ]
    resp = client.chat.completions.create(model=model, system=system, messages=anthro_msgs)
    return resp.choices[0].message.content

def cleanup(files: list):
    for f in files:
        try: os.unlink(f)
        except: pass


def process_continuous(audio_data, provider, img_proc):
    wav = save_wav(audio_data)
    try:
        txt = transcribe(open(wav,'rb').read(), st.session_state.openai_key)
        if not txt.strip(): return
        st.session_state.conversation.append({"role":"user","content":txt})
        st.write(f"**You:** {txt}")
        img = img_proc.get_image() if img_proc else None
        if provider == 'OpenAI':
            reply = get_openai_reply(st.session_state.conversation, st.session_state.openai_model, st.session_state.openai_key)
        else:
            reply = get_anthropic_reply(st.session_state.conversation, st.session_state.anthropic_model, st.session_state.anthropic_key, img)
        st.write(f"**AI:** {reply}")
        mp3 = generate_tts(reply, st.session_state.voice, st.session_state.openai_key)
        st.audio(mp3)
        st.session_state.conversation.append({"role":"assistant","content":reply})
    finally:
        cleanup([wav, mp3])


def process_recording(rec_proc, provider, img_proc):
    data = rec_proc.get_audio()
    if data is None: st.warning("No audio captured."); return
    process_continuous(data, provider, img_proc)


def process_upload(file, provider, img_proc):
    data = file.read()
    txt = transcribe(data, st.session_state.openai_key)
    if not txt.strip(): st.warning("No speech."); return
    st.write(f"**You:** {txt}")
    st.session_state.conversation.append({"role":"user","content":txt})
    img = img_proc.get_image() if img_proc else None
    if provider=='OpenAI': reply = get_openai_reply(st.session_state.conversation, st.session_state.openai_model, st.session_state.openai_key)
    else: reply = get_anthropic_reply(st.session_state.conversation, st.session_state.anthropic_model, st.session_state.anthropic_key, img)
    st.write(f"**AI:** {reply}")
    mp3 = generate_tts(reply, st.session_state.voice, st.session_state.openai_key)
    st.audio(mp3)
    st.session_state.conversation.append({"role":"assistant","content":reply})
    cleanup([mp3])


def process_text(msg, provider, img_proc):
    st.write(f"**You:** {msg}")
    st.session_state.conversation.append({"role":"user","content":msg})
    img = img_proc.get_image() if img_proc else None
    if provider=='OpenAI': reply = get_openai_reply(st.session_state.conversation, st.session_state.openai_model, st.session_state.openai_key)
    else: reply = get_anthropic_reply(st.session_state.conversation, st.session_state.anthropic_model, st.session_state.anthropic_key, img)
    st.write(f"**AI:** {reply}")
    if st.session_state.openai_key:
        mp3 = generate_tts(reply, st.session_state.voice, st.session_state.openai_key)
        st.audio(mp3)
        cleanup([mp3])
    st.session_state.conversation.append({"role":"assistant","content":reply})

# ---------------------- Streamlit App ----------------------
st.set_page_config(page_title="Multi-AI Voice Chat", layout="wide")

# Initialize state
for key, val in {
    'conversation': [],
    'openai_key':'', 'anthropic_key':'',
    'openai_model':'gpt-4o', 'anthropic_model':'claude-sonnet-4-20250514',
    'voice':'alloy', 'provider':'OpenAI',
    'continuous':False, 'pending_speech':None
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# UI: Provider & Keys
st.title("🎙️ Universal Voice Chat")
col1, col2 = st.columns(2)
with col1:
    st.session_state.provider = st.radio("AI Provider", ["OpenAI","Anthropic"], index=0 if st.session_state.provider=='OpenAI' else 1)
with col2:
    st.session_state.openai_key = st.text_input("OpenAI Key", type="password", value=st.session_state.openai_key)
    st.session_state.anthropic_key = st.text_input("Anthropic Key", type="password", value=st.session_state.anthropic_key)

# Model & Voice
if st.session_state.provider=='OpenAI':
    st.selectbox("OpenAI Model", ["gpt-4o","gpt-4-turbo","gpt-3.5-turbo"], key='openai_model')
else:
    st.selectbox("Claude Model", ["claude-sonnet-4-20250514","claude-opus-4-20250514"], key='anthropic_model')
st.selectbox("Voice", ["alloy","echo","fable","onyx","nova","shimmer"], key='voice')
st.checkbox("Continuous Mode", key='continuous')

# Video for Claude
video_ctx = None
if st.session_state.provider=='Anthropic':
    video_ctx = webrtc_streamer(key='video', mode=WebRtcMode.SENDONLY,
                                media_stream_constraints={"video":True,"audio":False},
                                video_processor_factory=VideoRecorder)

# Continuous
if st.session_state.continuous:
    audio_ctx = webrtc_streamer(key='cont', mode=WebRtcMode.SENDONLY,
                                 media_stream_constraints={"audio":True},
                                 audio_processor_factory=ContinuousAudioProcessor)
    if st.session_state.pending_speech is not None and not st.session_state.get('processing',False):
        st.session_state.processing = True
        process_continuous(st.session_state.pending_speech, st.session_state.provider, video_ctx and video_ctx.video_processor)
        st.session_state.pending_speech = None
        st.session_state.processing = False

# Manual Recording
elif True:
    st.header("🎙️ Manual Voice Recording")
    rec_ctx = webrtc_streamer(key='rec', mode=WebRtcMode.SENDONLY,
                              media_stream_constraints={"audio":True},
                              audio_processor_factory=AudioRecorder)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Process Recording"):
            process_recording(rec_ctx.audio_processor, st.session_state.provider, video_ctx and video_ctx.video_processor)
    with col2:
        if st.button("Clear Recording"):
            rec_ctx.audio_processor.frames = []

# File Upload
st.header("📁 Upload Audio File")
uf = st.file_uploader("Choose audio", type=['wav','mp3','m4a','flac','ogg'])
if uf and st.button("Process Upload"):
    process_upload(uf, st.session_state.provider, video_ctx and video_ctx.video_processor)

# Text Input
st.header("⌨️ Text Input")	
txt = st.text_area("Message:")
if st.button("Send") and txt.strip():
    process_text(txt.strip(), st.session_state.provider, video_ctx and video_ctx.video_processor)

# Conversation Log
if st.session_state.conversation:
    st.header("💬 History")
    for m in st.session_state.conversation:
        role = "You" if m['role']=='user' else "AI"
        st.write(f"**{role}:** {m['content']}")
    if st.button("Clear Chat"):
        st.session_state.conversation = []
