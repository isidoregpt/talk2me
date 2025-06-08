import streamlit as st
import openai
import anthropic
import base64
import os
import time
import json
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
from PIL import Image
import tempfile
import requests
import numpy as np
from collections import deque

# Page settings
st.set_page_config(page_title="Multi-AI Voice Chat", layout="wide")
st.title("🎙️ Universal Voice Chat - OpenAI GPT-4o & Anthropic Claude")

# Define ICE configuration with TURN support
ICE_CONFIG = {
    "iceServers": [
        {"urls": "stun:stun.l.google.com:19302"},
        {
            "urls": "turn:openrelay.metered.ca:80",
            "username": "openrelayproject",
            "credential": "openrelayproject"
        },
        {
            "urls": "turn:openrelay.metered.ca:443",
            "username": "openrelayproject",
            "credential": "openrelayproject"
        }
    ],
    "iceCandidatePoolSize": 10
}

# Initialize session state
st.session_state.setdefault('conversation', [])
st.session_state.setdefault('selected_ai', 'OpenAI')
st.session_state.setdefault('continuous_mode', False)
st.session_state.setdefault('is_processing', False)
st.session_state.setdefault('openai_api_key', '')
st.session_state.setdefault('anthropic_api_key', '')

# API Key Input
with st.sidebar:
    st.header("🔐 API Keys")
    st.session_state['openai_api_key'] = st.text_input("OpenAI API Key", type="password", value=st.session_state['openai_api_key'])
    st.session_state['anthropic_api_key'] = st.text_input("Anthropic API Key", type="password", value=st.session_state['anthropic_api_key'])

# Audio Processor for recording
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.frames.append(frame.to_ndarray())
        return frame

    def get_audio(self):
        if self.frames:
            audio = np.concatenate(self.frames, axis=0)
            self.frames = []
            return audio
        return None

# Save audio
def save_audio(audio_data, sample_rate=48000):
    import wave
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio_int16 = (audio_data * 32767).astype(np.int16)
    with wave.open(tmp.name, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return tmp.name

# Transcribe using OpenAI Whisper
def transcribe_audio(filepath):
    headers = {"Authorization": f"Bearer {st.session_state['openai_api_key']}"}
    with open(filepath, 'rb') as f:
        files = {"file": ("audio.wav", f, "audio/wav")}
        data = {"model": "whisper-1", "response_format": "json"}
        response = requests.post("https://api.openai.com/v1/audio/transcriptions", headers=headers, files=files, data=data)
    return response.json().get("text", "") if response.ok else ""

# Call OpenAI Chat
def openai_chat(messages):
    client = openai.OpenAI(api_key=st.session_state['openai_api_key'])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7
    )
    return response.choices[0].message.content

# Call Claude
def claude_chat(messages):
    client = anthropic.Anthropic(api_key=st.session_state['anthropic_api_key'])
    content = [{"role": m['role'], "content": m['content']} for m in messages if m['role'] in ['user', 'assistant']]
    response = client.messages.create(model="claude-3-5-sonnet-20241022", messages=content, max_tokens=4000, system="You are a helpful AI.")
    return response.content[0].text

# UI Selection
st.sidebar.header("🤖 AI Provider")
ai_provider = st.sidebar.radio("Select: ", ["OpenAI", "Anthropic"])
st.session_state['selected_ai'] = ai_provider

# WebRTC Audio Stream
st.subheader("🎤 Record and Chat")
ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration=ICE_CONFIG,
    audio_processor_factory=AudioProcessor,
    async_processing=True,
)

# Handle processing
if ctx.audio_processor:
    if st.button("🗣️ Process Audio"):
        audio_data = ctx.audio_processor.get_audio()
        if audio_data is not None:
            audio_path = save_audio(audio_data)
            transcript = transcribe_audio(audio_path)
            st.write(f"**You:** {transcript}")
            st.session_state['conversation'].append({"role": "user", "content": transcript})

            if ai_provider == "OpenAI":
                reply = openai_chat(st.session_state['conversation'])
            else:
                reply = claude_chat(st.session_state['conversation'])

            st.session_state['conversation'].append({"role": "assistant", "content": reply})
            st.markdown(f"**AI:** {reply}")
        else:
            st.warning("No audio detected. Please try speaking again.")

# Show conversation history
if st.session_state['conversation']:
    st.subheader("📝 Conversation History")
    for msg in st.session_state['conversation']:
        speaker = "🧑 You" if msg['role'] == 'user' else "🤖 AI"
        st.markdown(f"**{speaker}:** {msg['content']}")
