import streamlit as st
import openai
import base64
import os
import time
import json
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import tempfile
import asyncio
import requests

# Page settings
st.set_page_config(page_title="Voice Chat with GPT-4o", layout="wide")
st.title("🎙️ Live Duplex Voice Chat with GPT-4o + Live Transcription")

# API Key
if 'openai_api_key' not in st.session_state:
    st.session_state['openai_api_key'] = ""
st.session_state['openai_api_key'] = st.text_input("🔑 Enter your OpenAI API key", type="password")

# Conversation log
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Voice model selection
voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
voice = st.selectbox("🎤 Select a voice for GPT-4o response", voice_options, index=0)

# Audio recorder
class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.audio_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.audio_frames.append(frame.to_ndarray().tobytes())
        return frame

    def get_audio_bytes(self):
        return b"".join(self.audio_frames)

st.header("🎧 Speak and Listen to GPT-4o")
ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    audio_processor_factory=AudioProcessor,
    async_processing=True,
)

if ctx.audio_processor and st.button("🗣️ Transcribe and Send"):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(ctx.audio_processor.get_audio_bytes())
        audio_path = tmp.name

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    # Re-encode to supported format (MP3) if needed
    import pydub
    audio_segment = pydub.AudioSegment.from_raw(audio_path, sample_width=2, frame_rate=48000, channels=1)
    converted_path = audio_path.replace(".mp3", "_converted.mp3")
    audio_segment.export(converted_path, format="mp3")

    with open(converted_path, "rb") as f:
        audio_data = f.read()

    # Transcribe using Whisper
    with st.spinner("🔎 Transcribing audio..."):
        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {st.session_state['openai_api_key']}"
        }
        files = {
            "file": ("audio.mp3", audio_data, "audio/mpeg")
        }
        data = {
            "model": "whisper-1",
            "response_format": "json"
        }
        response = requests.post(url, headers=headers, files=files, data=data)

    if not response.ok:
        st.error(f"Transcription failed. {response.status_code}: {response.text}")
        st.error("Transcription failed.")
    else:
        transcript = response.json().get("text", "")
        st.success("Transcription complete.")
        st.markdown(f"**🧑‍💼 You:** {transcript}")
        st.session_state['conversation'].append({"role": "user", "content": transcript})

        # Send to GPT-4o for reply
        client = openai.OpenAI(api_key=st.session_state['openai_api_key'])
        messages = st.session_state['conversation'].copy()

        with st.spinner("🤖 GPT-4o is responding with audio..."):
            reply = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format="audio",
                voice=voice
            )

        audio_reply = reply.content
        st.audio(audio_reply, format="audio/mp3")

        # Store text version of model response
        reply_text = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7
        ).choices[0].message.content

        st.markdown(f"**🤖 GPT-4o:** {reply_text}")
        st.session_state['conversation'].append({"role": "assistant", "content": reply_text})

# Conversation log
st.subheader("📝 Conversation Log")
for msg in st.session_state['conversation']:
    role = "🧑‍💼 You" if msg['role'] == 'user' else "🤖 GPT-4o"
    st.markdown(f"**{role}:** {msg['content']}")

# Download logs
if st.button("📥 Download Log as JSON"):
    filename = f"gpt4o_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    st.download_button("Download", json.dumps(st.session_state['conversation'], indent=2), filename, mime="application/json")

if st.button("📄 Download Log as Text"):
    log_text = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state['conversation']])
    st.download_button("Download", log_text, file_name="conversation_log.txt", mime="text/plain")

# Save locally
SAVE_DIR = "conversation_logs"
os.makedirs(SAVE_DIR, exist_ok=True)
if st.button("💾 Save Log to Disk"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(SAVE_DIR, f"session_{timestamp}.json"), "w") as f:
        json.dump(st.session_state['conversation'], f, indent=2)
    st.success("✅ Saved log to disk.")
