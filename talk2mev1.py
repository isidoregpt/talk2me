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

# Page settings
st.set_page_config(page_title="Voice Chat with GPT-4o", layout="wide")
st.title("🎙️ Real-Time Full Duplex Voice Chat with GPT-4o")

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

class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.audio_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.audio_frames.append(frame.to_ndarray().tobytes())
        return frame

    def get_audio_data(self):
        return b"".join(self.audio_frames)

st.header("🎧 Speak to GPT-4o")
ctx = webrtc_streamer(
    key="send-audio",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    audio_processor_factory=AudioProcessor,
    async_processing=True
)

if ctx.audio_processor and st.button("📤 Send to GPT-4o"):
    audio_bytes = ctx.audio_processor.get_audio_data()
    tmp_path = tempfile.mktemp(suffix=".wav")
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    with open(tmp_path, "rb") as f:
        audio_data = f.read()
    encoded_audio = base64.b64encode(audio_data).decode("utf-8")

    client = openai.OpenAI(api_key=st.session_state['openai_api_key'])
    messages = st.session_state['conversation'].copy()
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "Please respond to this audio."},
            {"type": "input_audio", "input_audio": {"data": encoded_audio, "format": "wav"}}
        ]
    })

    with st.spinner("🤖 GPT-4o is generating a reply..."):
        response = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": voice, "format": "mp3"},
            messages=messages,
            store=True
        )

    message = response.choices[0].message
    st.session_state['conversation'].append({"role": "user", "content": "[audio input]"})
    st.session_state['conversation'].append({"role": "assistant", "content": message['content']})

    audio_reply = base64.b64decode(message['audio']['data'])
    st.audio(audio_reply, format="audio/mp3")

# Display chat log
st.subheader("📝 Conversation Log")
for msg in st.session_state['conversation']:
    role = "🧑‍💼 You" if msg['role'] == 'user' else "🤖 GPT-4o"
    content = msg['content'] if isinstance(msg['content'], str) else msg['content'][0].get('text', '')
    st.markdown(f"**{role}:** {content}")

# Download logs
if st.button("📥 Download Conversation Log as JSON"):
    filename = f"gpt4o_convo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_log = json.dumps(st.session_state['conversation'], indent=2)
    st.download_button("Download", json_log, file_name=filename, mime="application/json")

if st.button("📄 Download Conversation Log as Text"):
    log_text = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state['conversation']])
    st.download_button("Download", log_text, file_name="conversation_log.txt", mime="text/plain")

# Optional: Save persistent session
SAVE_DIR = "conversation_logs"
os.makedirs(SAVE_DIR, exist_ok=True)
if st.button("💾 Save session to disk"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(SAVE_DIR, f"session_{timestamp}.json")
    with open(filepath, "w") as f:
        json.dump(st.session_state['conversation'], f, indent=2)
    st.success(f"Saved session as {filepath}")
