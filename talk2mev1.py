import streamlit as st
import openai
import base64
import tempfile
import os
import time
from io import BytesIO
import json
from datetime import datetime

# Set up page
st.set_page_config(page_title="Voice Chat with GPT-4o", layout="wide")
st.title("🎙️ Real-Time Voice Chat with GPT-4o")

# API key input
if 'openai_api_key' not in st.session_state:
    st.session_state['openai_api_key'] = ""
st.session_state['openai_api_key'] = st.text_input("🔑 Enter your OpenAI API key", type="password")

# Initialize chat log
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

# Initialize audio model options
voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
voice = st.selectbox("🎤 Select a voice for GPT-4o response", voice_options, index=0)

# Record audio input
recorded_audio = st.audio_recorder("🎙️ Hold to speak", format="audio/wav")

# Save and process audio input
if recorded_audio and st.session_state['openai_api_key']:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(recorded_audio.getbuffer())
        tmp_audio_path = tmp_audio.name

    with open(tmp_audio_path, "rb") as f:
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

    with st.spinner("🧠 GPT-4o is thinking and responding..."):
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

    audio_bytes = base64.b64decode(message['audio']['data'])
    st.audio(audio_bytes, format="audio/mp3")

# Display chat log
st.subheader("📝 Conversation Log")
for i, msg in enumerate(st.session_state['conversation']):
    role = "🧑‍💼 You" if msg['role'] == 'user' else "🤖 GPT-4o"
    content = msg['content'] if isinstance(msg['content'], str) else msg['content'][0].get('text', '')
    st.markdown(f"**{role}:** {content}")

# Download log
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
