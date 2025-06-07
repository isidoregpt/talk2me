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
import numpy as np

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
        # Convert audio frame to numpy array and store
        audio_array = frame.to_ndarray()
        self.audio_frames.append(audio_array)
        return frame

    def get_audio_data(self):
        if not self.audio_frames:
            return None
        
        # Concatenate all audio frames
        audio_data = np.concatenate(self.audio_frames, axis=0)
        return audio_data

def save_audio_as_wav(audio_data, sample_rate=48000, channels=1):
    """Save numpy array as WAV file without pydub"""
    import wave
    import struct
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    
    with wave.open(temp_file.name, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        # Convert float to int16
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        # Write audio data
        wav_file.writeframes(audio_data.tobytes())
    
    return temp_file.name

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
    if not st.session_state['openai_api_key']:
        st.error("Please enter your OpenAI API key first!")
    else:
        audio_data = ctx.audio_processor.get_audio_data()
        
        if audio_data is None or len(audio_data) == 0:
            st.warning("No audio data captured. Please try recording again.")
        else:
            try:
                # Save audio as WAV file
                audio_path = save_audio_as_wav(audio_data)
                
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()

                # Transcribe using Whisper
                with st.spinner("🔎 Transcribing audio..."):
                    url = "https://api.openai.com/v1/audio/transcriptions"
                    headers = {
                        "Authorization": f"Bearer {st.session_state['openai_api_key']}"
                    }
                    files = {
                        "file": ("audio.wav", audio_bytes, "audio/wav")
                    }
                    data = {
                        "model": "whisper-1",
                        "response_format": "json"
                    }
                    response = requests.post(url, headers=headers, files=files, data=data)

                if not response.ok:
                    st.error(f"Transcription failed. {response.status_code}: {response.text}")
                else:
                    transcript = response.json().get("text", "")
                    if transcript.strip():
                        st.success("Transcription complete.")
                        st.markdown(f"**🧑‍💼 You:** {transcript}")
                        st.session_state['conversation'].append({"role": "user", "content": transcript})

                        # Send to GPT-4o for reply
                        try:
                            client = openai.OpenAI(api_key=st.session_state['openai_api_key'])
                            messages = st.session_state['conversation'].copy()

                            with st.spinner("🤖 GPT-4o is thinking..."):
                                # Get text response first
                                text_response = client.chat.completions.create(
                                    model="gpt-4o",
                                    messages=messages,
                                    temperature=0.7
                                )
                                reply_text = text_response.choices[0].message.content

                            # Generate audio response
                            with st.spinner("🎵 Generating audio response..."):
                                audio_response = client.audio.speech.create(
                                    model="tts-1",
                                    voice=voice,
                                    input=reply_text
                                )
                                
                                # Save audio response to temporary file
                                audio_temp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                                audio_response.stream_to_file(audio_temp.name)
                                
                                # Play audio
                                st.audio(audio_temp.name, format="audio/mp3")

                            st.markdown(f"**🤖 GPT-4o:** {reply_text}")
                            st.session_state['conversation'].append({"role": "assistant", "content": reply_text})
                            
                            # Clean up temporary files
                            os.unlink(audio_temp.name)
                            
                        except Exception as e:
                            st.error(f"Error generating GPT-4o response: {str(e)}")
                    else:
                        st.warning("No speech detected. Please try again.")
                
                # Clean up temporary audio file
                os.unlink(audio_path)
                
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")

# Alternative: File upload method
st.header("📁 Alternative: Upload Audio File")
uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'm4a', 'flac'])

if uploaded_file is not None and st.session_state['openai_api_key']:
    if st.button("🎯 Transcribe Uploaded File"):
        with st.spinner("🔎 Transcribing uploaded audio..."):
            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {
                "Authorization": f"Bearer {st.session_state['openai_api_key']}"
            }
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }
            data = {
                "model": "whisper-1",
                "response_format": "json"
            }
            response = requests.post(url, headers=headers, files=files, data=data)

        if not response.ok:
            st.error(f"Transcription failed. {response.status_code}: {response.text}")
        else:
            transcript = response.json().get("text", "")
            if transcript.strip():
                st.success("Transcription complete.")
                st.markdown(f"**🧑‍💼 You:** {transcript}")
                st.session_state['conversation'].append({"role": "user", "content": transcript})

                # Generate GPT-4o response
                try:
                    client = openai.OpenAI(api_key=st.session_state['openai_api_key'])
                    messages = st.session_state['conversation'].copy()

                    with st.spinner("🤖 GPT-4o is responding..."):
                        text_response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=messages,
                            temperature=0.7
                        )
                        reply_text = text_response.choices[0].message.content

                        # Generate audio response
                        audio_response = client.audio.speech.create(
                            model="tts-1",
                            voice=voice,
                            input=reply_text
                        )
                        
                        audio_temp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                        audio_response.stream_to_file(audio_temp.name)
                        st.audio(audio_temp.name, format="audio/mp3")

                    st.markdown(f"**🤖 GPT-4o:** {reply_text}")
                    st.session_state['conversation'].append({"role": "assistant", "content": reply_text})
                    
                    os.unlink(audio_temp.name)
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

# Conversation log
st.subheader("📝 Conversation Log")
for msg in st.session_state['conversation']:
    role = "🧑‍💼 You" if msg['role'] == 'user' else "🤖 GPT-4o"
    st.markdown(f"**{role}:** {msg['content']}")

# Download logs
if st.session_state['conversation'] and st.button("📥 Download Log as JSON"):
    filename = f"gpt4o_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    st.download_button(
        "📥 Download JSON", 
        json.dumps(st.session_state['conversation'], indent=2), 
        filename, 
        mime="application/json"
    )

if st.session_state['conversation'] and st.button("📄 Download Log as Text"):
    log_text = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state['conversation']])
    st.download_button(
        "📄 Download Text", 
        log_text, 
        file_name=f"conversation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 
        mime="text/plain"
    )

# Clear conversation
if st.session_state['conversation'] and st.button("🗑️ Clear Conversation"):
    st.session_state['conversation'] = []
    st.success("Conversation cleared!")
    st.rerun()

# Instructions
st.sidebar.header("ℹ️ How to Use")
st.sidebar.markdown("""
1. **Enter your OpenAI API key** in the text field
2. **Select a voice** for GPT-4o responses  
3. **Method 1: Live Recording**
   - Click "Start" to begin recording
   - Speak into your microphone
   - Click "Stop" then "Transcribe and Send"
4. **Method 2: File Upload**
   - Upload an audio file (WAV, MP3, M4A, FLAC)
   - Click "Transcribe Uploaded File"
5. **Listen** to GPT-4o's audio response
6. **Download** conversation logs if needed

**Note:** Live recording works best in Chrome/Edge browsers.
""")

st.sidebar.header("🔧 Troubleshooting")
st.sidebar.markdown("""
- **No audio detected:** Check microphone permissions
- **Transcription failed:** Verify API key is correct
- **Poor audio quality:** Try uploading a file instead
- **Browser issues:** Use Chrome or Edge for best results
""")
