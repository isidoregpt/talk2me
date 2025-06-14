import streamlit as st
import openai
import anthropic
import base64
import os
import time
import json
from datetime import datetime
import tempfile
import requests
import numpy as np
from PIL import Image
import io
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import queue
import threading
from typing import Optional, Dict, List, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Multi-AI Voice Chat", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        font-weight: bold;
    }
    .recording-indicator {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .conversation-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .ai-message {
        background-color: #f1f8e9;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎙️ Universal Voice Chat - OpenAI GPT-4o & Anthropic Claude")

# Initialize session state with proper defaults
def init_session_state():
    defaults = {
        'conversation': [],
        'selected_ai': "OpenAI",
        'openai_api_key': '',
        'anthropic_api_key': '',
        'audio_queue': queue.Queue(),
        'image_queue': queue.Queue(),
        'processing': False,
        'last_audio': None,
        'last_image': None,
        'recording_start': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# API Client Management with Caching
@st.cache_resource
def get_openai_client(api_key: str) -> Optional[openai.OpenAI]:
    """Get cached OpenAI client"""
    try:
        return openai.OpenAI(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return None

@st.cache_resource
def get_anthropic_client(api_key: str) -> Optional[anthropic.Anthropic]:
    """Get cached Anthropic client"""
    try:
        return anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        logger.error(f"Failed to initialize Anthropic client: {e}")
        return None

# Audio Processing Class
class AudioProcessor:
    def __init__(self):
        self.audio_buffer = []
        self.is_recording = False

    def process_audio_frame(self, frame):
        """Process audio frames from WebRTC"""
        sound = frame.to_ndarray()
        self.audio_buffer.append(sound)
        return frame

    def get_audio_bytes(self) -> bytes:
        """Convert audio buffer to bytes"""
        if not self.audio_buffer:
            return b''

        audio_data = np.concatenate(self.audio_buffer)
        # Normalize and convert to 16-bit PCM
        audio_data = (audio_data * 32767).astype(np.int16)

        # Create WAV file in memory
        import wave
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_data.tobytes())

        buffer.seek(0)
        return buffer.read()

    def clear_buffer(self):
        """Clear audio buffer"""
        self.audio_buffer = []

# Video Processing Class
class VideoProcessor:
    def __init__(self):
        self.latest_frame = None

    def recv(self, frame):
        """Process video frames from WebRTC"""
        self.latest_frame = frame.to_image()
        return frame

    def get_latest_image(self) -> Optional[bytes]:
        """Get the latest captured frame as JPEG bytes"""
        if self.latest_frame:
            buffer = io.BytesIO()
            self.latest_frame.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            return buffer.read()
        return None

# Initialize processors
audio_processor = AudioProcessor()
video_processor = VideoProcessor()

# API Functions with Proper Error Handling
def transcribe_audio(audio_bytes: bytes, api_key: str) -> Optional[str]:
    """Transcribe audio using OpenAI Whisper"""
    client = get_openai_client(api_key)
    if not client:
        st.error("Failed to initialize OpenAI client")
        return None

    try:
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name

        # Transcribe
        with open(tmp_file_path, 'rb') as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )

        # Cleanup
        os.unlink(tmp_file_path)
        return response

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        st.error(f"Transcription failed: {str(e)}")
        return None

def generate_tts(text: str, voice: str, api_key: str) -> Optional[bytes]:
    """Generate text-to-speech using OpenAI"""
    client = get_openai_client(api_key)
    if not client:
        return None

    try:
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )

        # Convert to bytes
        audio_data = b''
        for chunk in response.iter_bytes():
            audio_data += chunk

        return audio_data

    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        st.error(f"TTS generation failed: {str(e)}")
        return None

def get_openai_response(messages: List[Dict], model: str, api_key: str) -> Optional[str]:
    """Get response from OpenAI"""
    client = get_openai_client(api_key)
    if not client:
        return None

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=4000
        )
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        st.error(f"OpenAI API call failed: {str(e)}")
        return None

def get_anthropic_response(messages: List[Dict], model: str, api_key: str, 
                          image_bytes: Optional[bytes] = None) -> Optional[str]:
    """Get response from Anthropic with optional vision"""
    client = get_anthropic_client(api_key)
    if not client:
        return None

    try:
        # Convert messages format
        anthropic_messages = []
        system_message = "You are a helpful AI assistant."

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Add image to latest message if provided
        if image_bytes and anthropic_messages:
            latest_msg = anthropic_messages[-1]
            if latest_msg["role"] == "user":
                image_b64 = base64.b64encode(image_bytes).decode()
                latest_msg["content"] = [
                    {"type": "text", "text": latest_msg["content"]},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64
                        }
                    }
                ]

        response = client.messages.create(
            model=model,
            max_tokens=4000,
            system=system_message,
            messages=anthropic_messages
        )

        return response.content[0].text

    except Exception as e:
        logger.error(f"Anthropic API call failed: {e}")
        st.error(f"Anthropic API call failed: {str(e)}")
        return None

# Main UI Components
def render_ai_selection():
    """Render AI provider selection"""
    st.header("🤖 Choose Your AI Assistant")
    col1, col2 = st.columns(2)

    with col1:
        ai_provider = st.radio(
            "Select AI Provider:",
            ["OpenAI", "Anthropic"],
            index=0 if st.session_state['selected_ai'] == "OpenAI" else 1,
            key="ai_provider_radio"
        )
        st.session_state['selected_ai'] = ai_provider

    with col2:
        if ai_provider == "OpenAI":
            st.info("🔹 **OpenAI GPT-4o**\n- Native voice input/output\n- Multiple voice options\n- Real-time audio generation")
        else:
            st.info("🔹 **Anthropic Claude**\n- Latest Claude models\n- Advanced reasoning\n- Vision capabilities")

    return ai_provider

def render_api_configuration():
    """Render API key configuration"""
    st.header("🔑 API Configuration")
    col1, col2 = st.columns(2)

    with col1:
        openai_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            value=st.session_state.get('openai_api_key', ''),
            help="Required for voice processing and OpenAI models"
        )
        if openai_key:
            st.session_state['openai_api_key'] = openai_key

    with col2:
        anthropic_key = st.text_input(
            "Anthropic API Key", 
            type="password",
            value=st.session_state.get('anthropic_api_key', ''),
            help="Required for Claude models"
        )
        if anthropic_key:
            st.session_state['anthropic_api_key'] = anthropic_key

def render_model_settings(ai_provider: str):
    """Render model-specific settings"""
    if ai_provider == "OpenAI":
        st.subheader("🎤 OpenAI Settings")
        col1, col2 = st.columns(2)
        with col1:
            voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
            voice = st.selectbox("Select Voice", voice_options, index=0)
        with col2:
            model = st.selectbox("GPT Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], index=0)
        return voice, model, None
    else:
        st.subheader("🧠 Anthropic Settings")
        col1, col2 = st.columns(2)
        with col1:
            model = st.selectbox(
                "Claude Model", 
                ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
                index=0
            )
            voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
            voice = st.selectbox("TTS Voice (OpenAI)", voice_options, index=0)
        with col2:
            enable_webcam = st.checkbox("📹 Enable Webcam for Vision", value=False)
        return voice, model, enable_webcam

def render_audio_interface():
    """Render audio recording interface"""
    st.subheader("🎤 Voice Recording")

    # Audio recorder using WebRTC
    webrtc_ctx = webrtc_streamer(
        key="audio-recorder",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"audio": True, "video": False},
        audio_frame_callback=audio_processor.process_audio_frame,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🎙️ Start Recording", disabled=webrtc_ctx.state.playing):
            audio_processor.clear_buffer()
            audio_processor.is_recording = True
            st.session_state['recording_start'] = time.time()

    with col2:
        if st.button("⏹️ Stop Recording", disabled=not webrtc_ctx.state.playing):
            audio_processor.is_recording = False
            audio_bytes = audio_processor.get_audio_bytes()
            if audio_bytes:
                st.session_state['last_audio'] = audio_bytes
                st.success("Recording saved!")

    with col3:
        if st.session_state.get('last_audio'):
            st.audio(st.session_state['last_audio'], format='audio/wav')

def render_video_interface():
    """Render video capture interface"""
    st.subheader("📹 Webcam for Vision")

    # Video capture using WebRTC
    webrtc_ctx = webrtc_streamer(
        key="video-capture",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"audio": False, "video": True},
        video_processor_factory=lambda: video_processor,
    )

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📸 Capture Image"):
            image_bytes = video_processor.get_latest_image()
            if image_bytes:
                st.session_state['last_image'] = image_bytes
                st.success("Image captured!")
            else:
                st.warning("No image available. Make sure webcam is active.")

    with col2:
        if st.session_state.get('last_image'):
            st.image(st.session_state['last_image'], caption="Captured Image", width=200)

def render_conversation():
    """Render conversation history"""
    st.subheader("💬 Conversation History")
    
    if st.session_state['conversation']:
        conversation_html = "<div class='conversation-container'>"
        for msg in st.session_state['conversation']:
            if msg['role'] == 'user':
                conversation_html += f"<div class='user-message'><strong>You:</strong> {msg['content']}</div>"
            else:
                conversation_html += f"<div class='ai-message'><strong>AI:</strong> {msg['content']}</div>"
        conversation_html += "</div>"
        st.markdown(conversation_html, unsafe_allow_html=True)
    else:
        st.info("No conversation yet. Start by recording your voice or typing a message!")

def process_user_input(user_input: str, ai_provider: str, voice: str, model: str, 
                      enable_webcam: bool = False):
    """Process user input and generate AI response"""
    if not user_input.strip():
        st.warning("Please provide some input!")
        return

    # Add user message to conversation
    st.session_state['conversation'].append({
        'role': 'user',
        'content': user_input,
        'timestamp': datetime.now().isoformat()
    })

    # Prepare messages for AI
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Provide clear, concise, and helpful responses."}
    ]
    
    # Add conversation history (last 10 messages to avoid token limits)
    recent_conversation = st.session_state['conversation'][-10:]
    for msg in recent_conversation:
        messages.append({
            "role": msg['role'],
            "content": msg['content']
        })

    # Generate AI response
    with st.spinner(f"Generating response from {ai_provider}..."):
        if ai_provider == "OpenAI":
            api_key = st.session_state.get('openai_api_key')
            if not api_key:
                st.error("Please provide OpenAI API key")
                return
            
            response = get_openai_response(messages, model, api_key)
        else:
            api_key = st.session_state.get('anthropic_api_key')
            if not api_key:
                st.error("Please provide Anthropic API key")
                return
            
            image_bytes = st.session_state.get('last_image') if enable_webcam else None
            response = get_anthropic_response(messages, model, api_key, image_bytes)

    if response:
        # Add AI response to conversation
        st.session_state['conversation'].append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })

        # Generate TTS if OpenAI key is available
        openai_key = st.session_state.get('openai_api_key')
        if openai_key:
            with st.spinner("Generating speech..."):
                tts_audio = generate_tts(response, voice, openai_key)
                if tts_audio:
                    st.audio(tts_audio, format='audio/mp3')

        st.rerun()

def main():
    """Main application function"""
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # AI Selection
        ai_provider = render_ai_selection()
        
        st.divider()
        
        # API Configuration
        render_api_configuration()
        
        st.divider()
        
        # Model Settings
        voice, model, enable_webcam = render_model_settings(ai_provider)
        
        st.divider()
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state['conversation'] = []
            st.rerun()

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Audio Interface
        render_audio_interface()
        
        # Process audio button
        if st.button("🎯 Process Audio", disabled=not st.session_state.get('last_audio')):
            if not st.session_state.get('openai_api_key'):
                st.error("OpenAI API key required for audio transcription")
            else:
                with st.spinner("Transcribing audio..."):
                    transcription = transcribe_audio(
                        st.session_state['last_audio'], 
                        st.session_state['openai_api_key']
                    )
                    if transcription:
                        process_user_input(transcription, ai_provider, voice, model, enable_webcam)

        # Text input
        st.subheader("💬 Text Input")
        text_input = st.text_area("Type your message:", height=100)
        if st.button("📤 Send Message"):
            process_user_input(text_input, ai_provider, voice, model, enable_webcam)

    with col2:
        # Video Interface (if enabled)
        if enable_webcam:
            render_video_interface()
        
        # Conversation Display
        render_conversation()

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>Multi-AI Voice Chat Application | Built with Streamlit</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
