import streamlit as st
import openai
import anthropic
import base64
import os
import time
import json
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, VideoProcessorBase, WebRtcMode
import av
from PIL import Image
import tempfile
import asyncio
import requests
import numpy as np
import threading
from collections import deque

# Page settings
st.set_page_config(page_title="Multi-AI Voice Chat", layout="wide")
st.title("🎙️ Universal Voice Chat - OpenAI GPT-4o & Anthropic Claude")

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []
if 'selected_ai' not in st.session_state:
    st.session_state['selected_ai'] = "OpenAI"
if 'continuous_mode' not in st.session_state:
    st.session_state['continuous_mode'] = False
if 'is_processing' not in st.session_state:
    st.session_state['is_processing'] = False

# AI Provider Selection
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
        st.info("🔹 **Anthropic Claude 4 Sonnet**\n- Latest Claude Sonnet 4 model\n- Advanced reasoning & performance\n- **Vision capabilities with webcam**\n- Text-to-speech via OpenAI")

# API Keys Section
st.header("🔑 API Configuration")
col1, col2 = st.columns(2)

with col1:
    openai_key = st.text_input(
        "OpenAI API Key", 
        type="password",
        value=st.session_state.get('openai_api_key', ''),
        help="Required for voice processing and OpenAI models"
    )
    st.session_state['openai_api_key'] = openai_key

with col2:
    anthropic_key = st.text_input(
        "Anthropic API Key", 
        type="password",
        value=st.session_state.get('anthropic_api_key', ''),
        help="Required for Claude models"
    )
    st.session_state['anthropic_api_key'] = anthropic_key

# Model-specific settings
if ai_provider == "OpenAI":
    st.subheader("🎤 OpenAI Settings")
    col1, col2 = st.columns(2)
    with col1:
        voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        voice = st.selectbox("Select Voice", voice_options, index=0)
    with col2:
        gpt_model = st.selectbox("GPT Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], index=0)
else:
    st.subheader("🧠 Anthropic Settings")
    col1, col2 = st.columns(2)
    with col1:
        claude_model = st.selectbox(
            "Claude Model", 
            ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"],
            index=0
        )
        # Voice for TTS (using OpenAI for audio generation)
        voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        voice = st.selectbox("TTS Voice (OpenAI)", voice_options, index=0)
    with col2:
        st.info("💡 Claude responses will be converted to speech using OpenAI's TTS")
        enable_webcam = st.checkbox("📹 Enable Webcam for Claude Vision", value=True)
        continuous_mode = st.checkbox("🔄 Continuous Conversation Mode", value=False, 
                                    help="Talk naturally without pressing buttons - like chatting with a human!")
        if enable_webcam:
            st.success("🎥 Webcam enabled - Claude can see you!")
        if continuous_mode:
            st.success("🗣️ Continuous mode - Just talk naturally!")

# Audio and Video Processing Classes with Voice Activity Detection
class ContinuousAudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.audio_buffer = deque(maxlen=48000 * 10)  # 10 seconds at 48kHz
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_counter = 0
        self.silence_threshold = 48000 * 2  # 2 seconds of silence
        self.volume_threshold = 0.01  # Minimum volume to consider as speech
        self.min_speech_length = 48000 * 1  # Minimum 1 second of speech
        
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio_array = frame.to_ndarray()
        
        # Add to circular buffer
        for sample in audio_array.flatten():
            self.audio_buffer.append(sample)
        
        # Simple voice activity detection
        volume = np.sqrt(np.mean(audio_array**2))
        
        if volume > self.volume_threshold:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_buffer = list(self.audio_buffer)  # Start with recent context
            else:
                self.speech_buffer.extend(audio_array.flatten())
            self.silence_counter = 0
        else:
            if self.is_speaking:
                self.silence_counter += len(audio_array.flatten())
                self.speech_buffer.extend(audio_array.flatten())
                
                # End of speech detected
                if self.silence_counter > self.silence_threshold:
                    if len(self.speech_buffer) > self.min_speech_length:
                        # Speech segment ready for processing
                        st.session_state['pending_speech'] = np.array(self.speech_buffer)
                    self.is_speaking = False
                    self.speech_buffer = []
                    self.silence_counter = 0
        
        return frame
    
    def get_audio_data(self):
        if self.speech_buffer and len(self.speech_buffer) > self.min_speech_length:
            return np.array(self.speech_buffer)
        return None

class AudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.audio_frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio_array = frame.to_ndarray()
        self.audio_frames.append(audio_array)
        return frame

    def get_audio_data(self):
        if not self.audio_frames:
            return None
        audio_data = np.concatenate(self.audio_frames, axis=0)
        return audio_data

class VideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.latest_frame = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # Convert BGR to RGB for PIL
        img_rgb = img[:, :, ::-1]
        self.latest_frame = Image.fromarray(img_rgb)
        return frame

    def get_latest_image(self):
        return self.latest_frame

def save_audio_as_wav(audio_data, sample_rate=48000, channels=1):
    """Save numpy array as WAV file"""
    import wave
    
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    
    with wave.open(temp_file.name, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)
        
        wav_file.writeframes(audio_data.tobytes())
    
    return temp_file.name

def transcribe_audio(audio_bytes, openai_key):
    """Transcribe audio using OpenAI Whisper"""
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {openai_key}"}
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {"model": "whisper-1", "response_format": "json"}
    
    response = requests.post(url, headers=headers, files=files, data=data)
    
    if response.ok:
        return response.json().get("text", "")
    else:
        raise Exception(f"Transcription failed: {response.status_code} - {response.text}")

def generate_tts(text, voice, openai_key):
    """Generate text-to-speech using OpenAI"""
    client = openai.OpenAI(api_key=openai_key)
    
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text
    )
    
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    response.stream_to_file(temp_file.name)
    return temp_file.name

def get_openai_response(messages, model, openai_key):
    """Get response from OpenAI"""
    client = openai.OpenAI(api_key=openai_key)
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7
    )
    
    return response.choices[0].message.content

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    import io
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def get_anthropic_response_with_vision(messages, model, anthropic_key, image=None):
    """Get response from Anthropic Claude with optional vision"""
    client = anthropic.Anthropic(api_key=anthropic_key)
    
    # Convert OpenAI format to Anthropic format
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
    
    # Add vision to the latest user message if image provided
    if image and anthropic_messages:
        latest_msg = anthropic_messages[-1]
        if latest_msg["role"] == "user":
            image_b64 = encode_image_to_base64(image)
            latest_msg["content"] = [
                {
                    "type": "text",
                    "text": latest_msg["content"]
                },
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

# Main Voice Interface
st.header("🎧 Voice Interaction")

# Initialize session state for pending speech
if 'pending_speech' not in st.session_state:
    st.session_state['pending_speech'] = None

# Webcam for Anthropic models
video_ctx = None
if ai_provider == "Anthropic" and 'enable_webcam' in locals() and enable_webcam:
    st.subheader("📹 Webcam Feed for Claude Vision")
    
    video_ctx = webrtc_streamer(
        key="video_stream",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )
    
    if video_ctx.video_processor:
        st.success("📷 Webcam active - Claude can see you!")
    else:
        st.info("📷 Click 'START' to activate webcam for Claude vision")

# Continuous conversation mode for Anthropic
if ai_provider == "Anthropic" and 'continuous_mode' in locals() and continuous_mode:
    st.subheader("🔄 Continuous Conversation Mode")
    st.info("💬 **Just start talking!** Claude will listen, see you, and respond automatically.")
    
    # Status indicator
    status_placeholder = st.empty()
    
    if not st.session_state['is_processing']:
        status_placeholder.success("🎤 Listening... speak naturally!")
    else:
        status_placeholder.warning("🤖 Processing your message...")
    
    # Continuous audio stream
    continuous_ctx = webrtc_streamer(
        key="continuous_audio",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        audio_processor_factory=ContinuousAudioProcessor,
        async_processing=True,
    )
    
    # Process pending speech automatically
    if (st.session_state['pending_speech'] is not None and 
        not st.session_state['is_processing'] and
        st.session_state['openai_api_key'] and
        st.session_state['anthropic_api_key']):
        
        st.session_state['is_processing'] = True
        status_placeholder.warning("🤖 Processing your message...")
        
        # Process the speech
        process_continuous_speech(
            st.session_state['pending_speech'], 
            ai_provider, 
            video_ctx,
            status_placeholder
        )
        
        # Clear the pending speech
        st.session_state['pending_speech'] = None
        st.session_state['is_processing'] = False
        
        # Refresh the page to continue listening
        st.rerun()

def process_continuous_speech(audio_data, ai_provider, video_ctx, status_placeholder):
    """Process speech in continuous mode"""
    try:
        status_placeholder.info("🔎 Transcribing your speech...")
        
        # Save and transcribe audio
        audio_path = save_audio_as_wav(audio_data)
        
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        
        transcript = transcribe_audio(audio_bytes, st.session_state['openai_api_key'])
        
        if not transcript.strip():
            status_placeholder.success("🎤 Listening... speak naturally!")
            return
            
        # Show what was heard
        st.success(f"👂 Heard: {transcript}")
        st.session_state['conversation'].append({"role": "user", "content": transcript})
        
        # Get webcam image
        current_image = None
        if video_ctx and video_ctx.video_processor:
            current_image = video_ctx.video_processor.get_latest_image()
        
        status_placeholder.info("🧠 Claude is thinking...")
        
        # Get Claude response
        messages = st.session_state['conversation'].copy()
        response_text = get_anthropic_response_with_vision(
            messages, 
            claude_model, 
            st.session_state['anthropic_api_key'],
            current_image
        )
        
        status_placeholder.info("🎵 Generating speech response...")
        
        # Generate and play audio response
        audio_file = generate_tts(
            response_text, 
            voice, 
            st.session_state['openai_api_key']
        )
        
        # Display response
        st.markdown(f"**🤖 Claude:** {response_text}")
        st.audio(audio_file, format="audio/mp3")
        
        st.session_state['conversation'].append({"role": "assistant", "content": response_text})
        
        # Cleanup
        os.unlink(audio_path)
        os.unlink(audio_file)
        
        status_placeholder.success("🎤 Listening... speak naturally!")
        
    except Exception as e:
        st.error(f"Error in continuous mode: {str(e)}")
        status_placeholder.success("🎤 Listening... speak naturally!")

# Manual conversation mode (original functionality)
else:
    st.subheader("🎙️ Manual Voice Recording")
    st.info("💡 Click buttons to record and process your voice messages")

# Live Recording (for manual mode or non-Anthropic)
if not (ai_provider == "Anthropic" and 'continuous_mode' in locals() and continuous_mode):
    ctx = webrtc_streamer(
        key="audio_recorder",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=256,
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        audio_processor_factory=AudioProcessor,
        async_processing=True,
    )

    if ctx.audio_processor:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗣️ Process Recording", use_container_width=True):
                process_audio_recording(ctx.audio_processor, ai_provider, video_ctx)
        with col2:
            if st.button("🗑️ Clear Audio Buffer", use_container_width=True):
                ctx.audio_processor.audio_frames = []
                st.success("Audio buffer cleared!")

def process_audio_recording(audio_processor, ai_provider, video_ctx=None):
    """Process recorded audio with optional video"""
    if not st.session_state['openai_api_key']:
        st.error("OpenAI API key required for audio processing!")
        return
        
    if ai_provider == "Anthropic" and not st.session_state['anthropic_api_key']:
        st.error("Anthropic API key required for Claude!")
        return
    
    audio_data = audio_processor.get_audio_data()
    
    if audio_data is None or len(audio_data) == 0:
        st.warning("No audio data captured. Please record some audio first.")
        return
    
    # Get webcam image if available
    current_image = None
    if ai_provider == "Anthropic" and video_ctx and video_ctx.video_processor:
        current_image = video_ctx.video_processor.get_latest_image()
        if current_image:
            st.success("📸 Captured webcam image for Claude!")
            # Show preview of captured image
            st.image(current_image, caption="Image sent to Claude", width=300)
    
    try:
        # Save and transcribe audio
        audio_path = save_audio_as_wav(audio_data)
        
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        
        with st.spinner("🔎 Transcribing audio..."):
            transcript = transcribe_audio(audio_bytes, st.session_state['openai_api_key'])
        
        if not transcript.strip():
            st.warning("No speech detected. Please try again.")
            return
            
        st.success("✅ Transcription complete!")
        st.markdown(f"**🧑‍💼 You:** {transcript}")
        
        # Add to conversation
        st.session_state['conversation'].append({"role": "user", "content": transcript})
        
        # Get AI response
        messages = st.session_state['conversation'].copy()
        
        with st.spinner(f"🤖 {ai_provider} is thinking..."):
            if ai_provider == "OpenAI":
                response_text = get_openai_response(
                    messages, 
                    gpt_model, 
                    st.session_state['openai_api_key']
                )
            else:
                response_text = get_anthropic_response_with_vision(
                    messages, 
                    claude_model, 
                    st.session_state['anthropic_api_key'],
                    current_image
                )
        
        # Generate audio response
        with st.spinner("🎵 Generating audio response..."):
            audio_file = generate_tts(
                response_text, 
                voice, 
                st.session_state['openai_api_key']
            )
            st.audio(audio_file, format="audio/mp3")
        
        st.markdown(f"**🤖 {ai_provider}:** {response_text}")
        st.session_state['conversation'].append({"role": "assistant", "content": response_text})
        
        # Cleanup
        os.unlink(audio_path)
        os.unlink(audio_file)
        
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")

# File Upload Alternative (always available)
st.header("📁 Upload Audio File")
uploaded_file = st.file_uploader(
    "Choose an audio file", 
    type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
    help="Upload an audio file to transcribe and chat"
)

if uploaded_file is not None:
    if st.button("🎯 Process Uploaded Audio", use_container_width=True):
        process_uploaded_file(uploaded_file, ai_provider)

def process_uploaded_file(uploaded_file, ai_provider, video_ctx=None):
    """Process uploaded audio file with optional video"""
    if not st.session_state['openai_api_key']:
        st.error("OpenAI API key required for audio processing!")
        return
        
    if ai_provider == "Anthropic" and not st.session_state['anthropic_api_key']:
        st.error("Anthropic API key required for Claude!")
        return
    
    # Get webcam image if available
    current_image = None
    if ai_provider == "Anthropic" and video_ctx and video_ctx.video_processor:
        current_image = video_ctx.video_processor.get_latest_image()
        if current_image:
            st.success("📸 Captured webcam image for Claude!")
            st.image(current_image, caption="Image sent to Claude", width=300)
    
    try:
        with st.spinner("🔎 Transcribing uploaded audio..."):
            transcript = transcribe_audio(
                uploaded_file.getvalue(), 
                st.session_state['openai_api_key']
            )
        
        if not transcript.strip():
            st.warning("No speech detected in the uploaded file.")
            return
            
        st.success("✅ Transcription complete!")
        st.markdown(f"**🧑‍💼 You:** {transcript}")
        
        # Add to conversation
        st.session_state['conversation'].append({"role": "user", "content": transcript})
        
        # Get AI response
        messages = st.session_state['conversation'].copy()
        
        with st.spinner(f"🤖 {ai_provider} is responding..."):
            if ai_provider == "OpenAI":
                response_text = get_openai_response(
                    messages, 
                    gpt_model, 
                    st.session_state['openai_api_key']
                )
            else:
                response_text = get_anthropic_response_with_vision(
                    messages, 
                    claude_model, 
                    st.session_state['anthropic_api_key'],
                    current_image
                )
        
        # Generate audio response
        with st.spinner("🎵 Generating audio response..."):
            audio_file = generate_tts(
                response_text, 
                voice, 
                st.session_state['openai_api_key']
            )
            st.audio(audio_file, format="audio/mp3")
        
        st.markdown(f"**🤖 {ai_provider}:** {response_text}")
        st.session_state['conversation'].append({"role": "assistant", "content": response_text})
        
        # Cleanup
        os.unlink(audio_file)
        
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")

# Text Input Alternative (always available)
st.header("⌨️ Text Input")
text_input = st.text_area(
    "Type your message:",
    height=100,
    placeholder="Type your message here and press Ctrl+Enter to send..."
)

if st.button("📝 Send Text Message", use_container_width=True) and text_input.strip():
    process_text_input(text_input.strip(), ai_provider, video_ctx)

def process_text_input(text, ai_provider, video_ctx=None):
    """Process text input with optional video"""
    if ai_provider == "OpenAI" and not st.session_state['openai_api_key']:
        st.error("OpenAI API key required!")
        return
        
    if ai_provider == "Anthropic" and not st.session_state['anthropic_api_key']:
        st.error("Anthropic API key required for Claude!")
        return
    
    # Get webcam image if available
    current_image = None
    if ai_provider == "Anthropic" and video_ctx and video_ctx.video_processor:
        current_image = video_ctx.video_processor.get_latest_image()
        if current_image:
            st.success("📸 Captured webcam image for Claude!")
            st.image(current_image, caption="Image sent to Claude", width=300)
    
    try:
        st.markdown(f"**🧑‍💼 You:** {text}")
        st.session_state['conversation'].append({"role": "user", "content": text})
        
        # Get AI response
        messages = st.session_state['conversation'].copy()
        
        with st.spinner(f"🤖 {ai_provider} is responding..."):
            if ai_provider == "OpenAI":
                response_text = get_openai_response(
                    messages, 
                    gpt_model, 
                    st.session_state['openai_api_key']
                )
            else:
                response_text = get_anthropic_response_with_vision(
                    messages, 
                    claude_model, 
                    st.session_state['anthropic_api_key'],
                    current_image
                )
        
        # Generate audio response
        if st.session_state['openai_api_key']:
            with st.spinner("🎵 Generating audio response..."):
                audio_file = generate_tts(
                    response_text, 
                    voice, 
                    st.session_state['openai_api_key']
                )
                st.audio(audio_file, format="audio/mp3")
                os.unlink(audio_file)
        
        st.markdown(f"**🤖 {ai_provider}:** {response_text}")
        st.session_state['conversation'].append({"role": "assistant", "content": response_text})
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Conversation Display
if st.session_state['conversation']:
    st.header("💬 Conversation History")
    
    for i, msg in enumerate(st.session_state['conversation']):
        role = "🧑‍💼 You" if msg['role'] == 'user' else "🤖 AI Assistant"
        
        with st.container():
            if msg['role'] == 'user':
                st.markdown(f"**{role}:** {msg['content']}")
            else:
                st.markdown(f"**{role}:** {msg['content']}")
        st.divider()

# Conversation Management
if st.session_state['conversation']:
    st.header("📊 Conversation Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state['conversation'] = []
            st.success("Conversation cleared!")
            st.rerun()
    
    with col2:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_data = json.dumps(st.session_state['conversation'], indent=2)
        st.download_button(
            "📥 Download JSON",
            json_data,
            f"conversation_{timestamp}.json",
            "application/json",
            use_container_width=True
        )
    
    with col3:
        text_data = "\n\n".join([
            f"{msg['role'].title()}: {msg['content']}" 
            for msg in st.session_state['conversation']
        ])
        st.download_button(
            "📄 Download Text",
            text_data,
            f"conversation_{timestamp}.txt",
            "text/plain",
            use_container_width=True
        )

# Sidebar Information
st.sidebar.title("ℹ️ How to Use")
st.sidebar.markdown("""
### 🚀 Quick Start
1. **Choose AI Provider** (OpenAI or Anthropic)
2. **Enter API Keys** for your selected provider
3. **Select Model & Voice** settings
4. **Start Chatting** via voice, file, or text

### 🎙️ Conversation Modes
- **🔄 Continuous Mode**: Natural conversation with Claude (Anthropic only)
- **🎙️ Manual Recording**: Click to record and process (All models)
- **📁 File Upload**: Upload audio files (WAV, MP3, etc.)
- **⌨️ Text Input**: Type messages for text-only chat
- **📹 Webcam Vision**: Enable camera for Claude to see you

### 🤖 Natural Conversation Features
- **Voice Activity Detection**: Automatically detects when you start/stop talking
- **Seamless Flow**: No button pressing - just talk like with a human
- **Visual Context**: Claude sees you through webcam while you talk
- **Instant Responses**: Automatic transcription and audio replies

### 🤖 AI Models
- **OpenAI**: GPT-4o, GPT-4-turbo, GPT-3.5-turbo
- **Anthropic**: Claude-4-Sonnet, Claude-4-Opus, Claude-3.5-Sonnet

### 🔊 Audio Features
- Speech-to-text via OpenAI Whisper
- Text-to-speech with multiple voice options
- Full conversation audio playback
""")

st.sidebar.header("🔧 Troubleshooting")
st.sidebar.markdown("""
- **No audio detected**: Check microphone permissions
- **API errors**: Verify your API keys are correct
- **Recording issues**: Try file upload instead
- **Browser compatibility**: Use Chrome/Edge for best results
""")

st.sidebar.header("🔒 Privacy & Security")
st.sidebar.markdown("""
- API keys are stored in session only
- Audio files are processed temporarily
- Conversations are not permanently saved
- Clear conversation to remove chat history
""")
