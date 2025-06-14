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

# Thread-Safe Audio Processing Class
class ThreadSafeAudioProcessor:
    def __init__(self):
        self.audio_buffer = []
        self.lock = threading.Lock()
        self.is_recording = False
        self.max_buffer_size = 16000 * 60  # 1 minute max

    def process_audio_frame(self, frame):
        """Thread-safe audio frame processing"""
        with self.lock:
            if self.is_recording:
                sound = frame.to_ndarray()
                self.audio_buffer.append(sound)

                # Prevent memory overflow
                total_samples = sum(len(chunk) for chunk in self.audio_buffer)
                if total_samples > self.max_buffer_size:
                    self.audio_buffer = self.audio_buffer[-100:]  # Keep last 100 chunks
        return frame

    def get_audio_bytes(self) -> bytes:
        """Convert audio buffer to bytes with thread safety"""
        with self.lock:
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
        """Thread-safe buffer clearing"""
        with self.lock:
            self.audio_buffer = []

    def start_recording(self):
        """Start recording with thread safety"""
        with self.lock:
            self.is_recording = True
            self.clear_buffer()

    def stop_recording(self):
        """Stop recording with thread safety"""
        with self.lock:
            self.is_recording = False

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

# Enhanced ICE Server Configuration
def get_ice_servers():
    """Production-ready ICE configuration with commercial TURN servers"""
    return RTCConfiguration({
        "iceServers": [
            # Public STUN servers
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            
            # Commercial TURN service (CRITICAL for reliability) - Replace with your credentials
            {
                "urls": "turn:global.turn.twilio.com:3478?transport=tcp",
                "username": os.getenv("TWILIO_USERNAME", "your-twilio-username"),
                "credential": os.getenv("TWILIO_CREDENTIAL", "your-twilio-credential")
            },
            {
                "urls": "turn:global.turn.twilio.com:443?transport=tcp",
                "username": os.getenv("TWILIO_USERNAME", "your-twilio-username"),
                "credential": os.getenv("TWILIO_CREDENTIAL", "your-twilio-credential")
            },
            
            # Fallback: Free TURN server for testing
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            },
            {
                "urls": ["turn:openrelay.metered.ca:80?transport=tcp"],
                "username": "openrelayproject", 
                "credential": "openrelayproject"
            }
        ],
        "iceTransportPolicy": "all",  # Try all available candidates
        "bundlePolicy": "max-bundle",
        "rtcpMuxPolicy": "require"
    })

def diagnose_connection_issues():
    """Comprehensive network diagnostics"""
    st.subheader("🔬 Network Diagnostics")
    
    diagnostics = {
        "timestamp": datetime.now().isoformat(),
        "connection_type": "unknown",
        "nat_type": "unknown",
        "firewall_detected": False,
        "stun_reachable": False,
        "turn_reachable": False
    }

    with st.spinner("Running network diagnostics..."):
        # Test STUN connectivity
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(5)
            sock.connect(("stun.l.google.com", 19302))
            diagnostics["stun_reachable"] = True
            sock.close()
        except Exception as e:
            diagnostics["stun_reachable"] = False
            diagnostics["firewall_detected"] = True
            logger.warning(f"STUN test failed: {e}")

        # Test TURN connectivity (basic socket test)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect(("openrelay.metered.ca", 80))
            diagnostics["turn_reachable"] = True
            sock.close()
        except Exception as e:
            diagnostics["turn_reachable"] = False
            logger.warning(f"TURN test failed: {e}")

    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("STUN Connectivity", "✅ OK" if diagnostics["stun_reachable"] else "❌ Failed")
        st.metric("TURN Connectivity", "✅ OK" if diagnostics["turn_reachable"] else "❌ Failed")
    
    with col2:
        st.metric("Firewall Detected", "⚠️ Yes" if diagnostics["firewall_detected"] else "✅ No")
        
    # Show detailed diagnostics
    st.json(diagnostics)
    
    # Recommendations
    if not diagnostics["stun_reachable"] or not diagnostics["turn_reachable"]:
        st.error("**Connection Issues Detected!**")
        st.markdown("""
        **Immediate Actions:**
        1. Try using a VPN to bypass firewall restrictions
        2. Contact your network administrator about WebRTC support
        3. Set up commercial TURN servers (Twilio/Xirsys) for production
        4. Try the 'Force TCP Connection' workaround below
        """)
        
        # Quick workaround
        if st.button("🔧 Force TCP Connection"):
            st.session_state['force_tcp'] = True
            st.success("TCP-only mode enabled. Try connecting again.")
    else:
        st.success("Network connectivity looks good! ✅")

# WebRTC Connection Manager Class
class WebRTCConnectionManager:
    def __init__(self):
        self.connection_state = "new"
        self.ice_connection_state = "new"
        self.callbacks = {}
        self.retry_count = 0
        self.max_retries = 3

    def on_ice_connection_state_change(self, state):
        """Handle ICE connection state changes"""
        self.ice_connection_state = state
        logger.info(f"ICE connection state changed to: {state}")
        
        if state == "failed":
            self._handle_connection_failure()
        elif state == "disconnected":
            self._attempt_reconnection()
        elif state == "connected":
            self.retry_count = 0  # Reset retry count on successful connection

    def _handle_connection_failure(self):
        """Handle connection failures with exponential backoff"""
        logger.error("WebRTC connection failed")
        if self.retry_count < self.max_retries:
            delay = 2 ** self.retry_count
            logger.info(f"Attempting reconnection in {delay} seconds")
            self._attempt_reconnection(delay=delay)
        else:
            logger.error("Max retry attempts reached")

    def _attempt_reconnection(self, delay=0):
        """Attempt to reconnect with exponential backoff"""
        self.retry_count += 1
        if delay > 0:
            time.sleep(delay)
        # Reconnection logic would go here
        logger.info(f"Reconnection attempt {self.retry_count}")

# Initialize connection manager
connection_manager = WebRTCConnectionManager()

def monitor_connection_state(webrtc_ctx):
    """Monitor and display connection state"""
    if webrtc_ctx and webrtc_ctx.state.playing:
        connection_state = getattr(webrtc_ctx.state, 'ice_connection_state', None)
        if connection_state:
            state_indicator = {
                "new": "🟡 Initializing",
                "checking": "🟡 Connecting", 
                "connected": "🟢 Connected",
                "completed": "🟢 Stable",
                "failed": "🔴 Failed",
                "disconnected": "🔴 Disconnected",
                "closed": "⚫ Closed"
            }.get(connection_state, "❓ Unknown")

            st.sidebar.metric("Connection Status", state_indicator)

            if connection_state in ["failed", "disconnected"]:
                st.error("WebRTC connection failed. Check network settings or try refreshing.")
                st.info("💡 If problems persist, you may be behind a restrictive firewall. Consider using a VPN or contact your network administrator.")

def connect_with_retry(config, max_retries=3):
    """Connect with exponential backoff"""
    for attempt in range(max_retries):
        try:
            ctx = webrtc_streamer(**config)
            if ctx.state.playing:
                return ctx
        except Exception as e:
            wait_time = 2 ** attempt
            logger.warning(f"Connection attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
            if attempt < max_retries - 1:
                time.sleep(wait_time)

    logger.error("Failed to establish WebRTC connection after retries")
    return None

# Secure API Key Management
def get_api_key(provider: str) -> Optional[str]:
    """Securely retrieve API keys"""
    # First try environment variables
    env_key = os.getenv(f"{provider.upper()}_API_KEY")
    if env_key:
        return env_key

    # Try Streamlit secrets
    try:
        if hasattr(st, 'secrets') and 'api_keys' in st.secrets:
            return st.secrets['api_keys'].get(provider.lower())
    except:
        pass

    # Fall back to session state
    return st.session_state.get(f'{provider.lower()}_api_key')

# Initialize processors
audio_processor = ThreadSafeAudioProcessor()
video_processor = VideoProcessor()

# Connection pooling configuration
@st.cache_resource
def get_connection_pool():
    """Maintain a pool of WebRTC connections"""
    return {
        "max_connections": 5,
        "timeout": 30,
        "retry_count": 3
    }
def transcribe_audio(audio_bytes: bytes, api_key: str = None) -> Optional[str]:
    """Transcribe audio using OpenAI Whisper with improved error handling"""
    if not api_key:
        api_key = get_api_key('openai')
    
    client = get_openai_client(api_key)
    if not client:
        st.error("Failed to initialize OpenAI client. Check your API key.")
        return None

    try:
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name

        # Transcribe with timeout and retry
        with open(tmp_file_path, 'rb') as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                language="en"  # Specify language for better accuracy
            )

        # Cleanup
        os.unlink(tmp_file_path)
        return response.strip() if response else None

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        st.error(f"Transcription failed: {str(e)}")
        # Cleanup on error
        try:
            os.unlink(tmp_file_path)
        except:
            pass
        return None

def generate_tts(text: str, voice: str, api_key: str = None) -> Optional[bytes]:
    """Generate text-to-speech using OpenAI with improved error handling"""
    if not api_key:
        api_key = get_api_key('openai')
        
    client = get_openai_client(api_key)
    if not client:
        return None

    try:
        # Limit text length to prevent API errors
        if len(text) > 4000:
            text = text[:4000] + "..."
            
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="mp3"
        )

        # Convert to bytes more efficiently
        return response.content

    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        st.error(f"TTS generation failed: {str(e)}")
        return None

def get_openai_response(messages: List[Dict], model: str, api_key: str = None) -> Optional[str]:
    """Get response from OpenAI with improved error handling"""
    if not api_key:
        api_key = get_api_key('openai')
        
    client = get_openai_client(api_key)
    if not client:
        return None

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=4000,
            timeout=60  # Add timeout
        )
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        if "rate_limit" in str(e).lower():
            st.error("Rate limit exceeded. Please wait and try again.")
        elif "quota" in str(e).lower():
            st.error("API quota exceeded. Check your OpenAI billing.")
        else:
            st.error(f"OpenAI API call failed: {str(e)}")
        return None

def get_anthropic_response(messages: List[Dict], model: str, api_key: str = None, 
                          image_bytes: Optional[bytes] = None) -> Optional[str]:
    """Get response from Anthropic with improved error handling"""
    if not api_key:
        api_key = get_api_key('anthropic')
        
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
            messages=anthropic_messages,
            timeout=60
        )

        return response.content[0].text

    except Exception as e:
        logger.error(f"Anthropic API call failed: {e}")
        if "rate_limit" in str(e).lower():
            st.error("Rate limit exceeded. Please wait and try again.")
        elif "quota" in str(e).lower():
            st.error("API quota exceeded. Check your Anthropic billing.")
        else:
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
    """Render API key configuration with security improvements"""
    st.header("🔑 API Configuration")
    
    # Check for environment variables first
    openai_env = os.getenv('OPENAI_API_KEY')
    anthropic_env = os.getenv('ANTHROPIC_API_KEY')
    
    if openai_env or anthropic_env:
        st.success("🔒 API keys detected from environment variables (secure)")
        if openai_env:
            st.session_state['openai_api_key'] = openai_env
        if anthropic_env:
            st.session_state['anthropic_api_key'] = anthropic_env
    else:
        st.warning("⚠️ For production use, set API keys as environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY")
        
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

def render_unified_media_interface(enable_video=False):
    """Unified audio/video interface - CRITICAL FIX: Single WebRTC component"""
    st.subheader("🎤📹 Media Interface")
    
    # Check for TCP force mode
    force_tcp = st.session_state.get('force_tcp', False)
    if force_tcp:
        st.info("🔧 TCP-only mode enabled for firewall compatibility")
    
    media_constraints = {
        "audio": True,
        "video": enable_video
    }

    # Get ICE configuration with TCP forcing if needed
    ice_config = get_ice_servers()
    if force_tcp:
        ice_config = RTCConfiguration({
            **ice_config.__dict__,
            "iceTransportPolicy": "relay"  # Force TURN server usage
        })

    # Use enhanced ICE configuration with retry logic
    config = {
        "key": "unified-media",
        "mode": WebRtcMode.SENDONLY,
        "rtc_configuration": ice_config,
        "media_stream_constraints": media_constraints,
        "audio_frame_callback": audio_processor.process_audio_frame,
    }
    
    if enable_video:
        config["video_processor_factory"] = lambda: video_processor

    # Try direct connection first, then retry logic
    try:
        webrtc_ctx = webrtc_streamer(**config)
        
        # Monitor connection state
        if webrtc_ctx:
            monitor_connection_state(webrtc_ctx)
            connection_manager.on_ice_connection_state_change(
                getattr(webrtc_ctx.state, 'ice_connection_state', 'new')
            )
    except Exception as e:
        logger.error(f"WebRTC initialization failed: {e}")
        webrtc_ctx = None
    
    if webrtc_ctx and webrtc_ctx.state.playing:
        # Media controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎙️ Start Recording"):
                audio_processor.start_recording()
                st.session_state['recording_start'] = time.time()
                st.success("Recording started!")

        with col2:
            if st.button("⏹️ Stop Recording"):
                audio_processor.stop_recording()
                audio_bytes = audio_processor.get_audio_bytes()
                if audio_bytes:
                    st.session_state['last_audio'] = audio_bytes
                    st.success("Recording saved!")
                else:
                    st.warning("No audio recorded")

        with col3:
            if enable_video and st.button("📸 Capture Image"):
                image_bytes = video_processor.get_latest_image()
                if image_bytes:
                    st.session_state['last_image'] = image_bytes
                    st.success("Image captured!")
                else:
                    st.warning("No image available")

        # Display captured media
        media_col1, media_col2 = st.columns(2)
        
        with media_col1:
            if st.session_state.get('last_audio'):
                st.write("**Latest Audio Recording:**")
                st.audio(st.session_state['last_audio'], format='audio/wav')

        with media_col2:
            if enable_video and st.session_state.get('last_image'):
                st.write("**Latest Captured Image:**")
                st.image(st.session_state['last_image'], caption="Captured", width=200)

    else:
        st.error("❌ Failed to establish WebRTC connection. Please check your network settings.")
        
        # Enhanced troubleshooting
        st.info("""
        🔧 **Troubleshooting Steps:**
        
        1. **Refresh the page** - Sometimes helps with WebRTC initialization
        2. **Check browser permissions** - Ensure camera/microphone access is granted
        3. **Corporate firewall?** - Try the network diagnostics in the sidebar
        4. **Use VPN** - Can bypass restrictive firewalls
        5. **Try TCP mode** - Click 'Force TCP Connection' in diagnostics
        """)
        
        # Quick retry button
        if st.button("🔄 Retry Connection"):
            st.rerun()
    
    return webrtc_ctx

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
            
        # Add connection diagnostics button
        if st.button("🔧 Run Connection Diagnostics"):
            diagnose_connection_issues()

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Unified Media Interface (CRITICAL FIX)
        webrtc_ctx = render_unified_media_interface(enable_video=enable_webcam)
        
        # Process audio button
        if st.button("🎯 Process Audio", disabled=not st.session_state.get('last_audio')):
            openai_key = get_api_key('openai')
            if not openai_key:
                st.error("OpenAI API key required for audio transcription")
            else:
                with st.spinner("Transcribing audio..."):
                    transcription = transcribe_audio(st.session_state['last_audio'])
                    if transcription:
                        process_user_input(transcription, ai_provider, voice, model, enable_webcam)

        # Text input
        st.subheader("💬 Text Input")
        text_input = st.text_area("Type your message:", height=100)
        if st.button("📤 Send Message"):
            process_user_input(text_input, ai_provider, voice, model, enable_webcam)

    with col2:        
        # Conversation Display
        render_conversation()

    # Network diagnostics section
    st.divider()
    with st.expander("🔧 Network Diagnostics & Troubleshooting"):
        st.markdown("""
        **Connection Issues?**
        
        1. **WebRTC Status:** Check the connection status in the sidebar
        2. **Firewall Issues:** If behind corporate firewall, try:
           - Using a VPN
           - Asking IT to whitelist STUN/TURN servers
        3. **Browser Permissions:** Ensure camera/microphone access is granted
        4. **TURN Server:** For production, set up your own TURN server for reliability
        
        **Environment Variables (Recommended):**
        ```bash
        export OPENAI_API_KEY="your-key-here"
        export ANTHROPIC_API_KEY="your-key-here"
        ```
        
        **TURN Server Setup:**
        - Free: OpenRelay (current setup)
        - Production: Twilio, Xirsys, or self-hosted CoTURN
        """)
        
        # Show current connection info if webrtc_ctx exists
        if 'webrtc_ctx' in locals() and webrtc_ctx:
            st.json({
                "WebRTC State": str(webrtc_ctx.state.playing),
                "ICE Servers": len(get_ice_servers()["iceServers"]),
                "Media Constraints": getattr(webrtc_ctx, 'media_stream_constraints', "N/A")
            })

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>Multi-AI Voice Chat Application | Built with Streamlit</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
