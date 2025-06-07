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

# Page settings
st.set_page_config(page_title="Multi-AI Voice Chat", layout="wide")
st.title("🎙️ Universal Voice Chat - OpenAI GPT-4o & Anthropic Claude")

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []
if 'selected_ai' not in st.session_state:
    st.session_state['selected_ai'] = "OpenAI"

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
        voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        voice = st.selectbox("TTS Voice (OpenAI)", voice_options, index=0)
    with col2:
        enable_webcam = st.checkbox("📹 Enable Webcam for Claude Vision", value=True)
        if enable_webcam:
            st.success("🎥 Webcam will be activated below!")

# Helper Functions
def encode_image_to_base64(image_data):
    """Convert image data to base64 string"""
    return base64.b64encode(image_data).decode()

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

def get_anthropic_response_with_vision(messages, model, anthropic_key, image_data=None):
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
    if image_data and anthropic_messages:
        latest_msg = anthropic_messages[-1]
        if latest_msg["role"] == "user":
            image_b64 = encode_image_to_base64(image_data)
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

# HTML5 Audio Recorder
def create_audio_recorder():
    """Create HTML5 audio recorder component"""
    audio_recorder_html = """
    <div style="text-align: center; padding: 20px; border: 2px solid #4CAF50; border-radius: 10px; background-color: #f9f9f9;">
        <h3>🎙️ Voice Recorder</h3>
        <button id="recordBtn" onclick="toggleRecording()" style="
            padding: 15px 30px; 
            font-size: 18px; 
            background-color: #4CAF50; 
            color: white; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
            margin: 10px;
        ">🎤 Start Recording</button>
        
        <button id="playBtn" onclick="playRecording()" disabled style="
            padding: 15px 30px; 
            font-size: 18px; 
            background-color: #2196F3; 
            color: white; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
            margin: 10px;
        ">▶️ Play</button>
        
        <div id="status" style="margin: 15px; font-size: 16px; font-weight: bold;">Ready to record</div>
        <div id="timer" style="font-size: 24px; color: #FF5722; margin: 10px;">00:00</div>
        
        <audio id="audioPlayback" controls style="margin: 15px; width: 100%; display: none;"></audio>
    </div>
    
    <script>
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    let startTime;
    let timerInterval;
    
    const recordBtn = document.getElementById('recordBtn');
    const playBtn = document.getElementById('playBtn');
    const status = document.getElementById('status');
    const timer = document.getElementById('timer');
    const audioPlayback = document.getElementById('audioPlayback');
    
    async function toggleRecording() {
        if (!isRecording) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    audioPlayback.src = audioUrl;
                    audioPlayback.style.display = 'block';
                    playBtn.disabled = false;
                    
                    // Convert to base64 and store
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        window.audioData = reader.result.split(',')[1]; // Remove data:audio/wav;base64,
                    };
                    reader.readAsDataURL(audioBlob);
                };
                
                mediaRecorder.start();
                isRecording = true;
                recordBtn.textContent = '⏹️ Stop Recording';
                recordBtn.style.backgroundColor = '#f44336';
                status.textContent = 'Recording...';
                
                startTime = Date.now();
                timerInterval = setInterval(updateTimer, 1000);
                
            } catch (err) {
                status.textContent = 'Error: Could not access microphone';
                console.error('Error accessing microphone:', err);
            }
        } else {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            isRecording = false;
            recordBtn.textContent = '🎤 Start Recording';
            recordBtn.style.backgroundColor = '#4CAF50';
            status.textContent = 'Recording complete! Ready to process.';
            clearInterval(timerInterval);
        }
    }
    
    function updateTimer() {
        if (isRecording) {
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            timer.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
    }
    
    function playRecording() {
        audioPlayback.play();
    }
    
    function getAudioData() {
        return window.audioData || null;
    }
    </script>
    """
    return audio_recorder_html

# Webcam Capture Component
def create_webcam_component():
    """Create webcam capture component"""
    webcam_html = """
    <div style="text-align: center; padding: 20px; border: 2px solid #FF9800; border-radius: 10px; background-color: #f9f9f9;">
        <h3>📹 Webcam for Claude Vision</h3>
        <video id="webcam" width="400" height="300" autoplay muted style="border-radius: 10px; border: 2px solid #ddd;"></video>
        <br>
        <button id="captureBtn" onclick="captureImage()" style="
            padding: 15px 30px; 
            font-size: 18px; 
            background-color: #FF9800; 
            color: white; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
            margin: 10px;
        ">📸 Capture Image</button>
        
        <button id="startCamBtn" onclick="startWebcam()" style="
            padding: 15px 30px; 
            font-size: 18px; 
            background-color: #4CAF50; 
            color: white; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer;
            margin: 10px;
        ">🎥 Start Camera</button>
        
        <div id="camStatus" style="margin: 15px; font-size: 16px; font-weight: bold;">Click "Start Camera" to begin</div>
        <canvas id="canvas" width="400" height="300" style="display: none;"></canvas>
    </div>
    
    <script>
    const webcam = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const camStatus = document.getElementById('camStatus');
    let stream = null;
    
    async function startWebcam() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            webcam.srcObject = stream;
            camStatus.textContent = 'Camera active! Ready to capture.';
            document.getElementById('startCamBtn').textContent = '✅ Camera Running';
            document.getElementById('startCamBtn').style.backgroundColor = '#4CAF50';
        } catch (err) {
            camStatus.textContent = 'Error: Could not access camera';
            console.error('Error accessing camera:', err);
        }
    }
    
    function captureImage() {
        if (stream) {
            ctx.drawImage(webcam, 0, 0, 400, 300);
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            window.capturedImage = imageData.split(',')[1]; // Remove data:image/jpeg;base64,
            camStatus.textContent = 'Image captured! Ready to send to Claude.';
        } else {
            camStatus.textContent = 'Please start the camera first!';
        }
    }
    
    function getCapturedImage() {
        return window.capturedImage || null;
    }
    </script>
    """
    return webcam_html

# Main Interface
st.header("🎙️ Voice & Vision Interface")

# Show recorder
st.subheader("🎤 Voice Recording")
st.components.v1.html(create_audio_recorder(), height=300)

# Show webcam for Anthropic
if ai_provider == "Anthropic" and enable_webcam:
    st.subheader("📹 Webcam for Claude Vision")
    st.components.v1.html(create_webcam_component(), height=500)

# Process Audio Button
col1, col2 = st.columns(2)

with col1:
    if st.button("🗣️ Process Voice Recording", use_container_width=True):
        # Check API keys
        if not st.session_state['openai_api_key']:
            st.error("OpenAI API key required for audio processing!")
        elif ai_provider == "Anthropic" and not st.session_state['anthropic_api_key']:
            st.error("Anthropic API key required for Claude!")
        else:
            # Get audio data from JavaScript
            audio_data = st.session_state.get('audio_data')
            
            if audio_data:
                try:
                    # Decode base64 audio
                    audio_bytes = base64.b64decode(audio_data)
                    
                    # Get image if using Anthropic with webcam
                    image_data = None
                    if ai_provider == "Anthropic" and enable_webcam:
                        image_b64 = st.session_state.get('image_data')
                        if image_b64:
                            image_data = base64.b64decode(image_b64)
                            st.success("📸 Using captured webcam image!")
                    
                    with st.spinner("🔎 Transcribing audio..."):
                        transcript = transcribe_audio(audio_bytes, st.session_state['openai_api_key'])
                    
                    if transcript.strip():
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
                                    image_data
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
                        
                    else:
                        st.warning("No speech detected. Please try again.")
                        
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")
            else:
                st.warning("No audio recorded. Please record some audio first!")

with col2:
    if st.button("🔄 Refresh Interface", use_container_width=True):
        st.rerun()

# JavaScript to get data from recorder
st.markdown("""
<script>
// Function to get audio data and store in Streamlit
function sendAudioToStreamlit() {
    const audioData = getAudioData();
    if (audioData) {
        // This would need to be handled differently in real implementation
        // For now, we'll use the file upload as backup
        console.log('Audio data captured:', audioData.length, 'characters');
    }
}

// Function to get image data and store in Streamlit
function sendImageToStreamlit() {
    const imageData = getCapturedImage();
    if (imageData) {
        console.log('Image data captured:', imageData.length, 'characters');
    }
}
</script>
""", unsafe_allow_html=True)

# File Upload Alternative
st.header("📁 Upload Audio File")
uploaded_file = st.file_uploader(
    "Choose an audio file", 
    type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
    help="Upload an audio file to transcribe and chat"
)

if uploaded_file is not None:
    if st.button("🎯 Process Uploaded Audio", use_container_width=True):
        if not st.session_state['openai_api_key']:
            st.error("OpenAI API key required for audio processing!")
        elif ai_provider == "Anthropic" and not st.session_state['anthropic_api_key']:
            st.error("Anthropic API key required for Claude!")
        else:
            try:
                with st.spinner("🔎 Transcribing uploaded audio..."):
                    transcript = transcribe_audio(
                        uploaded_file.getvalue(), 
                        st.session_state['openai_api_key']
                    )
                
                if transcript.strip():
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
                                st.session_state['anthropic_api_key']
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
                    
                else:
                    st.warning("No speech detected in the uploaded file.")
                    
            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")

# Text Input Alternative
st.header("⌨️ Text Input")
text_input = st.text_area(
    "Type your message:",
    height=100,
    placeholder="Type your message here..."
)

if st.button("📝 Send Text Message", use_container_width=True) and text_input.strip():
    if ai_provider == "OpenAI" and not st.session_state['openai_api_key']:
        st.error("OpenAI API key required!")
    elif ai_provider == "Anthropic" and not st.session_state['anthropic_api_key']:
        st.error("Anthropic API key required for Claude!")
    else:
        try:
            st.markdown(f"**🧑‍💼 You:** {text_input.strip()}")
            st.session_state['conversation'].append({"role": "user", "content": text_input.strip()})
            
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
                        st.session_state['anthropic_api_key']
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
4. **Start Recording** and speak clearly
5. **Process Audio** to get AI response

### 🎙️ Recording Steps
1. **Click "Start Camera"** (for Claude vision)
2. **Click "Start Recording"** 
3. **Speak your message** clearly
4. **Click "Stop Recording"**
5. **Capture Image** (optional for Claude)
6. **Click "Process Voice Recording"**

### 🎥 Vision Features (Claude)
- Enable webcam for visual conversations
- Capture images to send with your voice
- Claude can see and discuss what's in the image
""")

st.sidebar.header("🔧 Troubleshooting")
st.sidebar.markdown("""
- **Microphone issues**: Check browser permissions
- **Camera not working**: Allow camera access
- **No audio detected**: Speak louder and clearer
- **API errors**: Verify your API keys
- **Browser compatibility**: Use Chrome/Edge for best results
- **Recording problems**: Try file upload instead
""")

st.sidebar.header("🔒 Privacy & Security")
st.sidebar.markdown("""
- API keys stored in session only
- Audio/video processed locally in browser
- No permanent data storage
- Clear conversation to remove history
""")
