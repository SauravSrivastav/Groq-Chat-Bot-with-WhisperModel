import streamlit as st
import torch
from transformers import pipeline
import yt_dlp as youtube_dl
import tempfile
import os
from transformers.pipelines.audio_utils import ffmpeg_read

# Set page configuration
st.set_page_config(page_icon="🎙️", layout="wide", page_title="Whisper Transcription App")

# Custom theme
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Page title and description
st.title("Whisper Transcription App")
st.markdown("""
    Welcome to the Whisper Transcription App! Transcribe audio from file upload, microphone input, or YouTube videos.
""")

# Initialize session state for API key
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# Sidebar for configuration
st.sidebar.header("Configuration")

# API Key input
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
if api_key:
    st.session_state.api_key = api_key

# Add link to get API key
st.sidebar.markdown("""
    [Get your Groq API key here](https://console.groq.com/keys)
""")

# Model configuration
MODEL_NAME = "openai/whisper-large-v3"
BATCH_SIZE = 8
YT_LENGTH_LIMIT_S = 3600  # limit to 1 hour YouTube files

device = 0 if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    return pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        chunk_length_s=30,
        device=device,
    )

pipe = load_model()

def transcribe(audio_file, task):
    if audio_file is None:
        raise st.error("No audio file submitted! Please upload or record an audio file before submitting your request.")

    with st.spinner('Transcribing...'):
        text = pipe(audio_file, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]
    return text

def download_yt_audio(yt_url, filename):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': filename,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([yt_url])
        except youtube_dl.utils.DownloadError as err:
            st.error(f"Error downloading YouTube audio: {str(err)}")
            return None
    return filename

def yt_transcribe(yt_url, task):
    with st.spinner('Downloading YouTube audio...'):
        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, "audio.wav")
            audio_file = download_yt_audio(yt_url, filepath)
            
            if audio_file is None:
                return None

            with open(filepath, "rb") as f:
                inputs = f.read()

    inputs = ffmpeg_read(inputs, pipe.feature_extractor.sampling_rate)
    inputs = {"array": inputs, "sampling_rate": pipe.feature_extractor.sampling_rate}

    with st.spinner('Transcribing...'):
        text = pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]
    return text

# Main app
tab1, tab2, tab3 = st.tabs(["File Upload", "Microphone", "YouTube"])

with tab1:
    st.header("Transcribe from File")
    uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'm4a'])
    task = st.radio("Task", ["transcribe", "translate"], horizontal=True)
    
    if st.button("Transcribe File"):
        if uploaded_file is not None:
            text = transcribe(uploaded_file, task)
            st.text_area("Transcription:", value=text, height=300)
        else:
            st.warning("Please upload an audio file.")

with tab2:
    st.header("Transcribe from Microphone")
    audio_bytes = st.audio_recorder()
    task = st.radio("Task", ["transcribe", "translate"], horizontal=True, key="mic_task")
    
    if st.button("Transcribe Audio"):
        if audio_bytes:
            text = transcribe(audio_bytes, task)
            st.text_area("Transcription:", value=text, height=300)
        else:
            st.warning("Please record some audio first.")

with tab3:
    st.header("Transcribe from YouTube")
    yt_url = st.text_input("YouTube URL")
    task = st.radio("Task", ["transcribe", "translate"], horizontal=True, key="yt_task")
    
    if st.button("Transcribe YouTube Video"):
        if yt_url:
            text = yt_transcribe(yt_url, task)
            if text:
                st.text_area("Transcription:", value=text, height=300)
        else:
            st.warning("Please enter a YouTube URL.")

# Footer
st.markdown("---")
st.markdown("""
**Note:** This app uses the Whisper Large V3 model for transcription.
Developed based on Hugging Face Transformers and Streamlit.
""")
