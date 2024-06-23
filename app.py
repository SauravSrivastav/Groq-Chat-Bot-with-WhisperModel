import os
import streamlit as st
from typing import Generator
from groq import Groq
import json
from datetime import datetime
from fpdf import FPDF
import base64
import tempfile
import speech_recognition as sr
from pydub import AudioSegment
import io

# Set page configuration
st.set_page_config(page_icon="🤖", layout="wide", page_title="Groq AI Playground")

# Custom theme (unchanged)
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.bot {
        background-color: #475063;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .content {
        width: 80%;
    }
    .chat-message .content p {
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# Page title and description
st.title("Groq AI Playground")
st.markdown("""
    Welcome to the Groq AI Playground! Explore multiple language models and experience the power of Groq's API.
    Select a model, adjust parameters, start chatting with advanced AI models, or transcribe audio with Whisper.
""")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# Define model details
models = {
    "llama3-70b-8192": {"name": "LLaMA3-70b", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b", "tokens": 8192, "developer": "Meta"},
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
    "mixtral-8x7b-32768": {
        "name": "Mixtral-8x7b-Instruct-v0.1",
        "tokens": 32768,
        "developer": "Mistral",
    },
    "whisper-large-v3": {"name": "Whisper Large v3", "developer": "OpenAI"},
}

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

# Model selection
model_option = st.sidebar.selectbox(
    "Choose a model:",
    options=list(models.keys()),
    format_func=lambda x: models[x]["name"],
    index=0,
)

# Display model information
st.sidebar.markdown(f"""
**Model Information:**
- Name: {models[model_option]['name']}
- Developer: {models[model_option]['developer']}
""")

if model_option != "whisper-large-v3":
    # Max tokens slider
    max_tokens_range = models[model_option]["tokens"]
    max_tokens = st.sidebar.slider(
        "Max Tokens:",
        min_value=512,
        max_value=max_tokens_range,
        value=min(4096, max_tokens_range),
        step=512,
        help=f"Adjust the maximum number of tokens for the model's response. Max: {max_tokens_range}",
    )

    # Temperature slider
    temperature = st.sidebar.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Adjust the randomness of the model's responses. Higher values make output more random.",
    )

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

# Function to convert chat history to plain text
def chat_to_text(messages):
    text = "Groq AI Playground - Chat Export\n\n"
    for msg in messages:
        text += f"{msg['role'].capitalize()}: {msg['content']}\n\n"
    return text

# Function to convert chat history to PDF
def chat_to_pdf(messages):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Groq AI Playground - Chat Export", ln=1, align='C')
    for msg in messages:
        pdf.set_font("Arial", 'B', size=10)
        pdf.cell(200, 10, txt=f"{msg['role'].capitalize()}:", ln=1)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 10, txt=msg['content'])
        pdf.ln(5)
    return pdf.output(dest='S').encode('latin-1')

# Function to create a download link
def get_download_link(file_content, file_name, file_format):
    if file_format in ['txt', 'json']:
        b64 = base64.b64encode(file_content.encode('utf-8')).decode()
    else:  # pdf
        b64 = base64.b64encode(file_content).decode()
    
    mime_types = {
        'txt': 'text/plain',
        'json': 'application/json',
        'pdf': 'application/pdf'
    }
    mime = mime_types.get(file_format, 'application/octet-stream')
    
    href = f'<a href="data:{mime};base64,{b64}" download="{file_name}">Download {file_format.upper()} File</a>'
    return href

# Export chat functionality
st.sidebar.header("Export Chat")
export_format = st.sidebar.selectbox(
    "Choose export format:",
    options=["JSON", "TXT", "PDF"],
    index=0,
)

if st.sidebar.button("Export Chat"):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if export_format == "JSON":
            chat_export = {
                "model": st.session_state.selected_model,
                "timestamp": timestamp,
                "messages": st.session_state.messages
            }
            file_content = json.dumps(chat_export, indent=2)
            file_name = f"groq_chat_export_{timestamp}.json"
        elif export_format == "TXT":
            file_content = chat_to_text(st.session_state.messages)
            file_name = f"groq_chat_export_{timestamp}.txt"
        else:  # PDF
            file_content = chat_to_pdf(st.session_state.messages)
            file_name = f"groq_chat_export_{timestamp}.pdf"
        
        st.sidebar.markdown(
            get_download_link(file_content, file_name, export_format.lower()),
            unsafe_allow_html=True
        )
    except Exception as e:
        st.sidebar.error(f"An error occurred during export: {str(e)}")

# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

# Function for chat completion
def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Function to transcribe audio using Whisper
def transcribe_audio(audio_file, client):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        with open(tmp_file_path, 'rb') as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file
            )
        return transcription.text
    except Exception as e:
        st.error(f"An error occurred during transcription: {str(e)}")
        return None
    finally:
        os.unlink(tmp_file_path)

# Chat interface
if model_option != "whisper-large-v3":
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Enter your prompt here..."):
        if not st.session_state.api_key:
            st.error("Please enter your Groq API Key in the sidebar.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(prompt)

            # Create Groq client
            client = Groq(api_key=st.session_state.api_key)

            # Fetch response from Groq API
            try:
                chat_completion = client.chat.completions.create(
                    model=model_option,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )

                # Use the generator function to collect responses
                response_chunks = []
                with st.chat_message("assistant", avatar="🤖"):
                    message_placeholder = st.empty()
                    for chunk in generate_chat_responses(chat_completion):
                        response_chunks.append(chunk)
                        message_placeholder.markdown(''.join(response_chunks) + "▌")
                    full_response = ''.join(response_chunks)
                    message_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}", icon="❌")

else:
    # Whisper transcription interface
    st.header("Audio Transcription with Whisper")
    
    # Option to choose between file upload and voice recording
    input_option = st.radio("Choose input method:", ("Upload Audio File", "Record Voice"))
    
    if input_option == "Upload Audio File":
        uploaded_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'm4a'])
        
        if uploaded_file is not None:
            if not st.session_state.api_key:
                st.error("Please enter your Groq API Key in the sidebar.")
            else:
                # Create Groq client
                client = Groq(api_key=st.session_state.api_key)
                
                try:
                    with st.spinner('Transcribing...'):
                        transcription = transcribe_audio(uploaded_file, client)
                    
                    if transcription:
                        st.success("Transcription completed!")
                        st.text_area("Transcription:", value=transcription, height=300)
                        
                        # Offer download of transcription
                        st.download_button(
                            label="Download Transcription",
                            data=transcription,
                            file_name="transcription.txt",
                            mime="text/plain"
                        )
                except Exception as e:
                    st.error(f"An error occurred during transcription: {str(e)}")
    
    else:  # Record Voice
        st.write("Click the button below to start recording your voice:")
        
        if st.button("Start Recording"):
            with st.spinner("Recording... Speak now"):
                r = sr.Recognizer()
                with sr.Microphone() as source:
                    audio = r.listen(source, timeout=5)
                st.success("Recording complete!")
                
                # Convert audio to WAV
                audio_data = sr.AudioData(audio.frame_data, audio.sample_rate, audio.sample_width)
                wav_data = io.BytesIO(audio_data.get_wav_data())
                
                if not st.session_state.api_key:
                    st.error("Please enter your Groq API Key in the sidebar.")
                else:
                    # Create Groq client
                    client = Groq(api_key=st.session_state.api_key)
                    
                    try:
                        with st.spinner('Transcribing...'):
                            transcription = transcribe_audio(wav_data, client)
                        
                        if transcription:
                            st.success("Transcription completed!")
                            st.text_area("Transcription:", value=transcription, height=300)
                            
                            # Offer download of transcription
                            st.download_button(
                                label="Download Transcription",
                                data=transcription,
                                file_name="transcription.txt",
                                mime="text/plain"
                            )
                    except Exception as e:
                        st.error(f"An error occurred during transcription: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("""
**Note:** This is a demo application showcasing the capabilities of various AI models through the Groq API.
Developed by [Saurav Srivastav](https://github.com/SauravSrivastav)
""")
