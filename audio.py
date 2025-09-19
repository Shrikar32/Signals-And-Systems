import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import signal
from scipy.io import wavfile
import sounddevice as sd
import queue
import time
import io

# --- 1. App Configuration and Styling ---
st.set_page_config(
    page_title="Audio DFT Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Polished CSS for the 'Classic Dark Gold' theme
st.markdown("""
<style>
    body { background-color: #121212; color: #B0BEC5; }
    .stApp { background-color: #121212; }
    h1, h2, h3 { color: #D4AF37 !important; font-family: 'Segoe UI', sans-serif; }
    .stSidebar { background-color: #1E1E1E !important; }
    .stSlider label, .stSelectbox label, .stFileUploader label, .stRadio label { 
        color: #D4AF37 !important; 
    }
    .stButton>button {
        border: 2px solid #D4AF37;
        color: #D4AF37;
    }
    .stButton>button:hover {
        background-color: #D4AF37;
        color: #1E1E1E;
    }
</style>
""", unsafe_allow_html=True)

# Centralized Plotly layout configuration
PLOTLY_CONFIG = {
    'template': 'plotly_dark',
    'margin': dict(l=40, r=20, t=60, b=40),
    'font': dict(color='#B0BEC5', size=14)
}

# Initialize session state keys at the top level
if 'audio_q' not in st.session_state:
    st.session_state.audio_q = queue.Queue()
if 'stream_running' not in st.session_state:
    st.session_state.stream_running = False
if 'stream' not in st.session_state:
    st.session_state.stream = None


# --- 2. Main App Layout and Logic ---
st.title("🎙️ Audio DFT Analyzer")
st.write("An interactive lab to explore the frequency spectrum of live voice and uploaded audio files.")

audio_mode = st.radio(
    "Select Audio Source",
    ("Real-time Microphone", "Upload .wav File"),
    horizontal=True
)
st.divider()


# --- 3. Real-time Voice Analyzer Mode ---
if audio_mode == "Real-time Microphone":
    st.header("Real-time Voice Analyzer")
    with st.expander("Theory: The Spectrogram", expanded=True):
        st.markdown("""
        This tool captures audio from your microphone and performs a **Short-Time Fourier Transform (STFT)**. The result is a **Spectrogram**, which shows how the frequency content of your voice changes over time.
        **Note:** You must grant microphone permissions in your browser.
        """)

    with st.sidebar:
        st.title("Voice Analyzer Controls")
        audio_sample_rate = st.selectbox("Sample Rate (Hz)", [44100, 22050, 16000], index=0)

    # --- NEW: Audio Device Diagnostics ---
    with st.expander("Audio Device Diagnostics"):
        st.write("Available Input Devices:")
        try:
            devices = sd.query_devices()
            input_devices = [device['name'] for device in devices if device['max_input_channels'] > 0]
            if not input_devices:
                st.warning("No input devices found. Please ensure your microphone is connected and configured.")
            else:
                st.json(input_devices)
        except Exception as e:
            st.error(f"Could not query audio devices. Error: {e}")
            st.info("This may happen if you don't have a microphone connected or if system permissions are blocking access.")


    def audio_callback(indata, frames, time, status):
        """This function is called by the sounddevice stream for each audio block."""
        st.session_state.audio_q.put(indata[:, 0])

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Start Recording", key="start"):
            if not st.session_state.stream_running:
                try:
                    st.session_state.stream = sd.InputStream(callback=audio_callback, samplerate=audio_sample_rate, channels=1, dtype='float32')
                    st.session_state.stream.start()
                    st.session_state.stream_running = True
                    st.rerun() # Rerun to update the status message immediately
                except Exception as e:
                    st.error(f"Error starting audio stream: {e}")
        
        if st.button("Stop Recording", key="stop"):
            if st.session_state.stream_running:
                st.session_state.stream.stop(); st.session_state.stream_running = False
                with st.session_state.audio_q.mutex: st.session_state.audio_q.queue.clear()
                st.rerun() # Rerun to update the status message
    with col2:
        if st.session_state.stream_running: st.success("🔴 Recording... Speak into your microphone.")
        else: st.info("Press 'Start Recording' to begin.")

    # Real-time plot updates
    if st.session_state.stream_running:
        plot_placeholder = st.empty()
        full_buffer = np.array([])
        while st.session_state.stream_running:
            try:
                audio_data = st.session_state.audio_q.get(timeout=0.5)
                if len(full_buffer) < audio_sample_rate * 2: 
                    full_buffer = np.append(full_buffer, audio_data)
                else: 
                    full_buffer = np.append(full_buffer[len(audio_data):], audio_data)
                
                if len(full_buffer) > 1024:
                    f, t_spec, Sxx = signal.spectrogram(full_buffer, fs=audio_sample_rate)
                    with plot_placeholder.container():
                        fig_spec = go.Figure(data=go.Heatmap(z=10*np.log10(Sxx + 1e-9), x=t_spec, y=f, colorscale='Inferno'))
                        fig_spec.update_layout(title_text='Real-time Spectrogram of Your Voice', **PLOTLY_CONFIG)
                        st.plotly_chart(fig_spec, use_container_width=True)
            except queue.Empty:
                time.sleep(0.1)
                if not st.session_state.stream_running: 
                    break

# --- 4. Audio File Analyzer Mode ---
elif audio_mode == "Upload .wav File":
    st.header("Audio File Analyzer")
    with st.expander("Theory: Analyzing Static Audio Files", expanded=True):
        st.markdown("""
        This tool analyzes a standard `.wav` audio file, displaying its **Waveform** (time domain) and overall **Spectrum** (frequency domain).
        """)
        
    audio_file = st.file_uploader("Upload a .wav audio file", type=['wav'])

    if audio_file:
        st.audio(audio_file)
        with st.spinner("Analyzing audio file..."):
            try:
                sample_rate, audio_data = wavfile.read(io.BytesIO(audio_file.read()))
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
                duration = len(audio_data) / sample_rate
                t = np.linspace(0, duration, len(audio_data))
                dft = np.fft.rfft(audio_data)
                freqs = np.fft.rfftfreq(len(audio_data), 1/sample_rate)
                magnitude = np.abs(dft)
            except Exception as e:
                st.error(f"Could not process .wav file. Please ensure it is a valid PCM WAV file. Error: {e}")
                st.stop()

        # --- Plotting Results ---
        st.divider()
        st.subheader("Time Domain: Waveform")
        fig_time = go.Figure(data=go.Scatter(x=t, y=audio_data, line=dict(color='#D4AF37')))
        fig_time.update_layout(title_text="Audio Waveform", xaxis_title="Time (s)", yaxis_title="Amplitude", **PLOTLY_CONFIG)
        st.plotly_chart(fig_time, use_container_width=True)

        st.subheader("Frequency Domain: Spectrum")
        fig_freq = go.Figure(data=go.Scatter(x=freqs, y=magnitude, line=dict(color='#00CFE8')))
        fig_freq.update_layout(title_text="Overall Frequency Spectrum", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", 
                               xaxis_type="log", yaxis_type="log", **PLOTLY_CONFIG)
        st.plotly_chart(fig_freq, use_container_width=True)

