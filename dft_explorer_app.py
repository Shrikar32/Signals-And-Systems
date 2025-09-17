import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import signal

# --- NEW: Updated CSS with Classic Dark Gold Palette ---
def load_css():
    st.markdown("""
        <style>
        body {
            background-color: #121212; /* Very dark charcoal */
            color: #B0BEC5;           /* Cool silver/gray for text */
        }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1, h2, h3, h4, h5, h6 {
            color: #D4AF37 !important; /* Warm gold for headers */
            font-family: 'Segoe UI', sans-serif;
        }
        .stSlider label, .stSelectbox label, .stSidebar label {
            color: #D4AF37 !important; /* Warm gold for controls */
        }
        .stSidebar { background-color: #1E1E1E !important; } /* Slightly lighter charcoal for sidebar */
        </style>
    """, unsafe_allow_html=True)

# --- Page Config ---
st.set_page_config(page_title="DFT Explorer", layout="wide")
load_css()

# --- Title ---
st.title("⚡ Discrete Fourier Transform (DFT) Explorer ⚡")
st.write("An interactive tool to explore signals in time and frequency domains.")

# --- Definitions Expander ---
with st.expander("What is the Discrete Fourier Transform?"):
    st.markdown(r"""
    The **Discrete Fourier Transform (DFT)** converts a finite sequence of samples (Time Domain) into a sequence of complex numbers representing its frequency components (Frequency Domain). In essence, it breaks down a signal into the constituent sine and cosine waves that make it up.
    """)
    st.latex(r'X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-i \frac{2\pi}{N} k n}')
    st.markdown(r"""
    The **Inverse DFT (IDFT)** reconstructs the time-domain signal from its frequency components.
    """)
    st.latex(r'x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k \cdot e^{i \frac{2\pi}{N} k n}')

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Signal Parameters")
    signal_type = st.selectbox(
        "Choose a Signal",
        ("Sum of Sines", "Square Wave", "Sawtooth Wave", "Gaussian Pulse", "Amplitude Modulated (AM)", "Windowed Sine", "Signal with Noise", "Chirp Signal", "Sine with DC Offset")
    )
    # ... (The rest of the sidebar logic is unchanged)
    sampling_rate = st.slider("Sampling Rate (Hz)", 100, 4000, 1000)
    duration = st.slider("Signal Duration (s)", 1, 5, 2)
    if signal_type == "Sum of Sines":
        st.subheader("Sine Wave 1"); freq1 = st.slider("Frequency 1 (Hz)", 1, 500, 10, key='freq1'); amp1 = st.slider("Amplitude 1", 0.1, 5.0, 1.0, key='amp1')
        st.subheader("Sine Wave 2"); freq2 = st.slider("Frequency 2 (Hz)", 1, 500, 50, key='freq2'); amp2 = st.slider("Amplitude 2", 0.1, 5.0, 0.7, key='amp2')
    elif signal_type in ["Square Wave", "Sawtooth Wave"]:
        freq = st.slider("Frequency (Hz)", 1, 500, 15, key='freq_main'); amp = st.slider("Amplitude", 0.1, 5.0, 1.0, key='amp_main')
    elif signal_type == "Gaussian Pulse":
        center = st.slider("Pulse Center (s)", 0.0, duration, duration / 2, key='center'); std_dev = st.slider("Standard Deviation (σ)", 0.01, 1.0, 0.1, key='std_dev')
    elif signal_type == "Amplitude Modulated (AM)":
        freq_c = st.slider("Carrier Frequency (Hz)", 50, 500, 100, key='freq_c'); freq_m = st.slider("Message Frequency (Hz)", 1, 50, 5, key='freq_m'); mod_index = st.slider("Modulation Index", 0.0, 2.0, 1.0)
    elif signal_type == "Windowed Sine":
        freq = st.slider("Sine Frequency (Hz)", 1, 500, 50, key='freq_win'); window_type = st.selectbox("Window Type", ("Hann", "Hamming", "Rectangular (None)"))
    elif signal_type == "Signal with Noise":
        freq = st.slider("Signal Frequency (Hz)", 1, 500, 50, key='freq_noise'); amp = st.slider("Signal Amplitude", 0.1, 5.0, 1.0, key='amp_noise'); noise_amp = st.slider("Noise Amplitude", 0.0, 5.0, 0.8, key='noise_amp')
    elif signal_type == "Chirp Signal":
        start_freq = st.slider("Start Frequency (Hz)", 1, 1000, 1); end_freq = st.slider("End Frequency (Hz)", 1, 1000, 250)
    elif signal_type == "Sine with DC Offset":
        freq = st.slider("Frequency (Hz)", 1, 500, 20, key='freq_dc'); amp = st.slider("Amplitude", 0.1, 5.0, 1.0, key='amp_dc'); dc_offset = st.slider("DC Offset", -5.0, 5.0, 2.0)

# --- Signal Generation ---
n_samples = int(sampling_rate * duration)
t = np.linspace(0, duration, n_samples, endpoint=False)
signal_data = np.zeros(n_samples)
# ... (Signal generation logic is unchanged)
if signal_type == "Sum of Sines": signal_data = amp1 * np.sin(2 * np.pi * freq1 * t) + amp2 * np.sin(2 * np.pi * freq2 * t)
elif signal_type == "Square Wave": signal_data = amp * signal.square(2 * np.pi * freq * t)
elif signal_type == "Sawtooth Wave": signal_data = amp * signal.sawtooth(2 * np.pi * freq * t)
elif signal_type == "Gaussian Pulse": signal_data = np.exp(-0.5 * ((t - center) / std_dev)**2)
elif signal_type == "Amplitude Modulated (AM)": carrier = np.sin(2 * np.pi * freq_c * t); message = mod_index * np.sin(2 * np.pi * freq_m * t); signal_data = (1 + message) * carrier
elif signal_type == "Windowed Sine": pure_sine = np.sin(2 * np.pi * freq * t); window = {"Hann": np.hanning(n_samples), "Hamming": np.hamming(n_samples)}.get(window_type, np.ones(n_samples)); signal_data = pure_sine * window
elif signal_type == "Signal with Noise": clean_signal = amp * np.sin(2 * np.pi * freq * t); noise = noise_amp * np.random.randn(n_samples); signal_data = clean_signal + noise
elif signal_type == "Chirp Signal": signal_data = signal.chirp(t, f0=start_freq, f1=end_freq, t1=duration, method='linear')
elif signal_type == "Sine with DC Offset": signal_data = amp * np.sin(2 * np.pi * freq * t) + dc_offset

# --- Fourier Transform & Reconstruction ---
dft = np.fft.fft(signal_data)
# (Calculations are unchanged)
dft_magnitude = np.abs(dft) / n_samples
frequencies = np.fft.fftfreq(n_samples, 1 / sampling_rate)
positive_freq_indices = np.where(frequencies >= 0)
freq_axis = frequencies[positive_freq_indices]
mag_axis = dft_magnitude[positive_freq_indices]
reconstructed_signal = np.fft.ifft(dft)

# --- Layout & Plots ---
col1, col2 = st.columns(2)
font_dict = dict(color='#B0BEC5') # Silver/gray font for plots

with col1:
    st.subheader("⏱️ Original Signal (Time Domain)")
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=t, y=signal_data, mode='lines', name='Original', line=dict(color='#D4AF37', width=2))) # Warm Gold line
    fig_time.update_layout(template="plotly_dark", title=f"{signal_type} Signal", xaxis_title="Time (s)", yaxis_title="Amplitude", margin=dict(l=20, r=20, t=40, b=20), font=font_dict)
    st.plotly_chart(fig_time, use_container_width=True)

with col2:
    st.subheader("📡 Frequency Domain")
    fig_freq = go.Figure()
    fig_freq.add_trace(go.Bar(x=freq_axis, y=mag_axis, name="DFT Magnitude", marker_color='#D4AF37')) # Warm Gold bars
    fig_freq.update_layout(template="plotly_dark", title="DFT Magnitude Spectrum", xaxis_title="Frequency (Hz)", yaxis_title="Magnitude", margin=dict(l=20, r=20, t=40, b=20), font=font_dict)
    fig_freq.update_xaxes(range=[0, sampling_rate / 2])
    st.plotly_chart(fig_freq, use_container_width=True)

st.subheader("🔄 Reconstructed Signal")
fig_recon = go.Figure()
fig_recon.add_trace(go.Scatter(x=t, y=reconstructed_signal.real, mode='lines', name='Reconstructed', line=dict(color='#B0BEC5', width=2))) # Cool Silver line
fig_recon.update_layout(template="plotly_dark", title="Signal Reconstructed from DFT", xaxis_title="Time (s)", yaxis_title="Amplitude", margin=dict(l=20, r=20, t=40, b=20), font=font_dict)
st.plotly_chart(fig_recon, use_container_width=True)