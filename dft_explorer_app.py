import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import signal

# --- 1. App Configuration and Styling ---
st.set_page_config(
    page_title="DFT Explorer & Reconstruction Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Polished CSS for the 'Classic Dark Gold' theme
st.markdown("""
<style>
    /* Main theme colors */
    body { background-color: #121212; color: #B0BEC5; }
    .stApp { background-color: #121212; }
    h1, h2, h3 { color: #D4AF37 !important; font-family: 'Segoe UI', sans-serif; }
    
    /* Sidebar styling */
    .stSidebar { background-color: #1E1E1E !important; }
    .stSlider label, .stSelectbox label, .stCheckbox label { color: #D4AF37 !important; }
</style>
""", unsafe_allow_html=True)

# Centralized Plotly layout configuration for a consistent, professional look
PLOTLY_CONFIG = {
    'template': 'plotly_dark',
    'margin': dict(l=40, r=20, t=60, b=40),
    'font': dict(color='#B0BEC5', size=14)
}

# --- 2. Sidebar for User Controls ---
with st.sidebar:
    st.title("Signal Laboratory Controls")
    st.divider()

    st.header("Signal Generation")
    signal_type = st.selectbox(
        "Choose a Signal Type",
        ("Sum of Sines", "Square Wave", "Sawtooth Wave", "Gaussian Pulse", "Amplitude Modulated (AM)", "Windowed Sine", "Signal with Noise", "Chirp Signal", "Sine with DC Offset")
    )

    st.header("Signal Parameters")
    sampling_rate = st.slider("Sampling Rate (Hz)", 100, 4000, 1000)
    duration = st.slider("Signal Duration (s)", 1, 5, 2)
    
    # --- Dynamic UI for Signal Parameters ---
    params = {}
    if signal_type == "Sum of Sines":
        st.subheader("Sine Wave 1"); params['freq1'] = st.slider("Frequency 1 (Hz)", 1, 500, 10); params['amp1'] = st.slider("Amplitude 1", 0.1, 5.0, 1.0)
        st.subheader("Sine Wave 2"); params['freq2'] = st.slider("Frequency 2 (Hz)", 1, 500, 50); params['amp2'] = st.slider("Amplitude 2", 0.1, 5.0, 0.7)
    elif signal_type in ["Square Wave", "Sawtooth Wave"]:
        params['freq'] = st.slider("Frequency (Hz)", 1, 500, 15); params['amp'] = st.slider("Amplitude", 0.1, 5.0, 1.0)
    elif signal_type == "Gaussian Pulse":
        params['center'] = st.slider("Pulse Center (s)", 0.0, duration, duration / 2); params['std_dev'] = st.slider("Standard Deviation (σ)", 0.01, 1.0, 0.1)
    elif signal_type == "Amplitude Modulated (AM)":
        params['freq_c'] = st.slider("Carrier Frequency (Hz)", 50, 500, 100); params['freq_m'] = st.slider("Message Frequency (Hz)", 1, 50, 5); params['mod_index'] = st.slider("Modulation Index", 0.0, 2.0, 1.0)
    elif signal_type == "Windowed Sine":
        params['freq'] = st.slider("Sine Frequency (Hz)", 1, 500, 50); params['window_type'] = st.selectbox("Window Type", ("Hann", "Hamming", "Rectangular (None)"))
    elif signal_type == "Signal with Noise":
        params['freq'] = st.slider("Signal Frequency (Hz)", 1, 500, 50); params['amp'] = st.slider("Signal Amplitude", 0.1, 5.0, 1.0); params['noise_amp'] = st.slider("Noise Amplitude", 0.0, 5.0, 0.8)
    elif signal_type == "Chirp Signal":
        params['start_freq'] = st.slider("Start Frequency (Hz)", 1, 1000, 1); params['end_freq'] = st.slider("End Frequency (Hz)", 1, 1000, 250)
    elif signal_type == "Sine with DC Offset":
        params['freq'] = st.slider("Frequency (Hz)", 1, 500, 20); params['amp'] = st.slider("Amplitude", 0.1, 5.0, 1.0); params['dc_offset'] = st.slider("DC Offset", -5.0, 5.0, 2.0)

    st.divider()
    
    st.header("Frequency Filtering")
    filter_enabled = st.checkbox("Enable Band-Stop Filter", value=True)
    max_freq = sampling_rate / 2
    stop_band = st.slider(
        "Frequency Band to Remove (Hz)", 0.0, max_freq, (max_freq / 4, max_freq / 2),
        disabled=not filter_enabled
    )

# --- 3. Core Signal Processing and Calculations ---

# Generate time vector and signal waveform based on user input
n_samples = int(sampling_rate * duration)
t = np.linspace(0, duration, n_samples, endpoint=False)
signal_data = np.zeros(n_samples)
if signal_type == "Sum of Sines":
    signal_data = params['amp1'] * np.sin(2 * np.pi * params['freq1'] * t) + params['amp2'] * np.sin(2 * np.pi * params['freq2'] * t)
elif signal_type == "Square Wave":
    signal_data = params['amp'] * signal.square(2 * np.pi * params['freq'] * t)
elif signal_type == "Sawtooth Wave":
    signal_data = params['amp'] * signal.sawtooth(2 * np.pi * params['freq'] * t)
elif signal_type == "Gaussian Pulse":
    signal_data = np.exp(-0.5 * ((t - params['center']) / params['std_dev'])**2)
elif signal_type == "Amplitude Modulated (AM)":
    carrier = np.sin(2 * np.pi * params['freq_c'] * t)
    message = params['mod_index'] * np.sin(2 * np.pi * params['freq_m'] * t)
    signal_data = (1 + message) * carrier
elif signal_type == "Windowed Sine":
    pure_sine = np.sin(2 * np.pi * params['freq'] * t)
    if params['window_type'] == "Hann": window = np.hanning(n_samples)
    elif params['window_type'] == "Hamming": window = np.hamming(n_samples)
    else: window = np.ones(n_samples)
    signal_data = pure_sine * window
elif signal_type == "Signal with Noise":
    clean_signal = params['amp'] * np.sin(2 * np.pi * params['freq'] * t)
    noise = params['noise_amp'] * np.random.randn(n_samples)
    signal_data = clean_signal + noise
elif signal_type == "Chirp Signal":
    signal_data = signal.chirp(t, f0=params['start_freq'], f1=params['end_freq'], t1=duration, method='linear')
elif signal_type == "Sine with DC Offset":
    signal_data = params['amp'] * np.sin(2 * np.pi * params['freq'] * t) + params['dc_offset']

# Perform the DFT
dft_complex = np.fft.fft(signal_data)
frequencies = np.fft.fftfreq(n_samples, 1 / sampling_rate)

# Create a copy of the DFT for potential filtering
dft_filtered_complex = dft_complex.copy()

# Apply the frequency filter if enabled
if filter_enabled:
    # Find indices of frequencies within the stop band (for both positive and negative frequencies)
    filter_indices = np.where((np.abs(frequencies) >= stop_band[0]) & (np.abs(frequencies) <= stop_band[1]))
    dft_filtered_complex[filter_indices] = 0

# Perform the Inverse DFT on the (potentially filtered) spectrum
reconstructed_signal = np.fft.ifft(dft_filtered_complex)

# Prepare data for one-sided spectrum plots (Magnitude and Phase)
positive_freq_indices = np.where(frequencies >= 0)
freq_axis = frequencies[positive_freq_indices]
dft_positive = dft_complex[positive_freq_indices]
magnitude_spectrum = (np.abs(dft_positive) / n_samples) * 2
magnitude_spectrum[0] /= 2  # DC component is not doubled
phase_spectrum_unwrapped = np.unwrap(np.angle(dft_positive))


# --- 4. Main App Layout and Visualizations ---

st.title("Discrete Fourier Transform (DFT) Explorer")
st.write("A tool to visualize the journey of a signal from the time domain to the frequency domain and back.")

# --- Section 1: Signal Input ---
st.header("1. Signal Input: The Time Domain")
with st.expander("Theory: From Continuous to Discrete"):
    st.markdown("We start with a signal defined in the time domain. While real-world signals are continuous, computers must **sample** them to create a discrete sequence of points. This is the input for the DFT.")
fig_time = go.Figure()
fig_time.add_trace(go.Scatter(x=t, y=signal_data, mode='lines', name='Sampled Signal', line=dict(color='#D4AF37', width=2)))
fig_time.update_layout(title_text=f"Generated Signal: {signal_type}", xaxis_title="Time (s)", yaxis_title="Amplitude", **PLOTLY_CONFIG)
st.plotly_chart(fig_time, use_container_width=True)

# --- Section 2: DFT Analysis ---
st.header("2. DFT Analysis: The Frequency Domain")
with st.expander("Theory: Decomposing the Signal"):
    st.markdown(r"""
    The DFT analyzes the discrete signal to find its constituent frequencies. It produces a set of complex numbers, from which we can extract two key pieces of information:
    - **Magnitude:** The strength or amplitude of each frequency.
    - **Phase:** The starting offset of each frequency's wave.
    
    The DFT formula is: $X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-i \frac{2\pi}{N} k n}$
    """)
col1, col2 = st.columns(2)
with col1:
    fig_mag = go.Figure()
    fig_mag.add_trace(go.Bar(x=freq_axis, y=magnitude_spectrum, name="Magnitude", marker_color='#D4AF37'))
    fig_mag.update_layout(title_text="Magnitude Spectrum", xaxis_title="Frequency (Hz)", yaxis_title="Amplitude", **PLOTLY_CONFIG)
    fig_mag.update_xaxes(range=[0, max_freq])
    st.plotly_chart(fig_mag, use_container_width=True)
with col2:
    fig_phase = go.Figure()
    fig_phase.add_trace(go.Scatter(x=freq_axis, y=np.degrees(phase_spectrum_unwrapped), mode='lines', name="Phase", line=dict(color='#B0BEC5')))
    fig_phase.update_layout(title_text="Phase Spectrum", xaxis_title="Frequency (Hz)", yaxis_title="Phase (Degrees)", **PLOTLY_CONFIG)
    fig_phase.update_xaxes(range=[0, max_freq])
    st.plotly_chart(fig_phase, use_container_width=True)

# --- Section 3 & 4: Signal Reconstruction and Filtering ---
st.header("3 & 4. Signal Reconstruction & Frequency Manipulation")
with st.expander("Theory: Rebuilding and Filtering"):
    st.markdown(r"""
    The **Inverse DFT (IDFT)** uses the frequency domain data to reconstruct the original signal's discrete samples. This becomes powerful when we manipulate the frequency data before reconstruction, such as by performing a **band-stop filter**.
    
    Below, we show the final reconstructed signal in two forms: first as the raw discrete samples that the IDFT produces, and second as the continuous-like signal that is formed by connecting those samples.
    
    The IDFT formula is: $x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k \cdot e^{i \frac{2\pi}{N} k n}$
    """)

# Create a two-column layout for the final comparison
col3, col4 = st.columns(2)

# Left column: Discrete representation
with col3:
    fig_recon_discrete = go.Figure()
    # Add the original signal as a dotted reference line
    fig_recon_discrete.add_trace(go.Scatter(x=t, y=signal_data, mode='lines', name='Original Signal', line=dict(color='#B0BEC5', dash='dot')))
    # Add the final DISCRETE reconstructed signal
    fig_recon_discrete.add_trace(go.Scatter(x=t, y=reconstructed_signal.real, mode='markers', name='Discrete Samples', marker=dict(color='#D4AF37', size=5, line=dict(width=1, color='Black'))))
    
    title_text_discrete = "Discrete Samples of Reconstructed Signal"
    fig_recon_discrete.update_layout(title_text=title_text_discrete, xaxis_title="Time (s)", yaxis_title="Amplitude", **PLOTLY_CONFIG)
    st.plotly_chart(fig_recon_discrete, use_container_width=True)

# Right column: Continuous-like representation
with col4:
    fig_recon_continuous = go.Figure()
    # Add the original signal as a dotted reference line
    fig_recon_continuous.add_trace(go.Scatter(x=t, y=signal_data, mode='lines', name='Original Signal', line=dict(color='#B0BEC5', dash='dot')))
    # Add the final "continuous-like" reconstructed signal
    fig_recon_continuous.add_trace(go.Scatter(x=t, y=reconstructed_signal.real, mode='lines', name='Reconstructed Signal (Line)', line=dict(color='#00CFE8', width=2.5)))
    
    title_text_continuous = "Final Continuous-like Reconstructed Signal"
    fig_recon_continuous.update_layout(title_text=title_text_continuous, xaxis_title="Time (s)", yaxis_title="Amplitude", **PLOTLY_CONFIG)
    st.plotly_chart(fig_recon_continuous, use_container_width=True)

