import streamlit as st
import numpy as np
from PIL import Image
import requests
import io

# --- 1. App Configuration and Styling ---
st.set_page_config(
    page_title="2D Image DFT Analyzer",
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
    .stSlider label, .stSelectbox label, .stFileUploader label { 
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

# --- Cached function to download images for performance ---
@st.cache_data
def load_image_from_url(url):
    """Downloads and caches an image from a URL, returning a NumPy array."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert('L')
        return np.array(image)
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching image: {e}")
        return None

# --- 2. Sidebar for Controls ---
with st.sidebar:
    st.title("Image Lab Controls")
    st.divider()

    st.header("Image Source")
    EXAMPLE_IMAGES = {
        "Guitar & Amplifier": "https://images.pexels.com/photos/164727/pexels-photo-164727.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        "City Skyline": "https://images.pexels.com/photos/2103836/pexels-photo-2103836.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
        "Classic Car Poster": "https://images.pexels.com/photos/337909/pexels-photo-337909.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"
    }
    image_source = st.selectbox(
        "Choose an Image Source",
        ("Upload Your Own", "Guitar & Amplifier", "City Skyline", "Classic Car Poster")
    )
    
    st.divider()
    
    st.header("Frequency Filtering")
    st.info("Apply a filter in the frequency domain to see its effect on the reconstructed image.")
    filter_type = st.selectbox(
        "Select Filter Type",
        ("None", "Low-Pass (Blur)", "High-Pass (Edge Detection)")
    )
    
    filter_radius = st.slider("Filter Radius / Cutoff", min_value=1, max_value=200, value=30)


# --- 3. Main App Layout and Logic ---
st.title("🖼️ 2D DFT Image Analyzer")
st.write("An interactive lab to explore the spatial frequencies of images and the power of frequency-domain filtering.")

with st.expander("Theory: Understanding 2D Spatial Frequencies", expanded=True):
    st.markdown("""
    A 2D image can be decomposed into its constituent **spatial frequencies**. This is analogous to breaking down a sound wave into its audio frequencies, but instead of representing pitch, spatial frequencies represent patterns and details in the image.

    - **The Spectrum Image:** The 2D DFT produces a frequency spectrum. By convention, we display the zero-frequency component (the image's average brightness) at the center.
    - **Low Frequencies (Center):** Represent the large, smooth, and slowly changing areas, like a clear sky or a plain wall.
    - **High Frequencies (Edges):** Represent sharp edges, fine details, and textures. The farther from the center, the higher the frequency.
    - **Direction:** The orientation of features in the spectrum reveals the orientation of patterns in the original image. For example, the strong vertical lines of skyscrapers in a cityscape will create a bright horizontal line in the frequency spectrum.
    """)
    

# --- Load Image ---
image_array = None
if image_source == "Upload Your Own":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('L')
        image_array = np.array(image)
else:
    with st.spinner(f"Downloading '{image_source}'..."):
        image_array = load_image_from_url(EXAMPLE_IMAGES[image_source])

# --- Perform Analysis if Image is Loaded ---
if image_array is not None:
    # --- DFT Calculation ---
    dft = np.fft.fft2(image_array)
    dft_shifted = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shifted) + 1)
    
    # --- FIX: Normalize the magnitude spectrum for display ---
    # This scales the array values to the range [0.0, 1.0] which st.image expects.
    magnitude_spectrum_normalized = (magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum))


    # --- Filtering Logic ---
    if filter_type != "None":
        rows, cols = image_array.shape
        crow, ccol = rows // 2 , cols // 2
        
        # Create a circular mask
        y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
        mask = x*x + y*y <= filter_radius*filter_radius
        
        if filter_type == "High-Pass (Edge Detection)":
            mask = ~mask # Invert the mask

        # Apply mask and inverse DFT
        dft_filtered_shifted = dft_shifted * mask
        f_ishift = np.fft.ifftshift(dft_filtered_shifted)
        img_reconstructed = np.fft.ifft2(f_ishift)
        img_reconstructed = np.abs(img_reconstructed)
    else:
        img_reconstructed = image_array

    # --- Display Results ---
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Grayscale Image")
        st.image(image_array, use_column_width=True, caption="The input for our analysis.")
    with col2:
        st.subheader("Magnitude Spectrum")
        # Use the normalized spectrum for display
        st.image(magnitude_spectrum_normalized, use_column_width=True, caption="Visual representation of spatial frequencies.")

    st.header("Filtered Result")
    if filter_type != "None":
        # Also normalize the reconstructed image for safe display
        img_reconstructed_normalized = (img_reconstructed - np.min(img_reconstructed)) / (np.max(img_reconstructed) - np.min(img_reconstructed))
        st.image(img_reconstructed_normalized, use_column_width=True, caption=f"Image after applying a {filter_type} filter in the frequency domain.")
    else:
        st.info("Select a filter type from the sidebar to see the effect of frequency manipulation.")

