
import gradio as gr
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# 1. Initialize the Deep Learning Transformer Infrastructure
print("Initializing Amharic TrOCR Engine...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Loading a pre-trained Transformer OCR model fine-tuned for Ge'ez/Amharic sequences
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("TrainedAmharicModelPlaceholder/Or-Your-Weights").to(device)
print(f"TrOCR Model loaded successfully onto compute target: {device}")

# 2. Advanced Computer Vision Preprocessing Pipeline
def preprocess_document(pil_image):
    """
    Normalizes spatial arrays, enhances textual contrast, 
    and filters out document background noise.
    """
    # Convert incoming PIL Image buffer into an OpenCV NumPy matrix array
    open_cv_image = np.array(pil_image)
    if len(open_cv_image.shape) == 3:
        bgr_matrix = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    else:
        bgr_matrix = open_cv_image.copy()
        
    # Convert matrix to single-channel Grayscale representation
    gray = cv2.cvtColor(bgr_matrix, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to smooth out scanning dust artifacts
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Adaptive Thresholding to separate ink layers from underlying paper color shifts
    processed_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Return both the BGR display frame and the thresholded processing matrix
    return bgr_matrix, processed_thresh

# 3. Transformer Sequence Inference Logic
def execute_ocr(input_img):
    if input_img is None:
        return "⚠️ Please upload or capture an image array first.", None
        
    try:
        # Run local Computer Vision preprocessing matrix pipeline
        bgr_frame, prepared_matrix = preprocess_document(input_img)
        
        # Convert thresholded array back to a PIL container for the Hugging Face processor
        processed_pil = Image.fromarray(prepared_matrix).convert("RGB")
        
        # Extract pixel values and scale tensors for the Vision Transformer Encoder
        pixel_values = processor(images=processed_pil, return_tensors="pt").pixel_values.to(device)
        
        # Generate token identifiers via Autoregressive Decoding Loop
        generated_ids = model.generate(pixel_values)
        
        # Decode numeric text tokens back into human-readable Unicode Amharic text strings
        extracted_amharic_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        if not extracted_amharic_text.strip():
            extracted_amharic_text = "🔄 No clear tokens extracted. Ensure image text contains sharp contrast."
            
        return extracted_amharic_text, prepared_matrix
        
    except Exception as error_logs:
        return f"🚨 System Processing Exception: {str(error_logs)}", None

# 4. Craft a Premium Production Gradio Block Layout UI
with gr.Blocks(theme=gr.themes.Soft(primary_cube=gr.themes.colors.amber, secondary_cube=gr.themes.colors.stone)) as demo:
    gr.Markdown(
        """
        # 🇪🇹 Strategic Amharic Text Detection & Document OCR Pipeline
        This machine learning system ingests custom text arrays, isolates characters via **Adaptive Gaussian Matrix thresholding**, and transforms features into Amharic Unicode sequences utilizing an autogressive **Vision-Decoder Transformer** neural network.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Document Source Ingestion")
            source_input = gr.Image(type="pil", label="Upload Amharic Document Fragment")
            run_pipeline_btn = gr.Button("⚡ Initialize OCR Processing Pipeline", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Extracted Spatial Thresholding")
            cv_matrix_display = gr.Image(label="Processed Binary Feature Mask")
            
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📝 Decoded Unicode Output String")
            output_string_box = gr.Textbox(label="Detected Amharic Text Array", placeholder="ሰላም እንዴት ነህ?", lines=4, show_copy_button=True)

    # Wire up the execution trigger handles
    run_pipeline_btn.click(
        fn=execute_ocr,
        inputs=source_input,
        outputs=[output_string_box, cv_matrix_display]
    )

if __name__ == "__main__":
    demo.launch()
