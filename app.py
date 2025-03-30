import streamlit as st
import pymupdf 
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import io
import re

# Load tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("fine-tuned-bart-lora")
    model = AutoModelForSeq2SeqLM.from_pretrained("fine-tuned-bart-lora")
    model = torch.compile(model) 
    return tokenizer, model

tokenizer, model = load_model()

def preprocess_text(text):
    """Removes author names and unnecessary metadata from research papers."""
    text = re.sub(r"(?i)(?:by|authors?)\s*[:\n].*?\n\n", "", text, flags=re.DOTALL)  # Remove author names
    text = re.sub(r"\n\s*\n", "\n", text)  # Clean excessive whitespace
    return text

def extract_text_from_pdf(pdf_bytes):
    """Extracts text from PDF."""
    pdf_bytes.seek(0)  
    doc = pymupdf.open(stream=pdf_bytes.read(), filetype="pdf")
    full_text = "\n".join([page.get_text("text") for page in doc])
    return preprocess_text(full_text) if full_text else None

def extract_images_from_pdf(pdf_bytes):
    """Extracts images and captions from the PDF."""
    pdf_bytes.seek(0)
    doc = pymupdf.open(stream=pdf_bytes.read(), filetype="pdf")
    images = []
    
    for page in doc:
        img_list = page.get_images(full=True)
        for img_index, img in enumerate(img_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_data = base_image["image"]
            image = Image.open(io.BytesIO(image_data))
            caption = f"Figure {img_index+1}: (Caption not found)"
            lines = page.get_text("text").split("\n")
            for i, line in enumerate(lines):
                if f"Figure {img_index+1}" in line or "Fig." in line:
                    caption = line
                    break
            images.append({"image": image, "caption": caption})
    
    return images

def summarize_text(text):
    """Summarizes text using the fine-tuned BART model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(**inputs, max_length=256, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Streamlit UI
st.title("📄 Research Paper Summarizer")

uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type="pdf")

if uploaded_file:
    pdf_bytes = io.BytesIO(uploaded_file.read())

    with st.spinner("Processing... Extracting text and images..."):
        extracted_text = extract_text_from_pdf(pdf_bytes)
        extracted_images = extract_images_from_pdf(pdf_bytes)

        if extracted_text:
            summary = summarize_text(extracted_text)
            st.subheader("📌 Summary:")
            st.success(summary)

        if extracted_images:
            st.subheader("📷 Extracted Figures:")
            for img_info in extracted_images:
                st.image(img_info["image"], caption=img_info["caption"], use_container_width=True)

        if not (extracted_text or extracted_images):
            st.error("Could not extract content from the PDF. Try another file.")
