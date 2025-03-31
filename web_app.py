from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory, jsonify
import fitz
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import io
import re
import os
import base64
from PIL import Image
import tabula
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import uuid
import json
import shutil
from threading import Thread
import time

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create required directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Load tokenizer and model
def load_model():
    # Load from the base model
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model

tokenizer, model = load_model()

# Add these global variables after the app configuration
processing_status = {}
processing_results = {}

def preprocess_text(text):
    """Removes author names and unnecessary metadata from research papers."""
    text = re.sub(r"(?i)(?:by|authors?)\s*[:\n].*?\n\n", "", text, flags=re.DOTALL)
    text = re.sub(r"\n\s*\n", "\n", text)
    return text

def extract_text_from_pdf(pdf_bytes):
    """Extracts text from PDF."""
    pdf_bytes.seek(0)
    doc = fitz.open(stream=pdf_bytes.read(), filetype="pdf")
    full_text = "\n".join([page.get_text("text") for page in doc])
    return preprocess_text(full_text) if full_text else None

def extract_images_from_pdf(pdf_bytes, output_dir):
    """Extracts images and figure captions from the PDF and saves to disk."""
    pdf_bytes.seek(0)
    doc = fitz.open(stream=pdf_bytes.read(), filetype="pdf")
    images_info = []
    
    for page_num, page in enumerate(doc):
        img_list = page.get_images(full=True)
        
        for img_index, img in enumerate(img_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_data = base_image["image"]
                
                # Convert to PIL Image
                pil_image = Image.open(io.BytesIO(image_data))
                
                # Generate a unique filename
                img_filename = f"img_{uuid.uuid4()}.png"
                img_path = os.path.join(output_dir, img_filename)
                
                # Save image to disk
                pil_image.save(img_path)
                
                # Try to find a caption (look for "Figure", "Fig.", "Chart", "Diagram" in text)
                page_text = page.get_text("text")
                lines = page_text.split('\n')
                caption = None
                
                for i, line in enumerate(lines):
                    if re.search(r"(Figure|Fig\.|Chart|Diagram|Image)\s*\d+", line, re.IGNORECASE):
                        caption = line.strip()
                        # Try to get the next line if it seems to continue the caption
                        if i+1 < len(lines) and not re.match(r"(Figure|Fig\.|Chart|Diagram|Image)\s*\d+", lines[i+1], re.IGNORECASE) and len(lines[i+1].strip()) > 0:
                            caption += " " + lines[i+1].strip()
                        break
                
                if not caption:
                    caption = f"Figure {img_index+1} from page {page_num+1}"
                    
                images_info.append({
                    "filename": img_filename,
                    "caption": caption
                })
            except Exception as e:
                print(f"Error processing image: {str(e)}")
    
    return images_info

def extract_tables_from_pdf(pdf_path, output_dir):
    """Extracts tables from PDF using tabula-py and saves as HTML files."""
    tables_info = []
    try:
        # Read PDF with tabula
        dfs = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True, encoding='latin-1')
        
        # Process each table
        for i, df in enumerate(dfs):
            try:
                # Drop entirely empty rows and columns
                df = df.replace('', np.nan)
                df = df.dropna(how='all', axis=0)
                df = df.dropna(how='all', axis=1)
                
                # If table is not empty
                if not df.empty and df.size > 1:
                    # Convert DataFrame to HTML with proper styling
                    html_table = """
                    <div class="table-responsive">
                        <table class="pdf-table">
                            <thead>
                                <tr>
                                    {}
                                </tr>
                            </thead>
                            <tbody>
                                {}
                            </tbody>
                        </table>
                    </div>
                    """.format(
                        ''.join(f'<th>{col}</th>' for col in df.columns),
                        ''.join(
                            '<tr>{}</tr>'.format(
                                ''.join(f'<td>{str(cell)}</td>' for cell in row)
                            )
                            for row in df.values
                        )
                    )
                    
                    # Generate a unique filename
                    table_filename = f"table_{uuid.uuid4()}.html"
                    table_path = os.path.join(output_dir, table_filename)
                    
                    # Save HTML to disk
                    with open(table_path, 'w', encoding='utf-8') as f:
                        f.write(html_table)
                    
                    tables_info.append({
                        "filename": table_filename,
                        "caption": f"Table {i+1}",
                        "html": html_table  # Store the HTML directly in the info
                    })
            except Exception as e:
                print(f"Error processing table {i+1}: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Error extracting tables: {str(e)}")
    
    return tables_info

def summarize_text(text):
    """Summarizes text using the BART model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    summary_ids = model.generate(**inputs, max_length=256, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def create_result_directory():
    """Create a unique directory for storing result files."""
    result_id = str(uuid.uuid4())
    result_dir = os.path.join(app.config['OUTPUT_FOLDER'], result_id)
    os.makedirs(result_dir, exist_ok=True)
    return result_id, result_dir

def process_pdf(filepath, result_id):
    """Process PDF in a separate thread and update status."""
    try:
        processing_status[result_id] = {'status': 'processing', 'progress': 0}
        
        # Get the result directory path
        result_dir = os.path.join(app.config['OUTPUT_FOLDER'], result_id)
        
        # Extract text (20%)
        with open(filepath, 'rb') as pdf_file:
            pdf_bytes = io.BytesIO(pdf_file.read())
            extracted_text = extract_text_from_pdf(pdf_bytes)
        processing_status[result_id]['progress'] = 20
        
        if extracted_text:
            # Generate summary (40%)
            summary = summarize_text(extracted_text)
            processing_status[result_id]['progress'] = 40
            
            # Extract images (60%)
            with open(filepath, 'rb') as pdf_file:
                pdf_bytes = io.BytesIO(pdf_file.read())
                images_info = extract_images_from_pdf(pdf_bytes, result_dir)
            processing_status[result_id]['progress'] = 60
            
            # Extract tables (80%)
            tables_info = extract_tables_from_pdf(filepath, result_dir)
            processing_status[result_id]['progress'] = 80
            
            # Store results with proper file paths
            processing_results[result_id] = {
                'summary': summary,
                'images_info': images_info,
                'tables_info': tables_info,
                'result_id': result_id  # Add result_id to the results for URL generation
            }
            processing_status[result_id]['progress'] = 100
            processing_status[result_id]['status'] = 'completed'
        else:
            processing_status[result_id]['status'] = 'error'
            processing_status[result_id]['error'] = 'Could not extract text from PDF'
            
    except Exception as e:
        processing_status[result_id]['status'] = 'error'
        processing_status[result_id]['error'] = str(e)

@app.route('/status/<result_id>')
def get_status(result_id):
    """Get the current processing status."""
    if result_id not in processing_status:
        return jsonify({'status': 'not_found'})
    return jsonify(processing_status[result_id])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['pdf_file']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            try:
                result_id, result_dir = create_result_directory()
                
                # Store filename in session
                session['filename'] = file.filename
                session['result_id'] = result_id
                
                # Start processing in a separate thread
                thread = Thread(target=process_pdf, args=(filepath, result_id))
                thread.start()
                
                return jsonify({
                    'status': 'processing',
                    'result_id': result_id
                })
                
            except Exception as e:
                flash(f'Error processing PDF: {str(e)}')
                return jsonify({'status': 'error', 'message': str(e)})
        else:
            flash('Please upload a PDF file')
            return redirect(request.url)
            
    return render_template('index.html')

@app.route('/processing/<result_id>')
def processing(result_id):
    """Show the processing page with progress bar."""
    if result_id not in processing_status:
        flash('Invalid processing ID')
        return redirect(url_for('index'))
    return render_template('processing.html', result_id=result_id)

@app.route('/result')
def result():
    if 'result_id' not in session:
        flash('No summary available. Please upload a PDF first.')
        return redirect(url_for('index'))
    
    result_id = session['result_id']
    
    # Check if processing is complete
    if result_id in processing_results:
        result_data = processing_results[result_id]
        return render_template(
            'result.html',
            summary=result_data['summary'],
            filename=session['filename'],
            images=result_data['images_info'],
            tables=result_data['tables_info'],
            result_id=result_id
        )
    else:
        return render_template('processing.html', result_id=result_id)

@app.route('/output/<path:filename>')
def output_files(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True) 