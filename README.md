# Research Paper Summarizer

A web application that uses the pre-trained BART-CNN model to generate summaries of research papers from PDF files. It also extracts figures, diagrams, and tables from the PDFs.

## Features

- Upload research papers in PDF format
- Extract text from PDF documents
- Generate concise summaries using the BART-CNN model
- Extract and display figures, diagrams, and charts from the PDF
- Extract and display tables using tabula-py
- Modern and responsive user interface
- Copy summary to clipboard with a single click

## Setup and Installation

1. Clone this repository or download the source code

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Java Runtime Environment (JRE) is required for tabula-py:
   - Make sure you have Java installed on your system
   - For Windows: Download and install from https://www.java.com/download/
   - For Linux: `sudo apt install default-jre`
   - For macOS: `brew install java`

4. Run the Flask application:
   ```
   python web_app.py
   ```

5. Open your web browser and navigate to `http://127.0.0.1:5000/`

## How to Use

1. Click on "Choose a PDF file" or drag and drop a PDF file containing a research paper
2. Click "Generate Summary" to process the file
3. View the generated summary, extracted figures, and tables on the results page
4. Use the "Copy Summary" button to copy the text to your clipboard

## Technical Details

- **Backend**: Flask web framework
- **Model**: Pre-trained BART-CNN model from Facebook/Meta
- **PDF Processing**: 
  - PyMuPDF for text and image extraction
  - tabula-py for table extraction
- **Frontend**: HTML, CSS, and JavaScript

## Requirements

- Python 3.8 or higher
- PyTorch 2.0.0 or higher
- Transformers 4.30.0 or higher
- Flask 2.3.3
- PyMuPDF 1.23.8
- tabula-py 2.7.0
- Java Runtime Environment (JRE) 