import os
import pandas as pd
import xml.etree.ElementTree as ET

# Define the base directory where papers are stored
base_dir = "Papers" 

# Initialize list to store data
data = []

# Iterate through each paper directory
for paper_id in os.listdir(base_dir):
    paper_path = os.path.join(base_dir, paper_id)

    if os.path.isdir(paper_path):  # Ensure it's a directory
        xml_text, summary_text = "", ""

        # Extract text from XML file
        xml_folder = os.path.join(paper_path, "Documents_xml")
        if os.path.exists(xml_folder):
            for file in os.listdir(xml_folder):
                if file.endswith(".xml"):
                    xml_path = os.path.join(xml_folder, file)
                    try:
                        tree = ET.parse(xml_path)
                        root = tree.getroot()
                        xml_text = " ".join([elem.text for elem in root.iter() if elem.text])  # Extract all text
                    except Exception as e:
                        print(f"Error parsing XML {xml_path}: {e}")

        # Extract text from summary file
        summary_folder = os.path.join(paper_path, "summary")
        if os.path.exists(summary_folder):
            for file in os.listdir(summary_folder):
                if file.endswith(".txt"):  # Assuming summary files are text files
                    summary_path = os.path.join(summary_folder, file)
                    try:
                        with open(summary_path, "r", encoding="utf-8") as f:
                            summary_text = f.read().strip()
                    except Exception as e:
                        print(f"Error reading summary {summary_path}: {e}")

        # Append extracted data 
        if xml_text or summary_text:
            data.append([paper_id, xml_text, summary_text])

# Create DataFrame
df = pd.DataFrame(data, columns=["Paper_ID", "XML_Text", "Summary_Text"])

# Save to CSV
csv_path = "extracted_data.csv"
df.to_csv(csv_path, index=False, encoding="utf-8")

print(f"CSV file saved at: {csv_path}")
