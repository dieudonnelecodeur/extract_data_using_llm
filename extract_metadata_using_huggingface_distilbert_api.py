import os
import cv2
import pytesseract
from openai import OpenAI
import pdfplumber
from pdf2image import convert_from_path
import pandas as pd
import json
from pathlib import Path
from PIL import Image
import re
from pathlib import Path
import json
import openai
from pathlib import Path
from PyPDF2 import PdfReader
from unstructured.partition.pdf import partition_pdf
from tenacity import retry, wait_random_exponential, stop_after_attempt
from huggingface_hub import InferenceClient


# Get values from environment variable
HUGGINGFACE_PROVIDER = os.getenv('HUGGINGFACE_PROVIDER')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

client = InferenceClient(
    provider=HUGGINGFACE_PROVIDER,
    api_key=HUGGINGFACE_API_KEY,
)


# Configuration for Tesseract OCR(On Linux, you may need to set the path to the Tesseract binary in the tesseract_cmd variable)
#pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD', r'/usr/bin/tesseract')
pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD', r'C:/Tools/Tesseract-OCR/tesseract.exe')

# Configuration de Poppler pour pdf2image(On Linux, you may need to set the path to the Poppler binaries in the poppler_path variable)
poppler_path = r"C:\Tools\poppler\Library\bin"

def read_prompt(prompt_path: str):
    """
    Read the prompt for research paper parsing from a text file.
    """
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()
    
def extract_text_from_pdf(pdf_path: str):
    """
    Extract text content from a PDF file using the unstructured library.
    """
    elements = partition_pdf(pdf_path, strategy="hi_res")
    return "\n".join([str(element) for element in elements])

def process_directory_metadata(directory_path: str) -> str:
    """Traite tous les fichiers PDF du répertoire."""
    extracted_data = []
    for pdf_path in directory_path.glob("*.pdf"):
        print(f"Traitement du fichier : {directory_path.name}")
        result = extract_text_from_pdf(directory_path)
        extracted_data.append(result)
    
    # Concatenate all extracted data into a single string
    return "\n".join(extracted_data)

def extract_text_from_directory(directory_path: str) -> str:
    """
    Extract and concatenate text content from all PDF files in a directory using the unstructured library.

    Args:
        directory_path (str): Path to the directory containing PDF files.

    Returns:
        str: A single string containing the concatenated text content of all PDF files.
    """
    directory_path = Path(directory_path)  # Ensure directory_path is a Path object
    all_content = []  # List to store content from all PDFs

    # Iterate through all PDF files in the directory
    for pdf_path in directory_path.glob("*.pdf"):
        try:
            print(f"Extracting text from: {pdf_path.name}")
            elements = partition_pdf(str(pdf_path), strategy="hi_res")
            content = "\n".join([str(element) for element in elements])
            all_content.append(content)  # Append the content to the list
        except Exception as e:
            print(f"Error extracting text from {pdf_path.name}: {e}")

    # Concatenate all content into a single string
    return "\n".join(all_content)


def clean_text(text: str) -> str:
        """Nettoie le texte en supprimant les caractères spéciaux, les espaces inutiles et les retours à la ligne."""
        text = text.replace("\n", " ").replace("\x0c", " ")
        return " ".join(text.split())


@retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    try:
        return  client.question_answering(**kwargs)
    
    except Exception as e:
        print(f"Error during API call: {e}")
        raise


def extract_metadata(content: str, prompt: str)-> dict:
    """
    Use GPT model to extract metadata from the research paper content based on the given prompt.
    """

    # Read the prompt
    print(f"Reading prompt from {prompt}")
  
    prompt_data = clean_text(read_prompt(prompt))
    
    try:
        # Call the Bert model to extract metadata
        response = completion_with_backoff(
            question=prompt_data,
            context=content,
            model="distilbert/distilbert-base-cased-distilled-squad",
        )
        
        print(f"Response from the model: {response}")

        response_content = response.choices[0].message.content
        if not response_content:
            print("Empty response from the model")
            return {}

        # Remove any markdown code block indicators
        response_content = re.sub(r'```json\s*', '', response_content)
        response_content = re.sub(r'\s*```', '', response_content)
        
        print(f"Response content: {response_content}")

        # Attempt to parse JSON into dictionary
        try:
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Raw response: {response_content}")

            # Attempt to extract JSON from the response
            match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError as jde:
                    print(f"Failed to extract valid JSON from the response: {jde}")

            return {}

    except Exception as e:
        print(f"Error during extract metadata: {e}")
        return {}


def process_files_content(content: str, prompt_path: str, output_folder: str):
    """
    Process files contente through the entire pipeline.
    """
    print("Processing files contentes...")
    file_name = "output"

    try:
        # Ensure the extracted content is encoded in UTF-8
        content = content.encode('utf-8', errors='ignore').decode('utf-8')

        # Step 1: Read the prompt
        prompt = clean_text(read_prompt(prompt_path))
        print(f"Cleaned Prompt: {prompt}")

        # Step 2: Extract metadata using the model
        metadata = extract_metadata(content, prompt)
        if not metadata or not isinstance(metadata, dict):
            print("Failed to extract metadata or invalid metadata format.")
            return
        print("Metadata extracted successfully.")

        # Step 3: Save the result as a JSON file
        output_filename_json = file_name + '.json'
        output_path_json = os.path.join(output_folder, output_filename_json)

        # Save metadata as JSON
        with open(output_path_json, 'w', encoding='utf-8') as json_file:
            json.dump(metadata, json_file, indent=2, ensure_ascii=False)
        print(f"Metadata saved as JSON to {output_path_json}")

        # Step 4: Save the result as a CSV file
        output_filename_csv = file_name + '.csv'
        output_path_csv = os.path.join(output_folder, output_filename_csv)

        # Save metadata as CSV
        if isinstance(metadata, dict):
            df = pd.DataFrame([metadata])  # Wrap metadata in a list to create a DataFrame
        else:
            print("Metadata is not a valid dictionary for CSV export.")
            return

        df.to_csv(output_path_csv, index=False, encoding='utf-8')
        print(f"Metadata saved as CSV to {output_path_csv}")

    except Exception as e:
        print(f"Error processing paper: {e}")
        

def process_directory(prompt_path: str, directory_path: str, output_folder: str):
    """
    Process all PDF files in the given directory.
    """
    directory_path = Path(directory_path)  # Ensure directory_path is a Path object
    output_folder = Path(output_folder)  # Ensure output_folder is a Path object

    # Create the output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    content =""
    # Iterate through all PDF files in the directory
    for pdf_path in directory_path.glob("*.pdf"):
        content=extract_text_from_directory(directory_path)
        
    process_files_content(content, prompt_path, str(output_folder))

            
# Define paths
prompt_path = "./PROMPTS/prompt.txt"
directory_path = "./INPUT"
output_folder = "./OUTPUT/extracted_metadata"

# This test huggingface API call
# extract_metadata(clean_text(extract_text_from_directory(directory_path)),prompt_path)

process_directory(prompt_path, directory_path, output_folder)