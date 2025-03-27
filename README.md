# extract_data_using_llm

Extraction automatique d'informations de documents non structurés

# Extract Metadata Using Hugging Face DistilBERT API or Azure OpenAI

Ce projet vise à développer des méthodes pour extraire automatiquement des informations à partir de documents non structurés, tels que des fichiers PDF, et les convertir en formats structurés comme des tables CSV. Le traitement automatique des langues (TAL) est un domaine de recherche pluridisciplinaire à l'intersection de la linguistique et de l'informatique, qui vise à mettre au point des programmes informatiques pour l'analyse des langues par ordinateur. L'extraction d'informations de documents non structurés implique des défis tels que la reconnaissance optique de caractères (OCR), la gestion de la mise en page complexe des documents et l'interprétation du contenu textuel. Le projet se concentrera sur le développement de techniques d'OCR robustes, l'analyse syntaxique et sémantique du texte extrait, et la structuration des données en formats exploitables pour des applications ultérieures.

Ce projet vise à :

    Développer des méthodes pour extraire automatiquement des informations à partir de documents non structurés (PDF).
    Convertir ces informations en formats structurés (JSON).
    Structurer les données en formats exploitables.

## Features
- Extract text content from PDFs using the `unstructured` library.
- Perform metadata extraction using Hugging Face's `distilbert-base-cased-distilled-squad` model.
- Save extracted metadata as JSON and CSV files.
- Process multiple PDFs in a directory.

## Requirements
- Python 3.8 or higher max 3.12
- Hugging Face Inference API key
- Poppler for PDF-to-image conversion
- Tesseract OCR for image-based text extraction

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/extract-metadata.git
   cd extract-metadata
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Poppler:
   - On Windows, download Poppler from Poppler for Windows and add the bin folder to your system's PATH.
   - On Linux:
     ```bash
     sudo apt-get install tesseract-ocr
     ```

4. Install Tesseract OCR:
   - On Windows, download Tesseract from Tesseract OCR and add it to your system's PATH.
   - On Linux:
     ```bash
     sudo apt-get install tesseract-ocr
     ```

Usage:

1- Place your PDF files in the INPUT directory.
2- Create a text file containing your prompt in the PROMPTS directory (e.g., prompt.txt).
3- Run the script:

`python extract_metadata_using_huggingface_distilbert_api.py` or

`python extract_metadata_using_azure_openai_api.py`
