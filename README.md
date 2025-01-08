
# ACLBot

ACLBot is a web-based tool designed to assist users with information and guidance related to ACL (Anterior Cruciate Ligament) health and recovery. This application allows users to interact with a chatbot that can provide detailed responses about ACL, using three different AI models: Mistral AI, and Llama 3.1 and gemma2. Users can ask questions, get responses, and listen to the responses through audio.

## Features

- Chat with AI: Ask questions related to ACL, and the AI provides clear, easy-to-understand responses.
- Voice Responses: The AI's answers are also converted into speech, so you can listen to the information instead of just reading it.
- Choose Your AI Model: Users can select from Google Mistral AI, Llama 3.1, or gemma2 to get
- PDF Analysis: The chatbot is equipped to analyze and extract information from PDFs containing ACL-related content. It uses the provided PDFs as its source of knowledge.


## How It Works
1. Select AI Model: In the sidebar, choose the AI model you want to use:
2. Google Gemini: Requires a Google API Key.
Mistral AI or Llama 3.1: Both require a Groq API Key.

3. Enter API Key: Input the required API key in the sidebar for the selected AI model.
4. Ask Your Question: Type your question in the chat input area.
5. Listen to the Response: The AI provides a detailed answer and also converts the response into audio that you can play directly in the app.
6. Analyze PDFs: The app can process PDFs stored in a specified folder and use the information from them to provide accurate responses. It ensures all answers are backed by the source document, mentioning the source and page number.
## Requirements

To use this app, you need the following:

- Google API Key (for Google Gemini model).
- Groq API Key (for Mistral AI or Llama 3.1 models).
- PDF files containing ACL-related content in a specific folder.
- Access to a web browser to run the Streamlit application.
## Setup and Installation

1. Clone the Repository:
## Deployment

1. To deploy this project run


```bash
git clone https://github.com/Abdul-Basit7/ACLBot
cd <ACLBot>

```

2. Install Dependencies: Make sure you have Python installed, then run:

```bash
pip install -r requirements.txt
 ```

3. Set Up Environment Variables: Create a .env file in the root directory with the following variables:
```bash
QDRANT_URL=<your_qdrant_url>
QDRANT_API_KEY=<your_qdrant_api_key>
PDF_PATH=<path_to_your_pdf_directory>

```
4.Run the Application:
```bash
streamlit run app.py

```
## Troubleshooting


- Missing API Key: If you don’t provide an API key for the selected AI model, the app will prompt you to enter one.
- PDF Not Loaded: Ensure that the PDF files are in the correct folder path specified in the .env file.
- Slow Responses: The speed of the AI responses can depend on the size of the PDF files and the complexity of the question..

