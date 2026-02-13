# Chat With PDF File

Chat With PDF File is a Streamlit-based application that allows users to chat with a conversational AI model trained on PDF documents. The chatbot extracts information from uploaded PDF files and answers user questions using the **Groq API**.
<https://chat-with-pdf-file.streamlit.app/>


## Features

- **PDF Upload:** Users can upload multiple PDF files.
- **Text Extraction:** Extracts text from uploaded PDF files.
- **Conversational AI:** Uses the **Groq** conversational AI model to answer user questions.
- **Chat Interface:** Provides a chat interface to interact with the chatbot.

## Get Started

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (for dependency management)

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/gsamansharma/chatwithpdf.git
   cd chatwithpdf
   ```

2. **Install Dependencies:**

   ```bash
   uv sync
   ```

3. **Set up Environment Variables:**
   - Create a `.env` file in the root directory.
   - Add your Groq API key and Model.

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=llama-3.1-8b-instant
   ```
   > You can get your API key from [Groq Console](https://console.groq.com/).

4. **Run the Application:**

   ```bash
   uv run streamlit run app.py
   ```

5. **Usage:**
   - Upload PDF files using the sidebar.
   - Click "Submit & Process".
   - Chat with your documents!

## Project Structure

- `app.py`: Main application script.
- `.env`: Environment variables.
- `pyproject.toml`: Project configuration and dependencies.
- `README.md`: Documentation.

## Dependencies

Managed via `uv` in `pyproject.toml`. Key dependencies include:
- `streamlit`
- `langchain`
- `langchain-groq`
- `chromadb`
- `faiss-cpu`
- `sentence-transformers`

## Acknowledgments

- [Groq](https://groq.com/): For providing the underlying language model API.
- [Streamlit](https://streamlit.io/): For the user interface framework.
- [LangChain](https://langchain.com/): For the LLM orchestration.
