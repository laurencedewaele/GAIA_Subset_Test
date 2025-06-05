# Standard library imports
import warnings
from datetime import datetime
import requests


# Google-related imports
from google.genai import types

# OpenAI
import openai

# Langchain and LangGraph imports
from langchain_core.tools import tool
from langchain_community.document_loaders import UnstructuredExcelLoader

# Suppress warnings
warnings.filterwarnings("ignore")

@tool
def ask_youtube_video(youtube_url: str, question: str) -> str:
    """Provides answers to a question relating to the content of a YouTube video.
    Args:
        youtube_url: The URL of the YouTube video (string).
        question: The question to answer (string).
    Returns:
        string: The answer to the question.
    """
    prompt = """You are an expert video analyst.
                Make a sufficiently detailed description to then be able to answer the question: """ + question + \
             """Make a very precise and detailed description of the content
                appearing in the images, and especially its evolution.
                For example, you may notice: we see dogs and cats, there are German Shepherds and poodles,
                Siamese and Persians, then they are joined by a Malinois, then the poodle leaves.
                Pay close attention to what is simultaneously visible during the video.""" + \
                """Then, when you have all the elements, answer the quetion.
                Return only your answer, with this prefix: FINAL ANSWER:"""
    response = client_google.models.generate_content(
    model='models/gemini-2.0-flash',
    contents=types.Content(
        parts=[
            types.Part(
                file_data=types.FileData(file_uri=youtube_url)
            ),
            types.Part(text=prompt)
        ]))

    return response.text

@tool
def get_chessboard_description(image_url: str) -> str:
    """ Get a textual description of a chessboard image.
    Args:
        image: The URL of the chessboard image (string).
    Returns:
        string: The textual description of the chessboard.
    """
    # Get image from its URL
    image_bytes = requests.get(image_url).content
    image = types.Part.from_bytes(
    data=image_bytes, mime_type="image/jpeg"
    )

    question = """You are an expert chess assistant.
    An chessboard image is attached. You have to locate all the pieces of each camp on the chessboard.
    To locate the positions on the chessboard, take into account the notation used in the image.
    Explains how the squares of the chessboard are indexed, in a neutral way, without being from the point of view
    of one of the two camps. Specifies the reading direction (right/left/top/bottom) of the indexing.
    Then locate all the pieces of the 2 camps using the same perspective to index it,
    don't take into account the camp perspective :
    if two pieces, one from each side, are visually in the same column then they must have the same abscissa, and
    if they are visually on the same row then they must have the same ordinate.
    Give a synthetic response, with for each piece its color, name, position.
    Ensure to have located all the pieces with good indexing."""
    config = types.GenerateContentConfig(
        temperature=0.,          # Contrôle la créativité de la réponse
        max_output_tokens=500,    # Nombre maximum de tokens dans la réponse
    #    top_p=1,                # Nucleus sampling
    #    top_k=1                  # Nombre de tokens à considérer pour le sampling
    )
    #client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client_google.models.generate_content(
        model="gemini-2.0-flash",
        contents=[question, image],
        config=config
    )

    return response.text

@tool
def transcript_audio(audio_url: str) -> str:
    """Provides transcription of an audio file URL.
    Args:
        audio_url: The URL of the audio file (string).
    Returns:
        string: The transcript.
    """
    local_filename = "audio.mp3"

    response = requests.get(audio_url)
    if response.status_code == 200:
        with open(local_filename, 'wb') as f:
            f.write(response.content)

        with open(local_filename, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        return transcript.text

    else:
        return f"Failed to download the audio file. HTTP status code: {response.status_code}"

@tool
def get_file_content(file_url: str) -> str:
    """Fetch file content from its URL and return it as a string.
    Args:
        file_url: The URL of the file (string).
    Returns:
        string: The content of the file.
    """
    try:
        response = requests.get(file_url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Failed to fetch the content file: {e}"


@tool
def get_excel_file_content(file_url: str) -> str:
    """
    Returns the content of an Excel file.
    Args:
        file_url (str): The URL of the Excel file.
    Returns:
        str: The content of the Excel file with: sheet name, page content.
    """

    loader = UnstructuredExcelLoader(file_url, mode="elements")
    docs = loader.load()
    output = ''
    for d in docs:
        output += "Sheet: "+d.metadata['page_name']+" Content: "+d.page_content+"\n"

    return output

@tool
def list_addition(nbs: list) -> str:
    """
    Returns the sum of a list of numbers.
    Args:
        nbs (list): A list of numbers.
    Returns:
        str: The sum of the numbers.
    """
    return str(sum(nbs))