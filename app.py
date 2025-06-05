import os
import gradio as gr
import requests
import pandas as pd
import requests

# Google-related imports
from google import genai
from google.genai import types

# OpenAI
import openai

# Langchain and LangGraph imports
from langchain.schema import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
OPENAI_API_KEY = os.getenv('NEW_OPENAI_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

# Custom tools
@tool
def ask_youtube_video(youtube_url: str, question: str) -> str:
    """Provides answers to a question relating to the content of a YouTube video.
    Args:
        youtube_url: The URL of the YouTube video (string).
        question: The question to answer (string).
    Returns:
        string: The answer to the question.
    """
    client_google = genai.Client(api_key=GOOGLE_API_KEY)
    prompt = """You are an expert video analyst.
                Make a sufficiently detailed description to then be able to answer the question: """ + question + \
             """Make a very precise and detailed description of the content
                appearing in the images, and especially its evolution.
                For example, you may notice: we see dogs and cats, there are German Shepherds and poodles,
                Siamese and Persians, then they are joined by a Malinois, then the poodle leaves.
                Pay close attention to what is simultaneously visible during the video.""" + \
                """Then, when you have all the elements, answer the quetion.
                Return only your answer, with this prefix: FINAL ANSWER:"""
    config = types.GenerateContentConfig(
                        temperature=0.,          # Contrôle la créativité de la réponse
 #                       max_output_tokens=500,    # Nombre maximum de tokens dans la réponse
 #                       top_p=1,                # Nucleus sampling
 #                       top_k=1                  # Nombre de tokens à considérer pour le sampling
                    )
    response = client_google.models.generate_content(
                        model='models/gemini-2.5-flash-preview-05-20',# 'models/gemini-2.0-flash',
                        contents=types.Content(
                            parts=[
                                types.Part(
                                    file_data=types.FileData(file_uri=youtube_url)
                                ),
                                types.Part(text=prompt)
                            ]),
                        config=config)

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
    client_google = genai.Client(api_key=GOOGLE_API_KEY)
    question = """You are an expert chess assistant.
    An chessboard image is attached. You have to locate all the pieces of each camp on the chessboard.
    To locate the positions on the chessboard, take into account the notation used in the image.
    Explains how the squares of the chessboard are indexed, in a neutral way, without being from the point of view
    of one of the two camps. Specifies the reading direction (right/left/top/bottom) of the indexing.
    Pay very close attention to the indexing of the chessboard as well as its reading direction.
    Then locate all the pieces of the 2 camps using the same perspective to index it,
    don't take into account the camp perspective :
    if two pieces, one from each side, are visually in the same column then they must have the same abscissa, and
    if they are visually on the same row then they must have the same ordinate.
    To give the position of each piece, be very careful to do so using the chessboard indexing you just described.
    Remember to use the same point of view for both sides, do not turn the chessboard over.
    Give a synthetic response, with for each piece its color, name, position.
    Ensure to have located all the pieces with good indexing."""

    config = types.GenerateContentConfig(
        temperature=0.,          # Contrôle la créativité de la réponse
    #    max_output_tokens=500,    # Nombre maximum de tokens dans la réponse
    #    top_p=1,                # Nucleus sampling
    #    top_k=1                  # Nombre de tokens à considérer pour le sampling
    )

    response = client_google.models.generate_content(
        model="models/gemini-2.0-flash",
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
def sum_numbers(nbs: list) -> str:
    """
    This tool takes a list of numbers and returns their sum.
    You MUST use this tool for any addition task, no matter how simple.
    Args:
        nbs (list): A list of numbers.
    Returns:
        str: The sum of the numbers.
    """
    return str(sum(nbs))
##

# --- Gaia Agent Definition ---
class GaiaAgent:
    def __init__(self, chat, tools, system_prompt):
        self.agent = create_react_agent(
                    model=chat,
                    tools=tools,
                    state_modifier=system_prompt
                )
        print("GaiaAgent initialized.")

    def __call__(self, question: str, file_url: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        if file_url is None:
            messages = [HumanMessage(content=question)]
        else:
            messages = [HumanMessage(content=question + ' Additionnal information: '+file_url)]

        messages = self.agent.invoke({"messages": messages})

        response = messages["messages"][-1].content
        print("Response: ", response)

        _, _, result = response.partition("FINAL ANSWER:")

        try:
            fixed_answer = result.replace("*", "").strip()
        except Exception as e:
            fixed_answer = f"Error extracting fixed_answer: {e}"

        print(f"Agent returning fixed answer: {fixed_answer}")
        return fixed_answer

def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the GaiaAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    with open('./utils/prompt.txt', 'r', encoding='utf-8') as f:
        prompt = f.read()

    # Generate the chat interface, including the tools
    model = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        huggingfacehub_api_token=HF_TOKEN,
    )

    chat = ChatHuggingFace(llm=model, verbose=True, temperature=0, top_k=1,
    top_p = 1.0, do_sample=False, seed=42)

    web_search_tool = TavilySearch(max_results=5, time_range=None, topic="general", search_depth='advanced')

    api_wrapper = WikipediaAPIWrapper(top_k_results=5) #, doc_content_chars_max=3000)
    wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

    system_prompt = SystemMessage(content=prompt)

    tools = [wikipedia_tool, web_search_tool, ask_youtube_video, \
             get_chessboard_description, transcript_audio, get_file_content, \
             get_excel_file_content, sum_numbers]

    # 1. Instantiate Agent
    try:
        agent = GaiaAgent(chat, tools, system_prompt)
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        print("ITEM: ",item)
        task_id = item.get("task_id")
        question_text = item.get("question")
        if item.get("file_name") == '':
            file_url = None
        else:
            file_url = f"{DEFAULT_API_URL}/files/{task_id}"
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            nb_try = 0
            while nb_try <= 3:
                try:
                    submitted_answer = agent(question_text, file_url)
                    if submitted_answer:
                        nb_try = 4
                except Exception as e:
                    submitted_answer = str(e)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df

# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Gaia Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Gaia Agent Evaluation...")
    demo.launch(debug=True, share=False)