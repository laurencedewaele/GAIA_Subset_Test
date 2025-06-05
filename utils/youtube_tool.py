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
    config = types.GenerateContentConfig(temperature=0.)
    response = client_google.models.generate_content(
                        model='models/gemini-2.5-flash-preview-05-20',
                        contents=types.Content(
                            parts=[
                                types.Part(
                                    file_data=types.FileData(file_uri=youtube_url)
                                ),
                                types.Part(text=prompt)
                            ]),
                        config=config)

    return response.text