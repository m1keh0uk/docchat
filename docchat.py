import readline
import groq
from dotenv import load_dotenv
import pprint
    
load_dotenv()

def llm(messages, temperature=1):

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
    )
    print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content


def split_document_into_chunks(text, max_words=5, overlap=2):
    """
    Splits a document into word-based chunks with overlap.

    Parameters:
        text (str): The input document string.
        max_words (int): Max words per chunk.
        overlap (int): Number of words to overlap between chunks.

    Returns:
        List[str]: List of chunked text strings.

    Example:
        >>> text = "The quick brown fox jumps over the lazy dog in the backyard"
        >>> split_document_into_chunks(text, max_words=5, overlap=2)
        ['The quick brown fox jumps', 'fox jumps over the lazy', 'the lazy dog in the', 'in the backyard']
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap
    return chunks

def score_chunk_relevance(chunk: str, user_query: str) -> float:
    """
    Scores relevance of a chunk to the user query using Jaccard similarity.

    Parameters:
        chunk (str): A chunk of text from the document.
        user_query (str): The user's query.

    Returns:
        float: Relevance score between 0 and 1.

    Example:
        >>> score_chunk_relevance("The cat sat on the mat", "Where did the cat sit?")
        0.25

        >>> score_chunk_relevance("Photosynthesis occurs in chloroplasts", "Where does photosynthesis happen?")
        0.2

        >>> score_chunk_relevance("The mitochondria is the powerhouse of the cell", "What is the powerhouse of the cell?")
        0.5

        >>> score_chunk_relevance("", "What is this?")
        0.0

        >>> score_chunk_relevance("Just a bunch of random words", "")
        0.0

        >>> score_chunk_relevance("Apples and oranges are fruits", "Fruits like apples and oranges")
        0.6
    """
    # Tokenize and lowercase words
    chunk_words = set(chunk.lower().split())
    query_words = set(user_query.lower().split())

    # Jaccard similarity: intersection over union
    if not chunk_words or not query_words:
        return 0.0

    intersection = chunk_words & query_words
    union = chunk_words | query_words

    return (len(intersection) / len(union))




if __name__ == '__main__':

    client = groq.Groq()  

    # api_key=os.environ.get("GROQ_API_KEY"),  # This is the default and can be omitted
    messages = []
    messages.append(
            {
                "role": "system",
                "content": "You are a practical scientist, i need your honest opinion. you dont care about offending people."
            }
        )

    while True:
        text = input('docchat>')
        messages.append({
            'role': 'user',
            'content': text,
        })

        result = llm(messages)
        # FIXME
        #add assistant so that llm has access to whole convo history, and will know what it has previously said


        # print('result=', result)
        # pprint.pprint(messages)