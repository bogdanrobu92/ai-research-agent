from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from datetime import datetime

def save_to_file(data: str, filename: str = "output.txt"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    formatted_text = f"Research Result - {timestamp}\n{data}\n"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    return f"Result saved to {filename}"


save_to_file_tool = Tool(
    name="save_to_file",
    func=save_to_file,
    description="Use this tool to save research results, summaries, or any text data to a file. Input should be the text content to save. Returns confirmation of the saved file location."
)


search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="Search",
    func=search.run,
    description="Search the web for information"
)
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)