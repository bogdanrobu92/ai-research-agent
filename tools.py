from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
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

# Configure DuckDuckGo search with limits for faster results
ddg_search = DuckDuckGoSearchAPIWrapper(max_results=3, region="wt-wt", time="y")
search_tool = Tool(
    name="Search",
    func=ddg_search.run,
    description="Search the web for current information. Returns quick summaries from top search results."
)

# Configure Wikipedia with more content but still limited
api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=500,
    load_all_available_meta=False
)
wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper)