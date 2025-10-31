# AI Research Agent

A LangChain-based AI research assistant that uses multiple tools to search the web, query Wikipedia, and save research results to files.

## Features

- 🔍 **Web Search**: Uses DuckDuckGo to search for real-time information
- 📚 **Wikipedia Integration**: Queries Wikipedia for detailed information
- 💾 **File Saving**: Automatically saves research results to files
- 🤖 **Structured Output**: Returns well-formatted research responses with topics, summaries, sources, and tools used

## Prerequisites

- Python 3.13 or higher
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd "AI Agent Tutorial"
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

Run the agent:
```bash
python3 main.py
```

Enter your research query when prompted. The agent will:
1. Search for relevant information using available tools
2. Compile a structured research response
3. Optionally save results to a file if requested

### Example Queries

- "What is the capital of France?"
- "Tell me about great white sharks"
- "Research southeast Asia countries best for software development and save to a file"

## Project Structure

```
AI Agent Tutorial/
├── main.py              # Main application entry point
├── tools.py             # Tool definitions (search, Wikipedia, file saving)
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (not tracked)
├── .gitignore          # Git ignore patterns
└── README.md           # This file
```

## Tools Available

- **Search**: Web search using DuckDuckGo
- **Wikipedia**: Query Wikipedia articles
- **save_to_file**: Save research results to output.txt

## Technologies Used

- [LangChain](https://www.langchain.com/) - LLM application framework
- [OpenAI GPT-4o-mini](https://openai.com/) - Language model
- [DuckDuckGo Search](https://duckduckgo.com/) - Web search
- [Wikipedia API](https://www.mediawiki.org/wiki/API) - Knowledge base
- [Pydantic](https://docs.pydantic.dev/) - Data validation

## License

MIT License

