import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wikipedia_tool

# Load environment variables
load_dotenv()

# Define response structure
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Configure Streamlit page
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Setup LLM and parser
@st.cache_resource
def setup_agent():
    llm = ChatOpenAI(model="gpt-4o-mini")
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system", 
                """
                You are a research assistant that will help generate a research paper.
                Answer the user query and use necessary tools to get the information.
                After all tool operations are complete, wrap the final output in this format:\n{format_instructions}
                """,
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())
    
    tools = [search_tool, wikipedia_tool]
    agent = create_tool_calling_agent(
        llm=llm, 
        prompt=prompt,
        tools=tools
    )
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
    
    return agent_executor, parser

agent_executor, parser = setup_agent()

# Header
st.title("🔍 AI Research Agent")
st.markdown("Ask me anything and I'll research it for you using web search and Wikipedia!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Display structured response
            data = message["content"]
            st.markdown(f"**Topic:** {data['topic']}")
            st.markdown(f"**Summary:**\n{data['summary']}")
            
            if data['sources']:
                st.markdown("**Sources:**")
                for source in data['sources']:
                    st.markdown(f"- {source}")
            
            if data['tools_used']:
                st.markdown(f"**Tools Used:** {', '.join(data['tools_used'])}")
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Researching..."):
            try:
                # Invoke agent
                raw_response = agent_executor.invoke({"query": prompt})
                
                # Parse response
                structured_response = parser.parse(raw_response["output"])
                
                # Display structured response
                st.markdown(f"**Topic:** {structured_response.topic}")
                st.markdown(f"**Summary:**\n{structured_response.summary}")
                
                if structured_response.sources:
                    st.markdown("**Sources:**")
                    for source in structured_response.sources:
                        st.markdown(f"- {source}")
                
                if structured_response.tools_used:
                    st.markdown(f"**Tools Used:** {', '.join(structured_response.tools_used)}")
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": {
                        "topic": structured_response.topic,
                        "summary": structured_response.summary,
                        "sources": structured_response.sources,
                        "tools_used": structured_response.tools_used
                    }
                })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.markdown("**Raw Response:**")
                st.json(raw_response)

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This AI Research Agent uses:
    - 🔍 **DuckDuckGo Search** for web information
    - 📚 **Wikipedia** for detailed knowledge
    - 🤖 **GPT-4o-mini** for intelligent responses
    
    Your chat history persists during this session.
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("💡 **Tip:** Ask detailed questions for better research results!")

