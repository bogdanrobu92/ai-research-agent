import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.callbacks.base import BaseCallbackHandler
from tools import search_tool, wikipedia_tool
import time

# Load environment variables
load_dotenv()

# Define response structure
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Custom callback handler to track tool execution times
class TimingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.tool_times = {}
        self.current_tool = None
        self.current_tool_start = None
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "Unknown")
        self.current_tool = tool_name
        self.current_tool_start = time.time()
    
    def on_tool_end(self, output, **kwargs):
        if self.current_tool and self.current_tool_start:
            elapsed = time.time() - self.current_tool_start
            if self.current_tool not in self.tool_times:
                self.tool_times[self.current_tool] = []
            self.tool_times[self.current_tool].append(elapsed)
        self.current_tool = None
        self.current_tool_start = None

# Configure Streamlit page
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔍",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# AI Research Agent\nBuilt with LangChain and Streamlit"
    }
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Setup LLM and parser
@st.cache_resource
def setup_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system", 
                """
                You are a research assistant. Answer concisely using available tools.
                
                IMPORTANT RULES:
                - Use Search tool ONCE for web information
                - Use Wikipedia tool ONCE if needed for detailed context
                - Do NOT call the same tool multiple times
                - Be efficient and direct
                
                After gathering information, format your response as:\n{format_instructions}
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
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=False,
        max_iterations=5,  # Limit iterations to prevent overthinking
        early_stopping_method="generate"  # Stop early if possible
    )
    
    return agent_executor, parser

agent_executor, parser = setup_agent()

# Header
st.title("🔍 AI Research Agent")
st.markdown("Ask me anything and I'll research it for you using web search and Wikipedia!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            # Check if content is structured or plain text
            content = message["content"]
            if isinstance(content, dict):
                # Display structured response
                st.markdown(f"**Topic:** {content['topic']}")
                st.markdown(f"**Summary:**\n{content['summary']}")
                
                if content['sources']:
                    st.markdown("**Sources:**")
                    for source in content['sources']:
                        st.markdown(f"- {source}")
                
                if content['tools_used']:
                    st.markdown(f"**Tools Used:** {', '.join(content['tools_used'])}")
                
                # Display timing breakdown if available
                if 'timing_breakdown' in content:
                    breakdown = content['timing_breakdown']
                    st.markdown("---")
                    st.markdown("**⏱️ Performance Breakdown:**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if breakdown.get('tool_times'):
                            for tool_name, times in breakdown['tool_times'].items():
                                total_time = sum(times)
                                calls = len(times)
                                st.caption(f"🔧 {tool_name}: {total_time:.2f}s ({calls} call{'s' if calls > 1 else ''})")
                        
                        if 'llm_time' in breakdown:
                            st.caption(f"🤖 LLM Processing: {breakdown['llm_time']:.2f}s")
                    
                    with col2:
                        if 'parse_time' in breakdown:
                            st.caption(f"📊 Parsing: {breakdown['parse_time']:.3f}s")
                        if 'total_time' in breakdown:
                            st.caption(f"⏱️ **Total: {breakdown['total_time']:.2f}s**")
                elif 'execution_time' in content:
                    # Fallback for old messages without breakdown
                    exec_time = content['execution_time']
                    if exec_time < 60:
                        st.caption(f"⏱️ Response time: {exec_time:.2f} seconds")
                    else:
                        minutes = int(exec_time // 60)
                        seconds = exec_time % 60
                        st.caption(f"⏱️ Response time: {minutes}m {seconds:.2f}s")
            else:
                # Display plain text response
                st.markdown(content)
                
                # Check if execution_time was stored separately for plain text
                if 'execution_time' in message:
                    exec_time = message['execution_time']
                    if exec_time < 60:
                        st.caption(f"⏱️ Response time: {exec_time:.2f} seconds")
                    else:
                        minutes = int(exec_time // 60)
                        seconds = exec_time % 60
                        st.caption(f"⏱️ Response time: {minutes}m {seconds:.2f}s")
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
            # Start timing
            start_time = time.time()
            
            # Create timing callback
            timing_callback = TimingCallbackHandler()
            
            try:
                # Invoke agent with callback
                invoke_start = time.time()
                raw_response = agent_executor.invoke(
                    {"query": prompt},
                    {"callbacks": [timing_callback]}
                )
                agent_time = time.time() - invoke_start
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Try to parse as structured response
                try:
                    parse_start = time.time()
                    structured_response = parser.parse(raw_response["output"])
                    parse_time = time.time() - parse_start
                    
                    # Display structured response
                    st.markdown(f"**Topic:** {structured_response.topic}")
                    st.markdown(f"**Summary:**\n{structured_response.summary}")
                    
                    if structured_response.sources:
                        st.markdown("**Sources:**")
                        for source in structured_response.sources:
                            st.markdown(f"- {source}")
                    
                    if structured_response.tools_used:
                        st.markdown(f"**Tools Used:** {', '.join(structured_response.tools_used)}")
                    
                    # Calculate time breakdown
                    total_tool_time = sum(sum(times) for times in timing_callback.tool_times.values())
                    llm_processing_time = agent_time - total_tool_time
                    
                    # Display detailed timing breakdown
                    st.markdown("---")
                    st.markdown("**⏱️ Performance Breakdown:**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if timing_callback.tool_times:
                            for tool_name, times in timing_callback.tool_times.items():
                                total_time = sum(times)
                                calls = len(times)
                                st.caption(f"🔧 {tool_name}: {total_time:.2f}s ({calls} call{'s' if calls > 1 else ''})")
                        
                        st.caption(f"🤖 LLM Processing: {llm_processing_time:.2f}s")
                    
                    with col2:
                        st.caption(f"📊 Parsing: {parse_time:.3f}s")
                        st.caption(f"⏱️ **Total: {execution_time:.2f}s**")
                    
                    # Add assistant message to chat history with timing details
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": {
                            "topic": structured_response.topic,
                            "summary": structured_response.summary,
                            "sources": structured_response.sources,
                            "tools_used": structured_response.tools_used,
                            "execution_time": execution_time,
                            "timing_breakdown": {
                                "tool_times": dict(timing_callback.tool_times),
                                "llm_time": llm_processing_time,
                                "parse_time": parse_time,
                                "total_time": execution_time
                            }
                        }
                    })
                    
                except Exception:
                    # If parsing fails, display the raw output as plain text
                    output_text = raw_response.get("output", "")
                    st.markdown(output_text)
                    
                    # Display execution time for plain text responses too
                    if execution_time < 60:
                        st.caption(f"⏱️ Response time: {execution_time:.2f} seconds")
                    else:
                        minutes = int(execution_time // 60)
                        seconds = execution_time % 60
                        st.caption(f"⏱️ Response time: {minutes}m {seconds:.2f}s")
                    
                    # Add plain text to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": output_text,
                        "execution_time": execution_time
                    })
                
            except Exception as e:
                execution_time = time.time() - start_time
                st.error(f"Error: {str(e)} (after {execution_time:.2f}s)")
                if 'raw_response' in locals():
                    st.markdown("**Raw Response:**")
                    st.json(raw_response)

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This AI Research Agent uses:
    - 🔍 DuckDuckGo Search: for web information
    - 📚 Wikipedia: for detailed knowledge
    - 🤖 GPT-4o-mini: for intelligent responses
    
    Your chat history persists during this session.
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("💡 **Tip:** Ask detailed questions for better research results!")

