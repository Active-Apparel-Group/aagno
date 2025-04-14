import os
import tempfile
from typing import List

import nest_asyncio
import requests
import streamlit as st
from agentic_rag import get_agentic_rag_agent
from agentic_rag import get_reasoning_agent
from cookbook.agents.finance_agents import get_finance_agent, get_financial_datasets_agent
from agno.agent import Agent
from agno.document import Document
from agno.document.reader.csv_reader import CSVReader
from agno.document.reader.pdf_reader import PDFReader
from agno.document.reader.text_reader import TextReader
from agno.document.reader.website_reader import WebsiteReader
from agno.document.reader.docx_reader import DocxReader
from agno.document.reader.json_reader import JSONReader
from markdown_reader import MarkdownReader
from sql_reader import SQLScriptReader
from agno.document.chunking.agentic import AgenticChunking
from agno.document.chunking.document import DocumentChunking
from agno.document.chunking.fixed import FixedSizeChunking
from agno.document.chunking.recursive import RecursiveChunking
from agno.utils.log import logger
from utils import (
    CUSTOM_CSS,
    about_widget,
    add_message,
    display_tool_calls,
    export_chat_history,
    rename_session_widget,
    session_selector_widget,
)

CHUNKING_STRATEGIES = {
    "Agentic": AgenticChunking,
    "Fixed": FixedSizeChunking,
    "Recursive": RecursiveChunking,
    "Document": DocumentChunking,
}


nest_asyncio.apply()
st.set_page_config(
    page_title="Agentic RAG",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Add selector to set reasoning mode
if "use_reasoning_agent" not in st.session_state:
    st.session_state.use_reasoning_agent = False

st.sidebar.markdown("#### 🧠 Reasoning Agent")
st.session_state.use_reasoning_agent = st.sidebar.checkbox(
    "Enable Reasoning Agent",
    value=st.session_state.use_reasoning_agent
)

def restart_agent():
    logger.debug("---*--- Restarting agent ---*---")
    st.session_state["agentic_rag_agent"] = None
    st.session_state["agentic_rag_agent_session_id"] = None
    st.session_state["messages"] = []
    st.rerun()

def get_reader(file_type: str):
    strategy_name = st.session_state.chunking_strategy_per_type.get(file_type.lower(), "Fixed")
    chunking_strategy = CHUNKING_STRATEGIES[strategy_name]()

    readers = {
        "pdf": PDFReader(chunking_strategy=chunking_strategy),
        "csv": CSVReader(chunking_strategy=chunking_strategy),
        "txt": TextReader(chunking_strategy=chunking_strategy),
        "docx": DocxReader(chunking_strategy=chunking_strategy),
        "json": JSONReader(chunking_strategy=chunking_strategy),
        "markdown": MarkdownReader(chunking_strategy=chunking_strategy),
        "sql": SQLScriptReader(chunking_strategy=chunking_strategy),
    }
    return readers.get(file_type.lower(), None)


def initialize_agent(model_id: str, research_logic: str = None, research_tools: list = None):
    if (
        "agentic_rag_agent" not in st.session_state
        or st.session_state["agentic_rag_agent"] is None
    ):
        logger.info(f"---*--- Creating {model_id} Agent ---*---")
        reasoning = st.session_state.use_reasoning_agent
        agent: Agent = (
            get_reasoning_agent(model_id=model_id)
            if reasoning else
            get_agentic_rag_agent(
                model_id=model_id,
                session_id=st.session_state.get("agentic_rag_agent_session_id"),
                debug_mode=True
            )
        )
        if research_logic:
            agent.update_instructions([research_logic])
        if research_tools:
            agent.update_tools(research_tools)
        st.session_state["agentic_rag_agent"] = agent
        st.session_state["agentic_rag_agent_session_id"] = agent.session_id
    return st.session_state["agentic_rag_agent"]

def main():
    if "chunking_strategy_per_type" not in st.session_state:
        st.session_state.chunking_strategy_per_type = {
            "pdf": "Agentic",
            "csv": "Fixed",
            "txt": "Recursive",
            "docx": "Document",
            "json": "Recursive",
            "markdown": "Recursive",
            "sql": "Document",
        }
    st.markdown("<h1 class='main-title'>Agentic RAG </h1>", unsafe_allow_html=True)
    if "show_reasoning_trace" not in st.session_state:
        st.session_state["show_reasoning_trace"] = False
    
    st.sidebar.checkbox("Show Tool Reasoning Trace", key="show_reasoning_trace")

    
    if st.session_state.use_reasoning_agent:
        st.markdown("<p class='subtitle'>🧠 Reasoning Mode Enabled — Multi-step analysis active</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='subtitle'>🔍 Agentic RAG – Contextual knowledge-based retrieval</p>", unsafe_allow_html=True)

    st.markdown(
        "<p class='subtitle'>Your intelligent research assistant powered by Agno</p>",
        unsafe_allow_html=True,
    )

    model_options = {
        "o3-mini": "openai:o3-mini",
        "gpt-4o": "openai:gpt-4o",
        "gemini-2.0-flash-exp": "google:gemini-2.0-flash-exp",
        "claude-3-5-sonnet": "anthropic:claude-3-5-sonnet-20241022",
        "llama-3.3-70b": "groq:llama-3.3-70b-versatile",
    }
    selected_model = st.sidebar.selectbox(
        "Select a model",
        options=list(model_options.keys()),
        index=0,
        key="model_selector",
    )
    model_id = model_options[selected_model]

    agentic_rag_agent: Agent
    if (
        "agentic_rag_agent" not in st.session_state
        or st.session_state["agentic_rag_agent"] is None
        or st.session_state.get("current_model") != model_id
    ):
        logger.info("---*--- Creating new Agentic RAG  ---*---")
        agentic_rag_agent = initialize_agent(model_id=model_id)
        st.session_state["agentic_rag_agent"] = agentic_rag_agent
        st.session_state["current_model"] = model_id
    else:
        agentic_rag_agent = st.session_state["agentic_rag_agent"]

    try:
        st.session_state["agentic_rag_agent_session_id"] = (
            agentic_rag_agent.load_session()
        )
    except Exception:
        st.warning("Could not create Agent session, is the database running?")
        return

    agent_runs = agentic_rag_agent.memory.runs
    if len(agent_runs) > 0:
        logger.debug("Loading run history")
        st.session_state["messages"] = []
        for _run in agent_runs:
            if _run.message is not None:
                add_message(_run.message.role, _run.message.content)
            if (_run.response is not None):
                add_message("assistant", _run.response.content, _run.response.tools)
    else:
        logger.debug("No run history found")
        st.session_state["messages"] = []

    if prompt := st.chat_input("👋 Ask me anything!"):
        add_message("user", prompt)

    if "loaded_urls" not in st.session_state:
        st.session_state.loaded_urls = set()
    if "loaded_files" not in st.session_state:
        st.session_state.loaded_files = set()
    if "knowledge_base_initialized" not in st.session_state:
        st.session_state.knowledge_base_initialized = False

    st.sidebar.markdown("#### 📚 Document Management")
    input_url = st.sidebar.text_input("Add URL to Knowledge Base")
    if input_url and not prompt and not st.session_state.knowledge_base_initialized:
        if input_url not in st.session_state.loaded_urls:
            alert = st.sidebar.info("Processing URLs...", icon="ℹ️")
            if input_url.lower().endswith(".pdf"):
                try:
                    response = requests.get(input_url, stream=True, verify=False)
                    response.raise_for_status()

                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as tmp_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            tmp_file.write(chunk)
                        tmp_path = tmp_file.name

                    reader = PDFReader()
                    docs: List[Document] = reader.read(tmp_path)
                    os.unlink(tmp_path)
                except Exception as e:
                    st.sidebar.error(f"Error processing PDF: {str(e)}")
                    docs = []
            else:
                scraper = WebsiteReader(max_links=2, max_depth=1)
                docs: List[Document] = scraper.read(input_url)

            if docs:
                agentic_rag_agent.knowledge.load_documents(docs, upsert=True)
                st.session_state.loaded_urls.add(input_url)
                st.sidebar.success("URL added to knowledge base")
            else:
                st.sidebar.error("Could not process the provided URL")
            alert.empty()
        else:
            st.sidebar.info("URL already loaded in knowledge base")
        
    st.sidebar.markdown("#### 🧩 Chunking Strategies")
    for file_type in sorted(st.session_state.chunking_strategy_per_type.keys()):
        current_strategy = st.session_state.chunking_strategy_per_type[file_type]
        selected = st.sidebar.selectbox(
            f"{file_type.upper()} Chunking",
            options=list(CHUNKING_STRATEGIES.keys()),
            index=list(CHUNKING_STRATEGIES.keys()).index(current_strategy),
            key=f"chunking_{file_type}",
        )
        st.session_state.chunking_strategy_per_type[file_type] = selected


    # Update accepted types to include SQL files
    uploaded_files = st.sidebar.file_uploader(
        "Add Documents (.pdf, .csv, .txt, .docx, .json, .md, .sql)",
        type=["pdf", "csv", "txt", "docx", "json", "md", "sql"],
        accept_multiple_files=True
    )
    if uploaded_files and not prompt:
        progress_text = "Loading documents into knowledge base..."
        progress_bar = st.sidebar.progress(0, text=progress_text)

        total_files = len(uploaded_files)
        for idx, uploaded_file in enumerate(uploaded_files):
            file_identifier = f"{uploaded_file.name}_{uploaded_file.size}"
            if file_identifier not in st.session_state.loaded_files:
                file_type = uploaded_file.name.split(".")[-1].lower()
                reader = get_reader(file_type)
                if reader:
                    try:
                        docs = reader.read(uploaded_file)
                        agentic_rag_agent.knowledge.load_documents(docs, upsert=True)
                        st.session_state.loaded_files.add(file_identifier)
                        st.toast(f"✅ {uploaded_file.name} loaded", icon="📄")
                    except Exception as e:
                        st.toast(f"❌ Failed to load {uploaded_file.name}: {e}", icon="⚠️")
            progress_bar.progress((idx + 1) / total_files, text=progress_text)

        progress_bar.empty()
        st.sidebar.success("All documents added to knowledge base")
        st.session_state.knowledge_base_initialized = True

    # Safe clear mechanism
    if "confirm_clear_kb" not in st.session_state:
        st.session_state.confirm_clear_kb = False

    with st.sidebar:
        st.markdown("#### 🚨 Dangerous Actions")
        if not st.session_state.confirm_clear_kb:
            if st.button("🧨 Clear Knowledge Base", type="primary"):
                st.session_state.confirm_clear_kb = True
                st.warning("Are you sure? Click again to confirm.")
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("✅ Confirm Clear"):
                    agentic_rag_agent.knowledge.vector_db.delete()
                    st.session_state.loaded_urls.clear()
                    st.session_state.loaded_files.clear()
                    st.session_state.knowledge_base_initialized = False
                    st.success("Knowledge base cleared.")
                    st.session_state.confirm_clear_kb = False
            with col2:
                if st.button("❌ Cancel"):
                    st.session_state.confirm_clear_kb = False

    st.sidebar.markdown("#### ❓ Sample Questions")
    if st.sidebar.button("📝 Summarize"):
        add_message(
            "user",
            "Can you summarize what is currently in the knowledge base (use `search_knowledge_base` tool)?",
        )

    st.sidebar.markdown("#### 🛠️ Utilities")
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        if st.sidebar.button("🔄 New Chat", use_container_width=True):
            restart_agent()
    with col2:
        if st.sidebar.download_button(
            "💾 Export Chat",
            export_chat_history(),
            file_name="rag_chat_history.md",
            mime="text/markdown",
            use_container_width=True,
        ):
            st.sidebar.success("Chat history exported!")

    # Add a new section for "Gather Research" in the Streamlit UI
    st.sidebar.markdown("#### 📚 Gather Research")

    # Input for research topic
    research_topic = st.sidebar.text_input("Enter Research Topic")

    # Ensure research output is formatted for human readability and markdown storage
    if st.sidebar.button("Gather Research"):
        if research_topic:
            try:
                # Use the reasoning agent to gather research
                agent = initialize_agent(model_id="openai:gpt-4o")
                with st.spinner("🔄 Gathering research, please wait..."):
                    response = agent.run(f"Gather research on the topic: {research_topic}")

                # Format the research output for human readability and markdown storage
                formatted_response = f"### Research on {research_topic}\n\n" + "\n".join([f"- {line.strip()}" for line in response.splitlines() if line.strip()])

                # Display the research output
                st.markdown("### Research Output")
                st.markdown(formatted_response)

                # Add button to save research to vector DB
                if st.button("Add Research to Knowledge Base"):
                    try:
                        agentic_rag_agent.knowledge.load_documents([Document(content=formatted_response)], upsert=True)
                        st.success("Research added to knowledge base successfully!")
                    except Exception as e:
                        st.error(f"Failed to add research to knowledge base: {str(e)}")

                st.success("✅ Research completed successfully!")
            except Exception as e:
                st.error(f"Error gathering research: {str(e)}")
        else:
            st.warning("Please enter a research topic.")

    # Add a new section for "Research Agent Team" in the Streamlit UI
    st.sidebar.markdown("#### 🤝 Research Agent Team")

    # Input for research query
    research_query = st.sidebar.text_input("Enter Research Query")

    # Button to trigger the research agent team
    if st.sidebar.button("Run Research Agent Team"):
        if research_query:
            try:
                # Initialize the research agent team
                from agno.agent import Agent
                from agno.models.openai import OpenAIChat
                from agno.team.team import Team
                from agno.tools.duckduckgo import DuckDuckGoTools
                from agno.tools.yfinance import YFinanceTools

                # Define agents
                web_agent = Agent(
                    name="Web Agent",
                    role="Search the web for information",
                    model=OpenAIChat(id="gpt-4o"),
                    tools=[DuckDuckGoTools()],
                    instructions="""You are an experienced web researcher. Provide clear, concise, and well-sourced information.""",
                    show_tool_calls=True,
                    markdown=True,
                )

                finance_agent = Agent(
                    name="Finance Agent",
                    role="Analyze financial data",
                    model=OpenAIChat(id="gpt-4o"),
                    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True)],
                    instructions="""You are a financial analyst. Provide detailed and structured financial insights.""",
                    show_tool_calls=True,
                    markdown=True,
                )

                # Create the team
                research_team = Team(
                    members=[web_agent, finance_agent],
                    model=OpenAIChat(id="gpt-4o"),
                    mode="coordinate",
                    success_criteria="""Provide a comprehensive and well-structured research report.""",
                    instructions="""Coordinate the agents to deliver a cohesive research output.""",
                    show_tool_calls=True,
                    markdown=True,
                )

                # Run the team with the user's query
                # Correct the error by ensuring the response is converted to a string before calling replace
                response = research_team.run(research_query)

                # Convert response to string if necessary and format it
                if not isinstance(response, str):
                    response = str(response)

                formatted_response = f"### Research Report: {research_query}\n\n" + response.replace("\n", "\n\n")

                # Display the formatted results
                st.markdown("### Research Agent Team Results")
                st.markdown(formatted_response)
            except Exception as e:
                st.error(f"Error running research agent team: {str(e)}")
        else:
            st.warning("Please enter a research query.")

    # Add a new UI section for modifying the research agent instructions
    st.sidebar.markdown("#### 🛠️ Research Agent Configuration")

    # Dropdown for selecting research logic interpretation
    research_logic_options = [
        "Default Logic",
        "Aggressive Research",
        "Conservative Research",
        "Balanced Research"
    ]
    selected_research_logic = st.sidebar.selectbox(
        "Select Research Logic",
        options=research_logic_options,
        help="Choose how the research agent interprets and applies research logic."
    )

    # Multi-select toggles for research tools
    research_tool_options = [
            "Web Search",
            "Financial Analysis",
            "Market Trends",
            "Competitor Analysis",
            "Thinking Tools",
            "Financial Datasets",
    ]
    selected_research_tools = st.sidebar.multiselect(
        "Select Research Tools",
        options=research_tool_options,
        default=["Web Search"],
        help="Select the tools the research agent should use during research."
    )

    # Apply the custom configuration to the research agent
    if st.sidebar.button("Apply Configuration"):
        try:
            # Update the research agent's logic and tools
            agentic_rag_agent.update_instructions([selected_research_logic])
            agentic_rag_agent.update_tools(selected_research_tools)

            st.sidebar.success("Research agent configuration updated successfully!")
        except Exception as e:
            st.sidebar.error(f"Failed to update research agent configuration: {str(e)}")

    for message in st.session_state["messages"]:
        if message["role"] in ["user", "assistant"]:
            _content = message["content"]
            if _content is not None:
                with st.chat_message(message["role"]):
                    if "tool_calls" in message and message["tool_calls"]:
                        display_tool_calls(st.empty(), message["tool_calls"])
                    st.markdown(_content)

    last_message = (
        st.session_state["messages"][-1] if st.session_state["messages"] else None
    )
    if last_message and last_message.get("role") == "user":
        question = last_message["content"]
        with st.chat_message("assistant"):
            tool_calls_container = st.empty()
            resp_container = st.empty()
            with st.spinner("🤔 Thinking..."):
                response = ""
                try:
                    run_response = agentic_rag_agent.run(
                    question,
                    stream=True,
                    show_full_reasoning=st.session_state.show_reasoning_trace,
                    stream_intermediate_steps=st.session_state.show_reasoning_trace
                )

                    for _resp_chunk in run_response:
                        if _resp_chunk.tools and len(_resp_chunk.tools) > 0:
                            display_tool_calls(tool_calls_container, _resp_chunk.tools)

                        if _resp_chunk.content is not None:
                            response += _resp_chunk.content
                            resp_container.markdown(response)

                    add_message(
                        "assistant", response, agentic_rag_agent.run_response.tools
                    )
                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    add_message("assistant", error_message)
                    st.error(error_message)

    session_selector_widget(agentic_rag_agent, model_id)
    rename_session_widget(agentic_rag_agent)
    about_widget()

main()
