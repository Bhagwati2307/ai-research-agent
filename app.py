import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# 1. Page Configuration
st.set_page_config(page_title="AI Research Agent", page_icon="🔍", layout="centered")
st.title("🔍 Autonomous Research Agent")
st.markdown("I am an AI agent that can browse the web in real-time to answer your questions with citations.")

# 2. Sidebar for API Keys
with st.sidebar:
    st.header("Setup")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    tavily_api_key = st.text_input("Tavily API Key", type="password")
    st.info("These keys are required to power the GPT-4o model and Web Search.")

# 3. User Input
user_query = st.text_input("What would you like me to research?", placeholder="e.g., Latest breakthroughs in fusion energy 2026")

# 4. Agent Execution Logic
if st.button("Run Research"):
    if not openai_api_key or not tavily_api_key:
        st.warning("Please provide both API keys in the sidebar to continue.")
    else:
        try:
            # Initializing the LLM (The Brain)
            llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0.5)
            
            # Initializing the Search Tool (The Eyes)
            search_tool = TavilySearchResults(tavily_api_key=tavily_api_key)
            
            # Initializing the Agent with ReAct Logic
            agent = initialize_agent(
                tools=[search_tool],
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )
            
            # Displaying the Thinking Process
            with st.container():
                st_callback = StreamlitCallbackHandler(st.container())
                response = agent.run(user_query, callbacks=[st_callback])
                
                # Displaying Final Result
                st.subheader("Research Report:")
                st.success(response)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.divider()
st.caption("Built with LangChain, GPT-4o, and Streamlit.")