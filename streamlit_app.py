import asyncio
import inspect
import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


# Load keys from .env before importing your agent.py file.
# This does NOT change your agent code.
load_dotenv()


st.set_page_config(
    page_title="Research Agent",
    page_icon="📚",
    layout="wide",
)


st.title("📚 Research Agent")
st.caption("Generate research reports with citations, review steps, and image support.")


with st.sidebar:
    st.header("Settings")
    st.write("Keep your API keys in a `.env` file or paste them here for local testing.")

    openai_key = st.text_input("OPENAI_API_KEY", type="password")
    tavily_key = st.text_input("TAVILY_API_KEY", type="password")
    siliconflow_key = st.text_input("SILICONFLOW_API_KEY", type="password")

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key
    if siliconflow_key:
        os.environ["SILICONFLOW_API_KEY"] = siliconflow_key

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.session_id = f"streamlit_session_{uuid.uuid4().hex[:10]}"
        st.rerun()


async def maybe_await(value):
    """Handle ADK methods that may be sync or async depending on version."""
    if inspect.isawaitable(value):
        return await value
    return value


@st.cache_resource
def get_session_service():
    return InMemorySessionService()


@st.cache_resource
def get_runner():
    # Import only after dotenv/sidebar env setup.
    # Your existing code should be saved as agent.py in the same folder.
    from agent import app_name, root_agent

    session_service = get_session_service()
    runner = Runner(
        agent=root_agent,
        app_name=app_name,
        session_service=session_service,
    )
    return runner, session_service, app_name


async def ensure_session(session_service, app_name: str, user_id: str, session_id: str):
    existing_session = await maybe_await(
        session_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
    )

    if existing_session is None:
        await maybe_await(
            session_service.create_session(
                app_name=app_name,
                user_id=user_id,
                session_id=session_id,
            )
        )


async def run_agent(prompt: str) -> str:
    from agent import user_id as default_user_id

    runner, session_service, app_name = get_runner()

    if "session_id" not in st.session_state:
        st.session_state.session_id = f"streamlit_session_{uuid.uuid4().hex[:10]}"

    session_id = st.session_state.session_id
    user_id = default_user_id

    await ensure_session(session_service, app_name, user_id, session_id)

    user_content = types.Content(
        role="user",
        parts=[types.Part(text=prompt)],
    )

    final_response = ""

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_content,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            text_parts = []
            for part in event.content.parts:
                if getattr(part, "text", None):
                    text_parts.append(part.text)
            final_response = "\n".join(text_parts).strip()

    return final_response or "No final response was returned by the agent."


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


example_prompt = "Write a university-level research report about depression in Canada with trusted sources and images."

prompt = st.chat_input("Ask your research agent, for example: " + example_prompt)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running your research agent..."):
            try:
                response = asyncio.run(run_agent(prompt))
            except RuntimeError as error:
                response = f"Runtime error: {error}"
            except Exception as error:
                response = f"Error: {error}"

        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
