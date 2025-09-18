"""LangChain agent that runs the RecruitFlow pipeline via a single tool call."""
from __future__ import annotations

from typing import Any

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI

from agent.clients import API_VERSION, AZURE_BASE_URL
from agent.config import API_KEY, SETTINGS

from agent.lc_tools import (
    lc_run_pipeline,  # <-- NEW single tool that does everything
    # (Keep the other tools imported if you still want them available manually)
)

# System instructions: use only run_pipeline and return file paths
INTRO = (
    "You are RecruitFlow AI. Run the full recruiting pipeline end-to-end with one tool call.\n"
    "Call: run_pipeline(raw_text) -> returns the saved file paths (final.json, report.md).\n"
    "Do not call any other tools unless run_pipeline fails.\n"
    "Your final reply must be just the returned file paths."
)

# Prompt template with chat history and agent scratchpad
PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", INTRO),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),  # required by create_openai_tools_agent
    ]
)
# Factory: configure LLM, register tools, and build
def build_agent() -> AgentExecutor:
    llm = AzureChatOpenAI(
        api_key=API_KEY,
        azure_endpoint=AZURE_BASE_URL,
        api_version=API_VERSION,
        azure_deployment=SETTINGS.CHAT_DEPLOYMENT,
        temperature=0.0,
        timeout=180,
        max_retries=1,
    )

    tools = [
        lc_run_pipeline,  # only this is needed for normal runs
    ]

    agent = create_openai_tools_agent(llm, tools, PROMPT)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,          # less console noise
        max_iterations=4,       # unnecessary looping protection
        early_stopping_method="force",
        return_intermediate_steps=False,
    )
    return executor