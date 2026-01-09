"""Agent implementations for the multi-agent RAG flow.

This module defines three LangChain agents (Retrieval, Summarization,
Verification) and thin node functions that LangGraph uses to invoke them.
"""
import json
from typing import List

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..llm.factory import create_chat_model
from .prompts import (
    PLANNING_SYSTEM_PROMPT,
    RETRIEVAL_SYSTEM_PROMPT,
    SUMMARIZATION_SYSTEM_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
)
from .state import QAState
from .tools import retrieval_tool


def _extract_last_ai_content(messages: List[object]) -> str:
    """Extract the content of the last AIMessage in a messages list."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(msg.content)
    return ""

planning_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=PLANNING_SYSTEM_PROMPT,
)

retrieval_agent = create_agent(
    model=create_chat_model(),
    tools=[retrieval_tool],
    system_prompt=RETRIEVAL_SYSTEM_PROMPT,
)

summarization_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
)

verification_agent = create_agent(
    model=create_chat_model(),
    tools=[],
    system_prompt=VERIFICATION_SYSTEM_PROMPT,
)

def planning_node(state: QAState) -> QAState:
    
    question = state["question"]
    result = planning_agent.invoke({"messages": [HumanMessage(content=question)]})
    output_text = result["messages"][-1].content  # Get the assistant's final message content 
    Structured_plan = json.loads(output_text) # Parse JSON safely

    return {
        "plan": Structured_plan["plan"],
        "sub_questions": Structured_plan["sub_questions"]
    }



def retrieval_node(state: QAState) -> QAState:
        """Retrieval Agent node: gathers context from vector store.

        This node:
        - Sends the user's question and plan (if available) to the Retrieval Agent.
        - If a plan exists, uses sub_questions for multiple retrieval tool calls.
        - The agent uses the attached retrieval tool to fetch document chunks.
        - Extracts the tool's content (CONTEXT string) from the ToolMessage.
        - Stores the consolidated context string in `state["context"]`.
        """
        question = state["question"]
        plan = state.get("plan")
        sub_questions = state.get("sub_questions", [])

        # Build user content with question and plan if available
        user_content = f"Question: {question}"
        if plan:
            user_content += f"\n\nPlan: {plan}"
            if sub_questions:
                user_content += f"\n\nSub-questions to address:\n" + "\n".join(
                    f"- {sq}" for sq in sub_questions
                )

        result = retrieval_agent.invoke({"messages": [HumanMessage(content=user_content)]})

        messages = result.get("messages", [])
        context = ""

        # Collect all ToolMessage contents (from retrieval_tool calls)
        tool_contents = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_contents.append(str(msg.content))

        context = "\n\n".join(tool_contents) if tool_contents else ""

        return {
            "context": context,
        }


def summarization_node(state: QAState) -> QAState:
    """Summarization Agent node: generates draft answer from context.

    This node:
    - Sends question + context to the Summarization Agent.
    - Agent responds with a draft answer grounded only in the context.
    - Stores the draft answer in `state["draft_answer"]`.
    """
    question = state["question"]
    context = state.get("context")

    user_content = f"Question: {question}\n\nContext:\n{context}"

    result = summarization_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    messages = result.get("messages", [])
    draft_answer = _extract_last_ai_content(messages)

    return {
        "draft_answer": draft_answer,
    }


def verification_node(state: QAState) -> QAState:
    """Verification Agent node: verifies and corrects the draft answer.

    This node:
    - Sends question + context + draft_answer to the Verification Agent.
    - Agent checks for hallucinations and unsupported claims.
    - Stores the final verified answer in `state["answer"]`.
    """
    question = state["question"]
    context = state.get("context", "")
    draft_answer = state.get("draft_answer", "")

    user_content = f"""Question: {question}

Context:{context}

Draft Answer:{draft_answer}

Please verify and correct the draft answer, removing any unsupported claims."""

    result = verification_agent.invoke(
        {"messages": [HumanMessage(content=user_content)]}
    )
    messages = result.get("messages", [])
    answer = _extract_last_ai_content(messages)

    return {
        "answer": answer,
    }
