"""Agent implementations for the multi-agent RAG flow.

This module defines three LangChain agents (Retrieval, Summarization,
Verification) and thin node functions that LangGraph uses to invoke them.
"""
import json
from typing import Any, List
from pydantic import ValidationError
from ...models import PlanningOutput

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


def _content_to_text(content: Any) -> str:
    """Normalize Gemini text content, including structured content blocks."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)

    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text

    return str(content)


def _extract_last_ai_content(messages: List[object]) -> str:
    """Extract the content of the last AIMessage in a messages list."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return _content_to_text(msg.content)
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

    output_text = _content_to_text(result["messages"][-1].content).strip()
    if output_text.startswith("```json") and output_text.endswith("```"):
        output_text = output_text[7:-3].strip()
    elif output_text.startswith("```") and output_text.endswith("```"):
        output_text = output_text[3:-3].strip()

    try:
        parsed_json = json.loads(output_text)
        
        structured_plan = PlanningOutput(**parsed_json)

    except (json.JSONDecodeError, TypeError):
        raise ValueError(f"Planning agent returned invalid JSON:\n{output_text}")

    except ValidationError as e:
        raise ValueError(f"Planning agent failed schema validation:\n{e}")

    return {
        "plan": structured_plan.plan,
        "sub_questions": structured_plan.sub_questions,
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
        tool_call_count = 0
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_call_count += 1
                tool_contents.append(str(msg.content))

        context = "\n\n".join(tool_contents) if tool_contents else ""
        print(f"🔢tool_call_count: {tool_call_count}")
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
