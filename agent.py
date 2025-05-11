import functools
import os
from typing import Literal
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from databricks_langchain.genie import GenieAgent
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from mlflow.langchain.chat_agent_langgraph import ChatAgentState
from pydantic import BaseModel
import uuid

###################################################
## Create Genie agent
###################################################
GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID", "???????????")
genie_agent_description = (
    "The Genie agent runs SQL queries to retrieve campaign and keyword performance metrics. "
    "It should not generate insights or recommendations."
)

genie_agent = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    genie_agent_name="Genie",
    description=genie_agent_description,
    client=WorkspaceClient(
        host=os.getenv("DB_MODEL_SERVING_HOST_URL"),
        token=os.getenv("DATABRICKS_GENIE_PAT"),
    ),
)

###################################################
## Create a secondary LLM agent
###################################################
LLM_ENDPOINT_NAME = os.getenv("LLM_ENDPOINT_NAME", "databricks-claude-3-7-sonnet")
llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
llm_agent = create_react_agent(llm, tools=[])

llm_agent_description = (
    "The LLM agent specializes in summarizing results and generating business insights from data retrieved by Genie agent. "
    "In most cases, it should be used after the Genie agent has run, to explain results or suggest actions in natural language. "
    "It can also handle general or conversational queries that do not require data, such as greetings."
)

###################################################
## Supervisor: route to Genie or LLM or finish
###################################################
MAX_ITERATIONS = 3

worker_descriptions = {
    "Genie": genie_agent_description,
    "LLM": llm_agent_description,
}

formatted_descriptions = "\n".join(
    f"- {name}: {desc}" for name, desc in worker_descriptions.items()
)

system_prompt = f"""
You are a supervisor agent responsible for routing user questions to the most appropriate agent based on their intent.

Available agents:
{formatted_descriptions}

Routing rules:
1. If the user says hello or asks a vague or general question unrelated to data → route to **LLM**.
2. If the user asks for campaign metrics, keyword performance, budget breakdowns, or any query that requires structured data → route to **Genie**.
3. If Genie has already been invoked and the user’s original question requires business reasoning, strategy, or recommendation → route to **LLM**.
4. If Genie has already been invoked and the original question was just for a data lookup (e.g. “what is the X value?”) → route to **FINISH**.
5. Do **not** invoke Genie more than once per user query.

Examples:
Q: "What are our highest-impression keywords?" → Genie  
Q: "If I invest $1.4M, what keywords should I target?" → Genie → LLM  
Q: "What’s our ad spend by brand?" → Genie  
Q: "Hey there!" → LLM  
Q: "Can you help me?" → LLM  

Based on this logic, determine the next step.
Respond only with:
- next_node: Genie
- next_node: LLM
- next_node: FINISH
"""

options = ["FINISH"] + list(worker_descriptions.keys())
FINISH = {"next_node": "FINISH"}

def supervisor_agent(state):
    # Increment the iteration count for each decision step
    count = state.get("iteration_count", 0) + 1
    if count > MAX_ITERATIONS:
        return FINISH

    class nextNode(BaseModel):
        next_node: Literal[tuple(options)]

    history = [m.get("name") for m in state.get("messages", []) if m.get("role") == "assistant"]
    genie_called = "Genie" in history
    llm_called = "LLM" in history

    # Stop if both have been invoked
    if genie_called and llm_called:
        return FINISH

    banned = ["Genie"] if genie_called else []

    routing_context = f"\n(Previously invoked agents: {', '.join(history) or 'None'})\n"
    prompt_with_history = system_prompt + routing_context

    preprocessor = RunnableLambda(
        lambda state: [{"role": "system", "content": prompt_with_history}] + state["messages"]
    )

    supervisor_chain = preprocessor | llm.with_structured_output(nextNode)
    next_node = supervisor_chain.invoke(state).next_node

    if next_node in banned:
        return FINISH

    return {"iteration_count": count, "next_node": next_node}

###################################################
## Define each agent execution wrapper
###################################################
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {
        "messages": [
            {
                "role": "assistant",
                "content": result["messages"][-1].content,
                "name": name,
            }
        ]
    }

###################################################
## Final summarization with llm.invoke
###################################################
def final_answer(state):
    prompt = (
        "You are the final agent responsible for presenting the result to the user.\n"
        "Review the above assistant messages carefully.\n"
        "Do not ask the user any questions. Your only task is to summarize and recommend.\n"
        "Avoid repeating previous outputs verbatim. Do not restate earlier content without adding value.\n\n"
        "Generate a comprehensive, well-structured response that highlights:\n"
        "- Key insights or metrics found\n"
        "- Strategic recommendations if applicable\n"
        "- Clear categorization (e.g. Primary Keywords, Expansion Strategy, Budget Advice)\n\n"
        "If no relevant insights were found, say 'No actionable insight was found based on the data.'"
    )

    messages = state["messages"] + [{"role": "user", "content": prompt}]
    return {"messages": [llm.invoke(messages)]}

###################################################
## Construct the LangGraph workflow
###################################################
class AgentState(ChatAgentState):
    next_node: str
    iteration_count: int

genie_node = functools.partial(agent_node, agent=genie_agent, name="Genie")
llm_node = functools.partial(agent_node, agent=llm_agent, name="LLM")

workflow = StateGraph(AgentState)
workflow.set_entry_point("supervisor")
workflow.add_node("Genie", genie_node)
workflow.add_node("LLM", llm_node)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("final_answer", final_answer)

for agent_name in worker_descriptions:
    workflow.add_edge(agent_name, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next_node"],
    {**{k: k for k in worker_descriptions}, "FINISH": "final_answer"},
)
workflow.add_edge("final_answer", END)

multi_agent = workflow.compile()
