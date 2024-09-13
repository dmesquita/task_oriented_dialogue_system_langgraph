# https://langchain-ai.github.io/langgraph/tutorials/chatbots/information-gather-prompting/

import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import AzureChatOpenAI
from typing import List

from langchain_core.messages import SystemMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from typing import Literal

from langgraph.graph import END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
import uuid

_ = load_dotenv(find_dotenv()) # read local .env file

llm = AzureChatOpenAI(azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
                    openai_api_version="2023-09-01-preview",
                    openai_api_type="azure",
                    openai_api_key=os.environ.get('AZURE_OPENAI_API_KEY'),
                    azure_endpoint=os.environ.get('AZURE_OPENAI_ENDPOINT'),
                    temperature=0)

prompt_system_task = """Your job is to gather information from the user about the User Story they need to create.

You should obtain the following information from them:

- Objective: the goal of the user story. should be concrete enough to be developed in 2 weeks.
- Success criteria the sucess criteria of the user story
- Plan_of_execution: the plan of execution of the initiative

If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess. 
Whenever the user responds to one of the criteria, evaluate if it is detailed enough to be a criterion of a User Story. If not, ask questions to help the user better detail the criterion.
Do not overwhelm the user with too many questions at once; ask for the information you need in a way that they do not have to write much in each response. 
Always remind them that if they do not know how to answer something, you can help them.

After you are able to discern all the information, call the relevant tool."""

class UserStoryCriteria(BaseModel):
    """Instructions on how to prompt the LLM."""
    objective: str
    success_criteria: str
    plan_of_execution: str
llm_with_tool = llm.bind_tools([UserStoryCriteria])

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class StateSchema(TypedDict):
    messages: Annotated[list, add_messages]
    created_user_story: bool

workflow = StateGraph(StateSchema)

def domain_state_tracker(messages):
    return [SystemMessage(content=prompt_system_task)] + messages

def call_llm(state: StateSchema):
    """
    talk_to_user node function, adds the prompt_system_task to the messages,
    calls the LLM and returns the response
    """
    messages = domain_state_tracker(state["messages"])
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}

workflow.add_node("talk_to_user", call_llm)

workflow.add_edge(START, "talk_to_user")

def finalize_dialogue(state: StateSchema):
    """
    Add a tool message to the history so the graph can see that it`s time to create the user story
    """
    return {
        "messages": [
            ToolMessage(
                content="Prompt generated!",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        ]
    }

workflow.add_node("finalize_dialogue", finalize_dialogue)

prompt_generate_user_story = """Based on the following requirements, write a good user story:

{reqs}"""

def build_prompt_to_generate_user_story(messages: list):
    tool_call = None
    other_msgs = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls: #tool_calls is from the OpenAI API
            tool_call = m.tool_calls[0]["args"]
        elif isinstance(m, ToolMessage):
            continue
        elif tool_call is not None:
            other_msgs.append(m)
    return [SystemMessage(content=prompt_generate_user_story.format(reqs=tool_call))] + other_msgs

def call_model_to_generate_user_story(state):
    messages = build_prompt_to_generate_user_story(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}

workflow.add_node("create_user_story", call_model_to_generate_user_story)

def define_next_action(state) -> Literal["finalize_dialogue", END]:
    messages = state["messages"]

    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "finalize_dialogue"
    else:
        return END

workflow.add_conditional_edges("talk_to_user", define_next_action)

workflow.add_edge("finalize_dialogue", "create_user_story")
workflow.add_edge("create_user_story", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

while True:
    user = input("User (q/Q to quit): ")
    if user in {"q", "Q"}:
        print("AI: Byebye")
        break
    output = None
    for output in graph.stream(
        {"messages": [HumanMessage(content=user)]}, config=config, stream_mode="updates"
    ):
        last_message = next(iter(output.values()))["messages"][-1]
        last_message.pretty_print()

    if output and "create_user_story" in output:
        print("User story created!")

