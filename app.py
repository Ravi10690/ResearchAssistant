# importing the required modules

import loading_env
import streamlit as st
from langchain_openai import ChatOpenAI
from typing import List, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator
import time
from IPython.display import display, Image
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, SystemMessage,HumanMessage,trim_messages,RemoveMessage
import operator
from typing import List, Annotated
from typing_extensions import TypedDict
from langgraph.constants import Send
from prompts import (
    analyst_instructions,
    question_instructions,
    search_instructions,
    answer_instructions,
    section_writer_instructions,
    report_writer_instructions,
    intro_conclusion_instructions)
from langchain_community.tools.tavily_search import TavilySearchResults
from  langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import get_buffer_string


st.title("Research Assistant")
with st.sidebar:
    openai_key = st.text_input("OpenAI Key",type="password")
    openai_models = [
        "gpt-4o",
        "gpt-4o-mini"]

    # Display the models in a Streamlit select box
    openai_model = st.selectbox("Select an OpenAI Model", openai_models,index=1)
    max_analyst = st.number_input("Enter the number of analyst", min_value=1, max_value=5, value=1)

st.write(f"Selected OpenAI Model: {openai_model}, Number of Analyst: {max_analyst}")
# defining the llm 
llm = ChatOpenAI(model = openai_model, temperature= 0.5)

#define the persona of analyst
class Analyst(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(description="Description of the analyst focus, concerns, and motives.")
    experties: str = Field(description="The area of expertise of the analyst.")

    @property
    def persona(self) -> str:
        return f"\nName: {self.name} \nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\nExpertise: {self.experties}"

    @property
    def persona_markdown(self) -> str:
        return f"""
            **Name:** {self.name}  
            **Role:** {self.role}  
            **Affiliation:** {self.affiliation}  
            **Description:** {self.description}  
            **Expertise:** {self.experties}
            """

# defining the schema for ResearchState

class ResearchGraphState(TypedDict):
    topic:str # topic of the research
    max_analysts:int # maximum number of analyst
    human_analyst_feedback :str # feedback from the human for creation of analyst
    analysts:list[Analyst] # list of analyst persona
    sections:Annotated[list, operator.add] # list of sections in the research
    introduction: str # Introduction for the final report
    content: str # Content for the final report
    conclusion: str # Conclusion for the final report
    final_report: str # Final report




######## defining a function to create analyst ###############################################

# structure output of llm output
class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(description="Comprehensive list of analysts with their roles and affiliations.")

def create_analysts(state: ResearchGraphState):
    
    """ Create analysts """
    
    topic=state['topic']
    max_analysts=state['max_analysts']
    human_analyst_feedback=state.get('human_analyst_feedback', '')
        
    # Enforce structured output
    structured_llm = llm.with_structured_output(Perspectives)

    # System message
    system_message = analyst_instructions.format(topic=topic,human_analyst_feedback=human_analyst_feedback, max_analysts=max_analysts)
    # Generate question 
    analysts = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the set of analysts.")])
    
    # Write the list of analysis to state
    return {"analysts": analysts.analysts}




# asking for human feedback
def human_feedback(state: ResearchGraphState):
    """ Ask for human feedback """
    pass

def should_continue(state: ResearchGraphState):
    """ Return the next node to execute """

    # Check if human feedback
    human_analyst_feedback=state.get('human_analyst_feedback', None)
    if human_analyst_feedback:
        return "create_analysts"
    
    # Otherwise end
    return END



######################  defining another stategraph for conducting interview ###########################
class InterviewGraphState(MessagesState):
    max_number_turns: int
    context : Annotated[list, operator.add]
    analyst : Analyst
    interview :str
    sections : list
    topic :str

class InterviewOutputState(MessagesState):
    context : Annotated[list, operator.add]
    analyst : Analyst
    sections : list

# generating question by analyst for expert
def generate_question(state: InterviewGraphState):
    """ Generate a question for the analyst """
    
    analyst = state["analyst"]
    topic = state["topic"]
    messages = state["messages"]

    system_message = question_instructions.format(topic=topic, goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)]+messages)
    # print(question)
    return {"messages": [question]}

#creating query to search on web and gather result
class SearchQuery(BaseModel):
    search_query: str = Field(None,description="Search query to be executed")

tavily_search = TavilySearchResults(max_results=3)

def search_web(state: InterviewGraphState):
    
    """ Retrieve docs from web search """
    search_system_message  = SystemMessage(content=search_instructions)
    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_system_message]+state['messages'])
    
    # Search
    search_docs = tavily_search.invoke(search_query.search_query)

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 


def search_wikipedia(state: InterviewGraphState):
    
    """ Retrieve docs from wikipedia """

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions]+state['messages'])
    
    # Search
    search_docs = WikipediaLoader(query=search_query.search_query, 
                                  load_max_docs=2).load()

     # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 


# defining the answer to the question asked by the analyst
def generate_answer(state: InterviewGraphState):
    
    """ Generate an answer to the question """
    
    analyst = state["analyst"]
    topic = state["topic"]
    messages = state["messages"]
    context = state["context"]
    
    system_message = answer_instructions.format(topic=topic, goals=analyst.persona, context=context)
    answer = llm.invoke([SystemMessage(content=system_message)]+messages)
    answer.name = "expert"
    return {"messages": [answer]}





## define to dave interview
def save_interview(state: InterviewGraphState):
    # Get messages
    messages = state["messages"]
    
    # Convert interview to a string
    interview = get_buffer_string(messages)
    
    # Save to interviews key
    return {"interview": interview}


def route_messages(state: InterviewGraphState, 
                   name: str = "expert"):

    """ Route between question and answer """
    
    # Get messages
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns',2)

    # Check the number of expert answers 
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return 'save_interview'

    # This router is run after each question - answer pair 
    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]
    
    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return "ask_question"


def write_section(state: InterviewGraphState):

    """ Node to answer a question """

    # Get state
    interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]
   
    # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
    system_message = section_writer_instructions.format(focus=analyst.description)
    section = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Use this source to write your section: {context}")]) 
                
    # Append it to state
    return {"sections": [section.content]}

if "is_subgrpah_built" not in st.session_state:
    st.session_state.is_subgraph_built = False

if not st.session_state.is_subgraph_built:
    st.session_state.is_subgraph_built = True
    # Add nodes and edges 
    interview_builder = StateGraph(InterviewGraphState,output=InterviewOutputState)
    interview_builder.add_node("ask_question", generate_question)
    interview_builder.add_node("search_web", search_web)
    interview_builder.add_node("search_wikipedia", search_wikipedia)
    interview_builder.add_node("answer_question", generate_answer)
    interview_builder.add_node("save_interview", save_interview)
    interview_builder.add_node("write_section", write_section)

    # Flow
    interview_builder.add_edge(START, "ask_question")
    interview_builder.add_edge("ask_question", "search_web")
    interview_builder.add_edge("ask_question", "search_wikipedia")
    interview_builder.add_edge("search_web", "answer_question")
    interview_builder.add_edge("search_wikipedia", "answer_question")
    interview_builder.add_conditional_edges("answer_question", route_messages,['ask_question','save_interview'])
    interview_builder.add_edge("save_interview", "write_section")
    interview_builder.add_edge("write_section", END)
    # Interview 
    memory = MemorySaver()
    interview_graph = interview_builder.compile(checkpointer=memory).with_config(run_name="Conducting Interview")
    st.session_state.interview_graph = interview_graph

def initiate_all_interviews(state: ResearchGraphState):
    """ This is the "map" step where we run each interview sub-graph using Send API """    

    # Check if human feedback
    human_analyst_feedback=state.get('human_analyst_feedback',"")
    if human_analyst_feedback:
        # Return to create_analysts
        return "create_analysts"

    # Otherwise kick off interviews in parallel via Send() API
    else:
        topic = state["topic"]
        return [Send("conduct_interview", {"analyst": analyst,
                                           "messages": [HumanMessage(
                                               content=f"So you said you were writing an article on {topic}?")],
                                            
                                            "topic" : topic,
                                            "max_number_turns": 3
                                            }) for analyst in state["analysts"]]



def write_report(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Summarize the sections into a final report
    system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)    
    report = llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Write a report based upon these memos.")]) 
    return {"content": report.content}


def write_introduction(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Summarize the sections into a final report
    
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
    intro = llm.invoke([instructions]+[HumanMessage(content=f"Write the report introduction")]) 
    return {"introduction": intro.content}

def write_conclusion(state: ResearchGraphState):
    # Full set of sections
    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    # Summarize the sections into a final report
    
    instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
    conclusion = llm.invoke([instructions]+[HumanMessage(content=f"Write the report conclusion")]) 
    return {"conclusion": conclusion.content}


def finalize_report(state: ResearchGraphState):
    """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
    # Save full final report
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}


# Check if the graph is built
if "is_graph_built" not in st.session_state:
    st.session_state.is_graph_built = False

if not st.session_state.is_graph_built:
    st.session_state.is_graph_built = True
# Add nodes and edges 
    builder = StateGraph(ResearchGraphState)
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("human_feedback", human_feedback)
    builder.add_node("conduct_interview", st.session_state.interview_graph)
    builder.add_node("write_report",write_report)
    builder.add_node("write_introduction",write_introduction)
    builder.add_node("write_conclusion",write_conclusion)
    builder.add_node("finalize_report",finalize_report)

    # Logic
    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges("human_feedback", initiate_all_interviews, ["create_analysts", "conduct_interview"])
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_conclusion")
    builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
    builder.add_edge("finalize_report", END)
    # Compile
    memory = MemorySaver()
    graph = builder.compile(interrupt_before= ["human_feedback"], checkpointer=memory)
    # View

    # display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
    st.write("Graph built")
    st.session_state.graph = graph


default_values = {
    "feedbacks": [],
    "feedback_submitted": False,
    "show_feedback_form": False,
    "research_started": False,
    "analysts": [],
    "graph_input": None,
    "waiting_for_feedback": False,
    "feedback_completed": False,
    "is_subgraph_built": False
}
def initialize_session_state():
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value
def reset_session_state():
    for key, value in default_values.items():
        st.session_state[key] = value
# defining all the session state variables
thread = {"configurable" : {"thread_id": "1"}}
initialize_session_state()
# Defining the main headline and the sidebar


# defining the main research input section
topic = st.text_input("Enter the topic you want to research")
if st.button("Start Research"):

    st.session_state.research_started = True
    st.session_state.waiting_for_feedback = False 

if st.session_state.research_started:
    if not st.session_state.feedbacks and not st.session_state.waiting_for_feedback and not st.session_state.analysts:
        print("Invoking the graph")
        st.session_state.graph_input = {"topic": topic, "max_analysts": max_analyst}
        message = st.session_state.graph.invoke(st.session_state.graph_input, config=thread)
        st.session_state.analysts = message["analysts"]  # Store analysts
        st.session_state.waiting_for_feedback = True  # Wait for feedback

    if st.session_state.feedbacks and st.session_state.waiting_for_feedback and st.session_state.feedback_submitted:
        print("Invoking the graph2")
        st.session_state.graph.update_state(config=thread, values={"human_analyst_feedback": st.session_state.feedbacks[-1]}, as_node="human_feedback")
        st.session_state.graph_input = None
        new_message = st.session_state.graph.invoke(st.session_state.graph_input, config=thread)
        st.session_state.analysts = new_message["analysts"]  # Store analysts
    
    if  st.session_state.feedback_completed: #not st.session_state.waiting_for_feedback:
        print("Invoking the graph3")
        st.session_state.graph_input = None
        st.session_state.graph.update_state(thread, {"human_analyst_feedback": None}, as_node="human_feedback")
        final_message = st.session_state.graph.invoke(st.session_state.graph_input, config=thread)
        st.write(final_message['final_report'])



    # Display Analysts (if available)
    if not st.session_state.feedback_completed:
        if st.session_state.analysts:
            for analyst in st.session_state.analysts:
                st.write(analyst.persona_markdown)
            
        st.session_state.feedback_submitted = False
        col1, col2 = st.columns(2)
        with col1:
            if st.button("provide New feedback"):
                st.session_state.show_feedback_form = True
        with col2:
            if st.button("Skip Feedback"):
                st.session_state.waiting_for_feedback = False
                st.session_state.feedback_completed = True
                st.rerun()

        if st.session_state.show_feedback_form:
            with st.form("feedback_form"):
                feedback = st.text_input("Enter your feedback:")
                if st.form_submit_button("Submit Feedback"):
                    st.session_state.feedbacks.append(feedback)
                    st.session_state.show_feedback_form = False
                    st.session_state.feedback_submitted = True
                    st.session_state.show_ask_feedback_radio = False  # Show thank you message
                    st.rerun()



# Render the button in a div with custom CSS
if st.sidebar.button("End Research", key="end_research"):
    reset_session_state()
    st.rerun()