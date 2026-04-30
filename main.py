import os
import json
import requests
import numpy as np
from dotenv import load_dotenv
from typing import List, TypedDict, Annotated, Sequence

from utility import append_to_response, remove_think, get_context, compress_context
from preprocessing import create_chunks, init_chroma, load_docs

from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


from langchain_classic.retrievers import EnsembleRetriever 
from langchain_community.retrievers import BM25Retriever 
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage 
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool 
from langgraph.graph import StateGraph, END 
from langgraph.prebuilt import ToolNode 
from langgraph.graph.message import add_messages 
from langchain_core.documents import Document


# -------------------------------------------------------------------
# ENVIRONMENT & MODEL INITIALIZATION
# -------------------------------------------------------------------
load_dotenv()
MODEL_NAME       = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL  = os.getenv("OPENAI_BASE_URL", "https://aihubmix.com/v1")
PDF_DIR          = os.getenv("PDF_DIR", "PDFs/")
ALL_DOCS_JSON    = os.getenv("ALL_DOCS_JSON", "all_docs.json")
CHROMA_DB_PATH   = os.getenv("CHROMA_DB_PATH", "chromaDB/saved/")
COLLECTION_NAME  = os.getenv("COLLECTION_NAME", "LEARNING_DOCS")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# Learning subjects list (Software Testing focused)
subjects_list = ['Software Testing Fundamentals', 'Test Case Design', 'Black Box Testing', 
                 'White Box Testing', 'Integration Testing', 'System Testing', 
                 'Acceptance Testing', 'Test Automation', 'Performance Testing', 
                 'Security Testing', 'Test Management']

# Instantiate graph
class AgentState(TypedDict):
    """
    State dictionary storing chat messages and any user-specific data.

    Fields:
        messages: (Sequence[BaseMessage]): Conversation history for the agent 1.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Software Testing Learning Tools
@tool
def knowledge_search(query: str) -> List[dict]:
    """
    Search for software testing knowledge from local documents.

    Args:
        query (str): Natural language query about software testing topics.

    Returns:
        List[dict]: Top-matching chunks with 'text' and metadata.
    """
    chroma_store = init_chroma()
    docs = load_docs()

    if not docs:
        create_chunks(subjects_list)
        docs = load_docs()

    if not docs:
        print("⚠️ No documents found for search.")
        return []

    # Ensure Chroma is populated
    if chroma_store._collection.count() == 0:
        chroma_store.add_documents(docs)

    n_docs = len(docs)
    safe_k = max(1, min(5, n_docs))
    safe_fetch_k = max(safe_k, 5)

    try:
        bm25_ret = BM25Retriever.from_texts(
            [d.page_content for d in docs],
            metadatas=[d.metadata for d in docs],
            k=safe_k
        )

        vec_ret = chroma_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": safe_k,
                "fetch_k": safe_fetch_k,
                "lambda_mult": 0.5
            }
        )

        ensemble = EnsembleRetriever(
            retrievers=[bm25_ret, vec_ret],
            weights=[0.5, 0.5]
        )

        results: List[Document] = ensemble.invoke(query)
        context = [{"text": d.page_content, **d.metadata} for d in results]
        append_to_response([{"knowledge_search": context}], filename="check_agent_log.json")
        return context

    except ZeroDivisionError:
        print("❌ BM25 failed due to too few documents.")
        return []
    except Exception as e:
        print(f"❌ Unexpected error in knowledge_search: {e}")
        return []

# Software Testing Learning Tools
@tool
def generate_question(topic: str, difficulty: str = "medium") -> str:
    """
    Generate a test question about software testing concepts.

    Args:
        topic (str): The software testing topic.
        difficulty (str): Difficulty level - 'easy', 'medium', 'hard'.

    Returns:
        str: A generated question.
    """
    # This would use LLM to generate questions based on retrieved knowledge
    # For now, return a placeholder
    return f"What is the key concept in {topic} at {difficulty} level?"

@tool
def evaluate_answer(question: str, user_answer: str, correct_answer: str) -> dict:
    """
    Evaluate user's answer against software testing knowledge.

    Args:
        question (str): The question asked.
        user_answer (str): User's response.
        correct_answer (str): The correct answer.

    Returns:
        dict: Evaluation with score and feedback.
    """
    # Simple evaluation - in practice, use LLM for better assessment
    score = 0.5 if user_answer.lower() in correct_answer.lower() else 0.0
    feedback = "Good job!" if score > 0.5 else "Try again."
    return {"score": score, "feedback": feedback}

@tool
def provide_explanation(topic: str, context: str) -> str:
    """
    Provide detailed explanation for software testing concepts.

    Args:
        topic (str): The topic to explain.
        context (str): Retrieved context from documents.

    Returns:
        str: Detailed explanation.
    """
    # Use LLM to generate explanation based on context
    return f"Explanation for {topic}: {context[:200]}..."

@tool
def design_test_cases(requirements: str, code_snippet: str = "") -> List[dict]:
    """
    Design test cases based on requirements and/or code.

    Args:
        requirements (str): User requirements or specifications.
        code_snippet (str): Optional code snippet to analyze.

    Returns:
        List[dict]: List of designed test cases with details.
    """
    # This would use LLM to analyze requirements and generate test cases
    # For now, return placeholder test cases
    test_cases = [
        {
            "id": "TC001",
            "title": "Basic functionality test",
            "description": f"Test basic functionality based on: {requirements[:100]}...",
            "preconditions": "System is running",
            "steps": ["Step 1", "Step 2", "Step 3"],
            "expected_result": "Expected behavior",
            "priority": "High"
        }
    ]
    return test_cases


# Graph Nodes for Taking input  
def input_query(state: AgentState) -> AgentState:
    """
    This node takes user input and ensures a clear, well-formed query is generated
    for subsequent processing by the LLM. It ensures input is gathered effectively
    and refined based on conversational history.
    """

    response = ''
    if len(state["messages"]) == 0:
        response = "Enter your Query"
    else:
        response = remove_think(state["messages"][-1].content)
    user_input = input(f'🤖: {response}\n\nUser: ')
    query = HumanMessage(content=user_input)
    
   
    content = compress_context(state)
    prev_compressed_responses = AIMessage(content=content)

    append_to_response([{"input_query": query, "previous_context":prev_compressed_responses}], filename="check_agent_log.json")
    
    state["messages"] = []
    return {
        "messages": [query,prev_compressed_responses]
    }
 
# Agent for Query Redirection
def query_redirection_agent(state: AgentState) -> AgentState:
    """
    Entry node: Classify user intent for software testing education.
    """
    system_prompt = SystemMessage(
    content=(
            "You are a Software Testing Teaching Assistant. "
            "Analyze the user's latest message and conversation history to choose exactly one tool:\n"
            "  1. Calling Knowledge Search — for retrieving information from software testing documents.\n"
            "  2. Calling Generate Question — when user wants to practice software testing concepts.\n"
            "  3. Calling Evaluate Answer — when user provides an answer to evaluate.\n"
            "  4. Calling Provide Explanation — when user needs detailed explanation of testing concepts.\n"
            "  5. Calling Design Test Cases — when user uploads requirements or code for test case design.\n"
            "  6. Return 'Moving to Check_Node' as response — when context is sufficient.\n\n"
            f"Here is the conversational History : {get_context(state)}"
            "It is mandatory to choose one of 6 options"
    )
)

    llm_response = llm_query_redirector.invoke([system_prompt])
    # print('LLM 1 --> ',llm_response)
    append_to_response([{"query_redirection_agent":llm_response}], filename="check_agent_log.json")
    
    return {"messages": [llm_response]}
    # return {"messages": [AIMessage(content=llm_response.content, kwargs=llm_response.additional_kwargs)]}

# Graph Nodes for Checking Answer   
def check_content(state: AgentState)->AgentState:
    """
    Check if retrieved content is relevant to the software testing query.

    """
    final_prompt = SystemMessage( content=(
        "You are a Software Testing Teaching Assistant. "
        "Analyze the user's latest query, retrieved tool outputs, and conversation history to decide the next step:\n"
        "1. return 'expand_query' as response: \n"
        "- if the query is ambiguous\n"
        "- if context is not enough to answer the query.\n"
        "- if context is not relevant to the software testing topic.\n\n"
        "2. return 'answer_query' as response : if the retrieved content fully addresses the user's learning needs.\n\n"
        "If there has been more than 3 tries of 'expand_query' then Call 'answer_query' to avoid looping."
        "Remove reason content from response and return response in specified format only"
        f"Take help from this conversational history {[get_context(state)]} to decide which tool to call"
        )
    )
    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_BASE_URL
    )
    llm_response = llm.invoke([final_prompt])
    append_to_response([{"check_agent":llm_response}], filename="check_agent_log.json")

    return {"messages": [llm_response]}

# Graph Nodes for expanding query  
def expand_query(
  state: AgentState
) -> AgentState:
    """
    Craft a single, optimized software testing search query based on conversation history.

    Args:
        context_str: conversation history for the context
        temperature (float): Sampling temperature for the LLM.

    Returns:
        str: One optimized search query string.
    """

    # Build context from state

    # Prepare LLM prompts
    system = SystemMessage(
        content=(
            "You are a Software Testing Teaching Assistant. "
            "Your task is to produce exactly one search query that can retrieve software testing context. "
            "Use the full conversation context to fix gaps in understanding."
            "Ensure maximal relevance for software testing education."
        )
    )
    context_str = get_context(state,7)
    human = HumanMessage(
        content=(
            f"RECENT CONVERSATION:\n{context_str}\n\n."
            "Identify gaps in software testing context and create one optimized search query."
            "Include relevant testing concepts, methods, or tools with specificity."
            "Return only the query string, without explanations."
            "Return format must be like optimised_query: "
            "After the query their should be no additional content. "
            "Format must be like Expanded Query: "
        )
    )

    # Invoke LLM
    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_BASE_URL,
        temperature=0.8
    )
    response = llm.invoke([system, human])

    append_to_response(
        [{"expand_query": response}],
        filename="check_agent_log.json"
    )

    return {"messages": [response]}

# Graph Nodes for Answering the query
def answer_query(state: AgentState) -> AgentState:
    """

    This function generates a final response for the software testing query.

    The response:
      - Should be educational for software testing.
      - Should cite sources.
      - Should provide practical testing knowledge.
    
    Return: the string which contains the final answer.
    """
    context_str = get_context(state)
    final_prompt = SystemMessage(
        content=(
            "You are a Software Testing Teaching Assistant integrating tool outputs and conversation history. "
            "When crafting your answer:\n"
            "  • Be educational and practical for software testing.\n"
            "  • Structure with clear testing concepts and examples.\n"
            "  • Cite any sources used.\n"
            "  • Provide real-world testing scenarios when helpful.\n"
            "  • Suggest testing best practices or next learning steps.\n"
            f" Here is context required to answer the query {context_str}"
            "Format must be like Answer: "
        )
    )
    llm = ChatOpenAI(
        model=MODEL_NAME,
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_BASE_URL
    )
    llm_response = llm.invoke([final_prompt])
    append_to_response([{"answer_query": llm_response}], filename="check_agent_log.json")
    
    return {"messages": [llm_response]}

# Main Agent 
llm_query_redirector = ChatOpenAI(
    model=MODEL_NAME,
    openai_api_key=OPENAI_API_KEY,
    openai_api_base=OPENAI_BASE_URL
).bind_tools(
    [knowledge_search, generate_question, evaluate_answer, provide_explanation, design_test_cases]
)



# Instantiate graph
graph = StateGraph(AgentState)

# Register and wire nodes
graph.add_node('Input_Query', input_query)
graph.add_node('Query_Redirection_Agent', query_redirection_agent)

graph.add_node('Knowledge_Node', ToolNode([knowledge_search]))
graph.add_node('Question_Node', ToolNode([generate_question]))
graph.add_node('Evaluate_Node', ToolNode([evaluate_answer]))
graph.add_node('Explain_Node', ToolNode([provide_explanation]))
graph.add_node('Design_Test_Node', ToolNode([design_test_cases]))

graph.add_node('Check_Node', check_content)
graph.add_node('Expand_Query', expand_query)
graph.add_node('Answer_Query',answer_query)

# router for query redirection
def route_redirector(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    content = remove_think(last_msg.content)
    calls   = getattr(last_msg, "additional_kwargs", {}).get("tool_calls", [])
    if calls:
        tool_name = calls[0]["function"]["name"]
        if tool_name == "knowledge_search":
            return "Knowledge_Node"
        if tool_name == "generate_question":
            return "Question_Node"
        if tool_name == "evaluate_answer":
            return "Evaluate_Node"
        if tool_name == "provide_explanation":
            return "Explain_Node"
        if tool_name == "design_test_cases":
            return "Design_Test_Node"
    if "Check_Node" in content:
            return "Check_Node"  
    # fallback
    return "Knowledge_Node"

# router for checking Answer
def route_answer(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    content = remove_think(last_msg.content).lower()
    if content:
        if "expand_query" in content or "expand" in content:
            return "Expand_Query"
        if "answer_query" in content or "final" in content:
            return "Answer_Query"

       
        
    # fallback
    return "Answer_Query"

# Graph wiring
graph.set_entry_point('Input_Query')
graph.add_edge('Input_Query',"Query_Redirection_Agent")


graph.add_conditional_edges('Query_Redirection_Agent', route_redirector, {
    'Knowledge_Node': 'Knowledge_Node',
    'Question_Node': 'Question_Node',
    'Evaluate_Node': 'Evaluate_Node',
    'Explain_Node': 'Explain_Node',
    'Design_Test_Node': 'Design_Test_Node',
    'Check_Node':'Check_Node'
})

for node in ['Knowledge_Node', 'Question_Node', 'Evaluate_Node', 'Explain_Node', 'Design_Test_Node']:
    graph.add_edge(node, 'Check_Node')

graph.add_conditional_edges('Check_Node', route_answer, {
     
    "Expand_Query": "Expand_Query",
    "Answer_Query": "Answer_Query"
})
graph.add_edge("Expand_Query","Query_Redirection_Agent")
graph.add_edge("Answer_Query",END)

app = graph.compile()


# Invoking the Graph
if __name__ == "__main__":
 
    init_state = AgentState({"messages":[]})
    while True:
        init_state = app.invoke(init_state, config={"recursion_limit": 50})



