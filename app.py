import os
import json
import requests
import numpy as np
from dotenv import load_dotenv
import chainlit as cl
from typing import List, TypedDict, Annotated, Sequence

from utility import append_to_response, remove_think, get_context, compress_context
from preprocessing import create_chunks, init_chroma, load_docs

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain.docstore.document import Document

# -------------------------------------------------------------------
# ENVIRONMENT & MODEL INITIALIZATION
# -------------------------------------------------------------------
load_dotenv()
MODEL_NAME       = os.getenv("MODEL_NAME", "qwen/qwen3-32b")
SERPER_API_KEY   = os.getenv("SERPER_API_KEY")
PDF_DIR          = os.getenv("PDF_DIR", "PDFs/")
ALL_DOCS_JSON    = os.getenv("ALL_DOCS_JSON", "all_docs.json")
CHROMA_DB_PATH   = os.getenv("CHROMA_DB_PATH", "chromaDB/saved/")
COLLECTION_NAME  = os.getenv("COLLECTION_NAME", "RAG_DOCS")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
pdfs_list = ['TESLA']

# Instantiate graph
class AgentState(TypedDict):
    """
    State dictionary storing chat messages and any user-specific data.

    Fields:
        messages: (Sequence[BaseMessage]): Conversation history for the agent 1.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Hybrid Search Tool
@tool
def hybrid_search(query: str) -> List[dict]:
    """
    Hybrid retrieval combining BM25 and vector search (ChromaDB) over local PDFs.

    Args:
        query (str): Natural language query string from the user.

    Returns:
        List[dict]: Top-matching chunks with 'text' and associated metadata.
    """
    chroma_store = init_chroma()
    docs = load_docs()

    if not docs:
        create_chunks(pdfs_list)
        docs = load_docs()

    if not docs:
        print("⚠️ No documents found for search.")
        return []

    # Ensure Chroma is populated
    if chroma_store._collection.count() == 0:
        chroma_store.add_documents(docs)

    n_docs = len(docs)
    safe_k = max(1, min(5, n_docs))      # prevent k=0 or k > corpus
    safe_fetch_k = max(safe_k, 5)        # ensure fetch_k ≥ k

    try:
        # BM25 Retriever
        bm25_ret = BM25Retriever.from_texts(
            [d.page_content for d in docs],
            metadatas=[d.metadata for d in docs],
            k=safe_k
        )

        # Vector Retriever
        vec_ret = chroma_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": safe_k,
                "fetch_k": safe_fetch_k,
                "lambda_mult": 0.5  # optional tuning
            }
        )

        # Hybrid Ensemble
        ensemble = EnsembleRetriever(
            retrievers=[bm25_ret, vec_ret],
            weights=[0.5, 0.5]
        )

        results: List[Document] = ensemble.invoke(query)

        # Format output
        context = [{"text": d.page_content, **d.metadata} for d in results]
        append_to_response([{"hybrid_search": context}], filename="check_agent_log.json")
        return context

    except ZeroDivisionError:
        print("❌ BM25 failed due to too few documents (division by zero).")
        return []
    except Exception as e:
        print(f"❌ Unexpected error in hybrid_search: {e}")
        return []

# Web Search Tools
@tool
def google_search(query: str, num: int = 10) -> dict:
    """
    Web search via Serper API for real-time news and factual queries.

    Args:
        query (str): Search string to send to the API.
        num (int): Number of top results to retrieve (default: 5).
        country: 

    Returns:
        dict: Parsed JSON response from Serper with search results.
    """
    url = "https://google.serper.dev/search"

    payload = json.dumps({
     "q": query,
  "gl": "in",
  "num": num,
  "tbs": "qdr:w"
})
    headers = {
    'X-API-KEY': SERPER_API_KEY,
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    append_to_response([{"Google search": response.text}], filename="check_agent_log.json")

    return response.text

@tool
def wiki_lookup(title: str, language: str = "en") -> dict:
    """
    Fetch full Wikipedia page content using the MediaWiki Action API.

    Args:
        title (str): Title of the Wikipedia page.
        language (str): Language code (e.g., 'en', 'hi', 'fr').

    Returns:
        dict: Dictionary with page existence, title, extract (intro), content (wikitext), and URL.
    """
    api_url = f"https://{language}.wikipedia.org/w/api.php"
    session = requests.Session()

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info",
        "exintro": True,
        "explaintext": True,
        "titles": title,
        "inprop": "url",
    }

    try:
        response = session.get(url=api_url, params=params)
        # print("Wiki--> ", response)
        response.raise_for_status()
        data = response.json()

        pages = data.get("query", {}).get("pages", {})
        page = next(iter(pages.values()))

        if "missing" in page:
            return {
                "exists": False,
                "error": f"PageError: The page titled '{title}' does not exist."
            }

        append_to_response([{"Wikipedia": page.get("extract")}], filename="check_agent_log.json")
        return {
            "exists": True,
            "page_id": page.get("pageid"),
            "title": page.get("title"),
            "summary": page.get("extract"),
            "content_url": page.get("fullurl")
        }

    except requests.RequestException as e:
        return {
            "exists": False,
            "error": f"RequestError: {str(e)}"
        }


# Financial Metrics Tools
@tool
def company_overview(symbol: str) -> dict:
    """
    Fetch company info & key financial metrics for a ticker via Alpha Vantage.

    Args:
        symbol (str): Stock ticker, e.g. "IBM" or "AAPL".

    Returns:
        dict: Overview fields such as Name, Exchange, MarketCap,
              P/E & PEG ratios, Dividends, Margins, Growth rates,
              Analyst targets, Valuation ratios, and 52‑week highs/lows.
    """
    # ensure .env has ALPHAVANTAGE_API_KEY=<your_key>
  
 
    if not ALPHAVANTAGE_API_KEY:
        raise ValueError("Set ALPHAVANTAGE_API_KEY in your environment before calling this tool.")
    
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": ALPHAVANTAGE_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    append_to_response([{"Company Overviw": data}], filename="check_agent_log.json")

    return data

@tool
def sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe Ratio for a return series.

    Formula: mean(returns - rf) / std(returns - rf)

    Args:
        returns (List[float]): Portfolio return time series.
        risk_free_rate (float): Risk-free rate baseline (default: 0.0).

    Returns:
        float: Computed Sharpe ratio.

    Raises:
        ValueError: If insufficient data or zero volatility.
    """
    arr = np.array(returns, dtype=float)
    excess = arr - risk_free_rate
    if arr.size < 2 or np.std(excess, ddof=1) == 0:
        raise ValueError("Insufficient data or zero volatility for Sharpe Ratio.")
    return float(np.mean(excess) / np.std(excess, ddof=1))

@tool
def batting_average(port: List[float], bench: List[float]) -> float:
    """
    Compute the batting average: fraction of periods where portfolio beats benchmark.

    Args:
        port (List[float]): Portfolio return series.
        bench (List[float]): Benchmark return series of equal length.

    Returns:
        float: Proportion of periods where port > bench.

    Raises:
        ValueError: If series lengths differ or are empty.
    """
    p = np.array(port)
    b = np.array(bench)
    if p.size != b.size or p.size == 0:
        raise ValueError("Return series must be equal-length non-empty arrays.")
    return float(np.sum(p > b) / p.size)

@tool
def capture_ratios(port: List[float], bench: List[float]) -> dict:
    """
    Compute up- and down-market capture ratios.

    Args:
        port (List[float]): Portfolio return series.
        bench (List[float]): Benchmark return series of equal length.

    Returns:
        dict: Contains 'up_capture' and 'down_capture' ratios.

    Raises:
        ValueError: If series lengths differ or are empty.
    """
    p = np.array(port)
    b = np.array(bench)
    if p.size != b.size or p.size == 0:
        raise ValueError("Return series must be equal-length non-empty arrays.")
    up = p[b > 0].sum() / b[b > 0].sum() if np.any(b > 0) else None
    down = p[b < 0].sum() / b[b < 0].sum() if np.any(b < 0) else None
    return {"up_capture": up, "down_capture": down}

@tool
def tracking_error(port: List[float], bench: List[float]) -> float:
    """
    Calculate the tracking error: standard deviation of active returns (port - bench).

    Args:
        port (List[float]): Portfolio return series.
        bench (List[float]): Benchmark return series of equal length.

    Returns:
        float: Tracking error.

    Raises:
        ValueError: If fewer than two observations.
    """
    diff = np.array(port) - np.array(bench)
    if diff.size < 2:
        raise ValueError("Need at least two observations for tracking error.")
    return float(np.std(diff, ddof=1))

@tool
def max_drawdown(returns: List[float]) -> float:
    """
    Compute the maximum drawdown for a return series.

    Args:
        returns (List[float]): Portfolio return time series.

    Returns:
        float: Maximum peak-to-trough drawdown.

    Raises:
        ValueError: If the return series is empty.
    """
    r = np.array(returns)
    if r.size == 0:
        raise ValueError("Empty return series.")
    wealth = np.cumprod(1 + r)
    peak = np.maximum.accumulate(wealth)
    return float(((wealth - peak) / peak).min())


# Graph Nodes for Taking input  
def input_query(state: AgentState) -> AgentState:
    """
    This node takes user input and ensures a clear, well-formed query is generated
    for subsequent processing by the LLM. It ensures input is gathered effectively
    and refined based on conversational history.
    """
    append_to_response([{"input_query": state["messages"]}], filename="check_agent_log.json")
    
    return {
        "messages": [state]
    }
 
# Agent for Query Redirection
def query_redirection_agent(state: AgentState) -> AgentState:
    """
    Entry node: Classify user intent to select the appropriate retrieval or calculation tool.
    """
    system_prompt = SystemMessage(
    content=(
            "You are a Retrieval‑Augmented Generation orchestrator. "
            "Analyze the user’s latest message, the conversation history, and any prior tool outputs choose exactly one tool to be called for giving a clear answer to user:\n"
            "  1. Calling Financial Metrics— when the user gives input data like risk rate, returns and ask to calculate some ratio.\n"
            "  2. Calling Company Overview— when the user ask about specific ratios, technical indicators, or financial analyses.\n"
            "  3. Calling  Wikipedia Search— when the user explicitly mentions “wiki” or requests historical or contextual background.\n"
            "  4. Calling Google Search — when the user asks for “latest”, “current”, “news”, or any real‑time factual update.\n"
            "  5. Calling Hybrid Search — as a fallback for general document retrieval from local PDFs (e.g. annual reports or SEC filings).\n"
            "  6. Return 'Moving to Check_Node' as response — when if the existing conversation already contains the context required to answer the query.\n\n"
            f"Here is the conversational History : {get_context(state)}"
            "Don't change the user query such that it looses small details while passing it to tools"
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
    check_content: checks wether reterived content from tools is relevent to user query and previous conversion history

    """
    final_prompt = SystemMessage( content=(
         "You are a financial RAG assistant. "
        "Analyze the user’s latest query, retrieved tool outputs, and conversation history to decide the next step:\n"
        """  
        1. return  “expand_query” as response: 
        - if the user’s query is ambiguous
        - if user’s query is factually incorrect
        - if context is not enough to answer the query.
        - if context is not relevent to answer the query.\n"""

        """  
        2. return  “answer_query” as response : if the retrieved content fully addresses the user’s information needs.\n\n"""
        "If there has been more than 3 tries of “expand_query” then  Call  “answer_query” to avoid looping."
        "Remove reason content from response and return response in specified format only"
        f"Take help from this conversational history {[get_context(state)]} to decide which tool to call"
        )
    )
    llm = ChatGroq(model=MODEL_NAME)
    llm_response = llm.invoke([final_prompt])
    append_to_response([{"check_agent":llm_response}], filename="check_agent_log.json")

    return {"messages": [llm_response]}

# Graph Nodes for expanding query  
def expand_query(
  state: AgentState
) -> AgentState:
    """
    Craft a single, optimized financial search query based on the complete
    conversation history and prior user data.

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
            "You are a Financial Retrieval‑Augmented Generation assistant. "
            "Your task is to produce exactly one search query that can retrieve context required for asnwering user query with precision. "
            "Use the full conversation context—including any silent reasoning and tool outputs—to fix conceptual errors."
            "and ensure maximal relevance."
            
        )
    )
    context_str = get_context(state,7)
    human = HumanMessage(
        content=(
            f"RECENT CONVERSATION:\n{context_str}\n\n."
            "Identify gaps in the context for answering the user query, and create one optimized search query."
            "Include or correct any relevant tickers, ISINs, and financial terminology with specificity only if there absence have resulted in incorrect retrieval or any tool failure."
            "Return only the query string in question format, without explanations."
            "Return format must be like optimised_query: "
            "After the query their should be no additional content. "
            "Format must be like Expanded Query: "
        )
    )

    # Invoke LLM
    llm = ChatGroq(model=MODEL_NAME, temperature=0.8)
    response = llm.invoke([system, human])

    append_to_response(
        [{"expand_query": response}],
        filename="check_agent_log.json"
    )

    return {"messages": [response]}

# Graph Nodes for Answering the query
def answer_query(state: AgentState) -> AgentState:
    """

    This function takes the context_str which contains the conversation history and any tool results,
    and passes them to a language model to generate a concise, structured, and accurate final response
    for the user.

    The response:
      - Should be well-organized (e.g., using bullet points or headings).
      - Should cite tool outputs and sources.
      - Should avoid hallucinating or fabricating facts.
      - Should acknowledge missing or incomplete data.
    
    Return: the string which contains the final answer.
    """
    context_str = get_context(state)
    final_prompt = SystemMessage(
        content=(
            "You are a Financial RAG assistant integrating tool outputs and conversation history. "
            "When crafting your answer:\n"
            "  • Be concise yet thorough; structure with headings or bullet points when helpful.\n"
            "  • Cite any tool or external data you used, and link to sources if available.\n"
            "  • If data is missing or incomplete, acknowledge it explicitly.\n"
            "  • Maintain accuracy and clarity—do not hallucinate.\n"
            "  • Mention the source of the information like document and links.\n"
            "  • answer the query from context ask user for analysis or feedback\n"
            f" Here is context required to answer the query {context_str}"
            "Format must be like Answer: "
        )
    )
    llm = ChatGroq(model=MODEL_NAME)
    llm_response = llm.invoke([final_prompt])
    append_to_response([{"answer_query": llm_response}], filename="check_agent_log.json")
    
    return {"messages": [llm_response]}

# Main Agent 
llm_query_redirector = ChatGroq(model=MODEL_NAME).bind_tools(
    [hybrid_search, google_search, wiki_lookup, company_overview, sharpe_ratio,
     batting_average, capture_ratios, tracking_error, max_drawdown]
)



# Instantiate graph
graph = StateGraph(AgentState)

# Register and wire nodes
graph.add_node('Input_Query', input_query)
graph.add_node('Query_Redirection_Agent', query_redirection_agent)

graph.add_node('Hybrid_Node_ToolNode', ToolNode([hybrid_search]))
graph.add_node('Web_Node_ToolNode', ToolNode([google_search, wiki_lookup]))
graph.add_node('Fin_Node_ToolNode', ToolNode([company_overview,sharpe_ratio, batting_average,
                                     capture_ratios, tracking_error, max_drawdown]))

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
        # catch every finance‐related tool in FinNode
        if tool_name in (
            "company_overview",
            "sharpe_ratio",
            "batting_average",
            "capture_ratios",
            "tracking_error",
            "max_drawdown"
        ):
            return "Fin_Node_ToolNode"
        if tool_name in ("google_search", "wiki_lookup"):
            return "Web_Node_ToolNode"
        if tool_name == "hybrid_search":
            return "Hybrid_Node_ToolNode"
    if "Check_Node" in content:
            return "Check_Node"  
    # fallback
    return "Hybrid_Node_ToolNode"

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
    'Hybrid_Node_ToolNode': 'Hybrid_Node_ToolNode',
    'Web_Node_ToolNode': 'Web_Node_ToolNode',
    'Fin_Node_ToolNode': 'Fin_Node_ToolNode',
    'Check_Node':'Check_Node'
})

for node in ['Hybrid_Node_ToolNode', 'Web_Node_ToolNode', 'Fin_Node_ToolNode']:
    graph.add_edge(node, 'Check_Node')

graph.add_conditional_edges('Check_Node', route_answer, {
     
    "Expand_Query": "Expand_Query",
    "Answer_Query": "Answer_Query"
})
graph.add_edge("Expand_Query","Query_Redirection_Agent")
graph.add_edge("Answer_Query",END)

app = graph.compile()
chat_state = []

@cl.on_chat_start
async def setup():
    # Initialize state when chat starts
    global chat_state
    chat_state = AgentState({"messages": []})
    await cl.Message("✅ Ask me anything!").send()

@cl.on_message
async def chat(message: str):
    global chat_state
    state = chat_state

    content = compress_context(state)
    prev_compressed = AIMessage(content=content)
    query = HumanMessage(content=message.content)

    new_state = AgentState({"messages":[prev_compressed, query]})
    new_state = app.invoke(new_state, config={"recursion_limit": 50})
    chat_state = new_state

    # Extract latest message
    last_message = new_state["messages"][-1]["content"]  # Adjust key if structure differs

    await cl.Message(last_message).send()