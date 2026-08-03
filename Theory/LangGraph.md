# LangGraph: Stateful Multi-Agent Orchestration Framework

## 1. What is LangGraph?

**LangGraph** is an open-source framework developed by LangChain designed for building **stateful, multi-actor, agentic applications** using Large Language Models (LLMs).

While traditional LLM chains (like standard LangChain LCEL) are linear Directed Acyclic Graphs (DAGs) that execute steps sequentially, real-world autonomous agents require **loops, cycles, state persistence, branching logic, and human-in-the-loop interactions**. LangGraph models application logic as a **state machine graph**.

```text
┌──────────────────────────────────────────────────────────┐
│                      LangGraph State                     │
│  { question: "...", documents: [...], generation: "..." }│
└──────────────┬────────────────────────────▲──────────────┘
               │                            │
               ▼                            │ Updates State
      ┌────────────────┐           ┌────────┴───────┐
      │  Retrieve Node ├──────────►│  Grade Node    │
      └────────────────┘           └────────┬───────┘
                                            │
                                  Conditional Routing
                                  ┌─────────┴─────────┐
                                  ▼                   ▼
                          [Context Relevant]   [Context Irrelevant]
                                  │                   │
                                  ▼                   ▼
                         ┌────────────────┐  ┌─────────────────┐
                         │ Generate Node  │  │ Rewrite Query   │
                         └────────────────┘  └─────────────────┘
```

---

## 1.1 The LangGraph Mental Model: The Central Blackboard

To intuitively understand LangGraph, use the **Central Blackboard & Specialized Workers** mental model:

```mermaid
flowchart TD
    classDef board fill:#fff9c4,stroke:#fbc02d,stroke-width:3px;
    classDef worker fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef router fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    BOARD[("SHARED BLACKBOARD (GRAPH STATE)<br/>{ messages: [...], documents: [...], step: 2 }")]:::board

    WORKER1["Node 1: Retriever Worker<br/>Reads question -> Writes documents"]:::worker <-->|Read & Update State| BOARD
    WORKER2["Node 2: Evaluator Worker<br/>Reads documents -> Writes is_relevant"]:::worker <-->|Read & Update State| BOARD
    ROUTER["Conditional Edge Router<br/>Reads is_relevant -> Chooses next Worker"]:::router <-->|Inspect State| BOARD
    WORKER3["Node 3: Generator Worker<br/>Reads documents -> Writes final response"]:::worker <-->|Read & Update State| BOARD
```

### The 4 Mental Pillars:

1. **The Shared Blackboard (`State`)**:
   - Imagine a blackboard in the middle of a room holding all current facts, messages, and variables.
   - Nodes do not pass data directly to each other; they **read from** and **write updates to** this single shared blackboard.

2. **Specialized Workers (`Nodes`)**:
   - Each Node is an isolated worker with one job (e.g., *“I retrieve documents”*, *“I grade relevance”*, *“I generate text”*).
   - Every worker receives the blackboard, does its job, and writes its result back.

3. **Traffic Controllers (`Edges & Routers`)**:
   - **Fixed Edges**: Direct instructions (*“After Worker 1 finishes, send the board to Worker 2”*).
   - **Conditional Routers**: Traffic guards that look at the blackboard state (*“If `is_relevant == True`, route to Worker 3; else route to Query Rewriter”*).

4. **Snapshot Camera (`Checkpointer / Memory`)**:
   - After every worker finishes, LangGraph takes a snapshot photo of the blackboard.
   - This allows pausing execution for human review, rewinding time, or resuming multi-turn chat sessions seamlessly.

---

## 2. Core Concepts & Building Blocks

> [!IMPORTANT]
> LangGraph centers on three fundamental abstractions: **State**, **Nodes**, and **Edges**.

### 1. Graph State (`State`)
The **State** is a centralized data structure that represents the current snapshot of the system. Every node in the graph receives the current state, performs operations, and returns state updates.

### 2. Nodes (`Nodes`)
Nodes are standard functions (Python or TypeScript) that perform discrete tasks (e.g., querying a vector database, calling an LLM, parsing JSON, or performing web searches).
- Inputs: Current `State`
- Outputs: Dict containing state updates

### 3. Edges (`Edges`)
Edges define the control flow transitions between nodes:
- **Normal Edges**: Direct transitions (e.g., Node A $\rightarrow$ Node B).
- **Conditional Edges**: Dynamic routing decisions based on state evaluation (e.g., IF context is relevant $\rightarrow$ Generate, ELSE $\rightarrow$ Rewrite Query).
- **Entry & Exit Points**: Special sentinel nodes `START` and `END`.

### 4. Checkpointers & Persistence
LangGraph natively saves state snapshots at every step. This unlocks:
- **Thread Persistence**: Multi-turn conversation history.
- **Time-Travel**: Rewind execution to past states and re-run with modified inputs.
- **Human-in-the-Loop**: Pause execution before critical actions (e.g., sending an email or executing code) for user approval.

---

## 3. LangGraph Workflow Diagrams

### 3.1 Agentic RAG State Machine Workflow Diagram

```mermaid
flowchart TD
    classDef startend fill:#ffe0b2,stroke:#f57c00,stroke-width:2px;
    classDef nodeStyle fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;

    START_NODE([START]):::startend --> RETRIEVE[Retrieve Documents Node]:::nodeStyle
    RETRIEVE --> GRADE[Grade Documents Node]:::nodeStyle
    
    GRADE --> ROUTE{Is Context Relevant?}:::decision
    
    ROUTE -->|YES: Context Relevant| GENERATE[Generate Answer Node]:::nodeStyle
    ROUTE -->|NO: Context Irrelevant| REWRITE[Rewrite Query Node]:::nodeStyle
    
    REWRITE -->|Update State Query| RETRIEVE
    GENERATE --> END_NODE([END]):::startend
```

---

### 3.2 Self-RAG Self-Correction Workflow Diagram

```mermaid
flowchart TD
    classDef step fill:#e8eaf6,stroke:#3f51b5,stroke-width:2px;
    classDef check fill:#fff3e0,stroke:#e65100,stroke-width:2px;

    Q[User Input Query]:::step --> RET[Retrieve Chunks]:::step
    RET --> EVAL_DOCS{Grade Chunk Relevance}:::check
    
    EVAL_DOCS -->|Relevant| GEN[LLM Generation]:::step
    EVAL_DOCS -->|Irrelevant| REWRITE_Q[Rewrite Query]:::step --> RET
    
    GEN --> EVAL_HALLUCINATION{Check Hallucination}:::check
    
    EVAL_HALLUCINATION -->|Factually Grounded| FINAL[Output Answer]:::step
    EVAL_HALLUCINATION -->|Hallucination Detected| RE_GEN[Re-Generate Answer]:::step --> EVAL_HALLUCINATION
```

---

### 3.3 Corrective RAG (CRAG) Fallback Workflow Diagram

```mermaid
flowchart TD
    classDef node fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef router fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;

    Q[User Query]:::node --> VEC_SEARCH[Vector Database Search]:::node
    VEC_SEARCH --> EVAL_CONF{Evaluate Score Confidence}:::router
    
    EVAL_CONF -->|High Relevance| LOCAL_CTX[Use Vector Store Chunks]:::node
    EVAL_CONF -->|Low Relevance / Missing| WEB_SEARCH[Fallback Web Search API]:::node
    
    LOCAL_CTX --> LLM_GEN[LLM Synthesis]:::node
    WEB_SEARCH --> LLM_GEN
    LLM_GEN --> OUTPUT[Final Response]:::node
```

---

## 4. Code Implementation Examples

### 🐍 Python Implementation (`agentic_rag.py`)

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Define State Schema
class AgentState(TypedDict):
    question: str
    documents: List[str]
    generation: str
    is_relevant: bool

# 2. Initialize Models & Tools
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. Define Node Functions
def retrieve_node(state: AgentState):
    """Retrieve relevant chunks from vector store."""
    print("--- RETRIEVING DOCUMENTS ---")
    retrieved_docs = ["Chunk 1: Remote work policy...", "Chunk 2: Annual leave..."]
    return {"documents": retrieved_docs}

def grade_node(state: AgentState):
    """Grade whether retrieved context is sufficient."""
    print("--- GRADING CONTEXT ---")
    docs = state["documents"]
    # Check if context contains useful info
    is_good = len(docs) > 0 and "Remote work" in docs[0]
    return {"is_relevant": is_good}

def generate_node(state: AgentState):
    """Synthesize final answer using retrieved context."""
    print("--- GENERATING ANSWER ---")
    context = "\n".join(state["documents"])
    response = llm.invoke(f"Context: {context}\nQuestion: {state['question']}")
    return {"generation": response.content}

def rewrite_query_node(state: AgentState):
    """Rewrite question for better retrieval."""
    print("--- REWRITING QUERY ---")
    new_query = f"Detailed info on: {state['question']}"
    return {"question": new_query}

# 4. Define Conditional Edge Router
def decide_to_generate(state: AgentState):
    if state["is_relevant"]:
        return "generate"
    else:
        return "rewrite"

# 5. Build StateGraph Workflow
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_node)
workflow.add_node("generate", generate_node)
workflow.add_node("rewrite", rewrite_query_node)

# Add Edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade")

# Add Conditional Edge
workflow.add_conditional_edges(
    "grade",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

# Compile Application Graph
app = workflow.compile()

# Execute Graph
output = app.invoke({"question": "How many remote days do I get?"})
print("Final Output:", output["generation"])
```

---

### ⚡ TypeScript / JavaScript Implementation (`agentic_rag.ts`)

```typescript
import { StateGraph, Annotation, START, END } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";

// 1. Define State Annotation
const GraphAnnotation = Annotation.Root({
  question: Annotation<string>(),
  documents: Annotation<string[]>(),
  generation: Annotation<string>(),
  isRelevant: Annotation<boolean>(),
});

// 2. Define State Machine Graph
const workflow = new StateGraph(GraphAnnotation)
  .addNode("retrieve", async (state) => {
    console.log("--- RETRIEVING DOCUMENTS ---");
    return { documents: ["Remote work policy chunk..."] };
  })
  .addNode("grade", async (state) => {
    console.log("--- GRADING CONTEXT ---");
    const isGood = state.documents.length > 0;
    return { isRelevant: isGood };
  })
  .addNode("generate", async (state) => {
    console.log("--- GENERATING ANSWER ---");
    const llm = new ChatOpenAI({ modelName: "gpt-4o-mini" });
    const response = await llm.invoke(`Question: ${state.question}\nContext: ${state.documents.join("\n")}`);
    return { generation: response.content as string };
  })
  .addNode("rewrite", async (state) => {
    console.log("--- REWRITING QUERY ---");
    return { question: `Expanded query: ${state.question}` };
  })
  .addEdge(START, "retrieve")
  .addEdge("retrieve", "grade")
  .addConditionalEdges("grade", (state) => (state.isRelevant ? "generate" : "rewrite"), {
    generate: "generate",
    rewrite: "rewrite",
  })
  .addEdge("rewrite", "retrieve")
  .addEdge("generate", END);

// Compile & Execute
const app = workflow.compile();
const result = await app.invoke({ question: "Remote work allowance?" });
console.log(result.generation);
```

---

## 5. Key Agentic RAG Architecture Patterns Powered by LangGraph

| Pattern | Description | Key Advantage |
| :--- | :--- | :--- |
| **Self-RAG** | The model retrieves documents, grades chunk relevance, generates an answer, and grades whether the answer is hallucinated or factually grounded. | Eliminates hallucinations by self-correcting faulty generations. |
| **Corrective RAG (CRAG)** | Evaluates retrieval confidence score. If vector store retrieval score is low, falls back to web search (Tavily/Google API). | Prevents system failure when private knowledge base misses the topic. |
| **Adaptive RAG** | Classifies query intent first and dynamically routes simple questions directly to LLM, complex facts to Vector DB, and real-time events to Web Search. | Reduces latency and API tokens by selecting optimal retrieval path. |

---

## 6. LangChain Chains vs. LangGraph: Key Differences

| Feature | Standard LangChain LCEL | LangGraph |
| :--- | :--- | :--- |
| **Graph Topology** | Linear / DAG (Directed Acyclic Graph) | Cyclic Graph (Supports Loops & State Machines) |
| **State Management** | Passed through pipeline inputs/outputs | Centrally managed State object with full persistence |
| **Execution Control** | Fixed sequence | Dynamic conditional branching & re-execution |
| **Human-in-the-Loop** | Difficult to interrupt / resume | Native checkpointers (`interrupt_before`, `interrupt_after`) |
| **Multi-Agent Teams** | Complex custom code | Built-in Multi-Agent node collaboration graphs |

---

## 7. How LLMs are Used in LangGraph

In a LangGraph application, LLMs fulfill 3 distinct roles:

```mermaid
flowchart TD
    subgraph Roles ["Roles of LLMs in LangGraph"]
        R1["1. Node Processor<br/>(Generate answers, summarize, draft code)"]
        R2["2. Decision Router<br/>(Structured Output for Conditional Edges)"]
        R3["3. Tool Caller<br/>(Emit JSON schema requests to ToolNode)"]
    end
```

### 1. Supported LLM Providers & Initialization

LangGraph supports any LLM that integrates with LangChain's `BaseChatModel` interface:

```python
# OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Anthropic Claude
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

# Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Open-Source Local Models (Ollama)
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1", temperature=0)
```

### 2. Pattern A: LLM as Node Processor
Inside a node, the LLM takes state context and updates state fields:
```python
def generate_node(state: State):
    # LLM processes state and synthesizes output
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

### 3. Pattern B: LLM as Conditional Router (Structured Output)
Using `.with_structured_output(PydanticSchema)`, the LLM returns a strict JSON object used by conditional edges to route execution:
```python
from pydantic import BaseModel, Field

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeDocuments)

def grade_node(state: State):
    score = structured_llm_grader.invoke(...)
    return {"is_relevant": score.binary_score == "yes"}
```

### 4. Pattern C: LLM as Autonomous Tool Caller
Using `.bind_tools(tools)`, the LLM automatically emits tool call requests that LangGraph routes to the prebuilt `ToolNode`:
```python
llm_with_tools = llm.bind_tools([calculator_tool, web_search_tool])
```
