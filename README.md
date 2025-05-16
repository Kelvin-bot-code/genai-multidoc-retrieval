## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
Accessing and synthesizing information from multiple documents is crucial for research, but manual analysis is time-consuming. A multidocument retrieval agent can automate this process by:

   1.Parsing and indexing multiple research articles.
   
   2.Enabling users to ask queries in natural language.
   
   3.Providing synthesized, concise, and accurate responses from the indexed documents.
   
The effectiveness of the system will be evaluated through diverse queries to test its accuracy and relevance.

### DESIGN STEPS:
## STEP 1: Load and Parse Research Articles
Use LlamaIndex's document loaders to read and parse multiple research articles in PDF or text format.

## STEP 2: Create a Unified Index
Combine and index content from all documents using LlamaIndex to enable cross-document retrieval.

## STEP 3: Set Up a Query Engine
Configure a query engine to allow natural language questions and retrieve relevant content.

## STEP 4: Implement the Retrieval Agent
Create an agent using AgentRunner with the configured worker.
This agent handles user queries over the indexed papers.

## STEP 5: Evaluate the Agent
Ask natural language questions about each paper.
Print the agent's clean, synthesized responses.

### PROGRAM:
```py
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()

import nest_asyncio
nest_asyncio.apply()

urls = [
    "https://openreview.net/pdf?id=VtmBAGCN7o",
    "https://openreview.net/pdf?id=6PmJoRfdaK",
    "https://openreview.net/pdf?id=hSyW5go0v8",
]

papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "selfrag.pdf",
]

from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

from llama_index.llms.openai import OpenAI
llm = OpenAI(model="gpt-3.5-turbo")

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

# ✅ Set verbose=False to suppress function logs
agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools,
    llm=llm,
    verbose=False
)

agent = AgentRunner(agent_worker)

# === QUERY 1 ===
response1 = agent.query(
    "Tell me about the evaluation dataset used in LongLoRA, "
    "and then tell me about the evaluation results"
)
print("=" * 50)
print(response1)

# === QUERY 2 ===
response2 = agent.query("Give me a summary of both Self-RAG and LongLoRA")
print("=" * 50)
print(response2)

# === QUERY 3 ===
response3 = agent.query(
    "Tell me about the evaluation dataset used in MetaGPT and compare it against SWE-Bench"
)
print("=" * 50)
print(response3)

``` 
### OUTPUT:
![image](https://github.com/user-attachments/assets/e621c13a-e224-49f6-8fb7-c0b65816be0d)

### RESULT:
Thus, a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles is designed and implemented successfully.
