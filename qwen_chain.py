from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn
import os
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from pydantic import BaseModel, Field
from langchain.schema.runnable import RunnablePassthrough
from typing import Any, List, Union
from langchain_community.vectorstores import Qdrant


llm = OllamaLLM(model="qwen2.5:0.5b")

RAG_PROMPT_TEMPLATE = """\
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context, say you don't know.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
User Query:
{query}

Context:
{context}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="all-minilm:latest",
)


from langchain.vectorstores import FAISS
vectorstore = FAISS.from_texts(
    [
    "George Kittle is one of the most dynamic tight ends in the NFL, known for his versatility in both receiving and blocking.",
    "He was drafted by the San Francisco 49ers in the fifth round of the 2017 NFL Draft and quickly became a key player on their offense.",
    "Kittle holds the record for the most receiving yards in a single season by a tight end, with 1,377 yards in 2018."
], embedding=embeddings

)
retriever = vectorstore.as_retriever()


lcel_rag_chain = {"context": itemgetter("query") | retriever, "query": itemgetter("query")}| rag_prompt | llm

app = FastAPI()

class Input(BaseModel):
    query: str

class Output(BaseModel):
    output: Any

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")
    
add_routes(
    app,
    lcel_rag_chain.with_types(input_type=Input, output_type=Output).with_config(
        {"run_name": "Kittle"}
    )
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)