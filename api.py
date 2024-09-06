from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_model import setup_rag_model, qa_chainH, load_data, split_text

# Initialize FastAPI
app = FastAPI()

# Load the data and set up the RAG model (this runs once on startup)
data = load_data("path_to_your_csv_file.csv")
split_data = split_text(data)
qa_chain = setup_rag_model(split_data)

# Define request and response schemas
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/query", response_model=QueryResponse)
def query_rag_model(request: QueryRequest):
    try:
        # Use the qa_chainH function to generate an answer
        result = qa_chainH({"query": request.query})
        return QueryResponse(answer=result["result"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn api:app --reload
