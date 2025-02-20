import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

HF_TOKEN = os.getenv("HF_TOKEN")
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm():
    return HuggingFaceEndpoint(repo_id=HUGGINGFACE_REPO_ID, temperature=0.5, model_kwargs={"token": HF_TOKEN, "max_length": 512})

def set_custom_prompt():
    return PromptTemplate(
        template="""
        Use the provided context to answer the user's question.
        If the answer is not in the context, say: "I do not know based on the provided documents."
        No small talk.

        Context: {context}
        Question: {question}

        Answer:
        """,
        input_variables=["context", "question"]
    )

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

qa_chain = RetrievalQA.from_chain_type(llm=load_llm(), chain_type="stuff", retriever=db.as_retriever(search_kwargs={'k': 3}), chain_type_kwargs={'prompt': set_custom_prompt()})

user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})
print("RESULT:", response.get("result", "I do not know based on the provided documents."))
