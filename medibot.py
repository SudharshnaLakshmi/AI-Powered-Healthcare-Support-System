import os 
import streamlit as st
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
import datetime
from dotenv import load_dotenv, find_dotenv
import PyPDF2  # For PDF file processing
import docx  # For DOCX file processing

# Load environment variables
load_dotenv(find_dotenv())

HF_TOKEN = os.getenv("HF_TOKEN")
DB_FAISS_PATH = "vectorstore/db_faiss"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def init_session_state():
    session_state_vars = {
        "user_info": None,
        "chat_started": False,
        "messages": [],
        "interaction_stage": "initial",
        "followup_questions": [],
        "followup_responses": [],
        "current_query": "",
        "followup_counter": 0,
        "conversation_history": [],
        "current_conversation": [],
        "file_processed": False,  # Track if a file has been processed
        "file_response": ""  # Store the response from file processing
    }
    
    for var, default in session_state_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

CUSTOM_PROMPT_TEMPLATE = """
You are a medical assistant. Answer using the structured format below. DO NOT leave out any section.

1. **Explain possible causes** of the symptoms based on medical knowledge.
2. **Explicitly mention medications** retrieved from the medical database.  
   - Provide brand names and dosage recommendations from the medical database.
   - suggest home remedies.
3. **Recommend when to see a doctor.** 
4. **Disclaimer:** Consult a doctor before taking this medication.

💊 **Response Format Example:(Strictly Follow This!):**
---
**Possible Causes:**  
[Detailed explanation of potential medical conditions based on symptoms]  

**Medications:**  
- **Recommended treatment:** [Medication name with Dosage] (Dosage)  
- **Prescribed drugs:** [Medication name with Dosage] (Dosage)  
- **Alternative remedies:** [Home remedies if no medication is found]  

**When to See a Doctor:**  
- **Seek urgent care if:** [List of serious symptoms requiring medical attention]  

**Disclaimer:** Consult a doctor before taking any medication.  


---
Previous Interactions:
{context}

Current Question: {question}

Answer:
"""



def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

def load_llm():
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="text-generation",
        temperature=0.3,
        model_kwargs={"token": HF_TOKEN, "max_length": 1024}
    )

def load_gemini_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.7)


def add_to_conversation_history(role, content):
    st.session_state.conversation_history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    st.session_state.current_conversation.append({
        "role": role,
        "content": content
    })

def generate_followup_questions(user_query):
    gemini_llm = load_gemini_llm()
    
    history_context = "\n".join([
        f"{'User  ' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in st.session_state.current_conversation[-6:]
    ])
    
    prompt = f"""
    Conversation history:
    {history_context}
    
    Current query: {user_query}
    
    Generate three specific follow-up medical questions to better understand the patient's condition.
    Make questions progressive and based on the conversation history.
    Return exactly three questions, one per line.
    """
    
    response = gemini_llm.invoke(prompt)
    if response and hasattr(response, 'content'):
        questions = [q.strip('123.- ') for q in response.content.strip().split('\n') if q.strip()]
        return questions[:3]
    return []

# Modify the get_final_answer function to better handle file content
def get_final_answer(current_query, followup_qa_pairs):
    try:
        vectorstore = get_vectorstore()
        if not vectorstore:
            return "Failed to load the medical database. Please try again later."
        
        # Initialize qa_chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={
                "prompt": set_custom_prompt()
            }
        )
        
        # Check if this is a file analysis (longer text)
        is_file_analysis = len(current_query) > 500  # Arbitrary threshold
        
        # Build the context for the model
        if is_file_analysis:
            # Handle file content differently
            context = f"Medical report content: {current_query[:1000]}..."  # First 1000 chars
            analysis_prompt = """
            Please analyze this medical report and follow the structured response format below. 

            1. **Explain possible causes** of the conditions mentioned in the report.
            2. **Explicitly mention medications** retrieved from the medical database:
            - Provide brand names and dosage recommendations from medical database.
            - suggest home remedies.
            3. **Recommend when to see a doctor.** 
            4. **Disclaimer:** Consult a doctor before taking this medication.

            **Response Format Example:(Strictly Follow This!):**  
            ---
           **Possible Causes:**  
            [Detailed explanation of potential medical conditions based on symptoms]  

            **Medications:**  
            - **Recommended treatment:** [Medication name with Dosage] (Dosage)  
            - **Prescribed drugs:** [Medication name from the medical database with Dosage] (Dosage)  
            - **Alternative remedies:** [Home remedies]  

            **When to See a Doctor:**  
            - **Seek urgent care if:** [List of serious symptoms requiring medical attention]  

            **Disclaimer:** Consult a doctor before taking any medication.  

            ---

            **Medical Report Content:** {current_query}

            Provide a structured response based on the above format.
            """
        else:
            # Regular query handling
            context = (
                f"Current symptoms: {current_query}\n" +
                "\n".join([f"Q: {q}\nA: {a}" for q, a in followup_qa_pairs])
            )
            analysis_prompt = "Based on the symptoms provided, please explain possible causes and recommend medications."
        
        response = qa_chain.invoke({
            'query': f"{context}\n\n{analysis_prompt}"
        })
        
        return response['result'] if isinstance(response, dict) and 'result' in response else str(response)
        
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return "I apologize, but I encountered an error while processing your query. Please try again or rephrase your question."
        
        # Rest of the function remains the same
def chat_interface():
    if st.session_state.user_info is None:
        st.error("Please complete registration first.")
        return

    st.title("Medical Consultation")

    # Display user info in sidebar but keep it minimal
    st.sidebar.title("Session Info")
    st.sidebar.write(f"**Name**: {st.session_state.user_info['name']}")
    st.sidebar.write(f"**Age**: {st.session_state.user_info['age']}")
    st.sidebar.write(f"**Gender**: {st.session_state.user_info['gender']}")
    st.sidebar.write(f"**Registration time**: {st.session_state.user_info['registration_time']}")

    # File upload feature in the sidebar with a unique key
    uploaded_file = st.sidebar.file_uploader("Upload your medical report (PDF or DOCX)", type=["pdf", "docx"], key="file_uploader_1")
    if uploaded_file is not None and not st.session_state.file_processed:
        text = extract_text_from_file(uploaded_file)
        if text:
            # Use the extracted text for analysis
            st.session_state.current_query = "I have uploaded a medical report. Please analyze it."
            add_to_conversation_history("user", st.session_state.current_query)
            followup_qa_pairs = []  # No follow-up questions for file upload
            final_answer = get_final_answer(text, followup_qa_pairs)  # Use the extracted text for analysis
            add_to_conversation_history("assistant", final_answer)
            st.session_state.file_processed = True  # Mark file as processed
            st.session_state.file_response = final_answer  # Store the response
            st.rerun()  # Rerun to update the interface

    # Always display chat history
    display_chat_history()

    if st.session_state.interaction_stage == "initial":
        prompt = st.chat_input("Type your message here...")
        if prompt:
            st.session_state.current_query = prompt
            add_to_conversation_history("user", prompt)
            
            # Generate three follow-up questions
            st.session_state.followup_questions = generate_followup_questions(prompt)
            st.session_state.followup_responses = []
            st.session_state.followup_counter = 0
            st.session_state.interaction_stage = "followup"
            
            if st.session_state.followup_questions:
                add_to_conversation_history(
                    "assistant", 
                    st.session_state.followup_questions[0]
                )
            st.rerun()

    elif st.session_state.interaction_stage == "followup":
        response = st.chat_input("Type your response...")
        if response:
            add_to_conversation_history("user", response)
            st.session_state.followup_responses.append(response)
            st.session_state.followup_counter += 1
            
            if st.session_state.followup_counter < 3 and st.session_state.followup_questions:
                next_question = st.session_state.followup_questions[st.session_state.followup_counter]
                add_to_conversation_history("assistant", next_question)
            else:
                followup_qa_pairs = list(zip(
                    st.session_state.followup_questions,
                    st.session_state.followup_responses
                ))
                final_answer = get_final_answer(
                    st.session_state.current_query,
                    followup_qa_pairs
                )
                add_to_conversation_history("assistant", final_answer)
                st.session_state.interaction_stage = "await_new_query"
            st.rerun()

    elif st.session_state.interaction_stage == "await_new_query":
        new_query = st.chat_input("Type your message here...")
        if new_query:
            st.session_state.current_query = new_query
            add_to_conversation_history("user", new_query)
            
            st.session_state.followup_questions = generate_followup_questions(new_query)
            st.session_state.followup_responses = []
            st.session_state.followup_counter = 0
            st.session_state.interaction_stage = "followup"
            
            if st.session_state.followup_questions:
                add_to_conversation_history(
                    "assistant",
                    st.session_state.followup_questions[0]
                )
            st.rerun()

def user_registration():
    st.title("Medical Consultation")
    
    with st.form("registration_form"):
        name = st.text_input("Full Name")
        age = st.number_input("Age", min_value=0, max_value=120)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        submit_button = st.form_submit_button("Start Consultation")
        
        if submit_button and name and age > 0:
            st.session_state.user_info = {
                "name": name,
                "age": age,
                "gender": gender,
                "registration_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.chat_started = True
            
            # Add welcoming message to conversation
            welcome_message = "Hello! I'm your medical assistant. How can I help you today?"
            add_to_conversation_history("assistant", welcome_message)
        else:
            st.error("Please fill in all required fields.")

def display_chat_history():
    for message in st.session_state.current_conversation:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 1. Improve the PDF text extraction with better error handling
def extract_text_from_file(uploaded_file):
    """Extract text from PDF or DOCX file."""
    try:
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # Check if text was successfully extracted
            if not text.strip():
                return "The PDF appears to be empty or contains scanned images without text. Please upload a PDF with extractable text."
            
            return text.strip()
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        else:
            return "Unsupported file format. Please upload a PDF or DOCX file."
    except Exception as e:
        return f"Error processing file: {str(e)}. Please try another file."

def main():
    init_session_state()
    if not st.session_state.chat_started:
        user_registration()
    else:
        chat_interface()


if __name__ == "__main__":
    main()

