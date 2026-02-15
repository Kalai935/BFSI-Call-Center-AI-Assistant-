import streamlit as st
import json
import torch
import re
import os
from sentence_transformers import SentenceTransformer, util
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# --- FIXED IMPORT BELOW ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --------------------------
from langchain_community.document_loaders import TextLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel

# --- CONFIGURATION ---
DATASET_PATH = "data/bfsi_dataset.json"
KNOWLEDGE_PATH = "data/knowledge_base.txt"
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH = "./fine_tuned_slm"

st.set_page_config(page_title="BFSI Call Center Assistant", layout="wide")
st.title("🏦 BFSI Call Center AI Assistant")
st.markdown("*Architecture: Dataset Check (Tier 1) → Fine-Tuned SLM (Tier 2) → RAG (Tier 3)*")

# --- 1. SYSTEM LOADER ---
@st.cache_resource
def initialize_system():
    # A. Load Tier 1 Dataset & Embeddings
    if not os.path.exists(DATASET_PATH):
        st.error(f"Missing {DATASET_PATH}. Please run setup_data.py")
        return None
        
    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)
    
    # Load Embedding Model for Similarity
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    corpus_embeddings = embedder.encode([item['instruction'] for item in dataset], convert_to_tensor=True)
    
    # B. Load Tier 2/3 Model (SLM)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    try:
        # Try loading fine-tuned weights
        if os.path.exists(ADAPTER_PATH):
            base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
            model = PeftModel.from_pretrained(base, ADAPTER_PATH)
            status = "✅ Fine-Tuned Model Loaded"
        else:
            raise Exception("No Adapter")
    except:
        # Fallback to base model
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
        status = "⚠️ Base Model Loaded (Training skipped)"
    
    gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)

    # C. Load Tier 3 RAG Knowledge Base
    if os.path.exists(KNOWLEDGE_PATH):
        loader = TextLoader(KNOWLEDGE_PATH)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents(docs)
        rag_db = Chroma.from_documents(splits, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    else:
        rag_db = None
        st.error("Missing Knowledge Base file.")

    return dataset, embedder, corpus_embeddings, gen_pipeline, rag_db, status

# Load everything
sys_data = initialize_system()
if sys_data:
    dataset, embedder, dataset_embeddings, llm, rag_db, model_status = sys_data
    st.sidebar.success(model_status)

# --- 2. GUARDRAILS ---
def safety_check(query):
    unsafe_keywords = ["gamble", "bet", "illegal", "laundering", "fake", "hack", "kill"]
    if any(word in query.lower() for word in unsafe_keywords):
        return False
    return True

def mask_sensitive_data(text):
    # Mask numbers that look like Credit Cards
    text = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}\b', '[REDACTED_CARD]', text)
    return text

# --- 3. CORE LOGIC ---
def process_query(query):
    # Step 0: Safety
    if not safety_check(query):
        return "⚠️ Query rejected due to safety guidelines.", "Guardrails"

    # Step 1: Tier 1 (Dataset Match)
    query_vec = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_vec, dataset_embeddings)[0]
    best_score = torch.max(cos_scores).item()
    best_idx = torch.argmax(cos_scores).item()

    if best_score > 0.85: # High similarity threshold
        return dataset[best_idx]['output'], f"Tier 1 (Dataset Match: {best_score:.2f})"

    # Step 2: Check for Complex RAG Triggers (Tier 3)
    complex_triggers = ["policy", "penalty", "charge", "rule", "gold", "balance", "terms", "fee"]
    use_rag = any(trigger in query.lower() for trigger in complex_triggers)

    if use_rag and rag_db:
        docs = rag_db.similarity_search(query, k=2)
        context = "\n".join([d.page_content for d in docs])
        
        prompt = f"<|system|>You are a helpful BFSI assistant. Answer strictly using the Context.<|user|>Context: {context}\n\nQuestion: {query}<|assistant|>"
        response = llm(prompt)[0]['generated_text'].split("<|assistant|>")[-1].strip()
        return mask_sensitive_data(response), "Tier 3 (RAG + SLM)"

    # Step 3: Tier 2 (General SLM)
    prompt = f"<|system|>You are a helpful banking assistant.<|user|>{query}<|assistant|>"
    response = llm(prompt)[0]['generated_text'].split("<|assistant|>")[-1].strip()
    return mask_sensitive_data(response), "Tier 2 (General SLM)"

# --- 4. USER INTERFACE ---
query = st.text_input("Enter your banking query:", placeholder="e.g., What is the gold loan interest rate?")

if st.button("Submit Query"):
    if query:
        with st.spinner("Analyzing request..."):
            ans, source = process_query(query)
        
        st.subheader("Assistant Response:")
        st.write(ans)
        
        st.divider()
        st.caption("System Diagnostics:")
        st.info(f"Processing Logic Used: {source}")
    else:
        st.warning("Please enter a question.")