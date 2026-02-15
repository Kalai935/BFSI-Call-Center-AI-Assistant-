# BFSI Call Center AI Assistant

A lightweight, compliant, and efficient AI assistant for handling Banking, Financial Services, and Insurance (BFSI) call center queries with local deployment capabilities.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production-brightgreen.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

The BFSI Call Center AI Assistant is designed to provide fast, accurate, and standardized responses to common banking queries while ensuring strict compliance with financial regulations. The system uses a three-tier architecture to balance response accuracy, speed, and safety.

### Key Capabilities

- **Loan & Credit Services**: Eligibility checks, application status, EMI calculations
- **Account Management**: Balance inquiries, card blocking, account closure
- **Transaction Support**: Payment queries, transaction history, fund transfers
- **Policy Information**: Interest rates, penalties, terms and conditions
- **Customer Support**: General inquiries, complaint registration, feedback

### System Highlights

✅ **Local Deployment**: Runs entirely on your hardware - no cloud dependencies  
✅ **Compliance-First**: Built-in guardrails for financial regulations  
✅ **Fast Response**: Dataset-based matching for instant answers  
✅ **Scalable**: Handles high call volumes with efficient architecture  
✅ **Explainable**: Clear indication of which logic tier generated each response

---

## 🏗️ Architecture

The system implements a **three-tier response logic** as specified in the PRD:

```
User Query
    ↓
┌─────────────────────────────────────────┐
│  TIER 1: Dataset Similarity Match       │
│  • 150+ curated BFSI Q&A pairs          │
│  • Threshold: 85% similarity            │
│  • Response Time: <100ms                │
└─────────────────────────────────────────┘
    ↓ (No Match)
┌─────────────────────────────────────────┐
│  TIER 2: Fine-Tuned Local SLM           │
│  • TinyLlama 1.1B (LoRA adapted)        │
│  • Trained on Alpaca BFSI dataset       │
│  • Response Time: ~2-5s                 │
└─────────────────────────────────────────┘
    ↓ (Complex Query Detected)
┌─────────────────────────────────────────┐
│  TIER 3: RAG (Retrieval) Layer          │
│  • Policy documents retrieval           │
│  • Context-grounded generation          │
│  • Response Time: ~3-7s                 │
└─────────────────────────────────────────┘
    ↓
Final Response + Source Attribution
```

### Why This Architecture?

1. **Tier 1 (Dataset)**: Ensures consistent, pre-approved responses for common queries
2. **Tier 2 (SLM)**: Handles variations and edge cases while staying lightweight
3. **Tier 3 (RAG)**: Provides accurate answers for complex policy/document queries

---

## ✨ Features

### Core Functionality

- **Intelligent Query Routing**: Automatically selects the best response method
- **Semantic Similarity Matching**: Uses sentence-transformers for accurate dataset matching
- **Fine-Tuned Language Model**: Custom-trained TinyLlama for BFSI domain
- **Knowledge Base Retrieval**: ChromaDB-powered RAG for policy documents
- **Real-Time Processing**: Streamlit-based interactive interface

### Safety & Compliance

- **Input Sanitization**: Filters unsafe or out-of-scope queries
- **Data Masking**: Automatically redacts sensitive information (card numbers, account IDs)
- **No Hallucination**: Strict control over financial number generation
- **Audit Trail**: Every response tagged with its generation source

### Scalability

- **Version Control**: Dataset, model weights, and documents tracked separately
- **Easy Updates**: Modular design for policy/rate changes
- **Batch Processing**: Can be extended for bulk query handling
- **GPU Optional**: Works on CPU (slower) or GPU (faster)

---

## 📦 Requirements

### System Requirements

- **OS**: Linux, macOS, or Windows 10/11
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space (for models and cache)
- **GPU**: Optional (NVIDIA with CUDA support for faster inference)

### Dependencies

All dependencies are listed in `requirements.txt`:

```
torch
transformers
sentence-transformers
langchain
langchain-community
langchain-text-splitters
chromadb
streamlit
accelerate
bitsandbytes
peft
datasets
tiktoken
```

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Kalai935/BFSI-Call-Center-AI-Assistant-.git
cd BFSI_AI_ASSISTANT
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# For Linux/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note**: Installation may take 5-10 minutes depending on your internet speed.

---

## ⚡ Quick Start

### Option 1: Automated Setup (Recommended)

Run all setup steps with a single command:

```bash
# Step 1: Generate data files
python setup_data.py

# Step 2: Fine-tune the model (optional but recommended)
python train_slm.py

# Step 3: Launch the application
streamlit run app.py
```

### Option 2: Manual Setup

If you prefer step-by-step control:

```bash
# 1. Create data folder and files
python setup_data.py

# 2. (Optional) Skip training to use base model
# python train_slm.py

# 3. Start the application
streamlit run app.py
```

### Expected Output

After running `streamlit run app.py`, you should see:

```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Open the **Local URL** in your browser.

---

## 💻 Usage

### Web Interface

1. **Launch the App**:
   ```bash
   streamlit run app.py
   ```

2. **Enter Your Query** in the text input field:
   - Example: "What are the eligibility criteria for a personal loan?"
   - Example: "What is the penalty for prepaying a fixed-rate loan?"

3. **Submit** and view the response with source attribution

### Query Examples

#### Tier 1 Queries (Dataset Match)
```
"How do I block my debit card?"
"What documents are required for a car loan?"
"What is the daily ATM withdrawal limit?"
```

#### Tier 2 Queries (General SLM)
```
"Write a polite greeting for a customer calling about a complaint."
"Explain the benefits of a savings account."
```

#### Tier 3 Queries (RAG Knowledge Base)
```
"What is the prepayment penalty for fixed-rate loans?"
"Tell me about the gold loan scheme."
"What are the minimum balance charges for urban branches?"
```

### Understanding Output

Each response includes:

- **Response Text**: The actual answer to your query
- **Source Logic**: Which tier processed the query
  - `Tier 1 (Dataset Match: 0.92)` → Pre-stored answer with 92% similarity
  - `Tier 2 (General SLM)` → Generated by the fine-tuned model
  - `Tier 3 (RAG + SLM)` → Retrieved from knowledge base
  - `Guardrails` → Query rejected for safety reasons

---

## 📁 Project Structure

```
BFSI_Project/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
│
├── setup_data.py               # Data generation script
├── train_slm.py                # Model fine-tuning script
├── app.py                      # Main Streamlit application
│
├── data/
│   ├── bfsi_dataset.json       # Tier 1: 150+ Q&A pairs (Alpaca format)
│   └── knowledge_base.txt      # Tier 3: Policy documents
│
├── fine_tuned_slm/             # Tier 2: Fine-tuned model weights (after training)
│   ├── adapter_config.json
│   └── adapter_model.safetensors
│
└── bfsi_checkpoints/           # Training checkpoints (temporary)
```

---

## ⚙️ Configuration

### Model Selection

To use a different base model, edit `app.py`:

```python
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Current
# BASE_MODEL = "microsoft/phi-2"                   # Alternative
```

### Similarity Threshold

Adjust the Tier 1 matching sensitivity in `app.py`:

```python
if best_score > 0.85:  # Current: 85%
    return dataset[best_idx]['output'], ...
```

- **Higher (0.90+)**: More strict, fewer dataset matches
- **Lower (0.75-0.80)**: More lenient, more dataset matches

### RAG Chunk Size

Modify retrieval granularity in `app.py`:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Current
    chunk_overlap=50
)
```

---

## 🧪 Testing

### Basic Functionality Test

```bash
# Start the app
streamlit run app.py

# Test queries in browser:
# 1. "What are the eligibility criteria for a personal loan?"
#    → Should return Tier 1 response
#
# 2. "What is the gold loan LTV ratio?"
#    → Should return Tier 3 response with policy details
#
# 3. "Write a thank you message to a customer"
#    → Should return Tier 2 response
```

### Safety Guardrails Test

Try entering restricted keywords:

```
"How to hack a bank account"
"Illegal money laundering methods"
```

**Expected**: `⚠️ Query rejected due to safety guidelines.`

### Data Masking Test

Enter a query with fake sensitive data:

```
"My card number is 1234 5678 9012"
```

**Expected**: Response should show `[REDACTED_CARD]` instead of the number.

---

## 🔧 Troubleshooting

### Issue 1: ModuleNotFoundError

**Error**: `ModuleNotFoundError: No module named 'langchain_text_splitters'`

**Solution**:
```bash
pip install langchain-text-splitters --upgrade
```

### Issue 2: Out of Memory (OOM)

**Error**: `CUDA out of memory` or `RuntimeError: killed`

**Solution**:
1. Reduce batch size in `train_slm.py`:
   ```python
   per_device_train_batch_size=1  # Already minimal
   ```

2. Use CPU instead of GPU:
   ```python
   device_map="cpu"  # In app.py and train_slm.py
   ```

3. Close other applications to free RAM

### Issue 3: Model Download Stuck

**Error**: Download hangs at "Fetching model..."

**Solution**:
```bash
# Pre-download models manually
python -c "from transformers import AutoModel; AutoModel.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Issue 4: Streamlit Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Use a different port
streamlit run app.py --server.port 8502
```

### Issue 5: Training Takes Too Long

**Symptom**: `train_slm.py` running for 30+ minutes

**Solution**:
1. **Skip Training**: Use base model directly (still works!)
   ```bash
   # Just run the app without training
   streamlit run app.py
   ```

2. **Reduce Epochs**:
   ```python
   num_train_epochs=1  # Instead of 3
   ```

---

## 🚄 Performance Optimization

### For Faster Inference

1. **Enable GPU** (if available):
   - Ensure PyTorch is installed with CUDA
   - Check: `python -c "import torch; print(torch.cuda.is_available())"`

2. **Quantization** (Reduce model size):
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       BASE_MODEL,
       load_in_8bit=True  # Adds 8-bit quantization
   )
   ```

3. **Increase Batch Size** (if you have RAM):
   ```python
   per_device_train_batch_size=2  # Or 4
   ```

### For Production Deployment

1. **Use a Reverse Proxy** (e.g., Nginx) for SSL/HTTPS
2. **Add Authentication** using Streamlit Cloud or custom OAuth
3. **Database Integration** for logging queries and responses
4. **Load Balancing** for handling multiple concurrent users

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to new functions
- Update README if adding features
- Test on both CPU and GPU environments

---

## 📄 License

This project is licensed under the MIT License

---

## 📞 Support

For questions, issues, or feedback:

- **Email**: kalaimuthu098@gmail.com
- **Issues**: [GitHub Issues](https://github.com/Kalai935/BFSI-Call-Center-AI-Assistant-/issues)
---

## 🙏 Acknowledgments

- **Anthropic** for Claude AI assistance in development
- **Hugging Face** for model hosting and transformers library
- **LangChain** for RAG framework
- **Streamlit** for the web interface framework
- **TinyLlama Team** for the efficient base model

---

## 📊 Metrics & Performance

| Metric | Value |
|--------|-------|
| Avg. Response Time (Tier 1) | <100ms |
| Avg. Response Time (Tier 2) | 2-5s |
| Avg. Response Time (Tier 3) | 3-7s |
| Model Size | 1.1B parameters |
| RAM Usage | 4-6GB |
| Dataset Size | 150+ Q&A pairs |
| Supported Query Types | 10+ categories |

---

## 🗺️ Roadmap

- [ ] Add multilingual support (Hindi, Spanish, French)
- [ ] Implement voice input/output
- [ ] Add analytics dashboard
- [ ] Create REST API endpoint
- [ ] Integrate with CRM systems
- [ ] Add A/B testing framework
- [ ] Expand dataset to 500+ samples
- [ ] Add conversation memory/context

---

**Built for BFSI Call Centers - Lendkraft Technologies Solutions Pvt. Ltd (Technical Challenge)**
