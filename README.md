# LLM-Powered-Research-Paper-Summarizer
This project demonstrates how to build a lightweight Retrieval-Augmented Generation (RAG) pipeline using:
- `pdfminer.six` for PDF parsing
- `sentence-transformers` for dense embeddings
- `FAISS` for efficient similarity search
- `tiiuae/falcon-7b-instruct` as the open-source LLM

---

## 🚀 Features

- Extracts text from a PDF file
- Splits text into manageable chunks
- Generates vector embeddings using `all-MiniLM-L6-v2`
- Indexes embeddings using FAISS for similarity search
- Uses Falcon-7B-Instruct to answer questions based on retrieved content

---

## 📦 Installation

```bash
pip install -q faiss-cpu sentence-transformers transformers pdfminer.six accelerate
```

---

## 📁 Directory Structure

```
project/
├── a.pdf                # Your input PDF file
├── script.ipynb         # The main script or notebook
└── README.md            # This file
```

---

## 🔍 Usage

### 1. Extract Text from PDF

```python
from pdfminer.high_level import extract_text

pdf_path = "a.pdf"
raw_text = extract_text(pdf_path)
```

---

### 2. Split Text into Chunks

```python
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_text(raw_text)
```

---

### 3. Generate Embeddings and Create FAISS Index

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embed_model.encode(texts, convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
```

---

### 4. Load Falcon 7B-Instruct Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.float16
)
```

---

### 5. Ask Questions

```python
def ask_question(query, top_k=3):
    query_embedding = embed_model.encode([query])
    D, I = index.search(query_embedding, top_k)
    retrieved_chunks = [texts[i] for i in I[0]]
    context = "\n".join(retrieved_chunks)

    prompt = f"""Based on the context below, answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip()
```

---

### 6. Generate Summary

```python
summary = ask_question("Summarize the main points of the document.")
print(summary)
```

---

## 📌 Notes

- Ensure that your environment supports `torch_dtype=torch.float16` (use GPU).
- You may substitute Falcon-7B with another model like `mistralai/Mistral-7B-Instruct` if needed.
- Make sure the PDF is not image-based or use OCR before extraction.

---

## 📖 Acknowledgments

- [FAISS by Facebook AI](https://github.com/facebookresearch/faiss)
- [SentenceTransformers](https://www.sbert.net/)
- [Hugging Face Transformers](https://huggingface.co/transformers)
- [PDFMiner.six](https://github.com/pdfminer/pdfminer.six)
