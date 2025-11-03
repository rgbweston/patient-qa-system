"""
Optimized Patient QA System using BioBERT + FAISS Retrieval + LLM Generation

This script implements an optimized RAG pipeline for patient question-answering:
- Retrieval: BioBERT dense embeddings + FAISS vector search
- Chunking: Intelligent merging (minimum 10 words per chunk)
- Top-k: Increased to k=15
- Generation: Phi-4-mini-instruct LLM with optimized prompting
"""

import argparse
import zipfile
import re
import os
import pandas as pd
import torch
import faiss
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline, set_seed
import random


def safe_filename(name):
    """Clean filename for safe dictionary keys."""
    base_name = os.path.splitext(name)[0]
    return re.sub(r'[^\w\-_. ]', '_', base_name).strip().replace(" ", "_")


def load_and_chunk_documents(docs_path, min_words=10):
    """
    Load and intelligently chunk NHS documents.

    Improved chunking method:
    - Merge adjacent chunks if either are less than min_words
    - Reduces meaningless short chunks
    """
    docs = {}
    with zipfile.ZipFile(docs_path, 'r') as zip_ref:
        txt_files = [f for f in zip_ref.namelist()
                     if f.endswith('.txt') and not f.startswith('__MACOSX/') and '._' not in f]

        for fname in txt_files:
            raw = zip_ref.read(fname)
            try:
                content = raw.decode('utf-8')
            except UnicodeDecodeError:
                content = raw.decode('latin-1')

            cleaned_key = safe_filename(os.path.basename(fname))

            # Split on double newlines
            paragraphs = re.split(r'\n\s*\n', content.strip())

            # Intelligent chunking: merge short chunks
            merged_chunks = []
            accumulated_text = ""

            for raw_chunk in paragraphs:
                raw_chunk = raw_chunk.strip()
                candidate = (accumulated_text + " " + raw_chunk).strip() if accumulated_text else raw_chunk

                if len(candidate.split()) < min_words:
                    accumulated_text = candidate
                else:
                    merged_chunks.append(candidate)
                    accumulated_text = ""

            # Handle remaining accumulated text
            if accumulated_text:
                if merged_chunks:
                    merged_chunks[-1] += " " + accumulated_text.strip()
                else:
                    merged_chunks.append(accumulated_text.strip())

            docs[cleaned_key] = merged_chunks

    return docs


def embed_texts(texts, tokenizer, model, device, max_length=512):
    """Generate BioBERT embeddings for texts."""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs).last_hidden_state

    # Use CLS token embeddings
    return out[:, 0, :].cpu().numpy()


def retrieve_with_biobert(chunks, query, tokenizer_bio, model_bio, device, k=15):
    """
    Retrieve top-k chunks using BioBERT + FAISS.

    Returns deduplicated chunks.
    """
    if not chunks:
        return []

    # Embed all chunks
    embs = embed_texts(chunks, tokenizer_bio, model_bio, device)

    # Build FAISS index
    idx = faiss.IndexFlatL2(embs.shape[1])
    idx.add(embs)

    # Embed query
    q_emb = embed_texts([query], tokenizer_bio, model_bio, device)

    # Search
    _, inds = idx.search(q_emb.reshape(1, -1), min(k, len(chunks)))

    # Deduplicate
    seen, unique = set(), []
    for i in inds[0]:
        c = chunks[i]
        if c not in seen:
            unique.append(c)
            seen.add(c)

    return unique


def setup_biobert(device):
    """Load BioBERT model for embeddings."""
    print("Loading BioBERT...")
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model.to(device)
    model.eval()
    return tokenizer, model


def setup_llm(seed=42):
    """Load and configure Phi-4-mini-instruct LLM."""
    # Set random seeds for reproducibility
    set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_id = "microsoft/Phi-4-mini-instruct"
    device = 0 if torch.cuda.is_available() else -1

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    return pipe


def run_optimized_model(docs_path, test_csv_path, k=15, answer_limit=50):
    """Run optimized BioBERT + FAISS RAG model on test set."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    print(f"Loading test data from {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path)

    # Load and chunk documents
    print(f"Loading and chunking NHS documents from {docs_path}...")
    docs = load_and_chunk_documents(docs_path, min_words=10)
    print(f"Loaded {len(docs)} documents with improved chunking")

    # Setup BioBERT
    tokenizer_bio, model_bio = setup_biobert(device)

    # Setup LLM
    pipe = setup_llm()

    # Optimized system prompt
    base_system = (
        "You are a friendly, knowledgeable medical expert who explains health topics "
        "in a clear, compassionate, and patient-friendly way. Please provide your answer "
        "in 1-2 complete sentences unless the question requires elaboration."
    )

    # Initialize scorer
    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)

    generated, references = [], []

    print(f"\nProcessing {min(answer_limit, len(test_df))} questions...\n")

    for idx, row in enumerate(test_df.itertuples(index=False)):
        if idx >= answer_limit:
            break

        question, reference_text, disease = row.question, row.answer, row.disease

        # Fetch chunks
        chunks = docs.get(safe_filename(disease), [])
        if not chunks:
            print(f"Warning: No chunks found for disease '{disease}'")
            continue

        # Retrieve top-k using BioBERT + FAISS
        contexts = retrieve_with_biobert(chunks, question, tokenizer_bio, model_bio, device, k=k)

        # Build prompt and generate
        prompt = "\n\n".join(contexts)
        messages = [
            {"role": "system", "content": base_system},
            {"role": "user", "content": f"Context:\n{prompt}\n\nQuestion:\n{question}"}
        ]

        answer = pipe(
            messages,
            max_new_tokens=200,
            return_full_text=False,
            do_sample=False
        )[0]["generated_text"]

        print(f"[{idx+1}/{min(answer_limit, len(test_df))}] {question[:60]}...")
        print(f"Answer: {answer[:100]}...")
        print()

        generated.append(answer)
        references.append(reference_text)

    # Compute metrics
    print("\n" + "="*70)
    print("OPTIMIZED MODEL RESULTS")
    print("="*70)

    precisions = [
        scorer.score(ref, pred)["rougeLsum"].precision
        for ref, pred in zip(references, generated)
    ]
    mean_precision = sum(precisions) / len(precisions)
    print(f"Mean ROUGE-Lsum Precision: {mean_precision:.4f}")

    P, R, F1 = bert_score(generated, references, lang="en", device=str(device))
    print(f"BERTScore - Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
    print("="*70)

    return {
        'generated': generated,
        'references': references,
        'rouge_precision': mean_precision,
        'bert_f1': F1.mean().item()
    }


def main():
    parser = argparse.ArgumentParser(description='Optimized Patient QA System with BioBERT + FAISS')
    parser.add_argument('--docs_path', type=str, required=True,
                        help='Path to ZIP file containing NHS documents')
    parser.add_argument('--test_csv', type=str, required=True,
                        help='Path to test set CSV file')
    parser.add_argument('--k', type=int, default=15,
                        help='Number of top chunks to retrieve (default: 15)')
    parser.add_argument('--answer_limit', type=int, default=50,
                        help='Number of questions to process (default: 50)')

    args = parser.parse_args()

    results = run_optimized_model(
        docs_path=args.docs_path,
        test_csv_path=args.test_csv,
        k=args.k,
        answer_limit=args.answer_limit
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
