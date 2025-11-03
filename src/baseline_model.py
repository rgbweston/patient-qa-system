"""
Baseline Patient QA System using TF-IDF Retrieval + LLM Generation

This script implements a baseline RAG pipeline for patient question-answering:
- Retrieval: TF-IDF sparse retrieval (k=5 chunks)
- Chunking: Paragraph-based splitting on double newlines
- Generation: Phi-4-mini-instruct LLM with basic prompting
"""

import argparse
import zipfile
import re
import os
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
import numpy as np
import random


def safe_filename(name):
    """Clean filename for safe dictionary keys."""
    base_name = os.path.splitext(name)[0]
    return re.sub(r'[^\w\-_. ]', '_', base_name).strip().replace(" ", "_")


def load_documents(docs_path):
    """Load and chunk NHS documents from ZIP file."""
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

            # Original chunking: split on double newlines
            paragraphs = re.split(r'\n\s*\n', content.strip())
            docs[cleaned_key] = paragraphs

    return docs


def retrieve_with_tfidf(sentences, query, k=5):
    """Retrieve top-k most relevant chunks using TF-IDF."""
    k = min(k, len(sentences))
    vectorizer = TfidfVectorizer()
    sent_vecs = vectorizer.fit_transform(sentences)
    q_vec = vectorizer.transform([query])

    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(sent_vecs)
    _, idxs = nn.kneighbors(q_vec)

    return [sentences[i] for i in idxs[0]]


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


def run_baseline_model(docs_path, test_csv_path, k=5, answer_limit=50):
    """Run baseline TF-IDF RAG model on test set."""

    # Load test data
    print(f"Loading test data from {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path)

    # Load documents
    print(f"Loading NHS documents from {docs_path}...")
    docs = load_documents(docs_path)
    print(f"Loaded {len(docs)} documents")

    # Setup LLM
    pipe = setup_llm()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # System prompt
    base_system = (
        "You are a friendly, knowledgeable medical expert who explains health topics "
        "in a clear, compassionate, and patient-friendly way. Please provide your answer "
        "in 2-3 sentences."
    )

    # Initialize scorer
    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)

    generated, references = [], []

    print(f"\nProcessing {min(answer_limit, len(test_df))} questions...\n")

    for idx, row in enumerate(test_df.itertuples(index=False)):
        if idx >= answer_limit:
            break

        question, reference_text, disease = row.question, row.answer, row.disease

        # Fetch chunks and retrieve top-k via TF-IDF
        chunks = docs.get(safe_filename(disease), [])
        if not chunks:
            print(f"Warning: No chunks found for disease '{disease}'")
            continue

        contexts = retrieve_with_tfidf(chunks, question, k=k)

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
    print("BASELINE MODEL RESULTS")
    print("="*70)

    precisions = [
        scorer.score(ref, pred)["rougeLsum"].precision
        for ref, pred in zip(references, generated)
    ]
    mean_precision = sum(precisions) / len(precisions)
    print(f"Mean ROUGE-Lsum Precision: {mean_precision:.4f}")

    P, R, F1 = bert_score(generated, references, lang="en", device=device)
    print(f"BERTScore - Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")
    print("="*70)

    return {
        'generated': generated,
        'references': references,
        'rouge_precision': mean_precision,
        'bert_f1': F1.mean().item()
    }


def main():
    parser = argparse.ArgumentParser(description='Baseline Patient QA System with TF-IDF')
    parser.add_argument('--docs_path', type=str, required=True,
                        help='Path to ZIP file containing NHS documents')
    parser.add_argument('--test_csv', type=str, required=True,
                        help='Path to test set CSV file')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of top chunks to retrieve (default: 5)')
    parser.add_argument('--answer_limit', type=int, default=50,
                        help='Number of questions to process (default: 50)')

    args = parser.parse_args()

    results = run_baseline_model(
        docs_path=args.docs_path,
        test_csv_path=args.test_csv,
        k=args.k,
        answer_limit=args.answer_limit
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
