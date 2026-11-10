from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np


# -------------------------
# Load models
# -------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# -------------------------
# Read the document
# -------------------------

with open("data/documents/python.txt", "r") as file:
    document = file.read()


# -------------------------
# Split document into chunks
# -------------------------

chunks = document.split("\n\n")


# -------------------------
# Create chunk embeddings
# -------------------------

chunk_embeddings = model.encode(chunks)


# -------------------------
# Create BM25 index
# -------------------------

tokenized_chunks = [chunk.lower().split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)


# -------------------------
# Evaluation questions
# -------------------------

questions = [
    "What libraries are used for data analysis in Python?",
    "What programming paradigms does Python support?",
    "What file extension is commonly used for Python programs?"
]


answers = [
    "Python libraries such as NumPy and pandas are used for data analysis.",
    "Python supports object-oriented, procedural, and functional programming.",
    "Python programs are commonly stored using the .py file extension."
]


# -------------------------
# Verification threshold
# -------------------------

verification_threshold = 5.0


# -------------------------
# Process each question
# -------------------------

for question, answer in zip(questions, answers):

    print("\n" + "=" * 70)
    print(f"Question: {question}")
    print("=" * 70)


    # -------------------------
    # Dense Retrieval
    # -------------------------

    question_embedding = model.encode(question)

    similarities = []

    for chunk_embedding in chunk_embeddings:
        similarity = np.dot(
            question_embedding,
            chunk_embedding
        ) / (
            np.linalg.norm(question_embedding)
            * np.linalg.norm(chunk_embedding)
        )

        similarities.append(similarity)


    # -------------------------
    # BM25 Retrieval
    # -------------------------

    tokenized_question = question.lower().split()

    bm25_scores = bm25.get_scores(tokenized_question)


    # -------------------------
    # Hybrid Retrieval using RRF
    # -------------------------

    dense_ranking = np.argsort(similarities)[::-1]

    bm25_ranking = np.argsort(bm25_scores)[::-1]

    k = 60

    rrf_scores = np.zeros(len(chunks))


    # Dense contribution
    for rank, chunk_index in enumerate(dense_ranking):
        rrf_scores[chunk_index] += 1 / (k + rank + 1)


    # BM25 contribution
    for rank, chunk_index in enumerate(bm25_ranking):
        rrf_scores[chunk_index] += 1 / (k + rank + 1)


    # Final hybrid ranking
    hybrid_ranking = np.argsort(rrf_scores)[::-1]


    # Take top 3 candidates for reranking
    top_candidates = hybrid_ranking[:3]


    # -------------------------
    # Cross-Encoder Reranking
    # -------------------------

    reranker_inputs = []

    for chunk_index in top_candidates:
        reranker_inputs.append(
            (question, chunks[chunk_index])
        )


    reranker_scores = reranker.predict(reranker_inputs)


    reranked = sorted(
        zip(top_candidates, reranker_scores),
        key=lambda x: x[1],
        reverse=True
    )


    # Best evidence selected by cross-encoder
    best_chunk_index = reranked[0][0]


    # -------------------------
    # Citation Verification
    # -------------------------

    verification_input = [
        (answer, chunks[best_chunk_index])
    ]

    verification_score = reranker.predict(
        verification_input
    )[0]


    if verification_score >= verification_threshold:
        verification_result = "SUPPORTED"
    else:
        verification_result = "NOT SUPPORTED"


    # -------------------------
    # Final Result
    # -------------------------

    print(f"\nAnswer: {answer}")

    print(
        f"\nBest Evidence (Chunk {best_chunk_index + 1}):"
    )

    print(chunks[best_chunk_index])

    print(
        f"\nCitation Verification Score: "
        f"{verification_score:.4f}"
    )

    print(
        f"Citation Verification: "
        f"{verification_result}"
    )


print("\n" + "=" * 70)
print("Evaluation completed.")
print("=" * 70)