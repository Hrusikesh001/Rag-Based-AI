import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests


def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        },
        timeout=180
    )
    r.raise_for_status()
    return r.json()["embeddings"]


def inference(prompt):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        },
        timeout=180
    )
    r.raise_for_status()
    return r.json()["response"]





def verify_context(question, context_json):
    prompt = f"""
You are checking whether course transcripts contain evidence
that answers a student's question.

Student question:
{question}

Retrieved course transcripts:
{context_json}

Evaluate the evidence carefully.

- Return YES if at least one excerpt directly explains the main
  concept asked about or multiple excerpts provide enough evidence.
- A basic explanation is sufficient; every detail is not required.
- Ignore unrelated transcript excerpts.
- Return NO only if the relevant excerpts do not explain the
  requested concept or lack essential information.
- Do not use knowledge outside the provided transcripts.

Important example:
Question: What is the CSS box model?
Evidence: The transcript identifies content, padding, border,
and margin, and explains how these parts surround the content.
Decision: YES

Question: What is a JavaScript closure?
Evidence: The transcripts only discuss the CSS box model.
Decision: NO

Return exactly one word: YES or NO.

Decision:
"""

    try:
        result = inference(prompt).strip().upper()
        print(f"\n[DEBUG] Raw verifier response: {result!r}")

        # Accept only a standalone YES or NO answer.
        if result.rstrip(".,!?") == "YES":
            return True

        if result.rstrip(".,!?") == "NO":
            return False

        print("[DEBUG] Invalid verifier response; rejecting answer.")
        return False

    except Exception as e:
        print(f"Context verification failed: {e}")
        return False



def format_timestamp(seconds):
    seconds = int(float(seconds))
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{minutes:02d}:{remaining_seconds:02d}"


# Load saved transcript embeddings
df = joblib.load("embeddings_chunked.joblib")

# Ask the student a question
incoming_query = input("Ask a Question: ").strip()

if not incoming_query:
    raise SystemExit("Please enter a question.")

# Generate an embedding for the question
question_embedding = create_embedding([incoming_query])[0]

# Calculate cosine similarity with transcript embeddings
similarities = cosine_similarity(
    np.vstack(df["embedding"].values),
    [question_embedding]
).flatten()

# Retrieve the five most relevant chunks
top_results = 5
max_indx = similarities.argsort()[::-1][:top_results]

new_df = df.iloc[max_indx].copy()

# Use the highest-scoring chunk as the primary source
best_idx = int(np.argmax(similarities))
best_chunk = df.iloc[best_idx]
best_score = float(similarities[best_idx])

print(f"Similarity score: {best_score:.4f}")

# Provisional threshold; validate it using your own course questions
RELEVANCE_THRESHOLD = 0.45

if best_score < RELEVANCE_THRESHOLD:
    message = (
        "I couldn't find sufficiently relevant information in the "
        "available course transcripts to answer confidently."
    )
    print("\n" + message)

    with open("response.txt", "w", encoding="utf-8") as f:
        f.write(message)

    raise SystemExit(0)

print("\n--- Best Matching Transcript ---")
print(f"Video: {best_chunk['number']} — {best_chunk['title']}")
print(
    f"Timestamp: {format_timestamp(best_chunk['start'])} "
    f"- {format_timestamp(best_chunk['end'])}"
)
print(f"Transcript: {best_chunk['text']}")
print("\n--- Retrieved Transcript Chunks ---")
for _, row in new_df.iterrows():
    print(f"\nVideo: {row['number']} — {row['title']}")
    print(f"Time: {format_timestamp(row['start'])} - {format_timestamp(row['end'])}")
    print(f"Text: {row['text']}")

# Prepare retrieved context and convert timestamps BEFORE building the prompt
context_data = new_df[
    ["title", "number", "start", "end", "text"]
].copy()

context_data["start"] = context_data["start"].apply(format_timestamp)
context_data["end"] = context_data["end"].apply(format_timestamp)

context_json = context_data.rename(columns={
    "title": "video_title",
    "number": "video_number",
    "start": "start_timestamp",
    "end": "end_timestamp",
    "text": "transcript_text"
}).to_json(orient="records", force_ascii=False)

best_source = (
    f"Video {best_chunk['number']} — {best_chunk['title']}, "
    f"timestamp {format_timestamp(best_chunk['start'])}"
)

# Build the prompt
prompt = f'''You are an AI learning assistant for a technical course.

Your goal is to help students understand concepts covered in the course and find the relevant learning material.

Retrieved course context:
{context_json}

Student's question:
"{incoming_query}"

Primary recommended source:
{best_source}

Instructions:
1. Identify the subject and intent of the student's question. It may relate to any topic covered in the retrieved course material.
2. Answer directly in simple, clear English appropriate for a learner.
3. Use the retrieved transcript as evidence about what the instructor actually covered.
4. Answer only when the retrieved course context contains sufficient information to answer the student's question. Do not use general knowledge to answer questions when the answer is not supported by the retrieved context. If the answer is not present, follow Rule 14 exactly.
5. Adapt the explanation to the subject. Use a code example, command, formula, analogy, or step-by-step explanation when helpful.
6. Do not assume the question is about a particular technology or programming language unless the question or retrieved context supports that interpretation.
7. Recommend a source that directly supports the answer. Prefer the primary recommended source only when its transcript supports the answer. Copy the video number and title exactly as provided in the retrieved context.
8. Use the recommended source's exact start timestamp from the retrieved context. Never confuse a timestamp with a video number. Do not provide a timestamp unless the source supports the answer.
9. Describe what the transcript actually covers. Do not claim that a short excerpt explains an entire concept.
10. If the retrieved context is insufficient, follow Rule 14 exactly. Do not provide a general-knowledge answer, example, video recommendation, or timestamp.
11. If the question is unrelated to the course or the retrieved context does not contain sufficient evidence to answer it, follow Rule 14 exactly.
12. You may mention other retrieved videos as additional resources when they are relevant.
13. Never fabricate video titles, video numbers, timestamps, or claims about the instructor's explanation.
14. If the answer is not present in the retrieved context, respond exactly:

Answer not found in course material

Sorry, I couldn't find an explanation of this topic in the available course transcripts. Please try asking about another topic covered in the course.

Response format:

**Answer:** Explain the concept clearly and accurately.

**Example:** Provide a relevant example when useful. Omit this section if an example is unnecessary.

**Where to learn:** Video [exact number] — [exact title].

**Timestamp:** [exact start timestamp] — [brief description supported by the transcript.]'''

# Save the prompt for debugging
with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

# Generate and display the answer
if not verify_context(incoming_query, context_json):
    response = """Answer not found in course material

Sorry, I couldn't find an explanation of this topic in the available course transcripts. Please try asking about another topic covered in the course."""
else:
    response = inference(prompt)

print("\n" + response)

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response)


