
import os
import json
import requests
import pandas as pd
import joblib


# Chunking settings — starting values to test
TARGET_WORDS = 180
OVERLAP_WORDS = 30

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_BATCH_SIZE = 32


def create_embedding(text_list):
    """Generate embeddings using Ollama."""

    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": text_list
        },
        timeout=300
    )

    response.raise_for_status()
    data = response.json()

    if "embeddings" not in data:
        raise ValueError(f"Unexpected Ollama response: {data}")

    return data["embeddings"]


def create_transcript_chunks(
    segments,
    target_words=TARGET_WORDS,
    overlap_words=OVERLAP_WORDS
):
    """
    Combine consecutive transcript segments into larger chunks.
    Preserve video metadata and timestamps.
    """

    segments = sorted(
        segments,
        key=lambda item: float(item.get("start", 0))
    )

    # Ignore empty transcript segments.
    segments = [
        segment for segment in segments
        if str(segment.get("text", "")).strip()
    ]

    chunks = []
    start_idx = 0

    while start_idx < len(segments):
        selected = []
        word_count = 0
        end_idx = start_idx

        # Add segments until the target chunk size is reached.
        while end_idx < len(segments):
            segment = segments[end_idx]
            text = str(segment.get("text", "")).strip()

            selected.append(segment)
            word_count += len(text.split())
            end_idx += 1

            if word_count >= target_words:
                break

        if not selected:
            break

        # Keep metadata from the first segment, then update
        # the text and timestamps for the complete chunk.
        chunk = dict(selected[0])
        chunk["start"] = float(selected[0]["start"])
        chunk["end"] = float(selected[-1]["end"])
        chunk["text"] = " ".join(
            str(segment["text"]).strip()
            for segment in selected
        )

        # Useful for tracing a chunk back to its source segments.
        chunk["source_segment_count"] = len(selected)

        chunks.append(chunk)

        if end_idx >= len(segments):
            break

        # Move backwards to create an approximate word overlap.
        next_start = end_idx
        overlap_count = 0

        while next_start > start_idx + 1:
            previous_text = str(
                segments[next_start - 1].get("text", "")
            ).strip()

            previous_words = len(previous_text.split())

            if overlap_count + previous_words > overlap_words:
                break

            overlap_count += previous_words
            next_start -= 1

        # Guarantee progress even if one segment is very long.
        start_idx = max(start_idx + 1, next_start)

    return chunks


def main():
    json_folder = "jsons"

    if not os.path.isdir(json_folder):
        raise FileNotFoundError(
            f"Transcript folder not found: {json_folder}"
        )

    json_files = sorted(
        filename
        for filename in os.listdir(json_folder)
        if filename.lower().endswith(".json")
    )

    if not json_files:
        raise FileNotFoundError(
            f"No JSON transcript files found in {json_folder}"
        )

    all_chunks = []

    # Stage 1: Create larger chunks from each transcript.
    for json_file in json_files:
        file_path = os.path.join(json_folder, json_file)

        with open(file_path, "r", encoding="utf-8") as file:
            content = json.load(file)

        segments = content.get("chunks", [])

        if not segments:
            print(f"Skipping {json_file}: no transcript chunks found.")
            continue

        print(f"Chunking transcript: {json_file}")

        new_chunks = create_transcript_chunks(segments)

        for chunk in new_chunks:
            chunk["source_file"] = json_file
            all_chunks.append(chunk)

        print(
            f"  Original segments: {len(segments)}"
            f" | New chunks: {len(new_chunks)}"
        )

    if not all_chunks:
        raise ValueError("No transcript chunks were created.")

    # Stage 2: Generate embeddings for the NEW chunk text.
    texts = [chunk["text"] for chunk in all_chunks]

    print(f"\nGenerating embeddings for {len(texts)} chunks...")

    embeddings = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]
        embeddings.extend(create_embedding(batch))

        print(
            f"  Embedded {min(start + len(batch), len(texts))}"
            f" / {len(texts)}"
        )

    if len(embeddings) != len(all_chunks):
        raise ValueError(
            "Number of embeddings does not match number of chunks."
        )

    # Stage 3: Save the dataset in the format your retriever uses.
    for chunk_id, (chunk, embedding) in enumerate(
        zip(all_chunks, embeddings)
    ):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embedding

    df = pd.DataFrame.from_records(all_chunks)

    output_path = "embeddings_chunked.joblib"
    joblib.dump(df, output_path)

    print("\nChunking completed successfully!")
    print(f"Total chunks: {len(df)}")
    print(f"Saved dataset: {output_path}")
    print("\nPreview:")
    print(df[["chunk_id", "title", "start", "end", "text"]].head(3))


if __name__ == "__main__":
    main()
