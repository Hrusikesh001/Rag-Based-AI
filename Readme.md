# 🤖 AI Teaching Assistant — RAG-Based Learning Platform

> An AI-powered learning platform that understands course content from educational videos and helps students find relevant explanations using Retrieval-Augmented Generation (RAG).

---

## 📌 Overview

**AI Teaching Assistant** is a Retrieval-Augmented Generation (RAG) based learning platform designed to help students interact with educational video content using natural-language questions.

Instead of manually searching through hours of tutorial videos, students can simply ask a question. The system searches the course knowledge base, identifies the most relevant video segments, and uses a local Large Language Model (LLM) to generate a context-aware response along with the relevant tutorial and timestamp.

### Example

**Student:**

> How can I add a video to an HTML webpage?

**System:**

> The `<video>` element can be used to embed video content in an HTML page.  
> 📚 **Tutorial:** Pure HTML Media Player  
> ⏱️ **Timestamp:** 08:32 – 09:15

---

## ✨ Key Features

- 🎥 **Video-to-Text Processing** using FFmpeg and Whisper
- 📝 **Timestamped Transcripts** for course videos
- 🧠 **Semantic Search** using BGE-M3 embeddings
- 🔎 **Top-K Retrieval** using cosine similarity
- 🤖 **Local LLM Generation** using Llama 3.2
- 🔒 **Local AI Inference** through Ollama
- ⏱️ **Video Timestamp References** for retrieved content
- 📚 **Course-Specific Question Answering**
- 💾 **Persistent Embedding Storage** using Pandas and Joblib
- 🚫 **Out-of-Course Question Handling**

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │   Educational Videos │
                    └──────────┬───────────┘
                               │
                               ▼
                         ┌───────────┐
                         │  FFmpeg   │
                         └─────┬─────┘
                               │
                               ▼
                       ┌───────────────┐
                       │  Audio Files  │
                       └───────┬───────┘
                               │
                               ▼
                    ┌────────────────────┐
                    │  Whisper Large-v2  │
                    │   Speech-to-Text   │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Timestamped JSON   │
                    │     Transcripts    │
                    └─────────┬──────────┘
                              │
                              ▼
                       ┌────────────┐
                       │   BGE-M3   │
                       │ Embeddings │
                       └─────┬──────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Pandas + Joblib      │
                  │ Embedding Dataset    │
                  └──────────┬───────────┘
                             │
                             │
                    ┌────────▼─────────┐
                    │   Student Query  │
                    └────────┬─────────┘
                             │
                             ▼
                       ┌────────────┐
                       │   BGE-M3   │
                       │Query Vector│
                       └─────┬──────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Cosine Similarity    │
                  │   + Top-K Retrieval  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Retrieved Context    │
                  │ + Student Question   │
                  └──────────┬───────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    Llama 3.2     │
                    │  Local LLM       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Grounded Answer │
                    │ + Video Metadata │
                    └──────────────────┘
