---
title: SpendWise Backend
emoji: 💸
colorFrom: indigo
colorTo: viloet
sdk: docker
pinned: false
app_port: 7860
---

# SpendWise Backend — ML Categorisation Engine

This Space hosts the FastAPI backend and 5-layer ML pipeline for the SpendWise transaction categorisation system.

## 🚀 Deployment Info
- **SDK:** Docker
- **Port:** 7860
- **Base Image:** python:3.11-slim

## 🛠️ Setup Instructions
1. Ensure all environment variables (SUPABASE_URL, etc.) are added in the **Settings > Variables and Secrets** tab.
2. The API will be available at `https://vihaan2code-spendwise-backend.hf.space`.

## 📈 ML Assets
The following models are cached on first run:
- `intfloat/e5-large` (1.3GB)
- `facebook/bart-large-mnli` (1.6GB)
