# Project LLM - Cook Assistant

## General Description

This project is a cooking assistant built with a multi-agent LLM architecture.

- `backend/` is a Flask API with a router agent (`MultiAgent`) that decides when to use:
  - `AgentQuery` for Neo4j graph/database questions (text-to-Cypher flow),
  - `AgentScraping` for recipe search/scraping and ingestion.
- `cookassistant-ui/` is a Next.js chat UI.
- Neo4j is used as the knowledge graph database.

Typical request flow:
1. The frontend sends a chat request to `POST /message`.
2. The backend routes the request to the right agent/tool.
3. The final answer is returned to the frontend.

### What the agent can do

- Route each request to the most relevant tool (`query` or `scrape_recipe`) through `MultiAgent`.
- Answer graph/database questions by translating natural language into Neo4j queries (`AgentQuery`).
- Search JoCooks recipes from a dish name or ingredient-based request (`AgentScraping`).
- Scrape a recipe page and return a clean result (title, source, ingredients, instructions).
- Ingest scraped recipe content into Neo4j so it can be reused in later graph queries.
- Provide a direct conversational answer when no specialized tool is needed.

## Setup

### Env variables

Make sure you have a `.env` file, based on the example provided.

### Run with Docker (recommended)

```bash
docker compose up --build
```

Then access:
- Frontend: `http://localhost:5173`
- Neo4j Browser: `http://localhost:7474`
