# KiwiQ AI Platform

**Production-grade multi-agent orchestration platform — JSON-defined agents, multi-tier memory, and built-in observability.**

Battle-tested on 200+ enterprise AI agents. Define complex multi-step AI workflows as Python/JSON graph schemas with 24+ reusable node types, multi-provider LLM support, human-in-the-loop interactions, web scraping, RAG pipelines, and versioned customer data management. Ships with 27+ ready-to-use workflow definitions.

This platform powered [KiwiQ AI](https://www.linkedin.com/company/kiwiq-ai/) (Marketing AI Agents) in production and is now fully open-sourced.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-green.svg)](https://www.python.org/downloads/release/python-3120/)

---

## Features

- **SDK-first workflow engine** — Define workflows as Python/JSON graph schemas (not a visual builder), compiled to LangGraph and executed via Prefect
- **Multi-provider LLM support** — OpenAI, Anthropic, Google Gemini, Perplexity, Fireworks, AWS Bedrock
- **Human-in-the-Loop (HITL)** — Pause workflows for human review, input, or approval with real-time WebSocket streaming
- **24+ reusable node types** — LLM, routing, conditional branching, data transforms, scraping, code execution, sub-workflows, and more
- **27+ production workflow definitions** — Content creation, diagnostics, lead scoring, deep research, playbook generation, and more included out of the box
- **Multi-tier memory & state** — PostgreSQL for relational state, MongoDB for versioned documents, Weaviate for vector search, Redis for caching
- **RAG pipelines** — Document ingestion, vector search (Weaviate), and retrieval-augmented generation
- **Customer data management** — Versioned document storage in MongoDB with CRUD workflow nodes
- **Event-driven architecture** — RabbitMQ-based event bus for async processing between services
- **Observability** — Real-time progress via RabbitMQ events, WebSocket streaming, Prefect dashboard, structured logging
- **Web scraping** — LinkedIn profiles/companies, web crawling, AI-powered search engines
- **Sandboxed code execution** — Run user-defined Python safely within workflows
- **Billing & auth** — Stripe integration, JWT authentication, role-based access control

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI (kiwi_app)                      │
│  Auth │ Billing │ Workflow API │ RAG │ Data Jobs │ WebSocket │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ RabbitMQ │ │ Prefect  │ │  Redis   │
        │ (events) │ │ (orch)   │ │ (cache)  │
        └────┬─────┘ └────┬─────┘ └──────────┘
             │             │
             ▼             ▼
        ┌──────────────────────────┐
        │   Workflow Service       │
        │  ┌────────┐ ┌────────┐  │
        │  │LangGraph│ │ Nodes  │  │
        │  │ Engine  │ │Registry│  │
        │  └────────┘ └────────┘  │
        └──────────┬───────────────┘
                   │
     ┌─────────┬───┼───┬──────────┐
     ▼         ▼   ▼   ▼          ▼
┌────────┐┌───────┐┌────────┐┌──────────┐
│Postgres││MongoDB││Weaviate ││LLM APIs  │
│(state) ││(docs) ││(vector) ││(OpenAI…) │
└────────┘└───────┘└────────┘└──────────┘
```

## Quick Start

### Prerequisites

- Python 3.12
- [Poetry](https://python-poetry.org/docs/#installation)
- [Docker](https://docs.docker.com/get-docker/) & Docker Compose

### 1. Clone and install

```bash
git clone https://github.com/kiwiq-ai/kiwiq-oss.git
cd kiwiq-oss
poetry install
```

### 2. Configure environment

```bash
cp .env.sample .env
```

Edit `.env` and fill in required values:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (required for LLM nodes) |
| `ANTHROPIC_API_KEY` | Anthropic API key (optional) |
| `GOOGLE_API_KEY` | Google Gemini API key (optional) |
| `PPLX_API_KEY` | Perplexity API key (optional, for deep research workflows) |
| `POSTGRES_*` | PostgreSQL credentials |
| `MONGO_ROOT_*` | MongoDB credentials |
| `RABBITMQ_DEFAULT_*` | RabbitMQ credentials |
| `REDIS_PASSWORD` | Redis password |
| `SECRET_KEY` | JWT secret — generate with `openssl rand -hex 32` |

See [`.env.sample`](.env.sample) for the full list of configuration options.

### 3. Start services

```bash
# Development (all services including databases)
docker compose -f docker-compose-dev.yml up --build
```

### 4. Access the platform

- **API docs:** http://localhost:8000/docs
- **Prefect dashboard:** http://localhost:4201
- **RabbitMQ management:** http://localhost:15672
- **RedisInsight:** http://localhost:8001

## Using with Claude Code

This repository ships with a comprehensive [`CLAUDE.md`](CLAUDE.md) that provides Claude Code with full context about the project — architecture, commands, testing patterns, workflow development, and troubleshooting. To get started:

1. Install [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
2. Run `claude` from the repo root

Claude Code will automatically pick up `CLAUDE.md` and can help with local setup, running tests, modifying services, building workflows, debugging, and navigating the codebase.

## Docker Environments

### Development (`docker-compose-dev.yml`)

Full local environment with all services containerized:

| Service | Port | Description |
|---------|------|-------------|
| FastAPI App | 8000 | Core API server with live reload |
| PostgreSQL | 5432 | Relational data, workflow state, LangGraph checkpoints |
| MongoDB | 27017 | Customer data, workflow configs, prompt templates |
| Redis | 6379 | Caching and session management |
| RabbitMQ | 5672 / 15672 | Event streaming / management UI |
| Weaviate | 8080 | Vector database for RAG |
| Prefect Server | 4201 | Workflow orchestration dashboard |
| Prefect Agent | — | Workflow execution worker |

All data is persisted via named Docker volumes. Code changes are hot-reloaded via volume mounts.

### Production (`docker-compose.prod.yml`)

Production-hardened environment with:

- **Nginx + Certbot** for SSL/TLS termination
- **External managed databases** (PostgreSQL, MongoDB) — not containerized
- **Resource limits** on all containers (CPU and memory caps)
- **JSON-file logging** with rotation
- **Health checks** on critical services
- Tuned for **4x concurrent workflow execution**

## Running Without Docker

To run services directly on your host:

```bash
# Set Python path
export PYTHONPATH=$(pwd):$(pwd)/services

# Start FastAPI server
poetry run uvicorn kiwi_app.main:app --host 0.0.0.0 --port 8000 --reload

# Start Prefect worker (separate terminal)
poetry run python services/workflow_service/services/worker.py
```

You'll need PostgreSQL, MongoDB, Redis, RabbitMQ, and Weaviate running separately (locally or hosted).

## Repository Structure

```
kiwiq-oss/
├── libs/src/                        # Shared libraries
│   ├── db/                          #   PostgreSQL (SQLModel, Alembic migrations)
│   ├── mongo_client/                #   MongoDB client
│   ├── redis_client/                #   Redis client
│   ├── rabbitmq_client/             #   RabbitMQ event streaming
│   ├── weaviate_client/             #   Weaviate vector DB client
│   ├── global_config/               #   Global settings
│   └── global_utils/                #   Shared utilities
│
├── services/
│   ├── kiwi_app/                    # Core FastAPI application
│   │   ├── auth/                    #   Authentication & authorization
│   │   ├── billing/                 #   Stripe billing & credits
│   │   ├── workflow_app/            #   Workflow API, WebSockets, events
│   │   ├── data_jobs/               #   Data ingestion & RAG pipelines
│   │   └── rag_service/             #   RAG query endpoints
│   │
│   ├── workflow_service/            # Workflow engine
│   │   ├── registry/nodes/          #   24+ node implementations
│   │   ├── graph/                   #   GraphSchema → LangGraph builder
│   │   └── services/worker.py       #   Prefect worker entrypoint
│   │
│   ├── linkedin_integration/        # LinkedIn OAuth & API
│   └── scraper_service/             # Web scraping endpoints
│
├── standalone_test_client/          # Workflow SDK, 27+ workflow definitions
│   └── kiwi_client/
│       └── workflows/active/        #   Production workflow examples
│
├── tests/                           # 75+ unit & integration tests
├── docs/                            # 40+ pages of technical documentation
├── docker/                          # Dockerfiles & setup scripts
└── pyproject.toml                   # Poetry dependencies
```

## Node Types

Workflows are composed from reusable nodes defined in Python/JSON. Key categories:

| Category | Nodes | Description |
|----------|-------|-------------|
| **Core** | `input_node`, `output_node`, `router_node`, `map_list_router_node`, `if_else_condition`, `transform_data` | Flow control, routing, data transforms |
| **LLM** | `llm`, `prompt_constructor`, `prompt_template_loader` | Multi-provider LLM execution, prompt building |
| **Data** | `load_customer_data`, `store_customer_data`, `delete_customer_data`, `merge_aggregate` | MongoDB CRUD, data joins |
| **Scraping** | `linkedin_scraping`, `crawler_scraper`, `ai_answer_engine_scraper` | LinkedIn, web crawling, AI search |
| **Advanced** | `tool_executor`, `code_runner`, `workflow_runner`, `hitl_node` | Tool use, sandboxed code, sub-workflows, human review |

## Included Workflows

The [`standalone_test_client/`](standalone_test_client/) ships with **27+ production workflow definitions** — complete with graph schemas, LLM prompts, HITL test inputs, and runner scripts. These serve as both reference implementations and a starting point for building your own.

### Content Studio (11 workflows)

Workflows for content creation across blog and LinkedIn:

| Workflow | Description |
|----------|-------------|
| **Blog Brief to Blog** | Full blog generation from brief with knowledge base enrichment, SEO optimization, and HITL approval |
| **Blog AEO/SEO Scoring** | Content scoring framework — architecture, depth, discovery optimization, and internal structure |
| **Blog Content Calendar** | Generate topic suggestions for upcoming weeks based on posting frequency and customer context |
| **Blog User Input to Brief** | Research pipeline: Google search, Reddit via Perplexity, AI topic suggestions, HITL topic selection |
| **Blog Content Optimization** | Parallel analysis (structure, SEO, readability, content gaps) with competitive web search |
| **LinkedIn Content Creation** | Post creation with user profile/strategy integration and feedback loops |
| **LinkedIn Content Calendar** | Topic planning with strategy, drafts, and preference context |
| **LinkedIn Alternate Text** | Generate alternate text suggestions based on profile and playbook |
| **LinkedIn User Input to Brief** | Strategic brief generation from executive profiles and content strategy |

### Content Diagnostics (10 workflows)

AI visibility analysis, competitive intelligence, and content audits:

| Workflow | Description |
|----------|-------------|
| **Company AI Visibility** | Assess blog coverage and AI visibility in search/answer engines |
| **Executive AI Visibility** | Analyze executive LinkedIn presence for AI search recognition |
| **Competitor Content Analysis** | Parallel competitor strategy analysis via `map_list_router` |
| **Blog Content Analysis** | Classify blog posts by sales funnel stage, analyze by group |
| **LinkedIn Content Analysis** | Theme extraction and classification across LinkedIn posts |
| **Deep Research** | Comprehensive web research using Perplexity deep research models |
| **LinkedIn Scraping** | Profile and post data extraction with filtering and storage |
| **Orchestrator** | Coordinate multiple diagnostics workflows with conditional execution flags |
| **Company Analysis** | Extract business intelligence, Reddit research, strategic content pillars |

### Sales & Investor (2 workflows)

| Workflow | Description |
|----------|-------------|
| **Lead Scoring & Talking Points** | Parallel company qualification, ContentQ scoring, gap analysis, personalized talking points |
| **Investor Lead Scoring** | 100-point scoring framework across fund vitals, thesis alignment, partner value, with LinkedIn + Perplexity deep research |

### Playbooks (2 workflows)

| Workflow | Description |
|----------|-------------|
| **Blog Content Playbook** | Generate implementation strategies from company documents with actionable timelines |
| **LinkedIn Content Playbook** | Executive-focused content strategy from LinkedIn profile analysis |

### Labs & Utilities (3 workflows)

| Workflow | Description |
|----------|-------------|
| **File Summarisation** | Task-specific document synthesis with intelligent noise filtering |
| **On-Demand Research** | Flexible deep research with configurable save targets |
| **Auto User Onboarding** | Optional LinkedIn and blog/company onboarding flows with router-based gating |

Each workflow folder includes:
- `wf_*_json.py` — Graph schema definition (Python/JSON)
- `wf_llm_inputs.py` — LLM prompts, output schemas, templates
- `wf_testing/` — HITL inputs, runner script, test artifacts

See the [workflow onboarding guide](standalone_test_client/kiwi_client/workflows/active/ONBOARDING_WORKFLOW_TESTING.md) to get started.

## Testing

```bash
# Run all tests
PYTHONPATH=$(pwd):$(pwd)/services poetry run pytest

# Unit tests only
PYTHONPATH=$(pwd):$(pwd)/services poetry run pytest -m unit

# Integration tests only
PYTHONPATH=$(pwd):$(pwd)/services poetry run pytest -m integration

# Specific test file
PYTHONPATH=$(pwd):$(pwd)/services poetry run pytest tests/unit/services/workflow_service/test_example.py -v

# With coverage
PYTHONPATH=$(pwd):$(pwd)/services poetry run pytest --cov=services --cov=libs --cov-report=term-missing
```

See [`tests/`](tests/) for the full test suite (75+ test files).

## Documentation

| Topic | Location |
|-------|----------|
| **Workflow building guide** | [`docs/.../workflow_building_guide.md`](docs/design_docs/workflow_service_docs/workflow_builder_guides/workflow_building_guide.md) |
| **Node guides (24)** | [`docs/.../nodes/`](docs/design_docs/workflow_service_docs/workflow_builder_guides/nodes/) |
| **Nodes interplay** | [`docs/.../nodes_interplay_guide.md`](docs/design_docs/workflow_service_docs/workflow_builder_guides/nodes_interplay_guide.md) |
| **LLM models reference** | [`docs/.../llm_models_guide.md`](docs/design_docs/workflow_service_docs/workflow_builder_guides/nodes/llm_models_guide.md) |
| **Production deployment** | [`docs/prod/README_PROD.md`](docs/prod/README_PROD.md) |
| **Unit testing guide** | [`docs/README_unit_testing.md`](docs/README_unit_testing.md) |
| **Database setup** | [`docs/DB_SETUP.md`](docs/DB_SETUP.md) |
| **Customer data integration** | [`docs/design_docs/customer_data/`](docs/design_docs/customer_data/) |
| **Workflow SDK & examples** | [`standalone_test_client/`](standalone_test_client/) |

## Database Migrations

```bash
# Run pending migrations
PYTHONPATH=$(pwd):$(pwd)/services poetry run alembic upgrade head

# Create a new migration
PYTHONPATH=$(pwd):$(pwd)/services poetry run alembic revision --autogenerate -m "description"
```

## LLM Providers

| Provider | Models | Notable Features |
|----------|--------|-----------------|
| **OpenAI** | GPT-5, GPT-4.1, o3/o4-mini, GPT-4o | Web search, code interpreter, deep research |
| **Anthropic** | Claude Opus 4, Sonnet 4.5/4, Haiku 3.5 | Web search, code interpreter |
| **Google** | Gemini 2.5 Pro/Flash | Multimodal |
| **Perplexity** | Sonar Deep Research, Sonar Reasoning Pro | Built-in web search |
| **Fireworks** | DeepSeek R1 | Fast inference |
| **AWS Bedrock** | DeepSeek R1 | Managed deployment |

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Make your changes with tests
4. Run the test suite (`poetry run pytest`)
5. Submit a pull request

## Acknowledgements

The KiwiQ AI platform was designed, built from scratch and open-sourced by **[Raunak Bhandari](https://github.com/rbcorx)**, CTO and co-founder of [KiwiQ AI](https://www.linkedin.com/company/kiwiq-ai/), including the core platform, workflow engine, node system, infrastructure, shared libraries, many of the production workflows, and auxiliary systems such as scraping services and browser pools.

**Gaurav Kumar** and **Anish Bharadwaj** built and tested several of the marketing and deep market research focused workflows, helped with platform testing and productizing for our customers.

## Disclaimer

This project is provided for educational and research purposes only. Scraping websites may violate their Terms of Service. The authors and contributors are not responsible for any misuse of this software. Users are solely responsible for ensuring their use complies with all applicable laws, regulations, and platform policies. Use at your own risk.

## License

Apache 2.0 — see the [LICENSE](LICENSE) file for details.

## FAQ

### General Questions

**What is KiwiQ?**
KiwiQ is a production-grade multi-agent orchestration platform that uses JSON-defined agents, multi-tier memory, and built-in observability. It is designed for enterprise AI agents and supports complex multi-step workflows.

**What are the main features?**
- SDK-first workflow engine with Python/JSON graph schemas
- Multi-provider LLM support (OpenAI, Anthropic, Google Gemini, etc.)
- Human-in-the-Loop (HITL) capabilities
- 24+ reusable node types
- 27+ production workflow definitions
- Multi-tier memory and state management
- RAG pipelines
- Event-driven architecture
- Observability and monitoring

### Quick Start

**How do I get started?**
1. Clone the repository
2. Set up your environment variables
3. Run the Docker Compose setup
4. Access the web interface at http://localhost:8000

**What are the system requirements?**
- Python 3.12+
- Docker and Docker Compose
- PostgreSQL, MongoDB, Redis, Weaviate, RabbitMQ (included in Docker Compose)

### Node Types

**What node types are available?**
KiwiQ provides 24+ node types including:
- LLM nodes for AI-powered processing
- Routing nodes for conditional branching
- Data transform nodes
- Web scraping nodes
- Code execution nodes
- Sub-workflow nodes
- And more

**How do I create custom nodes?**
You can create custom nodes by extending the base node class and implementing the required methods. See the documentation for details.

### LLM Providers

**Which LLM providers are supported?**
- OpenAI
- Anthropic
- Google Gemini
- Perplexity
- Fireworks
- AWS Bedrock

**How do I configure a new LLM provider?**
Add your API keys to the environment variables and configure the provider in the workflow definition. See the LLM Providers section in the README for details.

### Database Setup

**What databases are required?**
- PostgreSQL for relational state
- MongoDB for versioned documents
- Weaviate for vector search
- Redis for caching

**How do I set up the databases?**
Use the provided Docker Compose file which includes all required databases. Alternatively, you can configure each database separately.

### Docker Environments

**How do I run KiwiQ with Docker?**
1. Clone the repository
2. Copy `.env.example` to `.env` and configure your environment variables
3. Run `docker-compose up -d`
4. Access the web interface at http://localhost:8000

**Can I run without Docker?**
Yes, see the "Running Without Docker" section in the README for instructions.

### Troubleshooting

**I am getting connection errors. What should I do?**
1. Check that all Docker containers are running: `docker-compose ps`
2. Verify your environment variables are set correctly
3. Check the logs: `docker-compose logs`

**How do I debug workflow issues?**
1. Check the Prefect dashboard for workflow execution details
2. Enable debug logging in your environment variables
3. Review the workflow definition for any syntax errors

**Where can I get help?**
- Check the documentation
- Open an issue on GitHub
- Join the community discussions
