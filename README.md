# GenAI_marketing_chatbot

This repository contains the full codebase and configuration for a custom-built, multi-agent GenAI system developed as part of the MIT GenAI Lab. The tool is designed for **Campbell’s marketing team** to analyze Amazon advertising data via a **chat interface powered by LangGraph and Databricks MLflow**.

---

## 🔍 What This Tool Does

In plain terms:  
**Ask a marketing question in English, and get back a business-ready, data-grounded answer.**

Example questions:
- “Which keywords had the highest ROAS last month?”
- “What is our ad spend trend over time?”
- “What campaigns should we reinvest in?”

Behind the scenes, the tool:
1. Interprets the user’s question
2. Retrieves structured data via SQL (Genie Agent)
3. Analyzes and explains the results (LLM Agent)
4. Summarizes it into a professional insight (Final Answer Agent)

All via **a simple, secure, chat-based interface**.

---

## ⚠️ Note on Environment & Reproducibility

This project is designed to run entirely **within the Databricks environment** and relies on:

- **Databricks-hosted MLflow model registry and agent endpoints**
- **Databricks-native LangGraph integration**
- **Genie SQL Agent with access to Campbell’s Amazon Ads data tables**
- **Unity Catalog for model versioning and governance**

As a result, this repository **cannot be directly executed or reproduced outside of Databricks**.

In addition:
- The advertising data used by the system is **internal to Campbell’s and hosted securely on Databricks**. It is not included in this repository.
- API tokens and endpoint configurations (e.g., Genie space, model URIs) are managed via Databricks secrets and will not be exposed here.

> If you're a reviewer or collaborator with Databricks access, please reach out for shared access to the workspace.

---

## 🚀 How to Run (Inside Databricks)

### 1. Set up dependencies
Upload repo files to a Databricks workspace and install the required packages:
```bash
pip install -r requirements.txt
```
### 2. Explore via notebook
Use the `interactive_chat` notebook to:
- Test agent responses
- Trace Genie → LLM → Final output
- Debug system behavior with full LangGraph step logs

### 3. Log & Deploy with MLflow
Log the agent:
```python
mlflow.pyfunc.log_model(...)
```
Register to Unity Catalog:
```python
mlflow.register_model(...)
```
Deploy to endpoint:
```python
agents.deploy(...)
```

> Once deployed, Databricks automatically generates a live **chat-based review app** for business users.

---

## 🗂️ Repository Structure

| File | Description |
|------|-------------|
| `agent.py` | LangGraph workflow with Supervisor, Genie, LLM, and Final Answer agents |
| `chat_model.py` | Custom MLflow ChatAgent wrapper (`LangGraphChatAgent`) |
| `config.yaml` | Configuration for deployment and runtime parameters |
| `config_utils.py` | Utility functions to parse config and load runtime context |
| `input_examples.py` | Sample input prompts for testing the model |
| `interactive_chat` | Notebook for local testing and debugging of agent traces |
| `requirements.txt` | Python dependencies for reproducing the environment |

---

## 🧠 Agent Architecture

- All queries start at the **Supervisor Agent**, which decides whether to call **Genie Agent** (for SQL) or **LLM Agent** (for reasoning).
- In cases where both are needed (e.g., data + interpretation), the system routes Genie → LLM → Final Answer Agent.
- The **Final Answer Agent** always produces the final response.
---

## 🔐 Security & Data

- No PII or sensitive customer data is used.
- All advertising metrics are anonymized and scoped to internal campaign-level data.
- All model inference and SQL execution happens **inside the Databricks workspace**.
---

## 📦 Deployment Summary

- ML model logged with `mlflow.pyfunc.log_model`
- Registered to `main.retail_media.campbells_genai_versionFinal` in Unity Catalog
- Deployed using `databricks.agents.deploy()` to expose a secure endpoint
- Interface accessed via auto-generated **Databricks review app**

---

MIT GenAI Lab • 2025  
Ellie Yang, Naiqi Zhang, Brenda Silva, Yu He
