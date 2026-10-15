from __future__ import annotations

import os
from typing import Any

from langsmith import Client, tracing_context

LANGSMITH_EU_ENDPOINT = "https://eu.api.smith.langchain.com"


def invoke_judge_model(
    *, model: Any, schema: type, prompt: str, metadata: dict,
    enable_langsmith_tracing: bool,
):
    client = None
    project = None
    if enable_langsmith_tracing:
        key = os.environ.get("LANGSMITH_API_KEY")
        project = os.environ.get("LANGSMITH_PROJECT")
        endpoint = os.environ.get("LANGSMITH_ENDPOINT", LANGSMITH_EU_ENDPOINT).rstrip("/")
        if not key or not project:
            raise RuntimeError("LangSmith tracing requires an API key and project name.")
        if endpoint != LANGSMITH_EU_ENDPOINT:
            raise RuntimeError("LLM judging requires the LangSmith EU endpoint.")
        client = Client(api_url=endpoint, api_key=key)
        client.read_project(project_name=project)
    tags = ["llm-judge", metadata["task"], "development"]
    with tracing_context(
        enabled=enable_langsmith_tracing, project_name=project, client=client,
        tags=tags, metadata=metadata,
    ):
        result = model.with_structured_output(schema).invoke(
            prompt, config={
                "run_name": f"{metadata['task']}-llm-judge", "tags": tags, "metadata": metadata,
            },
        )
    if client is not None:
        client.flush()
    return result
