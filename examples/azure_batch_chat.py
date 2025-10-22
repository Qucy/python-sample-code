import os
import json
import time
from typing import List, Dict

from dotenv import load_dotenv

load_dotenv()

from src.common.azure_identity import AzureIdentityUtil
from src.common.azure_openai_factory import AzureOpenAIClientFactory


def build_jsonl_lines(deployment: str, prompts: List[str], system_prompt: str = "You are helpful.") -> List[str]:
    lines = []
    for i, p in enumerate(prompts):
        body = {
            "model": deployment,  # Global-Batch deployment name
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": p},
            ],
        }
        entry = {
            "custom_id": f"task-{i}",
            "method": "POST",
            "url": "/chat/completions",
            "body": body,
        }
        lines.append(json.dumps(entry))
    return lines


def main():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    # IMPORTANT: Use the Global-Batch deployment name, not your regular chat deployment
    deployment_batch = os.getenv("AZURE_OPENAI_DEPLOYMENT_BATCH")

    if not endpoint or not api_version or not deployment_batch:
        raise RuntimeError(
            "Missing required env. Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT_BATCH."
        )

    # Choose auth: service principal preferred, else API key
    has_sp = all(os.getenv(v) for v in ["AZURE_CLIENT_ID", "AZURE_TENANT_ID", "AZURE_CLIENT_SECRET"])
    if has_sp:
        token_provider = AzureIdentityUtil.from_env().get_token_provider()
        factory = AzureOpenAIClientFactory.from_env_with_identity(token_provider)
    else:
        factory = AzureOpenAIClientFactory.from_env_with_api_key()

    client = factory.create_client()

    prompts = [
        "Say hello and mention Azure.",
        "Give a short fun fact about cloud computing.",
        "Respond politely to a goodbye.",
    ]

    # Build JSONL in-memory and write to a temporary file
    jsonl_lines = build_jsonl_lines(deployment_batch, prompts, system_prompt="Be concise.")
    input_path = "./batch_input.jsonl"
    with open(input_path, "w", encoding="utf-8") as f:
        for line in jsonl_lines:
            f.write(line + "\n")

    # Upload file to Azure OpenAI Files API for batch purpose
    with open(input_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    file_id = getattr(file_obj, "id", None)
    if not file_id:
        raise RuntimeError("Failed to upload input JSONL file for batch.")

    # Submit batch job against chat completions endpoint
    # Note: completion_window must be one of Azure-supported windows (e.g., "24h")
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/chat/completions",
        completion_window="24h",
        # Optional: set output expiration and anchor
        extra_body={
            "output_expires_after": {"seconds": 1209600, "anchor": "created_at"}
        },
    )

    batch_id = getattr(batch, "id", None)
    print("Submitted batch id:", batch_id)
    if not batch_id:
        raise RuntimeError("Batch submission did not return an id.")

    # Poll for completion
    terminal_states = {"completed", "failed", "cancelled"}
    status = getattr(batch, "status", None)
    print("Initial batch status:", status)

    while status not in terminal_states:
        time.sleep(5)
        batch = client.batches.retrieve(batch_id)
        status = getattr(batch, "status", None)
        print("Batch status:", status)

    # Fetch outputs
    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        print("No output_file_id present; batch may have failed or has no results.")
        return

    # Download results; content is JSONL with one line per request result
    # Depending on client version, .content may return a stream; handle generically
    resp = client.files.content(output_file_id)
    # Most versions return a raw bytes payload
    output_bytes = resp.read() if hasattr(resp, "read") else resp  # fallback
    text = (
        output_bytes.decode("utf-8") if isinstance(output_bytes, (bytes, bytearray)) else str(output_bytes)
    )

    # Parse JSONL and print assistant replies
    by_custom_id: Dict[str, str] = {}
    for line in text.splitlines():
        try:
            obj = json.loads(line)
            cid = obj.get("custom_id")
            body = obj.get("response", {}).get("body")
            # Attempt to extract chat content
            content = None
            if isinstance(body, dict):
                choices = body.get("choices")
                if isinstance(choices, list) and choices:
                    msg = choices[0].get("message")
                    if isinstance(msg, dict):
                        content = msg.get("content")
            by_custom_id[cid] = content or json.dumps(body)  # fallback to raw body JSON
        except Exception as e:
            print("Failed to parse output line:", e)

    print("\nBatch results (ordered by input prompts):")
    for i, p in enumerate(prompts):
        cid = f"task-{i}"
        reply = by_custom_id.get(cid, "<missing>")
        print(f"Prompt[{i}]: {p}\nReply[{i}]: {reply}\n")


if __name__ == "__main__":
    main()