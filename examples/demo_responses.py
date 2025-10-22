import os
import json

from dotenv import load_dotenv

load_dotenv()

from src.common.azure_identity import AzureIdentityUtil
from src.common.azure_openai_factory import AzureOpenAIClientFactory


def main():
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    if not deployment:
        raise ValueError(
            "AZURE_OPENAI_DEPLOYMENT must be set to your model deployment name."
        )

    has_sp = all(
        os.getenv(var) for var in ["AZURE_CLIENT_ID", "AZURE_TENANT_ID", "AZURE_CLIENT_SECRET"]
    )

    if has_sp:
        print("Using Microsoft Entra ID (service principal) authentication...")
        token_provider = AzureIdentityUtil.from_env().get_token_provider()
        factory = AzureOpenAIClientFactory.from_env_with_identity(token_provider)
    else:
        print("Using API key authentication...")
        factory = AzureOpenAIClientFactory.from_env_with_api_key()

    client = factory.create_client()

    # Enforce JSON output using response_format for Responses API
    # Also instruct via system message to ensure consistent JSON
    response = client.responses.create(
        model=deployment,
        input=[
            {
                "role": "system",
                "content": "You are a helpful assistant that ONLY outputs JSON objects.",
            },
            {
                "role": "user",
                "content": "Return a JSON with fields: topic (string), summary (string). Topic: Azure identity.",
            },
        ],
        response_format={"type": "json_object"},
    )

    # Extract JSON text robustly
    output_text = None
    if hasattr(response, "output_text"):
        output_text = response.output_text
    else:
        try:
            if hasattr(response, "output") and response.output:
                content = response.output[0].content if hasattr(response.output[0], "content") else None
                if content:
                    # Find the first output_text chunk
                    for chunk in content:
                        if getattr(chunk, "type", None) == "output_text" and hasattr(chunk, "text"):
                            output_text = chunk.text
                            break
        except Exception:
            pass

    if not output_text:
        # Fallback to string dump if SDK shape differs
        output_text = str(response)

    print("Raw JSON response text:")
    print(output_text)

    # Try to parse JSON string
    try:
        obj = json.loads(output_text)
        print("\nParsed JSON object:")
        print(json.dumps(obj, indent=2))
    except Exception as e:
        print("Failed to parse JSON:", e)


if __name__ == "__main__":
    main()