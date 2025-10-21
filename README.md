# Azure OpenAI Utilities (Python)

Two small utility classes to streamline connecting to Azure OpenAI:
- AzureIdentityUtil: sets up Microsoft Entra ID (Azure AD) credentials using `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, and `AZURE_CLIENT_SECRET`, and exposes a bearer token provider for the Cognitive Services scope.
- AzureOpenAIClientFactory: builds an `AzureOpenAI` client from the OpenAI Python library using either Entra ID token provider or API key.

## Project Layout

```
python-sample-code/
├─ src/
│  └─ utils/
│     ├─ __init__.py
│     ├─ azure_identity.py
│     └─ azure_openai_factory.py
├─ examples/
│  ├─ demo_chat.py
│  └─ demo_responses.py
├─ requirements.txt
└─ .env.example
```

## Prerequisites

- Python 3.10+
- An Azure OpenAI resource and a deployment (e.g. `gpt-4o`) in your Azure subscription
- One of:
  - Microsoft Entra ID service principal with the `Cognitive Services OpenAI User` role on the Azure OpenAI resource
  - Azure OpenAI API key

## Required Packages

- `openai` (Azure OpenAI support via `AzureOpenAI` client)
- `azure-identity` (for Entra ID credentials and token provider)
- `python-dotenv` (optional, used in examples to read `.env`)

Install them:

```
pip install -r requirements.txt
```

## Configure Environment

Copy `.env.example` to `.env` and fill in:

```
AZURE_CLIENT_ID=<your-service-principal-client-id>
AZURE_TENANT_ID=<your-tenant-id>
AZURE_CLIENT_SECRET=<your-service-principal-secret>

AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-10-21

# If using API key auth instead of Entra ID
AZURE_OPENAI_API_KEY=<your-azure-openai-key>

# Deployment name of the model to use (e.g., gpt-4o)
AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>
```

Notes:
- `AZURE_OPENAI_API_VERSION` should match a supported version for the features you use. Microsoft lists the latest GA/preview versions here: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle
- In Azure, the `model` parameter is your deployment name (not a raw model ID).

## Usage Examples

### Chat Completions (simple)

```python
import os
from dotenv import load_dotenv
from src.utils.azure_identity import AzureIdentityUtil
from src.utils.azure_openai_factory import AzureOpenAIClientFactory

load_dotenv()

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Prefer Entra ID if SP creds are present; otherwise fall back to API key
has_sp = all(os.getenv(v) for v in ["AZURE_CLIENT_ID", "AZURE_TENANT_ID", "AZURE_CLIENT_SECRET"])
if has_sp:
    token_provider = AzureIdentityUtil.from_env().get_token_provider()
    factory = AzureOpenAIClientFactory.from_env_with_identity(token_provider)
else:
    factory = AzureOpenAIClientFactory.from_env_with_api_key()

client = factory.create_client()
response = client.chat.completions.create(
    model=deployment,
    messages=[{"role": "user", "content": "Say hello from Azure OpenAI!"}],
)
print(response.choices[0].message.content)
```

### Responses API (modern, unified)

```python
import os
from dotenv import load_dotenv
from src.utils.azure_identity import AzureIdentityUtil
from src.utils.azure_openai_factory import AzureOpenAIClientFactory

load_dotenv()

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

has_sp = all(os.getenv(v) for v in ["AZURE_CLIENT_ID", "AZURE_TENANT_ID", "AZURE_CLIENT_SECRET"])
if has_sp:
    token_provider = AzureIdentityUtil.from_env().get_token_provider()
    factory = AzureOpenAIClientFactory.from_env_with_identity(token_provider)
else:
    factory = AzureOpenAIClientFactory.from_env_with_api_key()

client = factory.create_client()
resp = client.responses.create(model=deployment, input="Write a haiku about cloud identity.")
print(getattr(resp, "output_text", str(resp)))
```

## Design Notes

- Entra ID auth: This uses `azure.identity.ClientSecretCredential` and `get_bearer_token_provider` for the `https://cognitiveservices.azure.com/.default` scope. Pass the token provider to the `AzureOpenAI` client via `azure_ad_token_provider`.
- API key auth: Provide `AZURE_OPENAI_API_KEY` and the endpoint; the factory constructs `AzureOpenAI` with `api_key`.
- API version: Keep this configurable (`AZURE_OPENAI_API_VERSION`). Use the latest GA that supports the endpoints you need.

## References

- OpenAI Cookbook – Azure examples (AAD token provider):
  - Chat Completions: https://cookbook.openai.com/examples/azure/chat
  - Embeddings: https://cookbook.openai.com/examples/azure/embeddings
- Microsoft Learn – Azure OpenAI API version lifecycle: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle
- AAD token provider in `AzureOpenAI` (community/guide):
  - https://luke.geek.nz/azure/authenticating-azureopenai-python/
  - https://shweta-lodha.medium.com/authenticate-your-azure-openai-based-app-using-microsoft-entra-id-b64c15acc081