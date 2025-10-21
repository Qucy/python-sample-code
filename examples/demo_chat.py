import os

from dotenv import load_dotenv

load_dotenv()

from src.utils.azure_identity import AzureIdentityUtil
from src.utils.azure_openai_factory import AzureOpenAIClientFactory


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

    text = factory.quick_chat(deployment, "Say hello from Azure OpenAI!")
    print("Model reply:", text)


if __name__ == "__main__":
    main()