import os
import unittest
from unittest.mock import patch

from azure.identity import ClientSecretCredential
from src.common.azure_identity import AzureIdentityUtil


class TestAzureIdentityUtil(unittest.TestCase):
    def test_identity_created_with_env_vars(self):
        # Provide fake but well-formed Entra ID variables
        env = {
            "AZURE_CLIENT_ID": "00000000-0000-0000-0000-000000000000",
            "AZURE_TENANT_ID": "11111111-1111-1111-1111-111111111111",
            "AZURE_CLIENT_SECRET": "dummy-secret",
        }
        with patch.dict(os.environ, env, clear=False):
            identity = AzureIdentityUtil.from_env()
            self.assertIsInstance(identity, AzureIdentityUtil)
            self.assertIsInstance(identity.credential, ClientSecretCredential)

            # Token provider should be created and be callable
            provider = identity.get_token_provider()
            self.assertTrue(callable(provider))

    def test_missing_env_vars_raises(self):
        # Ensure missing or empty variables trigger a ValueError
        env = {
            "AZURE_CLIENT_ID": "",
            "AZURE_TENANT_ID": "",
            "AZURE_CLIENT_SECRET": "",
        }
        with patch.dict(os.environ, env, clear=False):
            with self.assertRaises(ValueError):
                AzureIdentityUtil.from_env()


if __name__ == "__main__":
    unittest.main()