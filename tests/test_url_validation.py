import unittest
from huggingface_downloader import is_valid_repo_id_or_url

class TestURLValidation(unittest.TestCase):

    def test_valid_repo_ids(self):
        valid_ids = [
            "username/repo-name",
            "organization/model-name",
            "user/repo/tree/main",
            "datasets/username/dataset-name",
            "spaces/username/space-name",
            "huggingface/transformers",
            "google/bert",
            "facebook/bart-large",
            "openai/gpt-3",
            "EleutherAI/gpt-neo-2.7B",
            "user-name/repo.name",
            "org_name/model_v2.0",
            "user-with-hyphens/repo-with-hyphens",
            "hyphenated-user/hyphenated-repo",
            "dots.in.name/more.dots.here",
            "mix-of_both/and-more_here",
            "numbers123/v2-model",
            "user/repo/tree/branch-with-hyphens",
            "organization/model/tree/v1.0-beta",
            "very-long-username-123456789/very-long-repo-name-987654321",  # Long but valid
            "user.name/repo_name-v1.0",  # Mix of allowed special characters
            "user.with.multiple.dots/repo.with.dots",
            "username/repo.name.with.many.dots",
            "openai-community/gpt2/tree/main",
        ]
        for repo_id in valid_ids:
            with self.subTest(repo_id=repo_id):
                self.assertTrue(is_valid_repo_id_or_url(repo_id), f"Failed for: {repo_id}")

    def test_valid_urls(self):
        valid_urls = [
            "https://huggingface.co/google/bert-base-uncased",
            "https://huggingface.co/facebook/bart-large/tree/v1.0",
            "huggingface.co/google/t5-v1_1-large",
            "www.huggingface.co/openai/gpt-3",
            "https://huggingface.co/EleutherAI/gpt-neo-2.7B/tree/main",
            "https://huggingface.co/datasets/username/dataset-name",
            "https://huggingface.co/spaces/username/space-name",
            "https://huggingface.co/user-name/repo-name",
            "https://huggingface.co/org-name/model-name/tree/branch-name",
            "http://www.huggingface.co/user.name/repo.name/tree/v1.0-beta",
            "https://huggingface.co/user.with.dots/repo.with.many.dots",
            "https://huggingface.co/openai-community/gpt2/tree/main",
        ]
        for url in valid_urls:
            with self.subTest(url=url):
                self.assertTrue(is_valid_repo_id_or_url(url), f"Failed for: {url}")

    def test_invalid_inputs(self):
        invalid_inputs = [
            "invalid_url",
            "https://example.com/repo",
            "user@email.com",
            "https://huggingface.co/",
            "huggingface.co",
            "username/repo/invalid",
            "org/model/tree/branch/subbranch",
            "https://huggingface.co/username",
            "http://huggingface.co/org/model/invalid",
            "username",  # Single segment is invalid
            "/repo-name",  # Starting with slash is invalid
            "username/",  # Ending with slash (without tree) is invalid
            "user/name/repo-name",  # Too many segments without 'tree'
            "-username/repo",  # Starting with hyphen
            "username/-repo",  # Segment starting with hyphen
            "user/repo/tree/-branch",  # Branch starting with hyphen
            ".user/repo",  # Starting with dot
            "user/.repo",  # Segment starting with dot
            "_user/repo",  # Starting with underscore
            "user/_repo",  # Segment starting with underscore
            "user/repo/",  # Ending with slash (without tree)
            "user//repo",  # Empty segment
            "user/repo/tree/",  # Empty branch name
            "user/repo/tree",  # Missing branch name
            "/",  # Just a slash
            "a/",  # Single character followed by slash
            "/a",  # Slash followed by single character
            "datasets/",  # Just the datasets prefix
            "spaces/",  # Just the spaces prefix
            "models/user/repo",  # Invalid prefix
            "a/b",  # Too short to be valid
            "user./repo",  # Ending with dot before slash
            "user/.repo",  # Starting with dot after slash
            "http://huggingface.co/gpt2/tree/main",  # Missing author
            "https://huggingface.co/gpt2",  # Missing author
            "gpt2/tree/main",  # Missing author
        ]
        for input_str in invalid_inputs:
            with self.subTest(input_str=input_str):
                self.assertFalse(is_valid_repo_id_or_url(input_str), f"Failed for: {input_str}")

    def test_url_encoded_inputs(self):
        encoded_inputs = [
            "username%2Frepo-name",
            "datasets%2Fusername%2Fdataset-name",
            "spaces%2Fusername%2Fspace-name",
            "https%3A%2F%2Fhuggingface.co%2Fgoogle%2Fbert-base-uncased",
            "user-name%2Frepo-with-hyphens",
            "user.name%2Frepo.with.dots"
        ]
        for input_str in encoded_inputs:
            with self.subTest(input_str=input_str):
                self.assertTrue(is_valid_repo_id_or_url(input_str), f"Failed for: {input_str}")

if __name__ == '__main__':
    unittest.main()
