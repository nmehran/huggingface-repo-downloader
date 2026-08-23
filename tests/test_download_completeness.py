import os
import tempfile
import unittest

from huggingface_downloader import file_is_complete, should_download_file


class TestDownloadCompleteness(unittest.TestCase):
    def test_file_is_complete_requires_matching_size(self):
        with tempfile.TemporaryDirectory() as output_dir:
            rel_path = "config.json"
            full_path = os.path.join(output_dir, rel_path)
            with open(full_path, "w", encoding="utf-8") as handle:
                handle.write("abc")

            self.assertTrue(
                file_is_complete(rel_path, {"size": 3}, output_dir)
            )
            self.assertFalse(
                file_is_complete(rel_path, {"size": 10}, output_dir)
            )
            self.assertFalse(
                file_is_complete("missing.json", {"size": 3}, output_dir)
            )

    def test_missing_file_is_always_downloaded_even_if_metadata_claims_it(self):
        with tempfile.TemporaryDirectory() as output_dir:
            current = {"weights.bin": {"size": 8, "blob_id": "abc"}}
            stored = {"weights.bin": {"size": 8, "blob_id": "abc"}}

            self.assertTrue(
                should_download_file("weights.bin", current, stored, output_dir)
            )

    def test_complete_non_lfs_file_is_not_redownloaded(self):
        with tempfile.TemporaryDirectory() as output_dir:
            rel_path = "config.json"
            with open(os.path.join(output_dir, rel_path), "w", encoding="utf-8") as handle:
                handle.write("abcd")

            current = {rel_path: {"size": 4, "blob_id": "abc"}}
            stored = {rel_path: {"size": 4, "blob_id": "abc"}}

            self.assertFalse(
                should_download_file(rel_path, current, stored, output_dir)
            )


if __name__ == "__main__":
    unittest.main()
