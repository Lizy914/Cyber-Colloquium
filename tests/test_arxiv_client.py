from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.discussion_app.arxiv_client import build_arxiv_query_candidates, parse_arxiv_feed, render_bibtex_entry, save_arxiv_metadata


SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <updated>2024-01-10T12:00:00Z</updated>
    <published>2024-01-08T12:00:00Z</published>
    <title>  A Test Paper for Cyber Colloquium  </title>
    <summary>  This is a short abstract.  </summary>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Lee</name></author>
    <category term="cs.AI" />
    <link title="pdf" href="https://arxiv.org/pdf/2401.12345v2.pdf" />
  </entry>
</feed>
"""


class ArxivClientTests(unittest.TestCase):
    def test_build_query_candidates_maps_common_chinese_research_terms(self) -> None:
        candidates = build_arxiv_query_candidates("研究Mamba模型在遥感图像变化检测中的应用，重点研究其优势和局限性")
        merged = " | ".join(candidates).lower()

        self.assertTrue(candidates)
        self.assertIn("mamba", merged)
        self.assertIn("remote sensing", merged)
        self.assertIn("change detection", merged)

    def test_parse_feed_and_render_bibtex(self) -> None:
        papers = parse_arxiv_feed(SAMPLE_FEED)

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].paper_id, "2401.12345v2")
        self.assertEqual(papers[0].authors, ("Alice Smith", "Bob Lee"))
        key, bibtex = render_bibtex_entry(papers[0])

        self.assertEqual(key, "smith2024a")
        self.assertIn("arXiv:2401.12345v2", bibtex)
        self.assertIn("Alice Smith and Bob Lee", bibtex)

    def test_save_metadata_writes_json(self) -> None:
        papers = parse_arxiv_feed(SAMPLE_FEED)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "arxiv_metadata.json"
            save_arxiv_metadata(papers, path)
            written = path.read_text(encoding="utf-8")

        self.assertIn("2401.12345v2", written)
        self.assertIn("A Test Paper for Cyber Colloquium", written)


if __name__ == "__main__":
    unittest.main()
