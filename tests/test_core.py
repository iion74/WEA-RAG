import unittest

from rag_eval.wea_eval import compute_coverage, decide_web_needed, score_answer_candidate
from util import exact_match_score, f1_score


class MetricTests(unittest.TestCase):
    def test_hotpot_normalization(self):
        self.assertTrue(exact_match_score("The Eiffel Tower", "Eiffel Tower"))
        self.assertEqual(f1_score("Paris, France", "Paris")[0], 2 / 3)


class WeaGateTests(unittest.TestCase):
    def test_coverage_is_surface_token_overlap(self):
        ratio, matched, missing = compute_coverage(
            "Who directed the motion picture?", "The film was directed by Jane Doe."
        )
        self.assertIn("directed", matched)
        self.assertIn("motion", missing)
        self.assertLess(ratio, 1.0)

    def test_gate_requires_low_coverage_and_missing_terms(self):
        use_web, _ = decide_web_needed(
            question="Who directed the film?",
            rag_context="partial context",
            coverage_ratio=0.70,
            missing_terms={"directed", "film"},
            gate_mode="heuristic",
            coverage_threshold=0.78,
            min_missing_terms=2,
            force_time_sensitive=True,
        )
        self.assertTrue(use_web)

    def test_long_answer_penalty(self):
        short_score, _ = score_answer_candidate("Who?", "Jane Doe", "Jane Doe")
        long_score, _ = score_answer_candidate(
            "Who?", "Jane Doe gave a very long unsupported explanation with many extra words", "Jane Doe"
        )
        self.assertGreater(short_score, long_score)

if __name__ == "__main__":
    unittest.main()
