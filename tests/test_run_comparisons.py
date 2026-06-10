import tempfile
import unittest
from pathlib import Path

from src.run_comparisons import run_all_variants


class TestRunComparisons(unittest.TestCase):
    def test_run_all_variants_creates_summary_and_plots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_all_variants(
                num_tasks=20,
                seed=42,
                num_ctrl=3,
                num_agents=5,
                output_dir=tmpdir,
            )
            root = Path(tmpdir)
            self.assertFalse(summary.empty)
            self.assertTrue((root / "summary_metrics.csv").exists())
            self.assertTrue((root / "comparison_metrics.png").exists())
            self.assertIn("current", summary["variant"].tolist())
            self.assertIn("best_quality_static", summary["variant"].tolist())
            self.assertIn("no_judge", summary["variant"].tolist())


if __name__ == "__main__":
    unittest.main()
