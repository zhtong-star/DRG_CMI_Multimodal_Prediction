"""Regression tests for manuscript experiment scheduling."""
import contextlib
import io
import unittest
from run_experiments import main

class ReleaseProtocolTests(unittest.TestCase):
    def commands(self, *extra):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["--cohorts", "ms", "apr", "--ablations",
                  "--clinicalbert_dir", "ClinicalBERT",
                  "--pretrained_word_vectors", "embeddings/clinical.bin",
                  "--run_prefix", "test", "--gradient_accumulation_steps", "2",
                  "--dry_run", *extra])
        return output.getvalue().strip().splitlines()

    def test_table_three_defaults_to_ms_only(self):
        commands = self.commands()
        self.assertEqual(len(commands), 13)
        self.assertEqual(sum("--cohort ms " in c for c in commands), 8)
        self.assertEqual(sum("--cohort apr " in c for c in commands), 5)
        self.assertTrue(all("--fold " not in c for c in commands))
        lggcmn = [c for c in commands if "--model LGGCMN " in c]
        self.assertEqual(len(lggcmn), 5)
        self.assertTrue(all("--logit_adjustment 1.0 " in c for c in lggcmn))

    def test_explicit_extra_apr_ablations_remain_available(self):
        commands = self.commands("--ablation_cohorts", "ms", "apr")
        self.assertEqual(len(commands), 16)

if __name__ == "__main__":
    unittest.main()
