import tempfile
import unittest
from pathlib import Path

from structure.analyzer import StructureAnalyzer


class StructureAnalyzerTests(unittest.TestCase):

    def test_structural_active_sites_are_returned(self):
        pdb = """\
ATOM      1  N   SER A   1      11.104  13.207   9.457  1.00 20.00           N
HETATM    2  O1  LIG A 101      11.800  13.200   9.450  1.00 20.00           O
END
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "smoke.pdb"
            path.write_text(pdb, encoding="ascii")

            result = StructureAnalyzer().analyze(str(path))

        self.assertIn("active_sites", result)
        self.assertEqual(result["active_sites"][0]["residue"], "SER")
        self.assertEqual(result["active_sites"][0]["chain"], "A")
        self.assertEqual(result["active_sites"][0]["confidence"], 90)


if __name__ == "__main__":
    unittest.main()
