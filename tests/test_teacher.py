from __future__ import annotations

import unittest

import numpy as np

from robustsep_pkg.targets.teacher import calibrated_cmykogv_lab


class TeacherProxyTests(unittest.TestCase):
    def test_calibrated_teacher_hits_anchor_exactly(self) -> None:
        anchor = np.zeros((16, 16, 7), dtype=np.float32)
        anchor[..., 0] = 0.2
        lab = np.zeros((16, 16, 3), dtype=np.float32)
        lab[..., 0] = 52.0
        lab[..., 1] = 3.0
        lab[..., 2] = -4.0

        rendered = calibrated_cmykogv_lab(anchor, anchor_cmykogv=anchor, anchor_lab=lab)

        np.testing.assert_allclose(rendered, lab, atol=1e-6)

    def test_calibrated_teacher_changes_with_ink_delta(self) -> None:
        anchor = np.zeros((16, 16, 7), dtype=np.float32)
        lab = np.zeros((16, 16, 3), dtype=np.float32)
        lab[..., 0] = 50.0
        changed = anchor.copy()
        changed[..., 4] = 0.1

        rendered = calibrated_cmykogv_lab(changed, anchor_cmykogv=anchor, anchor_lab=lab)

        self.assertGreater(float(np.abs(rendered - lab).mean()), 0.0)


if __name__ == "__main__":
    unittest.main()
