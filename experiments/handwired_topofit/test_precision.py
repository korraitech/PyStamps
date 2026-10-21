import json
from pathlib import Path
import unittest
import numpy as np
from experiments.handwired_topofit.cpu import source_search, reference_topofit, refine


class PrecisionWitnessTests(unittest.TestCase):
    def test_saved_actual_calibration_near_tie_changes_refinement(self):
        witness=json.loads(Path(__file__).with_name('near-tie.json').read_text())
        phase=(np.array(witness['real'])+1j*np.array(witness['imag']))[None,:]
        baseline=np.array(witness['baseline'])
        _,best=source_search(phase,baseline,witness['wraps'])
        self.assertEqual(best[0],witness['source_trial'])
        original=reference_topofit(phase[0],baseline,witness['wraps'])
        altered=refine(phase,baseline,witness['wraps'],np.array([witness['handwired_trial']]))
        self.assertGreater(abs(altered['C'][0]-original[1]),0.03)


if __name__=='__main__':
    unittest.main()
