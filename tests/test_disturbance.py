import csv
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from lift_tutorial.disturbance import FORCE_FIELDS, force_for_step, pulse_steps, validate_disturbance


class DisturbanceTests(unittest.TestCase):
    def test_pulse_boundaries_and_impulse(self):
        p = dict(start_s=.5, duration_s=.15, force_n=[2, 0, 0])
        forces = [force_for_step(p, i, 20, 200) for i in range(1, 201)]
        self.assertEqual([i+1 for i, f in enumerate(forces) if any(f)], [11, 12, 13])
        self.assertAlmostEqual(sum(f[0]/20 for f in forces), .3)
        self.assertEqual(force_for_step(None, 11, 20, 200), [0, 0, 0])

    def test_bad_schedule_and_nonhorizontal_force(self):
        for p in [dict(start_s=.51, duration_s=.15, force_n=[1,0,0]),
                  dict(start_s=9.9, duration_s=.15, force_n=[1,0,0]),
                  dict(start_s=.5, duration_s=.15, force_n=[1,0,1]),
                  dict(start_s=.5, duration_s=.15, force_n=[float('nan'),0,0])]:
            with self.assertRaises(ValueError):
                pulse_steps(p, 20, 200)

    def test_force_targets_only_cube_and_can_be_cleared(self):
        import numpy as np
        from lift_tutorial.backend import LiftBackend
        applied = np.zeros((3, 6))
        applied[0, :] = 7
        backend = LiftBackend.__new__(LiftBackend)
        backend.env = SimpleNamespace(cube_body_id=1, sim=SimpleNamespace(data=SimpleNamespace(xfrc_applied=applied)))
        backend.set_cube_force([2, 0, 0])
        np.testing.assert_array_equal(applied[1], [2,0,0,0,0,0])
        np.testing.assert_array_equal(applied[0], [7]*6)
        backend.set_cube_force([0,0,0])
        self.assertFalse(applied[1].any())

    def test_verifier_detects_missing_and_incorrect_force_record(self):
        from test_report import recording_fixture
        from lift_tutorial.report import load_inputs
        from lift_tutorial.verify import replay_trajectory
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            meta, _, _ = recording_fixture(root)
            meta['config']['disturbance'] = dict(start_s=.5, duration_s=.15, force_n=[1,0,0])
            (root/'run.json').write_text(json.dumps(meta))
            meta, rows = load_inputs(root)
            path = root/'episodes/centre_trial00.csv'
            with path.open() as f:
                reader = csv.DictReader(f)
                fields, samples = reader.fieldnames, list(reader)
            with self.assertRaises(ValueError):
                replay_trajectory(root, meta, rows[0])
            def save():
                with path.open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=fields+FORCE_FIELDS)
                    w.writeheader(); w.writerows(samples)
            for i, sample in enumerate(samples, 1):
                sample.update(dict(zip(FORCE_FIELDS, force_for_step(meta['config']['disturbance'], i, 20, 100))))
            save()
            replay_trajectory(root, meta, rows[0])
            samples[10]['push_fx_n'] = 0
            save()
            with self.assertRaises(ValueError):
                replay_trajectory(root, meta, rows[0])


if __name__ == '__main__':
    unittest.main()
