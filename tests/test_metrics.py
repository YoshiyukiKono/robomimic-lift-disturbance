import unittest

from lift_tutorial.metrics import HoldMetric


class HoldMetricTests(unittest.TestCase):
    def test_consecutive_hold_not_total_time(self):
        metric = HoldMetric(0.8, 0.05, 1.0, 20)
        for _ in range(19):
            self.assertFalse(metric.update(0.9))
        metric.update(0.8)
        for _ in range(19):
            self.assertFalse(metric.update(0.9))
        self.assertTrue(metric.update(0.9))
        self.assertEqual(metric.time_to_success_s, 2.0)
        self.assertEqual(metric.max_hold_s, 1.0)
        metric.update(0.8)
        self.assertTrue(metric.success)  # ever achieved, not final-state success

    def test_nonfinite_rejected(self):
        metric = HoldMetric(0.8, 0.05, 1.0, 20)
        with self.assertRaises(ValueError):
            metric.update(float("nan"))

    def test_round_hold_up_to_control_interval(self):
        metric = HoldMetric(0.8, 0.05, 0.11, 20)
        self.assertEqual(metric.required_steps, 3)

    def test_tiny_hold_still_needs_a_high_sample(self):
        metric = HoldMetric(0.8, 0.05, 1e-12, 20)
        self.assertFalse(metric.update(0.8))
        self.assertTrue(metric.update(0.9))


if __name__ == "__main__":
    unittest.main()
