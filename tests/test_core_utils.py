import tempfile
import unittest
from pathlib import Path

import numpy as np

from infer_wingsnet_full_volume import compute_starts, pad_volume_to_patch, unpad_volume
from smooth_airway_path import project_to_lumen, zyx_to_xyz


class CoreUtilityTests(unittest.TestCase):
    def test_compute_starts_covers_volume_end(self):
        self.assertEqual(compute_starts(64, 128, 32), [0])
        self.assertEqual(compute_starts(256, 128, 64), [0, 64, 128])
        self.assertEqual(compute_starts(250, 128, 64), [0, 64, 122])

    def test_pad_and_unpad_round_trip(self):
        volume = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
        padded, pads = pad_volume_to_patch(volume, (8, 6, 5))
        self.assertEqual(padded.shape, (8, 6, 5))
        restored = unpad_volume(padded, pads)
        np.testing.assert_array_equal(restored, volume)

    def test_zyx_to_xyz_uses_spacing_and_axis_order(self):
        coords_zyx = np.asarray([[2, 3, 4], [5, 6, 7]], dtype=np.float32)
        converted = zyx_to_xyz(coords_zyx, spacing_zyx=(10, 20, 30))
        expected = np.asarray([[120, 60, 20], [210, 120, 50]], dtype=np.float64)
        np.testing.assert_allclose(converted, expected)

    def test_project_to_lumen_moves_nearby_point_inside_mask(self):
        mask = np.zeros((8, 8, 8), dtype=bool)
        mask[4, 4, 4] = True
        coords = np.asarray([[4.0, 4.0, 6.0], [4.0, 4.0, 4.0]], dtype=np.float64)
        projected, changed, inside_ratio = project_to_lumen(coords, mask, max_distance=2.1)
        self.assertEqual(changed, 1)
        self.assertEqual(inside_ratio, 1.0)
        np.testing.assert_array_equal(np.rint(projected).astype(int), [[4, 4, 4], [4, 4, 4]])


class CheckpointStrictnessTests(unittest.TestCase):
    def test_torch_strict_checkpoint_rejects_missing_keys(self):
        torch = __import__("torch")
        from infer_wingsnet_full_volume import load_checkpoint

        model = torch.nn.Conv3d(1, 2, kernel_size=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "bad_checkpoint.pth"
            torch.save({"weight": torch.zeros_like(model.weight)}, checkpoint_path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(model, checkpoint_path, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
