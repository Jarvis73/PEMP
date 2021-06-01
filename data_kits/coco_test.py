import unittest
from pathlib import Path

import torch
from sacred import Experiment

from data_kits.datasets import load, data_ingredient
from utils.misc import MapConfig


class TestDataLoader(unittest.TestCase):
    def test_shape_dtype_pairs(self):
        ex = Experiment(name="Test", ingredients=[data_ingredient], save_git_info=False, base_dir=Path(__file__).parents[1])

        @ex.command
        def run_(_config):
            cfg = MapConfig(_config)
            ds, loader, number = load(cfg.data, "train", split=0, shot=1, query=1)
            ds.reset_sampler()
            ds.sample_tasks()

            (sup_rgb, sup_msk, qry_rgb), qry_msk, cls = next(iter(loader))
            self.assertEqual(list(sup_rgb.shape), [4, 1, 3, 401, 401])
            self.assertEqual(list(sup_msk.shape), [4, 1, 2, 401, 401])
            self.assertEqual(list(qry_rgb.shape), [4, 1, 3, 401, 401])
            self.assertEqual(list(qry_msk.shape), [4, 1, 401, 401])
            self.assertEqual(list(cls.shape), [4])
            for c in cls:
                self.assertIn(c, list(range(21, 81)))
            self.assertEqual(sup_rgb.dtype, torch.float32)
            self.assertEqual(sup_msk.dtype, torch.float32)
            self.assertEqual(qry_rgb.dtype, torch.float32)
            self.assertEqual(qry_msk.dtype, torch.int64)
            self.assertEqual(cls.dtype, torch.int64)

            # -------------------------------------------------------------------
            ds, loader, number = load(cfg.data, "test", split=0, shot=1, query=1, ret_name=True)
            ds.reset_sampler()
            ds.sample_tasks()

            (sup_rgb, sup_msk, qry_rgb), qry_msk, cls, _, _ = next(iter(loader))
            self.assertEqual(list(sup_rgb.shape), [1, 1, 3, 401, 401])
            self.assertEqual(list(sup_msk.shape), [1, 1, 2, 401, 401])
            self.assertEqual(list(qry_rgb.shape), [1, 1, 3, 401, 401])
            self.assertEqual(list(cls.shape), [1])
            self.assertIn(cls[0], list(range(1, 21)))
            self.assertEqual(sup_rgb.dtype, torch.float32)
            self.assertEqual(sup_msk.dtype, torch.float32)
            self.assertEqual(qry_rgb.dtype, torch.float32)
            self.assertEqual(qry_msk.dtype, torch.int64)
            self.assertEqual(cls.dtype, torch.int64)

            # -------------------------------------------------------------------
            ds.reset_sampler()
            ds.sample_tasks()

            target = [
                [19, 69914,  581501],
                [6,  35594, 53345],
                [11, 187348, 143445],
                [6,  457217, 315352],
                [12, 177489, 85803],
            ]

            i = 0
            for (sup_rgb, sup_msk, qry_rgb), qry_msk, cls, name1, name2 in loader:
                self.assertEqual(target[i][0], cls[0].item())
                self.assertEqual(target[i][1], name1[0].item())
                self.assertEqual(target[i][2], name2[0].item())

                i += 1
                if i == 5:
                    break

        ex.run("run_", config_updates={"data.dataset": "COCO"})
