import unittest
from pathlib import Path

import torch
from sacred import Experiment

from data_kits.datasets import data_ingredient, load
from utils.misc import MapConfig


class TestDataLoader(unittest.TestCase):
    def test_shape_dtype_pairs(self):
        ex = Experiment(name="Test", ingredients=[data_ingredient], save_git_info=False, base_dir=Path(__file__).parents[1])

        @ex.command
        def run_(_config):
            cfg = MapConfig(_config)

            # -------------------------------------------------------------------
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
                self.assertIn(c, list(range(6, 21)))
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
            self.assertIn(cls[0], list(range(1, 6)))
            self.assertEqual(sup_rgb.dtype, torch.float32)
            self.assertEqual(sup_msk.dtype, torch.float32)
            self.assertEqual(qry_rgb.dtype, torch.float32)
            self.assertEqual(qry_msk.dtype, torch.int64)
            self.assertEqual(cls.dtype, torch.int64)

            # -------------------------------------------------------------------
            ds.reset_sampler()
            ds.sample_tasks()

            target = [
                [5, ['2010_001367'], ['2009_004324']],
                [1, ['2007_002376'], ['2007_001761']],
                [5, ['2009_002649'], ['2009_001278']],
                [3, ['2009_000991'], ['2009_001314']],
                [1, ['2007_002376'], ['2010_000572']],
            ]

            i = 0
            for (sup_rgb, sup_msk, qry_rgb), qry_msk, cls, name1, name2 in loader:
                self.assertEqual(target[i][0], cls[0])
                self.assertEqual(target[i][1], name1[0])
                self.assertEqual(target[i][2], name2[0])

                i += 1
                if i == 5:
                    break

        ex.run("run_")
