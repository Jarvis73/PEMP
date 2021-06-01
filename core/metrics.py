import numpy as np


class FewShotMetric(object):
    def __init__(self, classes):
        self.classes = classes
        self.stat = np.zeros((self.classes + 1, 3))     # +1 for bg, 3 for tp, fp, fn

    def update(self, pred, ref, cls, verbose=0):
        pred = np.asarray(pred, np.uint8)
        ref = np.asarray(ref, np.uint8)
        for i, ci in enumerate(cls):    # iter on batch ==> [episode_1, episode_2, ...]
            p = pred[i]
            r = ref[i]
            for j, c in enumerate([0, int(ci)]):     # iter on class ==> [bg_cls, fg_cls]
                tp = int((np.logical_and(p == j, r != 255) * np.logical_and(r == j, r != 255)).sum())
                fp = int((np.logical_and(p == j, r != 255) * np.logical_and(r != j, r != 255)).sum())
                fn = int((np.logical_and(p != j, r != 255) * np.logical_and(r == j, r != 255)).sum())
                if verbose:
                    print(tp / (tp + fp + fn))
                self.stat[c, 0] += tp
                self.stat[c, 1] += fp
                self.stat[c, 2] += fn

    def mIoU(self, labels, binary=False):
        if binary:
            stat = np.c_[self.stat[0], self.stat[1:].sum(axis=0)].T     # [2, 3]
        else:
            stat = self.stat[labels]                                    # [N, 3]

        tp, fp, fn = stat.T                                             # [2 or N]
        mIoU_class = tp / (tp + fp + fn)                                # [2 or N]
        mean = mIoU_class.mean()                                        # scalar

        return mIoU_class, mean


class Accumulator(object):
    def __init__(self, **kwargs):
        self.values = kwargs
        self.counter = {k: 0 for k, v in kwargs.items()}
        for k, v in self.values.items():
            if not isinstance(v, (float, int, list)):
                raise TypeError(f"The Accumulator does not support `{type(v)}`. Supported types: [float, int, list]")

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(self.values[k], list):
                self.values[k].append(v)
            else:
                self.values[k] = self.values[k] + v
            self.counter[k] += 1

    def mean(self, key, axis=None):
        if isinstance(key, str):
            if isinstance(self.values[key], list):
                return np.array(self.values[key]).mean(axis)
            else:
                return self.values[key] / self.counter[key]
        else:
            return [self.mean(k, axis) for k in key]

    def std(self, key, axis=None):
        if isinstance(key, str):
            if isinstance(self.values[key], list):
                return np.array(self.values[key]).std(axis)
            else:
                raise RuntimeError("`std` is not supported for (int, float). Use list instead.")
        elif isinstance(key, (list, tuple)):
            return [self.mean(k) for k in key]
        else:
            TypeError(f"`key` must be a str/list/tuple, got {type(key)}")
