"""PushT trajectory endpoints aligned with the 10+25-action planner."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class PoseReadoutClips(Dataset):
    def __init__(self, folder):
        folder = Path(folder)
        self.arrays = {k: np.load(folder / (k+'.npy'), mmap_mode='r')
                       for k in ('pixels', 'action', 'state')}
        with np.load(folder/'episodes.npz') as episodes:
            offsets = episodes['offsets'].copy()
            lengths = episodes['lengths'].copy()
            ids = episodes['source_episode_ids'].copy()
        if (offsets.ndim != 1 or lengths.shape != offsets.shape or ids.shape != offsets.shape
                or not len(ids) or len(np.unique(ids)) != len(ids)
                or any(v.dtype.kind not in 'iu' for v in (offsets,lengths,ids))
                or np.any(lengths <= 0)):
            raise ValueError('Invalid episode metadata')
        if not np.array_equal(offsets, np.r_[0, np.cumsum(lengths)[:-1]]):
            raise ValueError('Episodes must partition contiguous arrays')
        total = int(lengths.sum())
        if (any(len(v) != total for v in self.arrays.values())
                or self.arrays['action'].shape != (total,2)
                or self.arrays['state'].ndim != 2 or self.arrays['state'].shape[1] < 5
                or self.arrays['pixels'].ndim != 4 or self.arrays['pixels'].shape[-1] != 3):
            raise ValueError('Inconsistent trajectory arrays')
        # Frame t+35 must exist; action t+35 is never included.
        rows = [(int(ep), int(start), int(offset+start))
                for ep, offset, length in zip(ids, offsets, lengths)
                for start in range(0, int(length)-35, 5)]
        self.windows = np.asarray(rows, dtype=np.int64).reshape(-1,3)
        if not len(self.windows):
            raise ValueError('No complete 35-action windows')

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode, step, start = self.windows[index]
        pixels = np.array(self.arrays['pixels'][start+np.array([0,5,10,35])])
        actions = np.array(self.arrays['action'][start:start+35], dtype=np.float32)
        state = np.array(self.arrays['state'][start+35], dtype=np.float64)
        target = np.r_[state[:4], np.sin(state[4]), np.cos(state[4])]
        if not np.isfinite(actions).all() or not np.isfinite(target).all():
            raise ValueError('Nonfinite training window')
        return (torch.from_numpy(pixels).permute(0,3,1,2),
                torch.from_numpy(actions).reshape(7,10), torch.from_numpy(target),
                torch.tensor([episode,step]))
