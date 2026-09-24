"""Paired reproducible RGB perturbations for observation-only stress tests."""
import numpy as np

def corrupt_rgb(pixels, *, seed, slot, sigma):
    pixels=np.asarray(pixels)
    if pixels.dtype!=np.uint8 or pixels.shape!=(224,224,3):raise ValueError('Expected uint8 RGB224 image')
    if sigma not in [0,8,24] or seed<0 or slot not in range(4):raise ValueError('Unsupported fixed stress condition')
    if sigma==0:return pixels.copy()
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([1112001,int(seed),int(slot)])))
    noise=rng.standard_normal(pixels.shape)
    return np.rint(np.clip(pixels.astype(np.float64)+sigma*noise,0,255)).astype(np.uint8)
