"""Shared four-image, three-action-block interface for frozen-perception fitting."""
import numpy as np


def starts(frame_count, action_count):
    if frame_count<1 or action_count<0:raise ValueError('invalid trajectory length')
    # t+15 must exist; actions t,...,t+14 are consumed, never t+15.
    return list(range(0,min(frame_count-15,action_count-14),5))


def dense_window(pixels,states,actions,start):
    if len(pixels)!=len(states) or start not in starts(len(pixels),len(actions)):
        raise ValueError('incomplete or misaligned window')
    frame_indices=start+np.arange(4)*5
    return (np.asarray(pixels[frame_indices]).copy(),
            np.asarray(actions[start:start+15]).reshape(3,10).copy(),
            np.asarray(states[frame_indices]).copy())


def recorded_window(pixels_stride5,states,actions,start):
    if len(pixels_stride5)!=(len(states)-1)//5+1 or start not in starts(len(states),len(actions)):
        raise ValueError('incomplete or misaligned recorded trajectory')
    frames=np.asarray(pixels_stride5[start//5:start//5+4]).copy()
    if len(frames)!=4:raise ValueError('missing endpoint image')
    return frames,np.asarray(actions[start:start+15]).reshape(3,10).copy(),np.asarray(states[start+np.arange(4)*5]).copy()
