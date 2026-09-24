"""Update-count scheduling independent of dataset size and validation cadence."""
import math

def update_lr(index,total,base=5e-5):
    if total<2 or not 0<=index<total:raise ValueError('invalid update index/budget')
    warmup=max(1,min(total-1,math.ceil(.1*total)))
    if index<warmup:return base*(index+1)/warmup
    return base*.5*(1+math.cos(math.pi*(index-warmup+1)/(total-warmup)))

def chunks(total,cadence):
    if total<2 or cadence<1:raise ValueError('invalid budget/cadence')
    return [(start,min(start+cadence,total)) for start in range(0,total,cadence)]

def cycling(loader):
    if len(loader)==0:raise ValueError('empty training loader')
    while True:yield from loader
