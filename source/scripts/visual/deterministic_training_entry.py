"""Explicit deterministic engineering launcher; leaves target source untouched."""
import os,runpy,sys
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
script=sys.argv[1];sys.argv=sys.argv[1:]
print('DETERMINISTIC_ENGINEERING_ENTRY: deterministic algorithms, math SDPA, CUBLAS :4096:8',flush=True)
runpy.run_path(script,run_name='__main__')
