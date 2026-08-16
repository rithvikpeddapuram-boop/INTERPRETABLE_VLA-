import torch
print('torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Running on CPU only (expected, since this is a local AMD-GPU machine).')

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

policy = SmolVLAPolicy.from_pretrained('lerobot/smolvla_libero')
print(policy)

from torch.func import jacrev

def f(x):
    return x.sum(dim=-1)

x = torch.randn(4, 8, requires_grad=True)
J = jacrev(f)(x)
print('Jacobian shape:', J.shape)
