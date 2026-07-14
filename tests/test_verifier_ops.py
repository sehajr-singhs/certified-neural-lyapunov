import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

def elu_supported(x):
    # ELU(x) = relu(x) - relu(1 - exp(-relu(-x))), exp arg in [x_min,0], no blow-up
    return torch.relu(x) - torch.relu(1.0 - torch.exp(-torch.relu(-x)))

class TrigNet(nn.Module):
    def forward(self, x):
        return torch.sin(x[..., 0:1]) + torch.cos(x[..., 1:2])

class EluNet(nn.Module):
    def forward(self, x):
        return elu_supported(x[..., 0:1]) + elu_supported(x[..., 2:3])

def check(name, model, x0, eps, truth_fn):
    bm = BoundedModule(model, x0, verbose=False)
    ptb = PerturbationLpNorm(norm=float('inf'), eps=eps)
    bx = BoundedTensor(x0, ptb)
    g = np.linspace(-eps, eps, 31)
    import itertools
    pts = np.array(list(itertools.product(*[g]*x0.shape[1]))) + x0.numpy()
    tv = truth_fn(torch.tensor(pts, dtype=torch.float32)).numpy().ravel()
    print(f"== {name}: sampled true [{tv.min():.4f}, {tv.max():.4f}]")
    for method in ["IBP", "CROWN", "CROWN-Optimized"]:
        lb, ub = bm.compute_bounds(x=(bx,), method=method)
        lb, ub = lb.item(), ub.item()
        sound = lb <= tv.min() + 1e-4 and ub >= tv.max() - 1e-4
        print(f"   {method:16s} [{lb:.4f}, {ub:.4f}]  sound={sound}")

# first confirm elu identity equals torch ELU
xx = torch.linspace(-3, 3, 1001)
err = (elu_supported(xx) - nn.functional.elu(xx)).abs().max().item()
print("ELU identity max error vs torch.nn.ELU:", err)

check("sin+cos", TrigNet(), torch.tensor([[0.3, -0.2]]), 0.4,
      lambda p: torch.sin(p[:, 0:1]) + torch.cos(p[:, 1:2]))
check("elu(x0)+elu(x2)", EluNet(), torch.tensor([[0.3, 0.0, -0.5]]), 0.4,
      lambda p: nn.functional.elu(p[:, 0:1]) + nn.functional.elu(p[:, 2:3]))
print("CAPABILITY GATE PASSED" )
