"""The scaling seam. Everything about "which coordinates vary and which are pinned
at the equilibrium" lives here, as data, so moving from the 2-D gen-5 slice to an
N-D slice is a change of `active_buses` in a config and nothing else.

The gen-5 slice of Cui and Zhang's Fig. 2 pins every generator at its equilibrium
except generator 5, and varies that bus's (delta, omega). SliceSpec generalizes
that: give it a list of active generator buses (1-indexed, so gen 5 is bus 5) and
it works out the raw-state dimensions that vary, builds the certification box
programmatically from the slice dimension, samples inside it, and embeds slice
coordinates back into the pinned 20-D state. No function here hardcodes 2.

Dimension convention, matching src/dynamics.py and the rest of the port: the raw
state is x = [delta_1..delta_n, omega_1..omega_n], so generator b (1-indexed)
occupies delta-index (b-1) and omega-index (b-1)+n. The certification region is the
offset box her Fig. 2 result uses, [x* + inner, x* + inner + rho] on every active
coordinate, an axis-aligned box sitting a fixed inner offset off the equilibrium so
the equilibrium (offset 0), where the reduced-coordinate Lie derivative is
structurally 0, is never inside it. For slice_dim active coordinates this is a
slice_dim-dimensional box; certify_box already splits on the widest active
dimension and leaves the zero-width pinned dimensions alone, so it certifies an
N-D box with no change.
"""
import numpy as np
import torch


class SliceSpec:
    """A projected slice of the full 2n-D state.

    active_buses: list of 1-indexed generator numbers whose (delta, omega) vary.
                  [5] reproduces the gen-5 2-D slice. [5, 9] would be the 4-D slice
                  varying gens 5 and 9, and so on, with no other code change.
    """

    def __init__(self, system, active_buses):
        self.n = int(system["n"])
        self.active_buses = [int(b) for b in active_buses]
        for b in self.active_buses:
            if not (1 <= b <= self.n):
                raise ValueError(f"active bus {b} out of range 1..{self.n}")
        # raw-state dims that vary: (delta, omega) for each active bus, in bus order
        dims = []
        for b in self.active_buses:
            gi = b - 1
            dims.append(gi)             # delta_b
            dims.append(gi + self.n)    # omega_b
        self.active_dims = dims
        self.slice_dim = len(dims)

    # --- region construction, programmatic in the slice dimension ---
    def box(self, xstar, inner, outer):
        """Offset box [x*+inner, x*+outer] on every active coordinate, pinned
        elsewhere. inner/outer are scalars applied to all active dims (an isotropic
        box, which is what the gen-5 result uses); pass sequences of length
        slice_dim for a per-coordinate box. Returns (xL, xU), each (1, 2n)."""
        xstar = xstar.reshape(1, -1)
        xL = xstar.clone()
        xU = xstar.clone()
        inner = self._as_vec(inner)
        outer = self._as_vec(outer)
        for k, d in enumerate(self.active_dims):
            xL[0, d] = xstar[0, d] + inner[k]
            xU[0, d] = xstar[0, d] + outer[k]
        return xL, xU

    def _as_vec(self, v):
        if np.isscalar(v):
            return [float(v)] * self.slice_dim
        v = list(v)
        assert len(v) == self.slice_dim, f"expected {self.slice_dim} offsets, got {len(v)}"
        return [float(x) for x in v]

    # --- sampling and embedding, so training/attack code never touches raw indices ---
    def sample(self, xstar, box, k, generator=None):
        """k uniform samples inside the box, pinned at x* on inactive dims. (k, 2n)."""
        xstar = xstar.reshape(1, -1)
        xL, xU = box
        x = xstar.repeat(k, 1).clone()
        for d in self.active_dims:
            r = torch.rand(k, generator=generator)
            x[:, d] = xL[0, d] + (xU[0, d] - xL[0, d]) * r
        return x

    def embed(self, xstar, coords):
        """Map slice coordinates (k, slice_dim) into full states (k, 2n), pinning
        every inactive dim at x*. Used by the PGD attack, which optimizes only the
        active coordinates and never leaves the slice."""
        xstar = xstar.reshape(1, -1)
        k = coords.shape[0]
        x = xstar.repeat(k, 1).clone()
        for j, d in enumerate(self.active_dims):
            x[:, d] = coords[:, j]
        return x

    def extract(self, x):
        """Pull the active coordinates (…, slice_dim) out of a full state (…, 2n)."""
        return x[..., self.active_dims]

    def grid(self, xstar, inner, outer, max_side):
        """Tile the box into a list of (xL, xU) subregions whose every active side is
        <= max_side. This is the initial dataset D of the training-time BaB, built
        programmatically for any slice dimension. Returns a list of (1, 2n) pairs."""
        xstar = xstar.reshape(1, -1)
        inner = self._as_vec(inner)
        outer = self._as_vec(outer)
        # per active dim, the split points
        axes = []
        for k in range(self.slice_dim):
            lo, hi = xstar[0, self.active_dims[k]].item() + inner[k], \
                     xstar[0, self.active_dims[k]].item() + outer[k]
            m = max(1, int(np.ceil((hi - lo) / max_side)))
            edges = np.linspace(lo, hi, m + 1)
            axes.append(list(zip(edges[:-1], edges[1:])))
        # cartesian product of per-axis cells
        boxes = []
        for cell in _product(axes):
            xL = xstar.clone(); xU = xstar.clone()
            for k, d in enumerate(self.active_dims):
                xL[0, d] = cell[k][0]; xU[0, d] = cell[k][1]
            boxes.append((xL, xU))
        return boxes

    def as_dict(self):
        return dict(active_buses=self.active_buses, active_dims=self.active_dims,
                    slice_dim=self.slice_dim, n=self.n)


def _product(axes):
    """Cartesian product of a list of lists, iterative so it scales past 2-D."""
    out = [()]
    for ax in axes:
        out = [tuple(list(p) + [c]) for p in out for c in ax]
    return out


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import load_system
    s = load_system()
    for buses in ([5], [5, 9], [1, 5, 9]):
        sp = SliceSpec(s, buses)
        print(f"buses {buses}: slice_dim={sp.slice_dim} active_dims={sp.active_dims}")
    sp = SliceSpec(s, [5])
    xstar = torch.zeros(1, 20)
    box = sp.box(xstar, 0.05, 0.05 + 2.0)
    print("gen-5 box active widths:", [round((box[1][0, d] - box[0][0, d]).item(), 3) for d in sp.active_dims])
    print("grid at max_side 1.0 gives", len(sp.grid(xstar, 0.05, 2.05, 1.0)), "cells")
