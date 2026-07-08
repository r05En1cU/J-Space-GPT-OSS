#!/usr/bin/env python3
"""Math-only smoke checks for the J-space helper routines."""

import torch

from jspace_gpt_oss import coordinate_patch, positive_sparse_pursuit, projection


def main() -> None:
    dictionary = torch.eye(4)
    target = torch.tensor([2.0, 0.0, 1.0, 0.0])
    result, reconstruction, residual = positive_sparse_pursuit(
        target=target,
        dictionary=dictionary,
        token_ids=[10, 11, 12, 13],
        k=2,
        normalize_atoms=False,
    )
    assert result.active_token_ids == [10, 12], result
    assert torch.allclose(reconstruction, target, atol=1e-4), reconstruction
    assert residual.norm().item() < 1e-4, residual

    h = torch.tensor([2.0, 2.0])
    v = torch.tensor([1.0, 0.0])
    assert torch.allclose(projection(h, v), torch.tensor([2.0, 0.0]))

    patched = coordinate_patch(torch.tensor([3.0, 1.0]), torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    assert torch.allclose(patched, torch.tensor([1.0, 3.0]), atol=1e-5), patched
    print("ok")


if __name__ == "__main__":
    main()
