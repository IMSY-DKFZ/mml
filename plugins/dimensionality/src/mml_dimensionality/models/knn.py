# LICENSE HEADER MANAGED BY add-license-header
#
# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: MIT
#

"""
The method and partially also the code is reused from:
"The Intrinsic Dimensionaity of Images and Its Impact On Learning"
by Phillip Pope, Chen Zhu, Ahmed Abdelkader, Micah Goldblum, Tom Goldstein (ICLR 2021, spotlight)
code: https://github.com/ppope/dimensions
"""

import torch


class KNNComputer(torch.nn.Module):
    """
    Using this hack for data parallel
    without checking for the sample itself
    """

    def __init__(self, sample_num: int, k: int = 1, cosine_dist=False):
        super(KNNComputer, self).__init__()

        self.K = k
        self.cosine_dist = cosine_dist
        self.register_buffer("num_computed", torch.zeros([]))

        if k == 1:
            self.register_buffer("min_dists", torch.full((sample_num,), float("inf")))
            self.register_buffer("nn_indices", torch.full((sample_num,), 0, dtype=torch.int64))
        else:
            self.register_buffer("min_dists", torch.full((sample_num, k), float("inf")))
            self.register_buffer("nn_indices", torch.full((sample_num, k), 0, dtype=torch.int64))

    def forward(self, x, x_idx_start, y, y_idx_start):
        # update the min dist for existing examples...
        x_bsize, y_bsize = x.size(0), y.size(0)
        x = x.view(x_bsize, -1)
        y = y.view(y_bsize, -1)
        if self.cosine_dist:
            x = x / x.norm(dim=1, keepdim=True)
            y = y / y.norm(dim=1, keepdim=True)
            dist = x.mm(y.t())

        else:
            # dist = torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=2)
            dist = torch.cdist(x, y, p=2, compute_mode="donot_use_mm_for_euclid_dist")

        if self.K == 1:
            new_min_dist, nn_idxes = torch.min(dist, dim=1)

            self.min_dists[x_idx_start : x_idx_start + x_bsize] = torch.min(
                new_min_dist, self.min_dists[x_idx_start : x_idx_start + x_bsize]
            )
            self.nn_indices[x_idx_start : x_idx_start + x_bsize] = nn_idxes + y_idx_start
        else:
            comp = torch.cat([dist, self.min_dists[x_idx_start : x_idx_start + x_bsize]], dim=1)
            # updated_min_dist, nn_idxes = torch.topk(comp, self.K, dim=1, largest=False)
            # check for repeated images
            sorted_dists, sorted_idxes = torch.sort(comp, dim=1, descending=False)
            updated_dist_list, nn_idx_list = [], []
            for row in range(sorted_dists.shape[0]):
                sidx = 1
                while sidx < sorted_dists.shape[1]:
                    if sorted_dists[row, sidx] == 0:
                        sidx += 1
                    else:
                        break
                updated_dist_list.append(sorted_dists[row, sidx - 1 : sidx - 1 + self.K])
                nn_idx_list.append(sorted_idxes[row, sidx - 1 : sidx - 1 + self.K])
            updated_min_dist = torch.stack(updated_dist_list)
            nn_idxes = torch.stack(nn_idx_list)

            self.min_dists[x_idx_start : x_idx_start + x_bsize] = updated_min_dist

            sample_idxes = (nn_idxes < y_bsize).int() * (nn_idxes + y_idx_start) + (
                nn_idxes >= y_bsize
            ).int() * self.nn_indices[x_idx_start : x_idx_start + x_bsize]
            self.nn_indices[x_idx_start : x_idx_start + x_bsize] = sample_idxes

    def get_mean_nn_dist(self, sidx, eidx):
        if self.K == 1:
            return torch.mean(self.min_dists[sidx:eidx])
