import torch


def space_neighbors(k, hist_t, hist_id, hist_pos, hist_mask, id_dist, id_adj):
    all_ts = hist_t.unique()
    batches, samples = hist_t.shape
    device = hist_t.device
    final_valid = torch.zeros((batches, samples, k), device=device, dtype=torch.bool)
    vec_size = hist_pos.shape[-1] + 1
    final_vec = torch.zeros((batches, samples, k, vec_size), device=device, dtype=torch.float)
    final_idx = torch.zeros((batches, samples, k), device=device, dtype=torch.long)
    for this_t in all_ts:
        # Get indices for all samples from this timestep
        outer_idxs = (hist_t == this_t).nonzero(as_tuple=True)
        bid = outer_idxs[0]  # Batch of each sample (flattened away)
        # Which samples are from the same batch? No cross-batch interactions
        # allowed
        cross_bids = torch.broadcast_tensors(*[bid.unsqueeze(x) for x in (0, 1)])
        valid_bid = (cross_bids[0] == cross_bids[1])
        # Use indices to get ID, position of each sample
        lcl_ids = hist_id[outer_idxs]
        lcl_pos = hist_pos[outer_idxs]
        lcl_mask = hist_mask[outer_idxs]
        # Create product tensor over sample IDs
        cross_ids = torch.broadcast_tensors(*[lcl_ids.unsqueeze(x) for x in (0, 1)])
        same_id = (cross_ids[0] == cross_ids[1])  # Samples can't be neighbors with themselves
        combined_idx = [bid, *cross_ids]  # Combine batch + crossed IDs to create 'lookup' index tensor
        lcl_dist = id_dist[combined_idx]  # Use index to lookup distance between each ID pair
        lcl_valid = id_adj[combined_idx] * valid_bid * ~same_id * lcl_mask.unsqueeze(0) * lcl_mask.unsqueeze(1)  # Lookup adj. info + same batch? + different ID? to determine each pair's validity
        big_dist = lcl_dist.max()*1e2
        mod_lcl_dist = lcl_dist + (big_dist * ~lcl_valid)  # Add big val to each dist so invalid ones get sorted to the end
        closest_dist, closest_idxs = mod_lcl_dist.topk(k, largest=False)  # Get valid samples w/ smallest distance
        closest_pos = lcl_pos[closest_idxs] - lcl_pos.unsqueeze(1)
        closest_vec = torch.cat((closest_dist.unsqueeze(-1), closest_pos), dim=-1)
        final_valid[outer_idxs] = torch.gather(lcl_valid, 1, closest_idxs)
        final_idx[outer_idxs] = outer_idxs[1][closest_idxs]
        final_vec[outer_idxs] = closest_vec
    return final_idx, final_vec, final_valid


def time_neighbors(k, num_groups, hist_t, hist_id, hist_mask):
    all_ids = hist_id.unique()
    batches, samples = hist_id.shape
    device = hist_id.device
    final_valid = torch.zeros((batches, samples, k), device=device, dtype=torch.bool)
    final_vec = torch.zeros((batches, samples, k), device=device, dtype=torch.float)
    final_idx = torch.zeros((batches, samples, k), device=device, dtype=torch.long)
    group_size = len(all_ids)//num_groups
    for group_start in range(0, len(all_ids), group_size):
        id_group = all_ids[group_start:group_start+group_size]
        in_group = (hist_id.unsqueeze(-1) == id_group).any(-1)
        outer_idxs = in_group.nonzero(as_tuple=True)
        bid = outer_idxs[0]  # Batch of each sample (flattened away)
        lcl_ids = hist_id[outer_idxs]
        lcl_ts = hist_t[outer_idxs]
        lcl_mask = hist_mask[outer_idxs]
        same_id = (lcl_ids.unsqueeze(0) == lcl_ids.unsqueeze(1))
        same_batch = (bid.unsqueeze(0) == bid.unsqueeze(1))
        same = same_id * same_batch
        lcl_valid = same * lcl_mask.unsqueeze(0) * lcl_mask.unsqueeze(1)
        time_diff = (lcl_ts.unsqueeze(0) - lcl_ts.unsqueeze(1))
        big_time = time_diff.max()*1e2
        mod_time_diff = time_diff.abs() + (big_time * ~same)
        closest_td, closest_idxs = mod_time_diff.topk(k, largest=False)
        final_valid[outer_idxs] = torch.gather(lcl_valid, 1, closest_idxs)
        final_idx[outer_idxs] = outer_idxs[1][closest_idxs]
        final_vec[outer_idxs] = closest_td
    final_vec = final_vec.unsqueeze(-1)
    return final_idx, final_vec, final_valid


def query(time_enc, key_mask, point_mask, key_ids, point_ids, keys, points):
    key_ts = keys.unsqueeze(2)
    key_ids = key_ids.unsqueeze(2)
    point_ts = points.unsqueeze(1)
    point_ids = point_ids.unsqueeze(1)
    # Calculate spatial distances
    dist_vec = key_ts - point_ts
    dist = dist_vec.abs()
    dist_vec = dist_vec.unsqueeze(-1)
    # Determine which distances matter
    same_id = (key_ids == point_ids)
    # Combine the above
    valid = same_id * point_mask.unsqueeze(1) * key_mask.unsqueeze(2)
    dist_vec = dist_vec.float()
    dist = dist.float()
    return valid, time_enc(dist_vec), dist
