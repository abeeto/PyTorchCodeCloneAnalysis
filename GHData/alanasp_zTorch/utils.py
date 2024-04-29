import numpy as np


def cap_profiles(profiles):
    for prof in profiles:
        for i in range(len(prof)):
            prof[i] = min(prof[i], 100)
            prof[i] = max(prof[i], 0)
    return profiles


def gen_init_profiles(base_profiles, count_per_base):
    profiles = list()
    for key in base_profiles:
        base = base_profiles[key]
        for i in range(count_per_base):
            profile = np.random.pareto(1.16, size=len(base)) + np.array(base)
            profiles.append(profile)
    profiles = np.array(cap_profiles(profiles))
    return profiles


def group_points(points, group_ids):
    groups = dict()
    for i in range(len(points)):
        gid = group_ids[i]
        if gid not in groups:
            groups[gid] = list()
        groups[gid].append(points[i])
    for gid in groups:
        groups[gid] = np.array(groups[gid])
    return groups


def count_deviations(points, aff_groups, centres, granularity):
    count = 0
    for pid in range(len(points)):
        min_dist = np.linalg.norm(points[pid] - centres[aff_groups[pid]])
        group = aff_groups[pid]
        for cid in range(len(centres)):
            dist = np.linalg.norm(points[pid] - centres[cid])
            if min_dist - dist > np.sqrt(points.shape[1])*granularity:
                min_dist = dist
                group = cid
        if group != aff_groups[pid]:
            count += 1
    return count
