import torch
import numpy as np
import random
from kornia import create_meshgrid
from lpm.error_function import ssim, psnr, mse


def finite_cone_formulation(top_point, base_center, radius):   
    height = torch.norm(top_point - base_center)
    slant_height = torch.sqrt(height**2 + radius**2)
    half_angle = torch.atan(radius / height)
    half_angle_degrees = torch.cos(half_angle)
    direction = base_center-top_point
    direction = direction / torch.norm(direction)
    
    return direction, height, half_angle_degrees

def points_in_finite_cone(points, apex, direction, angle_cosine, height):
    """Determine if points are within a finite cone defined by apex, direction, angle cosine, and height."""
    direction_normalized = direction / torch.norm(direction)
    vectors = points - apex
    vector_lengths = torch.norm(vectors, dim=1)
    vector_norms = vectors / vector_lengths.unsqueeze(1)
    dot_products = torch.sum(vector_norms * direction_normalized, dim=1)
    mask_angle = dot_products >= angle_cosine
    mask_height = vector_lengths <= height
    return mask_angle & mask_height

def get_points_in_cones(current_cam_pos, zones3d, gs_xyz, gs_opacity):
    in_front_zones3d_mask = torch.zeros_like(torch.tensor(gs_xyz.shape[0])).cuda()
    inside_zones3d_mask = torch.zeros_like(torch.tensor(gs_xyz.shape[0])).cuda()
    for region in zones3d:
        in_front_single_region_mask, inside_single_region_mask = get_points_in_single_cone(current_cam_pos.cuda(), region.cuda(), gs_xyz.cuda(), gs_opacity)
        in_front_zones3d_mask = torch.logical_or(in_front_zones3d_mask, in_front_single_region_mask)
        inside_zones3d_mask = torch.logical_or(inside_zones3d_mask, inside_single_region_mask)
        
    return in_front_zones3d_mask, inside_zones3d_mask

def get_points_in_single_cone(current_cam_pos, region, gs_xyz, gs_opacity):
    base_center = torch.tensor([(region[0]+region[3])/2, (region[1]+region[4])/2, (region[2]+region[5])/2]).cuda()
    radius = torch.min(torch.cat([torch.abs(torch.tensor([region[0]-region[3]])),torch.abs(torch.tensor([region[1]-region[4]])), torch.abs(torch.tensor([region[2]-region[5]]))])) / 2 
    distance_points2base = torch.sqrt((gs_xyz[:,0]-base_center[0])**2+(gs_xyz[:,1]-base_center[1])**2+(gs_xyz[:,2]-base_center[2])**2).cuda()
    inside_mask = torch.where(distance_points2base <= radius, True, False)
    gs_opacity_means = torch.mean(gs_opacity[inside_mask])
    gs_opacity = gs_opacity.squeeze()
    direction, height, half_angle_degrees = finite_cone_formulation(current_cam_pos, base_center, radius)   
    in_front_mask = points_in_finite_cone(gs_xyz, current_cam_pos, direction, half_angle_degrees, height)
    d = height - radius 
    if d > 0:
        distance_points2cam = torch.sqrt((gs_xyz[:,0]-current_cam_pos[0])**2+(gs_xyz[:,1]-current_cam_pos[1])**2+(gs_xyz[:,2]-current_cam_pos[2])**2).cuda()
        points_mask_less_distance = torch.where(distance_points2cam <= d, True, False)
        in_front_mask = torch.logical_and(in_front_mask, points_mask_less_distance)

    return in_front_mask, inside_mask
    


def get_rays_intersection(ray_group_a, ray_group_b):
    ray_group_a=ray_group_a.cuda()
    ray_group_b=ray_group_b.cuda()
    start_a = ray_group_a[:, :3]
    dir_a = ray_group_a[:, 3:]
    start_b = ray_group_b[:, :3]
    dir_b = ray_group_b[:, 3:]
    cross = torch.cross(dir_a, dir_b, dim=1)

    t_a = torch.sum(torch.cross(start_b - start_a, dir_b, dim=1) * cross, dim=1) / torch.sum(cross * cross, dim=1)
    t_b = torch.sum(torch.cross(start_a - start_b, dir_a, dim=1) * cross, dim=1) / torch.sum(cross * cross, dim=1)

    intersection_points_a = start_a + t_a.view(-1, 1) * dir_a
    intersection_points_b = start_b + t_b.view(-1, 1) * dir_b*(-1)
    intersection_points = torch.cat((intersection_points_a, intersection_points_b), dim=0)
    return intersection_points #unique_intersection_points

def region2zone_3d(current_view_cam, referred_view_cam, r0, r1):
    # Ensure rays are on CUDA and rearranged
    current_view_rays = current_view_cam.rays.squeeze(0).permute(1, 2, 0).cuda()
    referred_view_rays = referred_view_cam.rays.squeeze(0).permute(1, 2, 0).cuda()

    # Efficiently gather rays along the perimeter of the region
    def gather_rays(rays, region):
        top = rays[region[1], region[0]:region[2]]
        bottom = rays[region[3]-1, region[0]:region[2]]
        left = rays[region[1]:region[3], region[0]]
        right = rays[region[1]:region[3], region[2]-1]
        return torch.cat([top, bottom, left, right], dim=0)

    current_view_rays_in_region = gather_rays(current_view_rays, r0)
    referred_view_rays_in_region = gather_rays(referred_view_rays, r1)

    # Handle cases where no rays were selected
    if not current_view_rays_in_region.shape[0] or not referred_view_rays_in_region.shape[0]:
        return None

    # Equalize the number of rays in each set if they differ
    diff = current_view_rays_in_region.shape[0] - referred_view_rays_in_region.shape[0]
    if diff > 0:
        referred_view_rays_in_region = torch.cat([referred_view_rays_in_region, referred_view_rays_in_region[0].repeat(diff, 1)])
    elif diff < 0:
        current_view_rays_in_region = torch.cat([current_view_rays_in_region, current_view_rays_in_region[0].repeat(-diff, 1)])

    # Calculate the 3D intersection points of the rays
    intersection_points = get_rays_intersection(current_view_rays_in_region, referred_view_rays_in_region)
    
    # Determine the bounding zone in 3D space
    mins, _ = torch.min(intersection_points, dim=0)
    maxs, _ = torch.max(intersection_points, dim=0)
    
    return torch.cat([mins, maxs])

def zones3d_projection(current_view_cam, referred_view_cam, current_view_regions, referred_view_regions):
    """Calculates intersections for lists of regions from two different cameras."""
    zones_3d = []
    for r0, r1 in zip(current_view_regions, referred_view_regions):
        # Calculate the area of each region
        r0_area = (r0[2] - r0[0]) * (r0[3] - r0[1])
        r1_area = (r1[2] - r1[0]) * (r1[3] - r1[1])
        
        # Only proceed if both regions have non-zero area
        if r0_area > 0 and r1_area > 0:
            single_zone = region2zone_3d(current_view_cam, referred_view_cam, r0, r1)
            if single_zone is not None:
                zones_3d.append(single_zone)
    return zones_3d






