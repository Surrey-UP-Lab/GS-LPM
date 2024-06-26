import random
import torch
import torchvision
import cv2
import numpy as np 
from random import randint
from torch import nn
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation

from LightGlue.lightglue import LightGlue, SuperPoint,DISK, SIFT

from lpm.zones_projection import get_points_in_cones, zones3d_projection
from lpm.utils import get_errormap, set_rays, get_paired_views
from lpm.region_matching import get_paired_regions

class LPM:
    def __init__(self, scene, gaussian_model, extractor="SuperPoint", matcher="LightGlue", angle=45):        
        self.extractor = self.initialize_extractor(extractor)
        self.matcher = self.initialize_matcher(matcher)
        self.scene = scene
        self.cams = scene.getTrainCameras()
        self.cv2_macher = False
        self.gaussians = gaussian_model
        self.angle = angle
        self.extent = self.scene.cameras_extent
        self.set_up()

    def initialize_extractor(self, extractor):
        """Initialize the feature extractor based on the given type."""
        extractors = {
            "SuperPoint": SuperPoint(max_num_keypoints=None),
            "DISK": DISK(max_num_keypoints=None),
            "SIFT": SIFT(max_num_keypoints=None) 
        }
        if extractor not in extractors:
            raise ValueError("Could not recognize extractor type!")
        return extractors[extractor].cuda()
    
    def initialize_matcher(self, matcher):
        """Initialize the matcher based on the given type."""
        matchers = {
            "LightGlue": LightGlue(features='superpoint', depth_confidence=0.9, width_confidence=0.95).cuda()
        }
        if matcher not in matchers:
            raise ValueError("Could not recognize matcher type!")
        return matchers[matcher]
    
    def set_up(self):
        """Setup the initial configurations for LPM operations."""
        set_rays(self.scene)
        self.paired_views = get_paired_views(self.scene, self.extractor, self.matcher, self.angle, self.cv2_macher)
        del self.extractor
        del self.matcher
        torch.cuda.empty_cache()
        
        
    def find_neighbor_cam(self, viewpoint_cam):
        # Find the index of the provided camera in the list of cameras
        camera_index = self.cams.index(viewpoint_cam)
        
        # Ensure there are paired views available for the indexed camera
        if len(self.paired_views[camera_index]) > 0:
            # Randomly select an index from the list of paired views for the camera
            random_index = randint(0, len(self.paired_views[camera_index]) - 1)
        else:
            # Handle the case where there are no paired views
            random_index = -1  

        return camera_index, random_index
    
    def points_addition(self, densify_grad_threshold,  size_threshold, viewpoint_cam, current_view_index, sampled_index, image, gt_image, add_in_center=False, error_function="diff"):
        current_view_points, referred_view_points, referred_view_index = self.paired_views[current_view_index][sampled_index]  
        error_map = get_errormap(image, gt_image, error_function)
        current_view_regions, referred_view_regions = get_paired_regions(error_map, current_view_points.cuda(), referred_view_points.cuda())
        zones3d = zones3d_projection(viewpoint_cam, self.cams[referred_view_index], current_view_regions, referred_view_regions)
        if len(zones3d) > 0:
            self.lpm_densify_and_prune(densify_grad_threshold, 0.005, self.extent, size_threshold, zones3d)
            

    
    def lpm_densify_and_clone(self, grads, grad_threshold, scene_extent, zones3d, grad_ratio=0.5):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask, torch.max(self.gaussians.get_scaling, dim=1).values <= self.gaussians.percent_dense*scene_extent)  
        selected_pts_mask_zone3d = self.get_localized_points_mask(zones3d)
        selected_pts_mask_zones3d = torch.logical_and(selected_pts_mask_zone3d, torch.max(self.gaussians.get_scaling, dim=1).values <= self.gaussians.percent_dense*scene_extent)
        selected_pts_mask_lower_grad = torch.where(torch.norm(grads, dim=-1) >= (grad_threshold * grad_ratio), True, False)
        selected_pts_mask_zones_lower = torch.logical_and(selected_pts_mask_lower_grad, selected_pts_mask_zones3d)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_zones_lower)
        selected_pts_mask_extra = torch.where(torch.norm(grads, dim=-1) < grad_threshold, True, False)
        selected_pts_mask_extra = torch.logical_and(selected_pts_mask_extra, selected_pts_mask_zones_lower)
        new_xyz = self.gaussians._xyz[selected_pts_mask]
        new_features_dc = self.gaussians._features_dc[selected_pts_mask]
        new_features_rest = self.gaussians._features_rest[selected_pts_mask]
        new_opacities = self.gaussians._opacity[selected_pts_mask]
        new_scaling = self.gaussians._scaling[selected_pts_mask]
        new_rotation = self.gaussians._rotation[selected_pts_mask]
        self.gaussians.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        
        return selected_pts_mask_extra
    
    def lpm_densify_and_split(self, grads, grad_threshold, scene_extent, zones3d, grad_ratio=0.5, N=2):
        n_init_points = self.gaussians.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,torch.max(self.gaussians.get_scaling, dim=1).values > self.gaussians.percent_dense*scene_extent)
        condition_area = self.get_localized_points_mask(zones3d)
        selected_pts_mask_zones3d = torch.logical_and(torch.max(self.gaussians.get_scaling, dim=1).values > self.gaussians.percent_dense*scene_extent, condition_area)
        selected_pts_mask_lower_grad = torch.where(padded_grad >= grad_threshold*grad_ratio, True, False)
        selected_pts_mask_zones3d_lower = torch.logical_and(selected_pts_mask_lower_grad, selected_pts_mask_zones3d)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_zones3d_lower)
        selected_pts_mask_extra = torch.where(padded_grad < grad_threshold, True, False)
        selected_pts_mask_extra = torch.logical_and(selected_pts_mask_extra, selected_pts_mask_zones3d_lower)
        stds = self.gaussians.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.gaussians._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.gaussians.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.gaussians.scaling_inverse_activation(self.gaussians.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self.gaussians._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self.gaussians._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self.gaussians._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self.gaussians._opacity[selected_pts_mask].repeat(N,1)
        self.gaussians.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.gaussians.prune_points(prune_filter)
        return selected_pts_mask_extra
    
    def lpm_densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, zones3d):  #
        grads = self.gaussians.xyz_gradient_accum /  self.gaussians.denom
        grads[grads.isnan()] = 0.0
        extra_add_num = 0
        if len(zones3d) != 0:
            selected_pts_mask_clone_extra = self.lpm_densify_and_clone(grads, max_grad, extent, zones3d=zones3d)
            selected_pts_mask_split_extra = self.lpm_densify_and_split(grads, max_grad, extent, zones3d=zones3d)
            extra_add_num = int(torch.sum(selected_pts_mask_clone_extra)) + int(torch.sum(selected_pts_mask_split_extra))
        else:
            self.gaussians.densify_and_clone(grads, max_grad, extent)
            self.gaussians.densify_and_split(grads, max_grad, extent)
            
        prune_mask = (self.gaussians.get_opacity < min_opacity).squeeze()
        if extra_add_num != 0:
            values, indices = torch.topk(self.gaussians._opacity.squeeze(), extra_add_num, dim=0, largest=False)
            extra_threshold = values[-1]
            extra_prune_mask = torch.where(self.gaussians._opacity.squeeze() <= extra_threshold, True, False)
            prune_mask = torch.logical_or(prune_mask, extra_prune_mask)
        if max_screen_size:
            big_points_vs = self.gaussians.max_radii2D > max_screen_size
            big_points_ws = self.gaussians.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)           
        self.gaussians.prune_points(prune_mask)
        torch.cuda.empty_cache()
        
    def points_calibration(self, densify_grad_threshold, viewpoint_cam, current_view_index, sampled_index, image, gt_image, error_function="diff"):
        current_view_points, referred_view_points, referred_view_index = self.paired_views[current_view_index][sampled_index]  
        error_map = get_errormap(image, gt_image, error_function)
        current_view_regions, referred_view_regions = get_paired_regions(error_map, current_view_points.cuda(), referred_view_points.cuda())
        zones3d = zones3d_projection(viewpoint_cam, self.cams[referred_view_index], current_view_regions, referred_view_regions)
        if len(zones3d) > 0:
            self.reset_localized_points(viewpoint_cam.camera_center, zones3d)
    
    def reset_localized_points(self, current_cam_pos, zones3d, top_ratio=0.7):        
        gs_xyz = self.gaussians._xyz
        gs_opacity = self.gaussians._opacity
        in_front_zones3d_mask, inside_zones3d_mask = get_points_in_cones(current_cam_pos, zones3d, gs_xyz, gs_opacity)
        # selected_pts_mask = mask2
        inside_zones_points_num = self.gaussians._opacity.squeeze()[inside_zones3d_mask].shape[0]
        if inside_zones_points_num != 0:
            k = int(inside_zones_points_num*top_ratio)
            values, indices = torch.topk(self.gaussians._opacity.squeeze()[inside_zones3d_mask], k, dim=0, largest=True) #
            if values.shape[0] != 0:
                reset_threshold = values[-1]
                reset_points_mask = torch.where(self.gaussians._opacity.squeeze() >= reset_threshold, True, False)
                reset_points_mask = torch.logical_and(in_front_zones3d_mask, reset_points_mask)
                self.gaussians._opacity[reset_points_mask] = inverse_sigmoid(0.01 * torch.ones((self.gaussians._opacity[reset_points_mask].shape[0], 1), device="cuda"))
    
    def get_localized_points_mask(self, regions):
        """Aggregate area calculations over multiple boxes."""
        n_init_points =  self.gaussians.get_xyz.shape[0]
        selected_pts_mask = torch.zeros(n_init_points, device="cuda", dtype=torch.bool)

        # Logical OR to accumulate points falling in any of the boxes
        for region in regions:
            selected_pts_mask = torch.logical_or(selected_pts_mask, self.get_points_mask_per_region(region))
            # selected_pts_mask |= get_region_points_mask(region)
        return selected_pts_mask

    def get_points_mask_per_region(self, region):
        xyz = self.gaussians.get_xyz #n,3
        selected_pts_maskx1 = torch.where(region[0]<xyz[:,0], True, False)
        selected_pts_maskx2 = torch.where(xyz[:,0]<region[3], True, False)  
        selected_pts_masky1 = torch.where(region[1]<xyz[:,1], True, False)
        selected_pts_masky2 = torch.where(xyz[:,1]<region[4], True, False)
        selected_pts_maskz1 = torch.where(region[2]<xyz[:,2], True, False)
        selected_pts_maskz2 = torch.where(xyz[:,2]<region[5], True, False)
        selected_pts_mask = torch.logical_and(selected_pts_maskx1, selected_pts_maskx2)
        selected_pts_mask = torch.logical_and(selected_pts_mask, selected_pts_masky1)
        selected_pts_mask = torch.logical_and(selected_pts_mask, selected_pts_masky2)
        selected_pts_mask = torch.logical_and(selected_pts_mask, selected_pts_maskz1)
        selected_pts_mask = torch.logical_and(selected_pts_mask, selected_pts_maskz2)
        return selected_pts_mask
