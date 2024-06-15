import torch
import random 
import torchvision
import numpy as np 
from kornia import create_meshgrid
from LightGlue.lightglue.utils import load_image, rbd
from lpm.error_function import ssim, psnr, mse

def pix2ndc(v, S):
    return (v * 2.0 + 1.0) / S - 1.0


def get_errormap(image, gt_image, error_function="diff"):

    # Normalize the input images
    image_adjust = image / (torch.mean(image) + 0.01)
    gt_adjust = gt_image / (torch.mean(gt_image) + 0.01)

    # Compute the error map based on the specified error function
    error_map = {
        'diff': lambda img, gt: torch.abs(img - gt),
        'ssim': ssim,
        'psnr': psnr,
        'mse': mse
    }.get(error_function)

    if not error_map:
        raise ValueError("Unrecognized error function provided!")

    error_map = error_map(image_adjust, gt_adjust)
    error_map = torch.sum(error_map, dim=0)  # Sum over channels to flatten the error map

    # Threshold the error map to isolate significant errors
    threshold = torch.quantile(error_map, 0.4)  # Use quantile for adaptive thresholding
    error_mask = error_map > threshold

    # Process the error mask to find connected high-error regions
    kernel_size = (16, 16)  # (height, width)
    stride = (16, 16)  # (height, width)
    padding = (
        (image.shape[1] + stride[0] - 1) // stride[0] * stride[0] - image.shape[1],
        (image.shape[2] + stride[1] - 1) // stride[1] * stride[1] - image.shape[2]
    )

    # Pad error mask to match the size of the image processing grid
    error_mask = torch.nn.functional.pad(error_mask, (0, padding[1], 0, padding[0]), mode='constant', value=0)

    # Extract patches and determine if they significantly contain errors
    patches = error_mask.unfold(0, kernel_size[0], stride[0]).unfold(1, kernel_size[1], stride[1])
    patch_sums = patches.sum(dim=(2, 3))
    significant_patches = patch_sums > (kernel_size[0] * kernel_size[1] * 0.85)

    # Create a full-size mask from significant patch information
    error_map = significant_patches.repeat_interleave(kernel_size[0], dim=0).repeat_interleave(kernel_size[1], dim=1)
    error_map = error_map[:image.shape[1], :image.shape[2]].float()  # Trim to the original image size

    return error_map


def extract_features(image, extractor):
    """Extract features from the image using a specified extractor."""
    return extractor.extract(image)


def get_matched_points(current_image, referred_image, extractor, matcher, cv2_macher):
    """Match points between two images using specified feature extractor and matcher."""
    feats0 = extract_features(current_image, extractor)
    feats1 = extract_features(referred_image, extractor)
    if cv2_macher==False:
        matches01 = matcher({'image0': feats0, 'image1': feats1})
    else:        
        feats0des =feats0["descriptors"].squeeze(dim=0).detach().cpu().numpy()
        feats1des =feats1["descriptors"].squeeze(dim=0).detach().cpu().numpy()
        matches01 = matcher.match(feats0des,feats1des)
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # Remove batch dimension
    matches = matches01['matches']
    points0 = feats0['keypoints'][matches[..., 0]]
    points1 = feats1['keypoints'][matches[..., 1]]
    
    return points0.cpu(), points1.cpu()

def find_angles(camera_dierction, angle):
    """Calculate the mask of angles between points that are less than a specified angle."""
    vectors = camera_dierction
    norms = torch.norm(vectors, dim=1)
    cosine_similarity = torch.mm(vectors, vectors.T) / torch.ger(norms, norms)
    angles = torch.rad2deg(torch.acos(cosine_similarity.clamp(-1, 1)))  # Clamp for numerical stability
    return (angles < angle) & (angles > 0)
def get_paired_views(scene, extractor, matcher, angle, cv2_macher=False, condition_points_num=100, select_min_num=3):
    #move image to cpu to save gpu memory
    for i in range(len(scene.getTrainCameras())):
        scene.train_cameras[1.0][i].original_image = scene.train_cameras[1.0][i].original_image.cpu() 
    """Find views in the scene that match based on the angle and extract matched points."""
    train_cameras = scene.getTrainCameras()
    # Process all camera rays at once and find central rays
    camera_rays = torch.stack([camera.rays.cpu() for camera in train_cameras])
    camera_rays = camera_rays.permute(0, 1, 3, 4, 2).reshape(-1, *camera_rays.shape[2:4], camera_rays.shape[4])
    camera_center_rays = camera_rays[:, camera_rays.shape[1] // 2, camera_rays.shape[2] // 2, :]
    angles_mask = find_angles(camera_center_rays, angle)

    paired_views = []
    for i, current_cam in enumerate(train_cameras):
        valid_cameras = [cam for cam, mask in zip(train_cameras, angles_mask[i]) if mask]
        if not valid_cameras:
            continue  # Skip if no valid cameras

        selected_cameras = random.sample(valid_cameras, min(select_min_num, len(valid_cameras)))
        current_view_pair = []
        for referred_cam in selected_cameras:
            points0, points1 = get_matched_points(current_cam.original_image.cuda(), referred_cam.original_image.cuda(), extractor, matcher, cv2_macher)
            current_view_pair.append([points0, points1, train_cameras.index(referred_cam)])

        paired_views.append(current_view_pair)

    return paired_views

def set_rays_od(cams):
    for id, cam in enumerate(cams):
        rayd=1
        if rayd is not None:
            projectinverse = cam.projection_matrix.T.inverse()
            camera2wold = cam.world_view_transform.T.inverse()
            pixgrid = create_meshgrid(cam.image_height, cam.image_width, normalized_coordinates=False, device="cpu")[0]
            pixgrid = pixgrid.cuda()  # H,W,
            xindx = pixgrid[:,:,0] # x 
            yindx = pixgrid[:,:,1] # y
            ndcy, ndcx = pix2ndc(yindx, cam.image_height), pix2ndc(xindx, cam.image_width)
            ndcx = ndcx.unsqueeze(-1)
            ndcy = ndcy.unsqueeze(-1)# * (-1.0)
            ndccamera = torch.cat((ndcx, ndcy,   torch.ones_like(ndcy) * (1.0) , torch.ones_like(ndcy)), 2) # N,4 
            projected = ndccamera @ projectinverse.T 
            diretioninlocal = projected / projected[:,:,3:] #v 
            direction = diretioninlocal[:,:,:3] @ camera2wold[:3,:3].T 
            # rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)
            rays_d = direction
            cam.rayo = cam.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0).cpu()                               
            cam.rayd = rays_d.permute(2, 0, 1).unsqueeze(0).cpu()    
        else :
            cam.rayo = None
            cam.rayd = None

def set_rays(scene,resolution_scales=[1.0]):
    set_rays_od(scene.getTrainCameras())
    for resolution_scale in resolution_scales:
        for cam in scene.train_cameras[resolution_scale]:
            if cam.rayo is not None:
                cam.rays = torch.cat([cam.rayo, cam.rayd], dim=1) 


