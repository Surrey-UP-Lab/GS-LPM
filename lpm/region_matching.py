import cv2
import numpy as np
import torch
import random

def prepare_image_for_opencv(tensor):
    """Convert a PyTorch tensor to a numpy array formatted for OpenCV."""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return np.transpose(tensor.numpy(), (1, 2, 0))

def find_connected_components_and_bounding_boxes(mask, kernel_size=3):
    """
    Find connected components and their bounding boxes in a binary mask image.
    
    Args:
        mask (ndarray): Input image mask which may have multiple channels.
        kernel_size (int, optional): Size of the kernel used for morphological operations. Default is 3.
    
    Returns:
        list of tuples: Bounding boxes for each connected component in the format (x_min, y_min, x_max, y_max).
        ndarray: Processed binary mask, single-channel.
    """

    # Convert RGB mask to grayscale if necessary
    if mask.ndim == 3 and mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

    # Ensure mask is binary and convert to uint8
    if mask.dtype != np.uint8:
        mask = (mask * 255).clip(0, 255).astype(np.uint8)

    # Find connected components with statistics
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Extract bounding boxes, excluding the background label
    regions = [
        (stats[i, cv2.CC_STAT_LEFT], 
         stats[i, cv2.CC_STAT_TOP], 
         stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH], 
         stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT])
        for i in range(1, num_labels)  # Start from 1 to exclude background
    ]

    return regions, mask


def find_bounding_box(points):
    """Calculate the bounding box of the given points."""
    if points.numel() == 0:
        return None
    x_min = torch.min(points[:, 0])
    y_min = torch.min(points[:, 1])
    x_max = torch.max(points[:, 0])
    y_max = torch.max(points[:, 1])
    return (int(x_min.item()), int(y_min.item()), int(x_max.item()), int(y_max.item()))

def find_matching_region(points0, points1, region):
    """Finds bounding boxes of matched points in the specified region for two point sets."""
    # Determine which points are within the specified region
    in_region = (points0[:, 0] >= region[0]) & (points0[:, 0] <= region[2]) & \
                (points0[:, 1] >= region[1]) & (points0[:, 1] <= region[3])

    # Filter points that are in the region
    matched_points0 = points0[in_region]
    matched_points1 = points1[in_region]

    # Compute bounding boxes for the filtered points from both sets
    region0 = find_bounding_box(matched_points0)
    region1 = find_bounding_box(matched_points1)

    return region0, region1

def get_paired_regions(error_map, current_view_points, referred_view_points):
    """
    Match bounding boxes between two sets of points based on an error map.
    
    Args:
        error_map (Tensor): A 2D tensor representing an error map where non-zero values indicate errors.
        current_view_points (Tensor): Coordinates of points in the current view.
        referred_view_points (Tensor): Coordinates of points in the referred view.
    
    Returns:
        list: Bounding boxes in the current view that match with the referred view.
        list: Corresponding bounding boxes in the referred view.
    """

    # Expand and threshold the error map to ensure it is suitable for processing
    error_map = error_map.repeat(3, 1, 1)  # Repeat channels to make it RGB if needed for following operations
    error_map = (error_map > 0).float()  # Convert to a binary map

    # Prepare the error map for OpenCV processing, assuming this involves some normalization or conversion
    error_map = prepare_image_for_opencv(error_map)

    # Find connected components in the error map to get potential regions
    current_view_region_all, _ = find_connected_components_and_bounding_boxes(error_map)

    current_view_selected_region, referred_view_selected_region = [], []
    # Match regions between current and referred views based on their points
    for region in current_view_region_all:
        region0, region1 = find_matching_region(current_view_points, referred_view_points, region)
        if region1 is not None:
            current_view_selected_region.append(region0)
            referred_view_selected_region.append(region1)

    return current_view_selected_region, referred_view_selected_region

