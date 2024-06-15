import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def gaussian(window_size, sigma):
    """
    Generates a Gaussian kernel.
    
    Parameters:
        window_size (int): The size of the Gaussian kernel.
        sigma (float): The standard deviation of the Gaussian kernel.
        
    Returns:
        torch.Tensor: A 1D tensor representing the Gaussian kernel.
    """
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    """
    Creates a 2D Gaussian window suitable for convolution operations.
    
    Parameters:
        window_size (int): The size of the window.
        channel (int): The number of channels the window should accommodate.
        
    Returns:
        torch.Tensor: A 4D tensor representing the Gaussian window for each channel.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    """
    Computes the Structural Similarity Index (SSIM) between two images.
    
    Parameters:
        img1 (torch.Tensor): The first image.
        img2 (torch.Tensor): The second image.
        window_size (int): The size of the window to create the SSIM index.
        size_average (bool): If True, averages the SSIM over all pixels.
        
    Returns:
        torch.Tensor: The computed SSIM index map.
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """
    Helper function to compute SSIM values between two images, given a precomputed window.
    
    Parameters are similar to ssim().
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map

def mse(img1, img2):
    """
    Computes the Mean Squared Error between two images.
    
    Parameters:
        img1 (torch.Tensor): The first image.
        img2 (torch.Tensor): The second image.
        
    Returns:
        torch.Tensor: The Mean Squared Error.
    """
    return (img1 - img2) ** 2

def psnr(img1, img2):
    """
    Computes the Peak Signal-to-Noise Ratio between two images based on MSE.
    
    Parameters:
        img1 (torch.Tensor): The first image.
        img2 (torch.Tensor): The second image.
        
    Returns:
        float: The PSNR value.
    """
    mse_value = mse(img1, img2).mean().sqrt()
    return 20 * torch.log10(1.0 / mse_value)
