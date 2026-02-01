import cv2
import hashlib
import numpy as np
from contextlib import contextmanager


def apply_image_perturbation(
    image,
    category,
    ptype,
    variant,
    seed=42,
    past_image_list=None
):
    """
    Apply visual perturbation to the input image.
    
    Args:
        image (np.ndarray): Input image (uint8)
        category (str): 'degradation' or 'occlusion'
        ptype (str): Type inside category: e.g. 'blur', 'noise', 'full', 'partial'
        variant (str): Variant: e.g. 'gaussian', 'motion', 's&p', 'mask'
        seed (int): Random seed for reproducibility
        frame_idx (int): Current frame index
        interval (int): Apply perturbation every `interval` frames
    
    Returns:
        perturbed_image (np.ndarray): Output image after perturbation
    """
    mapping = {
        "degradation": {
            "blur": {
                "gaussian": gaussian_blur,
                "motion": motion_blur,
            },
            "noise": {
                "gaussian": gaussian_noise,
                "sAp": salt_and_pepper,
            }
        },
        "occlusion": {
            "full": {
                "black": full_black,
                "cross": cross_image,
            },
            "partial": {
                "exposure": partial_exposure,
                "mask": partial_mask,
            }
        }
    }

    try:
        func = mapping[category][ptype][variant]
    except KeyError:
        raise ValueError(f"Invalid combination: {category} / {ptype} / {variant}")
    
    if variant == "cross":
        img = func(image.copy(), seed, past_image_list)
    else:
        img = func(image.copy(), seed)
    
    return img


def string_to_seed(s: str, max_seed: int = 2**32 - 1) -> int:
    hash_bytes = hashlib.sha256(s.encode('utf-8')).digest()
    seed = int.from_bytes(hash_bytes[:8], 'big')
    return seed % max_seed


@contextmanager
def temp_numpy_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def gaussian_blur(image, seed):
    with temp_numpy_seed(seed):
        is_chw = image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]
        if is_chw:
            image = image.transpose(1, 2, 0)
        # kernels_choice = [(11, 11), (15, 15), (29, 29)]
        # sigmas_choice = [(5, 0), (0, 5)]  # (sigmaX, sigmaY)
        # kernel = kernels_choice[np.random.randint(3)]
        # sigmaX, sigmaY = sigmas_choice[np.random.randint(2)]
        kernel = (15, 15)
        sigmaX, sigmaY = (5, 5)
        img = cv2.GaussianBlur(image, kernel, sigmaX=sigmaX, sigmaY=sigmaY)
        if is_chw:
            img = img.transpose(2, 0, 1)
    return img


def motion_blur(image, seed):
    with temp_numpy_seed(seed):
        is_chw = image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]
        if is_chw:
            image = image.transpose(1, 2, 0)

        # directions_choice = ['horizontal', 'vertical', 'diagonal']
        # levels_choice = [15, 25, 50]
        # direction = np.random.choice(directions_choice)
        # k = np.random.choice(levels_choice)
        direction = 'horizontal'
        k = 25
        kernel = np.zeros((k, k), dtype=np.float32)
        if direction == 'horizontal':
            kernel[int((k - 1) / 2), :] = 1.0
        elif direction == 'vertical':
            kernel[:, int((k - 1) / 2)] = 1.0
        elif direction == 'diagonal':
            np.fill_diagonal(kernel, 1.0)
        kernel /= kernel.sum()
        img = cv2.filter2D(image, -1, kernel)

        if is_chw:
            img = img.transpose(2, 0, 1)
    return img


def gaussian_noise(image, seed):
    with temp_numpy_seed(seed):
        is_chw = image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]
        if is_chw:
            image = image.transpose(1, 2, 0)
        # std_choices = [25, 50, 75]
        # std = np.random.choice(std_choices)
        std = 50
        noise = np.random.normal(0, std, image.shape).astype(np.int16)
        img = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        if is_chw:
            img = img.transpose(2, 0, 1)
    return img


def salt_and_pepper(image, seed):
    with temp_numpy_seed(seed):
        is_chw = image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]
        if is_chw:
            image = image.transpose(1, 2, 0)
        # amount_choices=[0.2, 0.5, 0.8]
        # salt_vs_pepper_choices=[0.0, 0.5, 1.0]
        # amount = np.random.choice(amount_choices)
        # salt_vs_pepper = np.random.choice(salt_vs_pepper_choices)
        amount = 0.2
        salt_vs_pepper = 0.5
        img = image.copy()
        h, w = image.shape[:2]
        num_pixels = h * w
        num_salt = int(amount * num_pixels * salt_vs_pepper)
        num_pepper = int(amount * num_pixels * (1.0 - salt_vs_pepper))
        # add salt (white)
        salt_coords = (np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt))
        if image.ndim == 2:
            img[salt_coords] = 255
        else:
            img[salt_coords[0], salt_coords[1], :] = 255

        # add pepper (black)
        pepper_coords = (np.random.randint(0, h, num_pepper), np.random.randint(0, w, num_pepper))
        if image.ndim == 2:
            img[pepper_coords] = 0
        else:
            img[pepper_coords[0], pepper_coords[1], :] = 0
        
        if is_chw:
            img = img.transpose(2, 0, 1)

    return img


def full_black(image, seed):
    img = np.zeros_like(image)
    return img


def cross_image(image, seed, past_image_list=None):
    with temp_numpy_seed(seed):
        assert past_image_list is not None
        image_list = past_image_list
        img = image_list[-1]
    return img


def partial_exposure(image, seed):
    with temp_numpy_seed(seed):
        is_chw = image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]
        if is_chw:
            image = image.transpose(1, 2, 0)
        # intensity_choices = [0.7, 0.85, 1.0]
        h, w = image.shape[:2]
        min_dim = min(h, w)
        # radius_choices = [int(min_dim * 1 / 6), int(min_dim * 1 / 3), int(min_dim * 1 / 2)]
        # intensity = np.random.choice(intensity_choices)
        # radius = np.random.choice(radius_choices)

        intensity = 0.7
        radius = int(min_dim * 1 / 3)

        h, w = image.shape[:2]
        center_x = int(0.5*w)
        center_y = int(0.5*h)

        Y, X = np.ogrid[:h, :w]                
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        mask = np.exp(-(dist_from_center**2) / (2 * (radius**2)))
        mask = (mask * intensity).astype(np.float32)
        if image.ndim == 3 and image.shape[2] == 3:
            mask = np.expand_dims(mask, axis=2)

        image = image.astype(np.float32)
        img = image + mask * (255 - image)
        img = np.clip(img, 0, 255).astype(np.uint8)

        if is_chw:
            img = img.transpose(2, 0, 1)
    return img


def partial_mask(image, seed):
    with temp_numpy_seed(seed): 
        is_chw = image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]
        if is_chw:
            image = image.transpose(1, 2, 0)
        h, w = image.shape[:2]
        min_dim = min(h, w)
        
        # x_positions_choices = [int(w * 1 / 4), int(w * 2 / 4), int(w * 3 / 4)]
        # radius_choices = [int(min_dim * 1 / 6), int(min_dim * 1 / 3), int(min_dim * 1 / 2)]

        # x = np.random.choice(x_positions_choices)
        # r = np.random.choice(radius_choices)

        x = int(w * 2 / 4)
        r = int(min_dim * 1 / 6)

        img = image.copy()
        cv2.circle(img, (x,  h//2), r, (0, 0, 0), -1)
        
        if is_chw:
            img = img.transpose(2, 0, 1)
    return img

