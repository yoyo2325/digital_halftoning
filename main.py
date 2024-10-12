import os
import numpy as np
from PIL import Image

output_dirs = ['output_ordered_dithering', 'output_error_diffusion', 'output_dbs', 'output_dot_diffusion']
for output_dir in output_dirs:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def bayer_matrix(n):
    if n == 1:
        return np.array([[0]])
    else:
        smaller_matrix = bayer_matrix(n // 2)
        top_left = 4 * smaller_matrix
        top_right = top_left + 2
        bottom_left = top_left + 3
        bottom_right = top_left + 1
        return np.block([[top_left, top_right], [bottom_left, bottom_right]])

def ordered_dithering(img, N=4):
    img = img.convert('L')
    pixels = np.array(img)
    threshold_matrix = bayer_matrix(N)
    threshold_matrix = 255 * (threshold_matrix + 0.5) / (N**2)
    h, w = pixels.shape
    dithered_img = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            threshold = threshold_matrix[i % N, j % N]
            dithered_img[i, j] = 255 if pixels[i, j] > threshold else 0
            
    return Image.fromarray(dithered_img)

def error_diffusion(img):
    img = img.convert('L')
    pixels = np.array(img, dtype=np.float32)
    h, w = pixels.shape
    for i in range(h):
        for j in range(w):
            old_pixel = pixels[i, j]
            new_pixel = 255 if old_pixel > 128 else 0
            pixels[i, j] = new_pixel
            error = old_pixel - new_pixel
            
            if j + 1 < w:
                pixels[i, j + 1] += error * 7 / 16
            if i + 1 < h:
                if j > 0:
                    pixels[i + 1, j - 1] += error * 3 / 16
                pixels[i + 1, j] += error * 5 / 16
                if j + 1 < w:
                    pixels[i + 1, j + 1] += error * 1 / 16
                    
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))

def direct_binary_search(img):
    # TODO
    img = img.convert('L')
    pixels = np.array(img)
    dbs_img = np.where(pixels > 128, 255, 0).astype(np.uint8)
    
    return Image.fromarray(dbs_img)

def dot_diffusion(img, N=3):
    img = img.convert('L')
    pixels = np.array(img)
    h, w = pixels.shape
    dot_diffused_img = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(0, h, N):
        for j in range(0, w, N):
            avg = np.mean(pixels[i:i + N, j:j + N])
            dot_diffused_img[i:i + N, j:j + N] = 255 if avg > 128 else 0
            
    return Image.fromarray(dot_diffused_img)

image_folder = 'images'
for filename in os.listdir(image_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path)

        ordered_img = ordered_dithering(img)
        ordered_img.save(f'output_ordered_dithering/{filename}')

        error_diffused_img = error_diffusion(img)
        error_diffused_img.save(f'output_error_diffusion/{filename}')

        dbs_img = direct_binary_search(img)
        dbs_img.save(f'output_dbs/{filename}')

        dot_diffused_img = dot_diffusion(img)
        dot_diffused_img.save(f'output_dot_diffusion/{filename}')
