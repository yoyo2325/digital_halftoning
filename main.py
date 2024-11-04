import os
import numpy as np
from PIL import Image
import pandas as pd

output_dirs = [
    'output_ordered_dithering', 'output_error_diffusion', 'output_dbs', 'output_dot_diffusion'
]
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

def error_diffusion(img, kernel='floyd-steinberg'):
    img = img.convert('L')
    pixels = np.array(img, dtype=np.float32)
    h, w = pixels.shape
    kernels = {
        'floyd-steinberg': [(1, 0, 7/16), (1, -1, 3/16), (1, 1, 1/16), (0, 1, 5/16)],
        'stucki': [(1, 0, 8/42), (1, 1, 4/42), (1, -1, 2/42), (0, 1, 7/42)]
    }
    selected_kernel = kernels.get(kernel, kernels['floyd-steinberg'])

    for i in range(h):
        for j in range(w):
            old_pixel = pixels[i, j]
            new_pixel = 255 if old_pixel > 128 else 0
            pixels[i, j] = new_pixel
            error = old_pixel - new_pixel
            
            for dy, dx, weight in selected_kernel:
                if 0 <= i + dy < h and 0 <= j + dx < w:
                    pixels[i + dy, j + dx] += error * weight
                    
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))

def direct_binary_search(img, iterations=50, tolerance=1e-3):
    img = img.convert('L')
    pixels = np.array(img, dtype=np.float64)  # 使用 float64 類型
    dbs_img = np.where(pixels > 128, 255, 0).astype(np.uint8)
    
    for _ in range(iterations):
        total_error_reduction = 0  # 計算每次迭代的誤差變化
        for i in range(pixels.shape[0]):
            for j in range(pixels.shape[1]):
                current_error = np.abs(pixels - dbs_img).sum(dtype=np.float64)
                dbs_img[i, j] = 255 - dbs_img[i, j]  # 嘗試翻轉
                new_error = np.abs(pixels - dbs_img).sum(dtype=np.float64)
                error_reduction = current_error - new_error
                if error_reduction >= 0:
                    total_error_reduction += error_reduction  # 累加每次誤差減少值
                else:
                    dbs_img[i, j] = 255 - dbs_img[i, j]  # 若誤差增大，則還原翻轉

        # 若誤差減少小於閾值 tolerance，則提前停止
        if total_error_reduction < tolerance:
            print(f"Early stopping at iteration {_+1} due to minimal error reduction.")
            break
            
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

def calculate_hpsnr(original_img, processed_img):
    original = np.array(original_img.convert('L'), dtype=np.float32)
    processed = np.array(processed_img.convert('L'), dtype=np.float32)
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    hpsnr = 10 * np.log10(max_pixel ** 2 / mse)
    return hpsnr

results = []

image_folder = 'images'
for filename in os.listdir(image_folder):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path)

        # Ordered Dithering Experiments with different matrix sizes
        for N in [2, 4, 8, 16]:
            ordered_img = ordered_dithering(img, N)
            ordered_img.save(f'output_ordered_dithering/{filename.split(".")[0]}_ordered_N{N}.png')
            hpsnr_value = calculate_hpsnr(img, ordered_img)
            print(f'{filename} - Ordered Dithering (N={N}): HPSNR = {hpsnr_value:.2f}')
            results.append([filename, 'Ordered Dithering', f'N={N}', hpsnr_value])

        # Error Diffusion Experiments with different kernels
        for kernel in ['floyd-steinberg', 'stucki']:
            error_diffused_img = error_diffusion(img, kernel)
            error_diffused_img.save(f'output_error_diffusion/{filename.split(".")[0]}_error_{kernel}.png')
            hpsnr_value = calculate_hpsnr(img, error_diffused_img)
            print(f'{filename} - Error Diffusion (Kernel={kernel}): HPSNR = {hpsnr_value:.2f}')
            results.append([filename, 'Error Diffusion', f'Kernel={kernel}', hpsnr_value])

        # Direct Binary Search with different iteration counts
        # for iterations in [1, 10, 50, 100]:
        #     dbs_img = direct_binary_search(img, iterations)
        #     dbs_img.save(f'output_dbs/{filename.split(".")[0]}_dbs_iter{iterations}.png')
        #     hpsnr_value = calculate_hpsnr(img, dbs_img)
        #     print(f'{filename} - Direct Binary Search (Iterations={iterations}): HPSNR = {hpsnr_value:.2f}')
        #     results.append([filename, 'Direct Binary Search', f'Iterations={iterations}', hpsnr_value])

        # Dot Diffusion Experiments with different block sizes
        for N in [3, 5, 7]:
            dot_diffused_img = dot_diffusion(img, N)
            dot_diffused_img.save(f'output_dot_diffusion/{filename.split(".")[0]}_dotdiffusion_N{N}.png')
            hpsnr_value = calculate_hpsnr(img, dot_diffused_img)
            print(f'{filename} - Dot Diffusion (Block Size={N}): HPSNR = {hpsnr_value:.2f}')
            results.append([filename, 'Dot Diffusion', f'Block Size={N}', hpsnr_value])

# Save results to a CSV file for easy analysis
df = pd.DataFrame(results, columns=['Image', 'Method', 'Parameter', 'HPSNR'])
df.to_csv('halftoning_hpsnr_results.csv', index=False)
print("Results saved to 'halftoning_hpsnr_results.csv'.")
