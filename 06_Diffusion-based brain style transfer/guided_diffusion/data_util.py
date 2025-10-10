import math

import numpy as np
import cv2
import pywt
from PIL import Image


def equal(im, imin, imax):
    im[im > imax] = imax
    im[im < imin] = imin
    return (im - imin) / (imax - imin)


def wavelet_denoising(image, wavelet='db1', level=2, mode='soft'):
    image = np.array(image)

    coeffs = pywt.wavedec(image, wavelet, level=level)

    noise_std = np.std(coeffs[-1])

    threshold = noise_std * np.sqrt(2 * np.log(len(image)))
    #     threshold = np.sqrt(2 * np.log2(image.size))

    denoised_coeffs = [pywt.threshold(i, value=threshold, mode=mode) for i in coeffs]

    denoised_image = pywt.waverec(denoised_coeffs, wavelet)

    return denoised_image


def get_cond_image(img, eta=10., ksize=3, isTrain=True, type="allen"):
    img = np.array(img, dtype=np.uint8)
    # img = cv2.equalizeHist(img)
    if not isTrain:
        if type == "allen":
            img = equal(img, 2, 309.6) * 225  # allen
        elif type == "allen_dev":
            # wavelet = 'bior3.3'
            # mode = 'soft'
            # img = wavelet_denoising(img, wavelet, 10, mode)  # 小波
            img = cv2.bilateralFilter(img.astype('float32'), 5, 75, 75)  # 病理图像处理
        elif type == "lsfm":
            img_shrink = cv2.resize(img, (int(img.shape[1] / 3 * 2), int(img.shape[0] / 3 * 2)))
            img = cv2.resize(img_shrink, (img.shape[1], img.shape[0]))
            # img = cv2.bilateralFilter(img.astype('float32'), 3, 75, 75)  # 病理图像处理
    else:
        if type == "allen_dev":
            img = cv2.bilateralFilter(img.astype('float32'), 5, 75, 75)  # 病理图像处理

    x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=ksize)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=ksize)
    absX = cv2.convertScaleAbs(x)  # 转回unit8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    dst = np.where(dst > eta, dst, 0)

    if (np.max(dst) - np.min(dst)) != 0:
        if not isTrain:
            if type == "allen_dev":
                # dst = cv2.equalizeHist(dst)
                dst[dst > 150] = 150  # 180 150 125
                # dst = dst
            elif type == "allen":
                dst[dst > 125] = 125
            elif type == "lsfm":
                # dst = cv2.equalizeHist(dst)
                # dst[dst > 120] = 120  # 120
                dst[dst > 150] = 150  # recA
                # dst[dst < 10] = 10
        else:
            dst = dst
        dst = (dst - np.min(dst)) / (np.max(dst) - np.min(dst)) * 255
        dst = (dst - np.min(dst)) / (np.max(dst) - np.min(dst)) * 255

    return Image.fromarray(dst.astype(np.uint8)).convert("L")


def f_low_cut(fshift, r1, r2, size):
    fabs = np.abs(fshift)
    fangle = np.angle(fshift)

    row = math.floor(r1 / size)
    col = math.floor(r2 / size)
    fabs[r1 - row: r1 + row, r2 - col: r2 + col] = 0

    fshift_cut = fabs * np.exp(1j * fangle)
    return fshift_cut


def get_cond_image_fda(img, size=15):
    img = np.array(img, dtype=np.uint8)
    dmin = img.min()
    dmax = img.max()
    img = (img - dmin) / (dmax - dmin + 1)
    img = img.astype(np.float32)

    f_img = np.fft.fft2(img)
    f_img = np.fft.fftshift(f_img)

    row, col = img.shape[0], img.shape[1]
    r1 = math.floor(row / 2)
    r2 = math.floor(col / 2)
    f_img_cut = f_low_cut(f_img, r1, r2, size)

    f_img_cut = np.fft.ifftshift(f_img_cut)
    f_img_cut = np.fft.ifft2(f_img_cut)
    dst = np.abs(f_img_cut)

    if (np.max(dst) - np.min(dst)) != 0:
        dst = (dst - np.min(dst)) / (np.max(dst) - np.min(dst)) * 255

    # dst[img == 0] = 0
    return Image.fromarray(dst.astype(np.uint8)).convert("L")


def get_edge_image(img):
    img = np.array(img, dtype=np.uint8)
    img[img != 0] = 255
    ret, binary = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bg = np.zeros_like(img)
    cv2.drawContours(bg, contours, -1, 255, 1)  # 改变的是img这张图

    return Image.fromarray(bg.astype(np.uint8)).convert("L")
