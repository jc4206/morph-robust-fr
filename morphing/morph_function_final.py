# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:25:26 2019

@author: KellyUM
"""

import numpy as np
import cv2
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter


def merge_duplicates(Landmarks0, Landmarks1):
    if len(np.array(list(set(tuple(p) for p in Landmarks0)))) < len(Landmarks0):
        for p in Landmarks0:
            indices = np.where((Landmarks0 == p).all(axis=1))[0]
            if indices.shape[0] > 1:
                first_index = indices[0]
                second_index = indices[1]
                Landmarks0 = np.delete(Landmarks0, first_index, axis=0)
                Landmarks1[first_index] = 0.5 * (Landmarks1[first_index] + Landmarks1[second_index])
                Landmarks1 = np.delete(Landmarks1, second_index, axis=0)
    if len(np.array(list(set(tuple(p) for p in Landmarks1)))) < len(Landmarks1):
        for p in Landmarks1:
            indices = np.where((Landmarks1 == p).all(axis=1))[0]
            if indices.shape[0] > 1:
                first_index = indices[0]
                second_index = indices[1]
                Landmarks1 = np.delete(Landmarks1, first_index, axis=0)
                Landmarks0[first_index] = 0.5 * (Landmarks0[first_index] + Landmarks0[second_index])
                Landmarks0 = np.delete(Landmarks0, second_index, axis=0)
    return Landmarks0, Landmarks1


def morph(img0, img1, landmarks0, landmarks1, alpha, beta, blur_sigma=0):
    H, W, C = img0.shape
    img0_f = img0.astype(np.float32)
    img1_f = img1.astype(np.float32)

    landmarks0 = np.clip(landmarks0, 0, H)
    landmarks1 = np.clip(landmarks1, 0, H)

    # check for duplicate points
    landmarks0, landmarks1 = merge_duplicates(landmarks0, landmarks1)

    # Compute intermediate landmarks
    landmarks_warp = beta * landmarks0 + (1 - beta) * landmarks1
    tri = Delaunay(landmarks_warp)

    # Initialize accumulators
    warped0_accum = np.zeros((H, W, C), dtype=np.float32)
    warped1_accum = np.zeros((H, W, C), dtype=np.float32)
    mask_accum = np.zeros((H, W, C), dtype=np.float32)

    corner_indices = [len(landmarks_warp) - 4, len(landmarks_warp) - 3,
                      len(landmarks_warp) - 2, len(landmarks_warp) - 1]
    for triangle in tri.simplices:
        pts_warp = landmarks_warp[triangle].astype(np.float32)
        pts0 = landmarks0[triangle].astype(np.float32)
        pts1 = landmarks1[triangle].astype(np.float32)

        # Bounding box for triangle0
        xmin0 = int(np.floor(np.min(pts0[:, 0])))
        xmax0 = int(np.ceil(np.max(pts0[:, 0]))) + 1
        ymin0 = int(np.floor(np.min(pts0[:, 1])))
        ymax0 = int(np.ceil(np.max(pts0[:, 1]))) + 1
        # Bounding box for triangle1
        xmin1 = int(np.floor(np.min(pts1[:, 0])))
        xmax1 = int(np.ceil(np.max(pts1[:, 0]))) + 1
        ymin1 = int(np.floor(np.min(pts1[:, 1])))
        ymax1 = int(np.ceil(np.max(pts1[:, 1]))) + 1

        # Bounding box for triangle
        xmin = int(np.floor(np.min(pts_warp[:, 0])))
        xmax = int(np.ceil(np.max(pts_warp[:, 0]))) + 1
        ymin = int(np.floor(np.min(pts_warp[:, 1])))
        ymax = int(np.ceil(np.max(pts_warp[:, 1]))) + 1

        # if xmin < 0 or ymin < 0 or xmax > W or ymax > H:
        #     continue

        # Crop regions
        rect_warp = pts_warp - [xmin, ymin]
        rect0 = pts0 - [xmin0, ymin0]
        rect1 = pts1 - [xmin1, ymin1]

        # Affine transforms
        M0 = cv2.getAffineTransform(np.float32(rect0), np.float32(rect_warp))
        M1 = cv2.getAffineTransform(np.float32(rect1), np.float32(rect_warp))

        # print((xmax - xmin, ymax - ymin))
        # print((xmax0 - xmin0, ymax0 - ymin0))
        # if xmax - xmin <= 0 or ymax - ymin <= 0 or xmax0 - xmin0 <= 0 or ymax0 - ymin0 <= 0 or xmax1 - xmin1 <= 0 or ymax1 - ymin1 <= 0:
        #     print()
        #     continue
        # Warp only bounding box
        warped0 = cv2.warpAffine(img0_f[ymin0:ymax0, xmin0:xmax0], M0, (xmax - xmin, ymax - ymin),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        warped1 = cv2.warpAffine(img1_f[ymin1:ymax1, xmin1:xmax1], M1, (xmax - xmin, ymax - ymin),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Mask for triangle inside bounding box
        mask = np.zeros((ymax - ymin, xmax - xmin), dtype=np.float32)
        contour = [np.int32(rect_warp)]  # drawContours expects a list of contours
        cv2.drawContours(mask, contour, contourIdx=-1, color=1, thickness=cv2.FILLED)

        # Accumulate warped images and mask
        for c in range(C):
            warped0_accum[ymin:ymax, xmin:xmax, c] = np.maximum(
                warped0_accum[ymin:ymax, xmin:xmax, c],
                warped0[:, :, c] * mask)
            warped1_accum[ymin:ymax, xmin:xmax, c] = np.maximum(
                warped1_accum[ymin:ymax, xmin:xmax, c],
                warped1[:, :, c] * mask)
            if not any(idx in corner_indices for idx in triangle):
                mask_accum[ymin:ymax, xmin:xmax, c] += mask

    # Final morph
    morph_full = (alpha * warped0_accum + (1 - alpha) * warped1_accum)
    mask_accum = np.clip(mask_accum, 0, 1)
    if blur_sigma > 0:
        mask_accum = gaussian_filter(mask_accum, sigma=blur_sigma)

    inv_mask = np.ones((H, W, C), dtype=np.float32) - mask_accum

    # adjust face area to color-match outside
    mean0 = np.sum(mask_accum * warped0_accum) / np.sum(mask_accum)
    mean1 = np.sum(mask_accum * warped1_accum) / np.sum(mask_accum)

    morph_adjust0 = mask_accum * (morph_full - 0.5 * (mean1 - mean0) * np.ones((H, W, C), dtype=np.float32))
    morph_adjust1 = mask_accum * (morph_full - 0.5 * (mean0 - mean1) * np.ones((H, W, C), dtype=np.float32))
    morph_in0 = morph_adjust0 + inv_mask * warped0_accum
    morph_in1 = morph_adjust1 + inv_mask * warped1_accum

    return np.clip(morph_full, 0, 255).astype(np.uint8), np.clip(morph_in0, 0, 255).astype(np.uint8), np.clip(morph_in1,
                                                                                                              0,
                                                                                                              255).astype(
        np.uint8)



