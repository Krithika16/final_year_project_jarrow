import cv2 as cv
import tensorflow as tf
import numpy as np


def convert_to_cv_format(img):
    if tf.is_tensor(img):
        img = img.numpy()
    img = img.astype('uint8')
    if len(img.shape) == 2:
        return img
    elif len(img.shape) == 4:
        return img[0, :, :, 0]
    elif len(img.shape) == 3:
        if img.shape[-1] == 1 or img.shape[-1] == 3:
            return img[:, :, 0]
        else:
            return img[0, :, :]


def get_matches_ratio(original_img, transformed_img, apply_filter=True):
    sift = cv.SIFT_create()

    kp_org, des_org = sift.detectAndCompute(original_img, None)
    kp_tran, des_tran = sift.detectAndCompute(transformed_img, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_org, des_tran, k=2)

    good = []
    for m, n in matches:
        if apply_filter and m.distance < 0.7 * n.distance:
            good.append(m)
        elif not apply_filter:
            good.append(m)

    ratio = None
    if len(kp_org) != 0:
        ratio = good / len(kp_org)
        if ratio > 1:
            ratio = 1
    return ratio


def register_img(ref_img, transformed_img):
    orb_detector = cv.ORB_create(5000)

    kp1, d1 = orb_detector.detectAndCompute(transformed_img, None)
    kp2, d2 = orb_detector.detectAndCompute(ref_img, None)
    height, width = ref_img.shape

    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d1, d2)
    matches.sort(key=lambda x: x.distance)

    ratio = None
    if len(kp2) != 0:
        ratio = len(matches) / len(kp2)
        if ratio > 1:
            ratio = 1

    matches = matches[:int(len(matches) * 90)]
    no_of_matches = len(matches)

    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    homography, mask = cv.findHomography(p1, p2, cv.RANSAC)

    registered_img = cv.warpPerspective(transformed_img,
                                        homography, (width, height))
    return registered_img, ratio


def get_registation_error(ref_img, transformed_img):
    registered_img, ratio = register_img(ref_img, transformed_img)

    y_pred = np.ravel(registered_img.astype("float16"))
    y_true = np.ravel(ref_img.astype("float16"))
    loss = tf.keras.losses.MSE(y_true, y_pred)
    return loss, ratio
