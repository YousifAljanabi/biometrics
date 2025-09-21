import os
import math
import csv
import cv2
import numpy as np

def _ensure_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def _read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    return _ensure_gray(img)

def _crossing_number(block):
    p = (block > 0).astype(np.uint8)
    n = [p[0,0], p[0,1], p[0,2], p[1,2], p[2,2], p[2,1], p[2,0], p[1,0], p[0,0]]
    t = 0
    for i in range(8):
        if n[i] == 0 and n[i+1] == 1:
            t += 1
    return t

def extract_minutiae(binary, border=8, nms_dist=8):
    h, w = binary.shape
    pts = []
    for y in range(border, h-border):
        for x in range(border, w-border):
            if binary[y, x] == 0:
                continue
            block = binary[y-1:y+2, x-1:x+2]
            cn = _crossing_number(block)
            if cn == 1:
                pts.append((x, y, "ending"))
            elif cn == 3:
                pts.append((x, y, "bifurcation"))
    kept = []
    for x, y, t in pts:
        if all((x-x0)**2 + (y-y0)**2 >= nms_dist**2 for x0,y0,_ in kept):
            kept.append((x, y, t))
    return kept

def _pair_candidates(ptsA, ptsB, dist_thresh=20.0):
    if not ptsA or not ptsB:
        return np.empty((0,2),np.float32), np.empty((0,2),np.float32)
    A = np.array([(x,y) for x,y,_ in ptsA], np.float32)
    B = np.array([(x,y) for x,y,_ in ptsB], np.float32)
    pairsA, pairsB = [], []
    for i in range(A.shape[0]):
        d2 = np.sum((B - A[i])**2, axis=1)
        j = int(np.argmin(d2))
        if math.sqrt(float(d2[j])) <= dist_thresh:
            pairsA.append(A[i])
            pairsB.append(B[j])
    if not pairsA:
        return np.empty((0,2),np.float32), np.empty((0,2),np.float32)
    return np.vstack(pairsA), np.vstack(pairsB)

def _similarity_ransac(A_xy, B_xy, tol=10.0):
    if len(A_xy) < 2 or len(B_xy) < 2:
        return None, np.zeros(0,bool)
    M, inliers = cv2.estimateAffinePartial2D(
        A_xy, B_xy, method=cv2.RANSAC,
        ransacReprojThreshold=tol,
        maxIters=2000, refineIters=25, confidence=0.995
    )
    return M, inliers.ravel().astype(bool) if inliers is not None else np.zeros(len(A_xy),bool)

def visualize_match(imgA, imgB, A_xy, B_xy, inliers):
    a = cv2.cvtColor(imgA, cv2.COLOR_GRAY2BGR)
    b = cv2.cvtColor(imgB, cv2.COLOR_GRAY2BGR)
    h = max(a.shape[0], b.shape[0])
    w = a.shape[1] + b.shape[1]
    canvas = np.full((h, w, 3), 255, np.uint8)
    canvas[:a.shape[0], :a.shape[1]] = a
    canvas[:b.shape[0], a.shape[1]:a.shape[1]+b.shape[1]] = b

    offset = np.array([a.shape[1], 0], np.float32)
    for i in range(len(A_xy)):
        ptA = tuple(np.round(A_xy[i]).astype(int))
        ptB = tuple(np.round(B_xy[i] + offset).astype(int))
        color = (0,200,0) if inliers[i] else (0,0,255)
        cv2.line(canvas, ptA, ptB, color, 1, cv2.LINE_AA)
    return canvas

def _certainty_score(inliers, pairs, nA, nB):
    """Combine inlier ratio and minutiae coverage into a certainty measure."""
    if pairs == 0 or min(nA, nB) == 0:
        return 0.0
    frac_inliers = inliers / pairs
    coverage = inliers / min(nA, nB)
    return 0.5 * (frac_inliers + coverage)

def _certainty_label(cert):
    if cert >= 0.75:
        return "High"
    elif cert >= 0.5:
        return "Moderate"
    else:
        return "Low"

def match_bmp(pathA, pathB, outdir="out_match"):
    os.makedirs(outdir, exist_ok=True)
    imgA = _read_gray(pathA)
    imgB = _read_gray(pathB)

    # binarize
    _, binA = cv2.threshold(imgA, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, binB = cv2.threshold(imgB, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    minA = extract_minutiae(binA)
    minB = extract_minutiae(binB)

    A_xy, B_xy = _pair_candidates(minA, minB)
    M, inliers = _similarity_ransac(A_xy, B_xy)

    nA, nB = len(minA), len(minB)
    pairs = len(A_xy)
    n_inliers = int(inliers.sum())
    certainty = _certainty_score(n_inliers, pairs, nA, nB)
    label = _certainty_label(certainty)

    vis = visualize_match(binA, binB, A_xy, B_xy, inliers)
    out_vis = os.path.join(outdir, f"{os.path.basename(pathA)}_vs_{os.path.basename(pathB)}.png")
    cv2.imwrite(out_vis, vis)

    return {
        "fileA": pathA, "fileB": pathB,
        "minutiae_A": nA, "minutiae_B": nB,
        "pairs": pairs, "inliers": n_inliers,
        "certainty": round(certainty, 3),
        "certainty_label": label,
        "vis": out_vis
    }

if __name__ == "__main__":
    base = os.path.expanduser("~/Desktop/lab/shanshal.bmp")
    others = [
        os.path.expanduser("~/Desktop/lab/hd_rot90.bmp"),
        os.path.expanduser("~/Desktop/lab/hd_rot270_flip.bmp")
    ]
    for other in others:
        res = match_bmp(base, other, outdir="match_results")
        print(res)
