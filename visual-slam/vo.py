#!/usr/bin/env python3
"""Monocular visual odometry / mini-VSLAM front end, built from first principles.

Pipeline per frame pair (k-1 -> k):
  1. ORB detection + description                      (cv2.ORB_create)
  2. Brute-force Hamming matching + Lowe ratio test   (cv2.BFMatcher.knnMatch)
  3. Essential matrix with RANSAC                     (cv2.findEssentialMat)
  4. Decompose E -> (R, t) with cheirality check      (cv2.recoverPose)
  5. Triangulate inlier matches -> 3D landmarks       (cv2.triangulatePoints)
  6. Resolve the monocular scale ambiguity            (--scale gt | rel)
  7. Chain relative poses -> camera trajectory

Usage (KITTI odometry layout: <root>/sequences/00/image_0/*.png, calib.txt,
<root>/poses/00.txt):
    python vo.py --root /data/kitti --seq 00 --out results/seq00 --scale rel
"""
import argparse
import glob
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------- data ---
def load_kitti(root, seq):
    img_dir = os.path.join(root, "sequences", seq, "image_0")
    images = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not images:
        raise FileNotFoundError(f"no images in {img_dir}")
    with open(os.path.join(root, "sequences", seq, "calib.txt")) as f:
        for line in f:
            if line.startswith("P0:"):
                P0 = np.array(line.split()[1:], dtype=float).reshape(3, 4)
    K = P0[:, :3]
    gt = None
    pose_file = os.path.join(root, "poses", f"{seq}.txt")
    if os.path.exists(pose_file):
        gt = np.loadtxt(pose_file).reshape(-1, 3, 4)
    return images, K, gt


# ------------------------------------------------------------- geometry ---
def to_h(T34):
    T = np.eye(4)
    T[:3, :4] = T34
    return T


def match_orb(des1, des2, ratio=0.75):
    """kNN Hamming matching + Lowe ratio test. Returns list of (idx1, idx2)."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    out = []
    for pair in bf.knnMatch(des1, des2, k=2):
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            out.append((pair[0].queryIdx, pair[0].trainIdx))
    return out


def triangulate(K, R, t, pts1, pts2):
    """Triangulate with P1 = K[I|0], P2 = K[R|t]. Returns Nx3 points in the
    frame-1 camera frame, plus a validity mask (positive depth in both views,
    reprojection error < 2 px)."""
    if len(pts1) == 0:
        return np.zeros((0, 3)), np.zeros(0, bool)
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    Xh = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    X = (Xh[:3] / Xh[3]).T
    X2 = (R @ X.T + t.reshape(3, 1)).T
    ok = (X[:, 2] > 0) & (X2[:, 2] > 0)
    proj2 = (K @ X2.T).T
    proj2 = proj2[:, :2] / proj2[:, 2:3]
    proj1 = (K @ X.T).T
    proj1 = proj1[:, :2] / proj1[:, 2:3]
    err = np.maximum(np.linalg.norm(proj1 - pts1, axis=1),
                     np.linalg.norm(proj2 - pts2, axis=1))
    ok &= err < 2.0
    return X, ok


class MonoVO:
    def __init__(self, K, n_features=3000, scale_mode="rel", ratio=0.75,
                 tracker="orb", cam_height=1.65, min_parallax=1.0):
        self.K = K
        self.cam_height = cam_height      # metres above ground (KITTI cam0 ~1.65 m)
        self.min_parallax = min_parallax  # px; skip pairs with less median flow
        self.scale_hist = []
        self.skip = False
        self.tracker = tracker    # "orb": detect+match each frame; "klt": FAST + LK flow
        self.orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=12)
        self.fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        self.n_features = n_features
        self.scale_mode = scale_mode
        self.ratio = ratio
        self.prev = None          # (kps_or_pts, des, depth_prev)
        self.prev_img = None
        self.T_w_c = np.eye(4)    # camera-to-world pose of the latest frame
        self.landmarks = []       # world-frame 3D points (for visualisation)
        self.stats = []

    # --- front end: returns the point correspondences between k-1 and k ---
    def _detect_fast(self, img, existing=None):
        mask = np.full(img.shape[:2], 255, np.uint8)
        if existing is not None:
            for x, y in existing:
                cv2.circle(mask, (int(x), int(y)), 10, 0, -1)
        kps = sorted(self.fast.detect(img, mask), key=lambda k: -k.response)
        return np.float32([k.pt for k in kps[:self.n_features]]).reshape(-1, 2)

    def _associate(self, img):
        """-> (i0, i1, p0, p1, new_a, new_b): indices into the previous/new
        feature arrays, pixel coords, and the new frame's feature state."""
        if self.tracker == "orb":
            kps, des = self.orb.detectAndCompute(img, None)
            kp0, des0, _ = self.prev
            matches = match_orb(des0, des, self.ratio) if des is not None else []
            i0 = np.array([m[0] for m in matches], int)
            i1 = np.array([m[1] for m in matches], int)
            p0 = np.float32([kp0[i].pt for i in i0]).reshape(-1, 2)
            p1 = np.float32([kps[i].pt for i in i1]).reshape(-1, 2)
            return i0, i1, p0, p1, kps, des
        # KLT: track previous points with pyramidal Lucas-Kanade, forward-backward check
        pts0 = self.prev[0]
        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if len(pts0) == 0:
            pts = self._detect_fast(img)
            return (np.zeros(0, int),) * 2 + (np.zeros((0, 2), np.float32),) * 2 + (pts, None)
        pts1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_img, img, pts0, None, **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(img, self.prev_img, pts1, None, **lk)
        h, w = img.shape[:2]
        good = ((st.ravel() == 1) & (st2.ravel() == 1)
                & (np.linalg.norm(back - pts0, axis=1) < 1.0)
                & (pts1[:, 0] >= 0) & (pts1[:, 0] < w) & (pts1[:, 1] >= 0) & (pts1[:, 1] < h))
        i0 = np.where(good)[0]
        p0, p1 = pts0[i0], pts1[good]
        i1 = np.arange(len(p1))
        new_pts = p1
        if len(p1) < 0.5 * self.n_features:      # re-detect to replenish tracks
            new_pts = np.vstack([p1, self._detect_fast(img, p1)])
        return i0, i1, p0, p1, new_pts, None

    def step(self, img, gt_step_len=None):
        self.skip = False
        pose = self._step(img, gt_step_len)
        if not self.skip:          # skipped (near-static) pairs keep the old reference frame
            self.prev_img = img
        return pose

    def _step(self, img, gt_step_len=None):
        if self.prev is None:
            if self.tracker == "orb":
                a, b = self.orb.detectAndCompute(img, None)
            else:
                a, b = self._detect_fast(img), None
            self.prev = (a, b, np.full(len(a), np.nan))
            return self.T_w_c.copy()
        depth0 = self.prev[2]
        i0, i1, p0, p1, kps, des = self._associate(img)
        matches = i0
        if len(matches) >= 20 and self.min_parallax > 0:
            if np.median(np.linalg.norm(p1 - p0, axis=1)) < self.min_parallax:
                self.skip = True            # stationary / tiny baseline: E is degenerate
                self.stats.append((len(matches), 0, 0))
                return self.T_w_c.copy()
        if len(matches) < 20:   # tracking lost: keep last pose
            self.prev = (kps, des, np.full(len(kps), np.nan))
            self.stats.append((len(matches), 0, 0))
            return self.T_w_c.copy()

        E, mask = cv2.findEssentialMat(p0, p1, self.K, method=cv2.RANSAC,
                                       prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            self.prev = (kps, des, np.full(len(kps), np.nan))
            self.stats.append((len(matches), 0, 0))
            return self.T_w_c.copy()
        n_in, R, t, mask_pose = cv2.recoverPose(E, p0, p1, self.K, mask=mask)
        if n_in < 8:   # degenerate pair (e.g. no parallax): no valid pose, no landmarks
            self.prev = (kps, des, np.full(len(kps), np.nan))
            self.stats.append((len(matches), 0, 0))
            return self.T_w_c.copy()
        t = t.ravel()  # unit norm
        inl = mask_pose.ravel() > 0
        X, ok = triangulate(self.K, R, t, p0[inl].astype(float), p1[inl].astype(float))

        # ---- scale -----------------------------------------------------
        s = 1.0
        if self.scale_mode == "gt" and gt_step_len is not None:
            s = gt_step_len
        elif self.scale_mode == "ground":
            # camera height above the road fixes the metric scale (no GT needed):
            # take triangulated points in the lower image, find the dominant height
            # Y (camera y-axis points down), and scale so that Y == cam_height.
            pk = p0[inl][ok]
            Xk0 = X[ok]
            low = (pk[:, 1] > self.K[1, 2] + 0.15 * 2 * self.K[1, 2]) & (Xk0[:, 1] > 0) \
                  & (Xk0[:, 2] > 0)
            Yv = Xk0[low, 1]
            est = None
            if len(Yv) >= 15:
                cnt = (np.abs(Yv[:, None] - Yv[None, :]) < 0.08 * Yv[:, None]).sum(1)
                best = int(np.argmax(cnt))
                if cnt[best] >= 12:
                    h_unit = float(np.median(Yv[np.abs(Yv - Yv[best]) < 0.08 * Yv[best]]))
                    est = self.cam_height / h_unit
            if est is not None and 0.02 < est < 100:
                self.scale_hist.append(est)
            if self.scale_hist:
                s = float(np.median(self.scale_hist[-7:]))   # speed is smooth: median filter
            else:
                s = 1.0
        elif self.scale_mode == "rel":
            # depth of the same physical points in frame k-1, known (with scale)
            # from the previous step; compare with this step's unit-scale depth.
            d_prev = depth0[i0[inl]][ok]
            d_unit = X[ok, 2]
            good = np.isfinite(d_prev) & (d_unit > 1e-3)
            if good.sum() >= 8:
                s = float(np.median(d_prev[good] / d_unit[good]))
                s = float(np.clip(s, 0.2, 5.0))
            else:
                s = self.last_scale if hasattr(self, "last_scale") else 1.0
        self.last_scale = s
        # forward-motion sanity: monocular E is ill-conditioned for near-zero motion
        t_s = t * s

        # pose chaining: x_k = R x_{k-1} + t  =>  T_w_ck = T_w_c(k-1) * inv([R|t])
        T_rel = np.eye(4)
        T_rel[:3, :3], T_rel[:3, 3] = R, t_s
        self.T_w_c = self.T_w_c @ np.linalg.inv(T_rel)

        # ---- landmarks and depth bookkeeping for next step ---------------
        Xs = X * s
        Xk = (R @ Xs.T + t_s.reshape(3, 1)).T       # points in frame-k camera
        depth1 = np.full(len(kps), np.nan)
        idx_in = i1[inl][ok]
        depth1[idx_in] = Xk[ok, 2]
        far = Xk[:, 2] < 80
        keep = ok & far
        Xw = (self.T_w_c[:3, :3] @ Xk[keep].T).T + self.T_w_c[:3, 3]
        self.landmarks.append(Xw)
        self.prev = (kps, des, depth1)
        self.stats.append((len(matches), int(inl.sum()), int(ok.sum())))
        return self.T_w_c.copy()


# -------------------------------------------------------------- metrics ---
def umeyama(src, dst, with_scale=True):
    """Least-squares similarity alignment dst ~ s R src + t (both Nx3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (xs ** 2).sum() / len(src)
    s = (D * np.diag(S)).sum() / var if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def ate_rmse(est, gt, with_scale):
    s, R, t = umeyama(est, gt, with_scale)
    aligned = (s * (R @ est.T)).T + t
    return float(np.sqrt(((aligned - gt) ** 2).sum(1).mean())), aligned


def kitti_segment_error(poses_est, poses_gt, lengths):
    """KITTI-style translational error (%) over segments of given lengths (m)
    and rotational error (deg/m)."""
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(
        np.diff(poses_gt[:, :3, 3], axis=0), axis=1))])
    t_errs, r_errs = [], []
    for L in lengths:
        for a in range(0, len(dist), 10):
            b = np.searchsorted(dist, dist[a] + L)
            if b >= len(dist):
                break
            Tg = np.linalg.inv(poses_gt[a]) @ poses_gt[b]
            Te = np.linalg.inv(poses_est[a]) @ poses_est[b]
            E = np.linalg.inv(Te) @ Tg
            t_errs.append(np.linalg.norm(E[:3, 3]) / L)
            ang = np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1))
            r_errs.append(np.degrees(ang) / L)
    if not t_errs:
        return float("nan"), float("nan")
    return 100 * float(np.mean(t_errs)), float(np.mean(r_errs))


# ----------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seq", default="00")
    ap.add_argument("--out", default="results")
    ap.add_argument("--scale", choices=["gt", "rel", "ground", "none"], default="rel",
                    help="gt: oracle per-step scale from ground truth; "
                         "rel: scale propagated from triangulated depths "
                         "(pure monocular); ground: metric scale from the known "
                         "camera height over the road plane (no GT); none: unit translations")
    ap.add_argument("--cam-height", type=float, default=1.65,
                    help="camera height above ground in m (KITTI: ~1.65)")
    ap.add_argument("--min-parallax", type=float, default=1.0,
                    help="skip frame pairs whose median pixel flow is below this (0 = off)")
    ap.add_argument("--tracker", choices=["orb", "klt"], default="orb",
                    help="orb: ORB detect+match per frame; klt: FAST corners + "
                         "pyramidal Lucas-Kanade tracking (classic monocular VO recipe)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--features", type=int, default=3000)
    ap.add_argument("--seg-lengths", type=int, nargs="+",
                    default=[100, 200, 300, 400, 500, 600, 700, 800])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    images, K, gt = load_kitti(a.root, a.seq)
    n = len(images) if not a.max_frames else min(a.max_frames, len(images))
    vo = MonoVO(K, a.features, a.scale, tracker=a.tracker,
                cam_height=a.cam_height, min_parallax=a.min_parallax)
    poses = []
    for k in range(n):
        img = cv2.imread(images[k], cv2.IMREAD_GRAYSCALE)
        step = None
        if gt is not None and k > 0:
            step = float(np.linalg.norm(gt[k, :, 3] - gt[k - 1, :, 3]))
        poses.append(vo.step(img, step))
        if k % 50 == 0:
            print(f"frame {k}/{n}  stats(matches,inliers,3D)={vo.stats[-1] if vo.stats else None}")
    poses = np.array(poses)
    np.savetxt(os.path.join(a.out, "est_poses.txt"), poses[:, :3, :].reshape(n, 12))

    est_xyz = poses[:, :3, 3]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    if vo.landmarks:
        L = np.vstack(vo.landmarks)
        L = L[np.random.default_rng(0).permutation(len(L))[:30000]]
        ax[0].scatter(L[:, 0], L[:, 2], s=0.2, c="0.7", label="triangulated landmarks")
    ax[0].plot(est_xyz[:, 0], est_xyz[:, 2], "r-", label=f"estimate (scale={a.scale})")
    summary = [f"frames: {n}"]
    if gt is not None:
        gt_h = np.array([to_h(g) for g in gt[:n]])
        gt_xyz = gt_h[:, :3, 3]
        ax[0].plot(gt_xyz[:, 0], gt_xyz[:, 2], "k-", label="KITTI ground truth")
        ate_raw, _ = ate_rmse(est_xyz, gt_xyz, with_scale=False)
        ate_sim, aligned = ate_rmse(est_xyz, gt_xyz, with_scale=True)
        te, re = kitti_segment_error(poses, gt_h, [l for l in a.seg_lengths
                                                    if l < 0.9 * np.sum(np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1))])
        ax[1].plot(gt_xyz[:, 0], gt_xyz[:, 2], "k-", label="ground truth")
        ax[1].plot(aligned[:, 0], aligned[:, 2], "b-", label="estimate, Sim(3)-aligned")
        summary += [f"ATE RMSE (SE3 align only, includes scale drift): {ate_raw:.2f} m",
                    f"ATE RMSE (Sim3 aligned): {ate_sim:.2f} m",
                    f"KITTI t_err: {te:.2f} %   r_err: {re:.4f} deg/m"]
    raw_unit = "m" if a.scale in ("gt", "ground") else "arbitrary units (monocular scale)"
    for axx, title, unit in zip(ax, ["Trajectory (top-down, raw)", "Sim(3)-aligned"],
                                [raw_unit, "m"]):
        axx.set_title(title)
        axx.set_xlabel(f"x [{unit}]")
        axx.set_ylabel(f"z [{unit}]")
        axx.axis("equal")
        axx.grid(alpha=0.3)
        axx.legend(loc="best", fontsize=8)
    fig.suptitle(f"Monocular VO - seq {a.seq}")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "trajectory.png"), dpi=130)
    with open(os.path.join(a.out, "metrics.txt"), "w") as f:
        f.write("\n".join(summary) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()