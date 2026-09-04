"""
vpf_extract.py

在 vf_extract.py 的基础上，增加视口提取逻辑：
  - 使用等面积球面采样坐标（44个视口，FOV=30°）
  - 用 torch.grid_sample 批量并行提取，效率高
  - 将44个视口拼接成 4行×11列 的大帧（每个视口112×112）
  - 对外接口与 vf_extract.py 完全一致，额外返回 vpframes / vpframes_next
"""

import math
import os

import cv2
import numpy as np
import pandas as pd
import py360convert
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────
# 1. 等面积球面视口坐标生成（44个，FOV=30°）
# ──────────────────────────────────────────────

def generate_non_overlapping_coords(fov_deg=30.0):
    """
    按 cos(pitch) 自动缩减每行视口数，保证各纬度不重叠。
    返回 np.ndarray  shape=(N, 2)  列顺序: [yaw, pitch]（度）
    排列顺序：从上（pitch大）到下（pitch小），每行从左（yaw小）到右（yaw大）
    ——这样拼图时直接按顺序填就是"从左到右、从上到下"。
    """
    pitch_min = -90.0 + fov_deg / 2.0   # -75
    pitch_max =  90.0 - fov_deg / 2.0   #  75
    pitch_centers = np.arange(pitch_min, pitch_max + 1e-9, fov_deg)  # 步长=fov，严格不重叠

    coords = []
    for pitch in reversed(pitch_centers):           # 从高纬到低纬 → 图像从上到下
        cos_p  = np.cos(np.radians(pitch))
        num_yaw = max(1, int(np.floor(360.0 * cos_p / fov_deg)))
        yaw_centers = np.linspace(-180.0, 180.0, num_yaw, endpoint=False)
        yaw_centers += (360.0 / num_yaw) / 2.0     # 居中偏移
        for yaw in yaw_centers:                     # 从左到右
            coords.append([float(yaw), float(pitch)])

    return np.array(coords, dtype=np.float32)


# ──────────────────────────────────────────────
# 2. 高效视口提取器（预计算网格 + GPU批量采样）
# ──────────────────────────────────────────────

class ViewportExtractorE2P:
    """
    预计算每种分辨率的采样网格，之后用 grid_sample 批量提取所有视口。

    参数
    ----
    coords   : np.ndarray (N, 2)  [yaw, pitch]（度）
    fov_deg  : 视口水平/垂直 FOV（度）
    vp_size  : 每个视口的输出像素尺寸（正方形），默认 112
    device   : 'cuda' 或 'cpu'
    """

    def __init__(self, coords, fov_deg=30.0, vp_size=112, device='cpu'):
        self.coords   = coords.astype(np.float32)   # (N, 2)
        self.N        = len(coords)
        self.fov_deg  = float(fov_deg)
        self.vp_size  = int(vp_size)
        self.device   = torch.device(device)
        self._grid_cache = {}                       # (H,W) -> Tensor (N, vp_size, vp_size, 2)

    # ------------------------------------------------------------------
    def _build_grid(self, H, W):
        key = (H, W)
        if key in self._grid_cache:
            return self._grid_cache[key]

        sz   = self.vp_size
        fov  = self.fov_deg
        N    = self.N

        # 构造"坐标场"图：每像素存 (列索引u, 行索引v, 0)
        u_idx    = np.tile(np.arange(W, dtype=np.float32)[None, :], (H, 1))
        v_idx    = np.tile(np.arange(H, dtype=np.float32)[:, None], (1, W))
        coord_img = np.stack([u_idx, v_idx, np.zeros_like(u_idx)], axis=-1)  # (H,W,3)

        grids = np.empty((N, sz, sz, 2), dtype=np.float32)

        for i, (yaw_deg, pitch_deg) in enumerate(self.coords):
            # e2p 投影：得到每个输出像素对应的源坐标
            patch = py360convert.e2p(
                coord_img,
                (fov, fov),
                float(yaw_deg), float(pitch_deg),
                (sz, sz),
                in_rot_deg=0,
                mode='bilinear'
            )   # (sz, sz, 3)

            u_map = patch[..., 0]            # 源列坐标 [0, W-1]
            v_map = patch[..., 1]            # 源行坐标 [0, H-1]

            # 归一化到 [-1, 1]（align_corners=True）
            grids[i, :, :, 0] = 2.0 * u_map / max(W - 1, 1) - 1.0
            grids[i, :, :, 1] = 2.0 * v_map / max(H - 1, 1) - 1.0

        grid_t = torch.from_numpy(grids).to(self.device)
        self._grid_cache[key] = grid_t
        return grid_t

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract(self, frame_bgr):
        """
        从单帧 BGR uint8 图像提取所有视口。

        参数
        ----
        frame_bgr : np.ndarray  (H, W, 3)  uint8，OpenCV 读取的原始帧

        返回
        ----
        viewports : np.ndarray  (N, vp_size, vp_size, 3)  uint8  RGB
        """
        H, W = frame_bgr.shape[:2]
        grid  = self._build_grid(H, W)          # (N, sz, sz, 2)

        # BGR → RGB → float32 [0,1] → Tensor (1, 3, H, W)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frame_t   = torch.from_numpy(
            np.ascontiguousarray(frame_rgb.transpose(2, 0, 1))
        ).to(self.device).unsqueeze(0)          # (1, 3, H, W)

        # expand → (N, 3, H, W)，不复制内存
        frame_exp = frame_t.expand(self.N, 3, H, W)

        # 批量采样所有视口
        crops = F.grid_sample(
            frame_exp, grid,
            mode='bilinear', padding_mode='border', align_corners=True
        )   # (N, 3, sz, sz)

        # → numpy uint8  (N, sz, sz, 3)  RGB
        crops_np = (crops.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        return crops_np


# ──────────────────────────────────────────────
# 3. 将 N 个视口拼成一张大帧
# ──────────────────────────────────────────────

def assemble_viewport_frame(viewports, grid_rows=4, grid_cols=11, vp_size=112):
    """
    将 N=44 个视口（RGB uint8）按 grid_rows×grid_cols 顺序拼成大帧。

    参数
    ----
    viewports : np.ndarray  (N, vp_size, vp_size, 3)  RGB uint8
    grid_rows : 行数，默认 4
    grid_cols : 列数，默认 11
    vp_size   : 单个视口边长，默认 112

    返回
    ----
    big_frame : np.ndarray  (grid_rows*vp_size, grid_cols*vp_size, 3)  RGB uint8
    """
    N = viewports.shape[0]
    needed = grid_rows * grid_cols

    # 不足时用黑色填充
    if N < needed:
        pad = np.zeros((needed - N, vp_size, vp_size, 3), dtype=np.uint8)
        viewports = np.concatenate([viewports, pad], axis=0)

    rows = []
    for r in range(grid_rows):
        row_vps = viewports[r * grid_cols: (r + 1) * grid_cols]   # (grid_cols, sz, sz, 3)
        rows.append(np.concatenate(row_vps, axis=1))               # (sz, grid_cols*sz, 3)

    big_frame = np.concatenate(rows, axis=0)                       # (grid_rows*sz, grid_cols*sz, 3)
    return big_frame


# ──────────────────────────────────────────────
# 4. 原始帧提取（与 vf_extract.py 相同）
# ──────────────────────────────────────────────

def extract_frames(video_path, sampled_path, frame_interval, residual=False):
    try:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cap    = cv2.VideoCapture(str(video_path))
        frames = []

        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return frames

        frame_count = 0
        saved_frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cond = (
                (frame_count % frame_interval == 0 and not residual) or
                ((frame_count - 1) % frame_interval == 0 and residual)
            )
            if cond:
                frames.append(frame)
                saved_frame_count += 1
            frame_count += 1

        cap.release()
        frame_type = 'next frames' if residual else 'sampled frames'
        print(f'Extraction of {frame_type} for {video_name} completed!')
        return frames
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


def process_video_residual(video_type, video_name, framerate, video_path, sampled_path):
    """与 vf_extract.py 接口一致，返回 (frames, frames_next)"""
    if not os.path.exists(sampled_path):
        os.makedirs(sampled_path)
    frame_interval = math.ceil(framerate / 2) if framerate < 2 else int(framerate / 2)
    frames      = extract_frames(video_path, sampled_path, frame_interval, residual=False)
    frames_next = extract_frames(video_path, sampled_path, frame_interval, residual=True)
    return frames, frames_next


# ──────────────────────────────────────────────
# 5. 核心：帧列表 → 视口大帧列表
# ──────────────────────────────────────────────

def frames_to_vpframes(frames, extractor, grid_rows=4, grid_cols=11, vp_size=112,
                       save_dir=None, video_name='video', suffix=''):
    """
    将帧列表转换为视口拼接大帧列表。

    参数
    ----
    frames     : list of np.ndarray  BGR uint8
    extractor  : ViewportExtractorE2P 实例
    grid_rows  : 拼图行数（默认4）
    grid_cols  : 拼图列数（默认11）
    vp_size    : 单个视口尺寸（默认112）
    save_dir   : 若不为 None，则把大帧保存到该目录（BGR png）
    video_name : 保存文件名前缀
    suffix     : '_next' 或 ''

    返回
    ----
    vpframes : list of np.ndarray  (grid_rows*vp_size, grid_cols*vp_size, 3)  RGB uint8
    """
    vpframes = []
    for t, frame in enumerate(frames):
        viewports  = extractor.extract(frame)                           # (N, sz, sz, 3) RGB
        big_frame  = assemble_viewport_frame(viewports, grid_rows, grid_cols, vp_size)
        vpframes.append(big_frame)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fname = os.path.join(save_dir, f'vp_{video_name}_{t + 1}{suffix}.png')
            cv2.imwrite(fname, cv2.cvtColor(big_frame, cv2.COLOR_RGB2BGR))

    return vpframes


# ──────────────────────────────────────────────
# 6. 一站式接口：同时返回 frames/frames_next 和 vpframes/vpframes_next
# ──────────────────────────────────────────────

def process_video_with_viewports(
        video_type, video_name, framerate, video_path,
        sampled_path,
        fov_deg=30.0,
        vp_size=112,
        grid_rows=4,
        grid_cols=11,
        save_vp=False,
        device=None
):
    """
    完整流程：抽帧 + 视口提取。

    返回
    ----
    frames, frames_next, vpframes, vpframes_next
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 6.1 抽帧
    frames, frames_next = process_video_residual(
        video_type, video_name, framerate, video_path, sampled_path
    )

    # 6.2 生成视口坐标（等面积采样，44个）
    coords = generate_non_overlapping_coords(fov_deg=fov_deg)
    print(f'[VP] 视口坐标数: {len(coords)}  FOV={fov_deg}°  size={vp_size}px  grid={grid_rows}×{grid_cols}')

    # 6.3 初始化提取器（网格会在第一次 extract 时按实际分辨率预计算并缓存）
    extractor = ViewportExtractorE2P(coords, fov_deg=fov_deg, vp_size=vp_size, device=device)

    # 6.4 提取视口大帧
    save_dir = sampled_path if save_vp else None

    vpframes = frames_to_vpframes(
        frames, extractor, grid_rows, grid_cols, vp_size,
        save_dir=save_dir, video_name=video_name, suffix=''
    )
    vpframes_next = frames_to_vpframes(
        frames_next, extractor, grid_rows, grid_cols, vp_size,
        save_dir=save_dir, video_name=video_name, suffix='_next'
    )

    print(f'[VP] {video_name}: frames={len(frames)}, vpframes={len(vpframes)}')
    return frames, frames_next, vpframes, vpframes_next


# ──────────────────────────────────────────────
# 7. 测试入口
# ──────────────────────────────────────────────

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    video_type = 'test'
    if video_type == 'test':
        ugcdata = pd.read_csv("../../metadata/test_odv_videos.csv")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 预生成坐标和提取器（在循环外初始化，避免重复构建）
    coords    = generate_non_overlapping_coords(fov_deg=20.0)
    extractor = ViewportExtractorE2P(coords, fov_deg=20.0, vp_size=112, device=device)

    for i in range(len(ugcdata)):
        video_name = ugcdata['vid'][i]
        framerate  = ugcdata['framerate'][i]
        print(f'\nProcessing: {video_name}  fps={framerate}')

        video_path    = f"../../ugc_original_videos/{video_name}.mp4"
        sampled_path  = f'../../video_sampled_frame/original_sampled_frame/{video_name}/'
        vp_save_path  = f'../../video_sampled_frame/vp_sampled_frame/{video_name}/'

        # 抽帧
        frames, frames_next = process_video_residual(
            video_type, video_name, framerate, video_path, sampled_path
        )
        print(f'  frames={len(frames)}, frames_next={len(frames_next)}')

        # 提取视口大帧
        vpframes = frames_to_vpframes(
            frames, extractor,
            grid_rows=10, grid_cols=10, vp_size=112,
            save_dir=vp_save_path, video_name=video_name, suffix=''
        )
        vpframes_next = frames_to_vpframes(
            frames_next, extractor,
            grid_rows=10, grid_cols=10, vp_size=112,
            save_dir=vp_save_path, video_name=video_name, suffix='_next'
        )
        print(f'  vpframes={len(vpframes)}, vpframes_next={len(vpframes_next)}')

        # 可视化第一帧对比
        if len(frames) > 0 and len(vpframes) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))
            axes[0].imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
            axes[0].set_title('Original Frame')
            axes[0].axis('off')
            axes[1].imshow(vpframes[0])
            axes[1].set_title(f'Viewport Grid (5×20, FOV=20°, vp=112px)')
            axes[1].axis('off')
            plt.tight_layout()
            plt.show()

        break  # 只测试第一个视频
