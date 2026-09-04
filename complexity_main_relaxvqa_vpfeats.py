"""
main_relaxvqa_vpfeats.py

基于原始 main_relaxvqa_feats.py 的改进版本，针对全景视频（ODV-VQA）：

原始三条分支：
  ① 全局分支         : frame → Resize 224 → ResNet50 Stack + ViT Pool
  ② Sampled Fragment : frame_patches (Top-196, from frame diff positions)
                       → ResNet50 Stack + ViT Pool
  ③ Merged Fragment  : merged_frag (diff_frag + flow_frag)
                       → ResNet50 Pool + ViT Pool

改进方案：
  ① 全局分支         : 保持 frame 不变（ERP 帧保留整体场景结构）
  ② Sampled Fragment : 改用 vpframe_patches（视口帧 Top-196 patch）
                       → ResNet50 Stack + ViT Pool
  ③ Merged Fragment  : 改用 vpframe 的 diff_frag + flow_frag
                       → ResNet50 Pool + ViT Pool

改进原因：
  - 全景视频高纬度区域 ERP 投影失真严重，Frame Diff / Optical Flow 在极点附近不可靠
  - vpframe 是等面积采样视口拼接帧（无投影失真），更适合做局部时序特征
  - 视口边界不连续性影响微乎其微：process_patches 按残差能量取 Top-196，
    边界处残差接近 0，基本不会被选中
  - 分支① 保留 ERP 帧做全局感知，确保整体场景语义完整
"""

import argparse
import pandas as pd
import numpy as np
import os
from pathlib import Path
import scipy.io
import shutil
import torch
import time
import cv2
from torchvision import models, transforms

from utils.logger_setup import logger
from extractor.vf_extract import process_video_residual
from extractor.vpf_extract import (
    generate_non_overlapping_coords,
    ViewportExtractorE2P,
    frames_to_vpframes,
)
from extractor.visualise_vit_layer import VitGenerator
from relax_vqa import (
    get_deep_feature, process_video_feature,
    process_patches, get_frame_patches,
    flow_to_rgb, merge_fragments, concatenate_features,
)
from files.visualize_patches_paper import visualize_from_frames


# ──────────────────────────────────────────────────────────────
# 数据集元数据加载
# ──────────────────────────────────────────────────────────────

def load_metadata(video_type):
    print(f'video_type: {video_type}\n')
    if video_type == 'test':
        return pd.read_csv("../metadata/test_odv_videos.csv")
    elif video_type == 'resolution_ugc':
        resolution = '360P'
        return pd.read_csv(f"../metadata/YOUTUBE_UGC_{resolution}_metadata.csv")
    else:
        return pd.read_csv(f'../metadata/{video_type.upper()}_metadata.csv')


# ──────────────────────────────────────────────────────────────
# 视频路径解析（与原版完全一致）
# ──────────────────────────────────────────────────────────────

def get_video_paths(network_name, video_type, videodata, i):
    video_name   = videodata['vid'][i]
    video_width  = videodata['width'][i]
    video_height = videodata['height'][i]
    pixfmt       = videodata['pixfmt'][i]
    framerate    = videodata['framerate'][i]
    common_path  = os.path.join('..', 'video_sampled_frame')

    if video_type == 'test':
        video_path = f"../ugc_original_videos/{video_name}.mp4"
    elif video_type == 'konvid_1k':
        video_path = Path("E:/NR-360VQA/ReLaX-VQA-main/ugc_original_videos/KoNViD_1k_videos") / f"{video_name}.mp4"
    elif video_type in ('lsvq_train', 'lsvq_test', 'lsvq_test_1080P'):
        print(f'video_name: {video_name}')
        video_path = Path("D:/video_dataset/LSVQ") / f"{video_name}.mp4"
        print(f'video_path: {video_path}')
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    elif video_type == 'live_vqc':
        video_path = Path("D:/video_dataset/LIVE-VQC/video") / f"{video_name}.mp4"
    elif video_type == 'live_qualcomm':
        video_path = Path("D:/video_dataset/LIVE-Qualcomm") / f"{video_name}.yuv"
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    elif video_type == 'cvd_2014':
        video_path = Path("D:/video_dataset/CVD2014") / f"{video_name}.avi"
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    elif video_type == 'youtube_ugc':
        video_path = Path("D:/video_dataset/ugc-dataset/youtube_ugc/") / f"{video_name}.mkv"
        video_name = os.path.splitext(os.path.basename(video_path))[0]
    elif video_type == 'odv-vqa':
        video_path = Path("E:/NR-360VQA/ODV-VQA") / f"{video_name}.mp4"

    sampled_frame_path = os.path.join(common_path, f'relaxvqa', f'video_{str(i + 1)}')
    feature_name = f"{network_name}_feature_map"

    if video_type == 'resolution_ugc':
        resolution = '360P'
        video_path = Path(f"D:/video_dataset/ugc-dataset/youtube_ugc/original_videos/{resolution}") / f"{video_name}.mkv"
        sampled_frame_path = os.path.join(common_path, f'ytugc_sampled_frame_{resolution}', f'video_{str(i + 1)}')
        feature_name = f"{network_name}_feature_map_{resolution}"

    return video_name, video_path, sampled_frame_path, feature_name, video_width, video_height, pixfmt, framerate


# ──────────────────────────────────────────────────────────────
# 核心特征提取
# ──────────────────────────────────────────────────────────────

def extract_features(config, video_idx, vp_extractor):
    """
    三条分支特征提取（与原版结构对齐，仅替换分支②③的输入源）

    分支①  全局         : frame      → ResNet50 Stack + ViT Pool
    分支②  Sampled Frag : vpframe     → Top-196 patches → ResNet50 Stack + ViT Pool
    分支③  Merged Frag  : vpframe     → diff + flow → ResNet50 Pool + ViT Pool

    Parameters
    ----------
    config       : dict，来自 argparse
    video_idx    : int，当前视频索引
    vp_extractor : ViewportExtractorE2P，在循环外初始化，所有视频共用
    """
    video_type  = config['video_type']
    model_name  = config['model_name']
    target_size = config['target_size']   # 224
    patch_size  = config['patch_size']    # 16
    top_n = int((target_size / patch_size) ** 2)  # 14*14 = 196

    VP_SIZE = config['vp_size']   # 224，可被 patch_size=16 整除
    VP_ROWS = config['vp_rows']   # 4
    VP_COLS = config['vp_cols']   # 11

    start_time = time.time()

    # ── 路径与元数据 ──────────────────────────────────────────
    (video_name, video_path, sampled_frame_path, feature_name,
     video_width, video_height, pixfmt, framerate) = \
        get_video_paths(model_name, video_type, videodata, video_idx)

    # ── Step 1: 抽取原始帧 ───────────────────────────────────
    # frames / frames_next: list of np.ndarray (H, W, 3) BGR uint8
    frames, frames_next = process_video_residual(
        video_type, video_name, framerate, video_path, sampled_frame_path
    )
    logger.info(f'{video_name}  frames={len(frames)}')

    # ── Step 2: 提取视口拼接帧（分支②③使用） ───────────────
    # vpframe shape: (VP_ROWS*VP_SIZE, VP_COLS*VP_SIZE, 3)  RGB uint8
    # 默认 4*224=896 行，11*224=2464 列
    vpframes = frames_to_vpframes(
        frames, vp_extractor,
        grid_rows=VP_ROWS, grid_cols=VP_COLS, vp_size=VP_SIZE,
    )
    vpframes_next = frames_to_vpframes(
        frames_next, vp_extractor,
        grid_rows=VP_ROWS, grid_cols=VP_COLS, vp_size=VP_SIZE,
    )



    # ── 特征容器（与原版命名对齐） ────────────────────────────
    # 分支① 全局（原始帧）
    all_frame_activations_resnet = []   # ResNet50 layer-stack
    all_frame_activations_vit    = []   # ViT pool

    # 分支② Sampled Fragment（视口帧）
    all_frame_activations_sampled_resnet = []
    all_frame_activations_sampled_vit    = []

    # 分支③ Merged Fragment（视口帧）
    all_frame_activations_merged_resnet = []
    all_frame_activations_merged_vit    = []

    '''循环内 = 空间特征提取（每帧单独处理）'''
    for j, (frame, frame_next, vpframe, vpframe_next) in enumerate(
            zip(frames, frames_next, vpframes, vpframes_next)):

        # visualize_from_frames(vpframe, vpframe_next, video_name=video_name, frame_idx=j)

        frame_number  = j + 1
        # original_path 仅供 process_patches 内部临时文件命名使用
        original_path = os.path.join(sampled_frame_path, f'{video_name}_{frame_number}.png')

        # ══════════════════════════════════════════════════════
        # 分支①：原始 ERP 帧 → 全局特征
        #   与原版完全相同，保留 ERP 帧的整体场景语义
        # ══════════════════════════════════════════════════════
        frame_rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb_tensor = transforms.ToTensor()(frame_rgb).unsqueeze(0).to(device)

        # ResNet50 layer-stack（全局）
        activations_dict_resnet, _, _ = get_deep_feature(
            'resnet50', video_name, frame_rgb_tensor,
            frame_number, resnet50, device, 'layerstack'
        )
        all_frame_activations_resnet.append(activations_dict_resnet)

        # ViT pool（全局）
        activations_dict_vit, _, _ = get_deep_feature(
            'vit', video_name, frame_rgb_tensor,
            frame_number, vit, device, 'pool'
        )
        all_frame_activations_vit.append(activations_dict_vit)

        # ══════════════════════════════════════════════════════
        # 分支②③：视口拼接帧 → 局部时序特征
        #   vpframe 是无投影失真的等面积视口拼接帧（RGB uint8）
        # ══════════════════════════════════════════════════════
        '''residual video frames'''
        # vpframe → tensor（RGB，[0,1]，NCHW）
        vpframe_tensor      = transforms.ToTensor()(vpframe).unsqueeze(0).to(device)
        vpframe_next_tensor = transforms.ToTensor()(vpframe_next).unsqueeze(0).to(device)

        # —— Frame Differencing（在视口帧上计算） ——
        residual = torch.abs(vpframe_next_tensor - vpframe_tensor)
        _, diff_frag, positions = process_patches(
            original_path, 'frame_diff', residual, patch_size, target_size, top_n
        )

        # ── 分支②：Sampled Fragment ──────────────────────────
        # 取与 diff 相同位置的 Top-196 patch（视口帧的空间细节）
        frame_patches = get_frame_patches(vpframe_tensor, positions, patch_size, target_size)

        # ResNet50 layer-stack（sampled fragment）
        sampled_frag_activations_resnet, _, _ = get_deep_feature(
            'resnet50', video_name, frame_patches,
            frame_number, resnet50, device, 'layerstack'
        )
        all_frame_activations_sampled_resnet.append(sampled_frag_activations_resnet)

        # ViT pool（sampled fragment）
        sampled_frag_activations_vit, _, _ = get_deep_feature(
            'vit', video_name, frame_patches,
            frame_number, vit, device, 'pool'
        )
        all_frame_activations_sampled_vit.append(sampled_frag_activations_vit)

        # ── 分支③：Merged Fragment ────────────────────────────
        # Optical Flow 在视口帧上计算，无极点拉伸干扰
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(vpframe,      cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(vpframe_next, cv2.COLOR_BGR2GRAY),
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        opticalflow_rgb        = flow_to_rgb(flow)
        opticalflow_rgb_tensor = transforms.ToTensor()(opticalflow_rgb).unsqueeze(0).to(device)
        opticalflow_frag_path, flow_frag, _ = process_patches(
            original_path, 'optical_flow', opticalflow_rgb_tensor, patch_size, target_size, top_n
        )

        # 合并 diff_frag 与 flow_frag
        merged_frag = merge_fragments(diff_frag, flow_frag)

        # ResNet50 pool（merged fragment）
        merged_frag_activations_resnet, _, _ = get_deep_feature(
            'resnet50', video_name, merged_frag,
            frame_number, resnet50, device, 'pool'
        )
        all_frame_activations_merged_resnet.append(merged_frag_activations_resnet)

        # ViT pool（merged fragment）
        merged_frag_activations_vit, _, _ = get_deep_feature(
            'vit', video_name, merged_frag,
            frame_number, vit, device, 'pool'
        )
        all_frame_activations_merged_vit.append(merged_frag_activations_vit)

    '''循环外 = 时序聚合（把T帧的特征平均成一个视频级别的特征）'''
    print(f'video frame number: {len(all_frame_activations_resnet)}')

    # ── 时序聚合（与原版完全一致） ────────────────────────────
    # 分支①：全局
    averaged_frames_resnet = process_video_feature(
        all_frame_activations_resnet, 'resnet50', 'layerstack'
    )
    averaged_frames_vit = process_video_feature(
        all_frame_activations_vit, 'vit', 'pool'
    )

    # 分支②：sampled fragment
    averaged_frames_sampled_resnet = process_video_feature(
        all_frame_activations_sampled_resnet, 'resnet50', 'layerstack'
    )
    averaged_frames_sampled_vit = process_video_feature(
        all_frame_activations_sampled_vit, 'vit', 'pool'
    )

    # 分支③：merged fragment
    averaged_frames_merged_resnet = process_video_feature(
        all_frame_activations_merged_resnet, 'resnet50', 'pool'
    )
    averaged_frames_merged_vit = process_video_feature(
        all_frame_activations_merged_vit, 'vit', 'pool'
    )

    # sampled + merged 拼接（与原版 concatenate_features 逻辑一致）
    averaged_combined_feature_resnet = concatenate_features(
        averaged_frames_sampled_resnet, averaged_frames_merged_resnet
    )
    averaged_combined_feature_vit = concatenate_features(
        averaged_frames_sampled_vit, averaged_frames_merged_vit
    )

    # ── 最终特征拼接（维度结构与原版完全一致） ───────────────
    combined_features = torch.cat([
        torch.mean(averaged_frames_resnet,           dim=0),  # 分支① ResNet50 全局
        torch.mean(averaged_frames_vit,              dim=0),  # 分支① ViT     全局
        torch.mean(averaged_combined_feature_resnet, dim=0),  # 分支②③ ResNet50 局部
        torch.mean(averaged_combined_feature_vit,    dim=0),  # 分支②③ ViT     局部
    ], dim=0).view(1, -1)

    feats_npy = combined_features.cpu().numpy()

    # ── 可选：保存单视频 npy ──────────────────────────────────
    output_npy_path = f'../features/{video_type}/{model_name}/'
    os.makedirs(output_npy_path, exist_ok=True)
    # output_npy_name = f'{output_npy_path}video_{str(video_idx + 1)}_{feature_name}.npy'
    # np.save(output_npy_name, feats_npy)

    # ── 清理临时文件夹 ────────────────────────────────────────
    if os.path.exists(sampled_frame_path):
        shutil.rmtree(sampled_frame_path)

    run_time = time.time() - start_time
    logger.debug(f"Execution time for {video_name} feature extraction: {run_time:.4f} seconds")
    return feats_npy


# ──────────────────────────────────────────────────────────────
# 参数解析
# ──────────────────────────────────────────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu_id',      type=str,   default='2',
                        help='GPU ID to use (e.g., 0, 1, 2)')
    parser.add_argument('-device',      type=str,   default='gpu',
                        help='cpu or gpu')
    parser.add_argument('-model_name',  type=str,   default='relaxvqa')
    parser.add_argument('-target_size', type=int,   default=224)
    parser.add_argument('-patch_size',  type=int,   default=16)
    parser.add_argument('-video_type',  type=str,   default='test',
                        help='test | konvid_1k | lsvq_train | lsvq_test | '
                             'live_vqc | cvd_2014 | youtube_ugc | odv-vqa')
    # 视口参数
    parser.add_argument('-vp_fov',  type=float, default=10.0,
                        help='视口 FOV（度），默认 30°')
    parser.add_argument('-vp_size', type=int,   default=224,
                        help='单个视口边长（像素），须能被 patch_size 整除，默认 224')
    parser.add_argument('-vp_rows', type=int,   default=4,
                        help='视口拼图行数，默认 4')
    parser.add_argument('-vp_cols', type=int,   default=101,
                        help='视口拼图列数，默认 11')
    args = parser.parse_args()
    return args


# ==============================================================================
# 新增的测算函数 (请确保已经 pip install thop)
# ==============================================================================
def measure_backbone_complexity(config, resnet50, vit, device):
    try:
        from thop import profile
    except ImportError:
        print("请先安装 thop: pip install thop")
        return

    print("\n" + "═" * 60)
    print("📊 开始计算特征提取阶段 (Backbone) 的参数量与 FLOPs")
    print("═" * 60)

    target_size = config['target_size']  # 默认 224

    # 构造一个虚拟输入，模拟 224x224 的图像
    dummy_input = torch.randn(1, 3, target_size, target_size).to(device)

    # 1. 测算单次推理的 MACs 和 Params
    # 注意：thop 默认输出的是 MACs (乘加累积操作数)，通常 1 MAC ≈ 2 FLOPs
    macs_r, params_r = profile(resnet50, inputs=(dummy_input,), verbose=False)
    macs_v, params_v = profile(vit.model, inputs=(dummy_input,), verbose=False)

    flops_r = macs_r * 2
    flops_v = macs_v * 2

    # 2. 汇总参数量 (一套权重)
    total_params = params_r + params_v
    print(f"🔹 [模型参数量 Parameters]")
    print(f"   ResNet50 : {params_r / 1e6:>8.2f} M")
    print(f"   ViT-Base : {params_v / 1e6:>8.2f} M")
    print(f"   ----------------------------------")
    print(f"   总参数量 : {total_params / 1e6:>8.2f} M (仅统计一次共享权重)\n")

    # 3. 汇总计算量 (处理 1 帧视频需要的网络算力)
    # 你的架构针对每帧会运行 3 条分支（原图、Sampled Patch、Merged Patch）
    # 每条分支都独立过了一遍 ResNet50 和 ViT
    total_flops_per_frame = 3 * (flops_r + flops_v)
    print(f"🔹 [网络计算量 FLOPs (每处理 1 帧视频)]")
    print(f"   单次 ResNet50 (224x224) : {flops_r / 1e9:>8.2f} GFLOPs")
    print(f"   单次 ViT-Base (224x224) : {flops_v / 1e9:>8.2f} GFLOPs")
    print(f"   ----------------------------------")
    print(f"   3分支总计 FLOPs/帧      : {total_flops_per_frame / 1e9:>8.2f} GFLOPs\n")

    # 4. 视口参数的影响分析报告
    vp_size = config['vp_size']
    vp_rows = config['vp_rows']
    vp_cols = config['vp_cols']
    vp_h = vp_rows * vp_size
    vp_w = vp_cols * vp_size

    print(f"💡 [当前视口参数对复杂度的实际影响]")
    print(f"   - 设定的视口网格: {vp_rows}行 x {vp_cols}列, 单个Size={vp_size}")
    print(f"   - 生成的 vpframe 分辨率: {vp_w} x {vp_h}")
    print(f"   * 无论上述分辨率多大，送入网络的 Patch 始终是 Top-196，拼回 {target_size}x{target_size}。")
    print(f"   * 因此，改变视口参数【不会】改变上述的 GFLOPs 数据！")
    print(f"   * 但它会指数级增加 cv2.calcOpticalFlowFarneback 在 {vp_w}x{vp_h} 上的 CPU 计算耗时。")
    print("═" * 60 + "\n")


# ──────────────────────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_arguments()
    config = vars(args)

    # 设备
    device = (torch.device(f"cuda:{config['gpu_id']}")
              if config['device'] == 'gpu' and torch.cuda.is_available()
              else torch.device('cpu'))
    logger.info(f"ReLax-VQA (VP) --- video type: {config['video_type']}")
    print(f"Running on {'GPU' if device.type == 'cuda' else 'CPU'}")

    begin_time = time.time()

    # ── 加载预训练模型 ────────────────────────────────────────
    resnet50 = models.resnet50(pretrained=True).to(device)
    vit = VitGenerator('vit_base', 16, device, evaluate=True, random=False, verbose=True)

    # ==========================================================
    # 🎯 运行复杂度测算代码
    # ==========================================================
    measure_backbone_complexity(config, resnet50, vit, device)

    # 如果你只是想测算 FLOPs，你可以加一个 input() 暂停，或者 sys.exit() 提前退出
    # input("按 Enter 键继续执行后续的特征提取 (或者 Ctrl+C 退出)...")

    # ── 视口提取器（循环外初始化，网格只构建一次，所有视频共用） ──
    vp_coords = generate_non_overlapping_coords(fov_deg=config['vp_fov'])
    vp_extractor = ViewportExtractorE2P(
        coords=vp_coords,
        fov_deg=config['vp_fov'],
        vp_size=config['vp_size'],
        device=str(device),
    )
    print(f"[VP] 视口数={len(vp_coords)}  FOV={config['vp_fov']}°  "
          f"vp_size={config['vp_size']}  grid={config['vp_rows']}×{config['vp_cols']}")

    # ── 加载数据集元数据 ──────────────────────────────────────
    videodata = load_metadata(config['video_type'])
    feats_matrix = None

    for video_idx in range(len(videodata)):
        feats_npy = extract_features(config, video_idx, vp_extractor)

        average_data = np.mean(feats_npy, axis=0)
        if feats_matrix is None:
            feats_matrix = np.zeros((len(videodata),) + average_data.shape)
        feats_matrix[video_idx] = average_data

    print(f'All features shape: {feats_matrix.shape}')

    # ── 保存 .mat 文件 ────────────────────────────────────────
    mat_file_path = f"../features/"
    os.makedirs(mat_file_path, exist_ok=True)
    mat_file_name = (f"{mat_file_path}"
                     f"{config['video_type']}_{config['model_name']}_vpfeats.mat")
    scipy.io.savemat(mat_file_name, {config['video_type']: feats_matrix})
    print(f"Execution time for all feature extraction: {time.time() - begin_time:.4f} seconds\n")




# # ──────────────────────────────────────────────────────────────
# # 主程序
# # ──────────────────────────────────────────────────────────────
#
# if __name__ == '__main__':
#     args   = parse_arguments()
#     config = vars(args)
#
#     # 设备
#     device = (torch.device(f"cuda:{config['gpu_id']}")
#               if config['device'] == 'gpu' and torch.cuda.is_available()
#               else torch.device('cpu'))
#     logger.info(f"ReLax-VQA (VP) --- video type: {config['video_type']}")
#     print(f"Running on {'GPU' if device.type == 'cuda' else 'CPU'}")
#     logger.debug(f"Running on {'GPU' if device.type == 'cuda' else 'CPU'}")
#
#     begin_time = time.time()
#
#     # ── 加载预训练模型 ────────────────────────────────────────
#     resnet50 = models.resnet50(pretrained=True).to(device)
#     vit      = VitGenerator('vit_base', 16, device, evaluate=True, random=False, verbose=True)
#
#     # ── 视口提取器（循环外初始化，网格只构建一次，所有视频共用） ──
#     vp_coords    = generate_non_overlapping_coords(fov_deg=config['vp_fov'])
#     vp_extractor = ViewportExtractorE2P(
#         coords  = vp_coords,
#         fov_deg = config['vp_fov'],
#         vp_size = config['vp_size'],
#         device  = str(device),
#     )
#     print(f"[VP] 视口数={len(vp_coords)}  FOV={config['vp_fov']}°  "
#           f"vp_size={config['vp_size']}  grid={config['vp_rows']}×{config['vp_cols']}")
#
#     # ── 加载数据集元数据 ──────────────────────────────────────
#     videodata    = load_metadata(config['video_type'])
#     feats_matrix = None
#
#     for video_idx in range(len(videodata)):
#         feats_npy = extract_features(config, video_idx, vp_extractor)
#
#         average_data = np.mean(feats_npy, axis=0)
#         if feats_matrix is None:
#             feats_matrix = np.zeros((len(videodata),) + average_data.shape)
#         feats_matrix[video_idx] = average_data
#
#     print(f'All features shape: {feats_matrix.shape}')
#     logger.debug(f'\n All features shape: {feats_matrix.shape}')
#
#     # ── 保存 .mat 文件 ────────────────────────────────────────
#     mat_file_path = f"../features/"
#     os.makedirs(mat_file_path, exist_ok=True)
#     mat_file_name = (f"{mat_file_path}"
#                      f"{config['video_type']}_{config['model_name']}_vpfeats.mat")
#     scipy.io.savemat(mat_file_name, {config['video_type']: feats_matrix})
#     logger.debug(f'Successfully created {mat_file_name}')
#     logger.debug(f"Execution time for all feature extraction: "
#                  f"{time.time() - begin_time:.4f} seconds\n")
