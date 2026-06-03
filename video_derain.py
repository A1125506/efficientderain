import cv2
import numpy as np
import torch
import argparse
import os
import network
import utils

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def derain_frame(generator, frame, use_gpu=True):
    """對單張 frame 做去雨處理，回傳去雨後的彩色影像"""
    # 取得原始尺寸
    height_origin, width_origin = frame.shape[:2]

    # 調整尺寸為 16 的倍數（模型要求）
    height = height_origin
    width = width_origin
    if height % 16 != 0:
        height = ((height // 16) + 1) * 16
    if width % 16 != 0:
        width = ((width // 16) + 1) * 16
    frame_resized = cv2.resize(frame, (width, height))

    # BGR -> RGB，轉成 tensor
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    img = frame_rgb.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

    # 🚀 關鍵修復：如果啟用 GPU，將輸入張量送進 CUDA
    if use_gpu:
        img_tensor = img_tensor.cuda()

    # 模型推論
    with torch.no_grad():
        output = generator(img_tensor, img_tensor)

    # tensor -> numpy
    output = output * 255.0
    output = output.clone().data.permute(0, 2, 3, 1)
    
    # 如果在 GPU 上，需要先轉回 CPU 才能變成 NumPy
    if use_gpu:
        output = output.cpu()
        
    output = output.numpy()
    output = np.clip(output, 0, 255).astype(np.uint8)[0]

    # RGB -> BGR，還原原始尺寸
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    output_bgr = cv2.resize(output_bgr, (width_origin, height_origin))

    return output_bgr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video',  type=str, default='input.avi',  help='輸入影片路徑')
    parser.add_argument('--output_video', type=str, default='output.avi', help='輸出影片路徑')
    parser.add_argument('--load_name',    type=str, default='./models/v3_SPA/v3_SPA.pth', help='模型路徑')
    parser.add_argument('--no_gpu',       type=str2bool, default=False, help='是否強制使用 CPU')
    parser.add_argument('--split_view',   type=str2bool, default=True, help='是否輸出左右對比影片')
    
    # 模型參數（保持與 validation.py 一致）
    parser.add_argument('--color',        type=str2bool, default=True)
    parser.add_argument('--burst_length', type=int,      default=1)
    parser.add_argument('--blind_est',    type=str2bool, default=True)
    parser.add_argument('--kernel_size',  type=list,     default=[3])
    parser.add_argument('--sep_conv',     type=str2bool, default=False)
    parser.add_argument('--channel_att',  type=str2bool, default=False)
    parser.add_argument('--spatial_att',  type=str2bool, default=False)
    parser.add_argument('--upMode',       type=str,      default='bilinear')
    parser.add_argument('--core_bias',    type=str2bool, default=False)
    parser.add_argument('--init_type',    type=str,      default='xavier')
    parser.add_argument('--init_gain',    type=float,    default=0.02)
    opt = parser.parse_args()

    # 1. 載入模型與 GPU 判定
    print(f"載入模型：{opt.load_name}")
    generator = utils.create_generator(opt)
    
    # 🚀 關鍵修復：判定是否將模型推入 GPU 運算
    use_cuda = torch.cuda.is_available() and (not opt.no_gpu)
    if use_cuda:
        generator = generator.cuda()
        print("--- 偵測到可用的 GPU，已啟動 CUDA 硬體加速 ---")
    else:
        print("--- 目前使用 CPU 進行運算（速度較慢） ---")
        
    generator.eval()
    print("模型載入成功！")

    # 2. 開啟輸入影片
    cap = cv2.VideoCapture(opt.input_video)
    if not cap.isOpened():
        print(f"無法開啟影片：{opt.input_video}")
        exit()

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"影片資訊：{width}x{height}，{fps} fps，共 {total} 幀")

    # 3. 建立輸出影片（AVI + XVID）
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # 🎬 炫技功能：如果要輸出左右對比，影片輸出的寬度要乘以 2
    if opt.split_view:
        out_width = width * 2
        print("--- 已開啟左右對比模式：左邊為原始畫面，右邊為去雨成果 ---")
    else:
        out_width = width

    out = cv2.VideoWriter(opt.output_video, fourcc, fps, (out_width, height))

    # 4. 影格循環處理
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % 10 == 0 or frame_idx == total: # 減少刷屏頻率，每 10 幀印一次進度
            print(f"處理進度： {frame_idx}/{total} 幀...")

        # 去雨處理
        derained = derain_frame(generator, frame, use_gpu=use_cuda)

        # 🎬 炫技功能：將原圖與去雨圖水平拼接
        if opt.split_view:
            output_frame = np.hstack((frame, derained))
        else:
            output_frame = derained

        # 寫入輸出影片
        out.write(output_frame)

    cap.release()
    out.release()
    print(f"\n✨ 完成！輸出影片已儲存至：{opt.output_video}")