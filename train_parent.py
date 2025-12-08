from tkinter import S
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import time
from torch.optim import lr_scheduler
import seaborn as sns
import pandas as pd
import argparse
import os
from dataloader import SurgLoaderP
from loss import *
from tqdm import tqdm
import json
from model import LapFM
from sam2.build_sam import build_sam2
import hydra
from functools import partial
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from torchmetrics.classification import BinaryAccuracy
from monai.losses import DiceCELoss, DiceLoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks import one_hot
import cv2

# torch.set_num_threads(8)
# matplotlib.use('TkAgg')
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def visualize_training_sample(image, pred_mask, gt_mask, epoch, batch_idx, save_dir, image_id=None):
    """
    可视化训练过程中的样本预测结果
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 转换为numpy数组并调整维度
    image_np = image.cpu().squeeze().permute(1, 2, 0).numpy()
    # 反标准化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)
    
    # 预测mask转为numpy
    pred_mask_np = torch.sigmoid(pred_mask).cpu().squeeze().detach().numpy()
    pred_mask_binary = (pred_mask_np > 0.5).astype(np.uint8)
    
    # ground truth mask转为numpy
    gt_mask_np = gt_mask.cpu().squeeze().numpy().astype(np.uint8)
    
    # 从image_id中提取数据集信息
    dataset_name = "Unknown"
    if image_id:
        if 'cholecseg8k' in image_id.lower():
            dataset_name = "CholecSeg8k"
        elif 'dresden' in image_id.lower():
            dataset_name = "Dresden"
        elif 'endo2023' in image_id.lower() or 'endovis2023' in image_id.lower():
            dataset_name = "EndoVis2023"
        elif 'm2caiseg' in image_id.lower():
            dataset_name = "M2CAISeg"
        elif 'autolaparo' in image_id.lower():
            dataset_name = "AutoLaparo"
    
    # 创建可视化图像
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 原图
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Ground Truth
    axes[1].imshow(gt_mask_np, cmap='gray')
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # 预测结果
    im = axes[2].imshow(pred_mask_np, cmap='viridis', vmin=0, vmax=1)
    axes[2].set_title('Prediction (Probability)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    # 添加colorbar
    plt.colorbar(im, ax=axes[2], shrink=0.8)
    
    # 叠加显示（原图+预测轮廓+GT轮廓）
    overlay = image_np.copy()
    if np.any(pred_mask_binary):
        try:
            pred_contours = cv2.findContours(pred_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            overlay = cv2.drawContours((overlay * 255).astype(np.uint8), pred_contours, -1, (0, 255, 0), 2)
        except:
            pass
    
    if np.any(gt_mask_np):
        try:
            gt_contours = cv2.findContours(gt_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            overlay = cv2.drawContours(overlay, gt_contours, -1, (255, 0, 0), 2)
        except:
            pass
    
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (Red: GT, Green: Pred)', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    # 设置主标题，包含更详细的信息
    title_text = f'Epoch {epoch}, Batch {batch_idx}'
    if image_id:
        title_text += f'\nDataset: {dataset_name} | Image ID: {image_id}'
    plt.suptitle(title_text, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # 保存图像，文件名包含数据集信息
    if image_id:
        clean_image_id = image_id.replace('/', '_').replace('\\', '_')
        save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_batch_{batch_idx:03d}_{dataset_name}_{clean_image_id}.png')
    else:
        save_path = os.path.join(save_dir, f'epoch_{epoch:03d}_batch_{batch_idx:03d}_sample.png')
    
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def plot_training_curves(train_history, save_dir):
    """
    绘制训练曲线
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(train_history['loss']) + 1)
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss曲线
    axes[0, 0].plot(epochs, train_history['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Dice Score曲线
    axes[0, 1].plot(epochs, train_history['dice'], 'g-', label='Training Dice Score', linewidth=2)
    axes[0, 1].set_title('Training Dice Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 学习率曲线
    axes[1, 0].plot(epochs, train_history['lr'], 'r-', label='Learning Rate', linewidth=2)
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # 训练时间曲线
    if 'epoch_time' in train_history:
        axes[1, 1].plot(epochs, train_history['epoch_time'], 'm-', label='Epoch Time', linewidth=2)
        axes[1, 1].set_title('Training Time per Epoch')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 保存训练历史到CSV
    df_history = pd.DataFrame(train_history)
    csv_path = os.path.join(save_dir, 'training_history.csv')
    df_history.to_csv(csv_path, index=False)
    print(f"Training history saved to {csv_path}")

def save_training_summary(train_history, save_dir, args):
    """
    保存训练总结信息
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建总结信息
    summary = {
        'Training Configuration': {
            'Dataset': args.dataset,
            'Batch Size': args.batch,
            'Learning Rate': args.lr,
            'Epochs': args.epoch,
            'Segmentation Class': args.seg_cls,
            'Segmentation Channel': args.seg_channel,
            'Pretrain Weight': args.pretrain_weight
        },
        'Training Results': {
            'Final Loss': train_history['loss'][-1] if train_history['loss'] else 'N/A',
            'Best Loss': min(train_history['loss']) if train_history['loss'] else 'N/A',
            'Final Dice Score': train_history['dice'][-1] if train_history['dice'] else 'N/A',
            'Best Dice Score': max(train_history['dice']) if train_history['dice'] else 'N/A',
            'Final Learning Rate': train_history['lr'][-1] if train_history['lr'] else 'N/A',
            'Total Training Time': f"{sum(train_history['epoch_time']):.2f} seconds" if 'epoch_time' in train_history else 'N/A'
        }
    }
    
    # 保存为JSON文件
    summary_path = os.path.join(save_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Training summary saved to {summary_path}")

def train_model(model, optimizer, scheduler, num_epochs=5, args=None):
    
    since = time.time()
    
    best_model_wts = model.state_dict()
    best_loss = float('inf')
    counter = 0
    
    # 创建可视化保存目录
    vis_save_dir = os.path.join('training_visualization', f'channel_{args.seg_channel}_class_{args.seg_cls}')
    sample_vis_dir = os.path.join(vis_save_dir, 'sample_predictions')
    curves_dir = os.path.join(vis_save_dir, 'curves')
    
    # 初始化训练历史记录
    train_history = {
        'loss': [],
        'dice': [],
        'mse': [],
        'lr': [],
        'epoch_time': []
    }
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)  

            running_loss_all = []
            running_loss_mse = []
            running_dice_c1_avg = []

            # 使用tqdm显示进度条
            data_loader_with_progress = tqdm(dataloaders[phase], desc=f'Epoch {epoch}')

            # Iterate over data
            for batch_idx, data_dict in enumerate(data_loader_with_progress):      
                # wrap them in Variable
                img = Variable(data_dict["image"].cuda())
                mask_map = Variable(data_dict["mask"].cuda())
                image_ids = data_dict.get("image_id", [None] * img.size(0))  # 获取image_id

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_mask_c1, pred_iou = model(x=img, p_cls=args.seg_channel, c_cls=args.seg_cls)
                    c1_loss = dice_ce_loss_m(pred_mask_c1, mask_map)
                    score_mask_c1 = accuracy_metric_m(pred_mask_c1, mask_map)
                    mse = mse_loss(pred_iou, score_mask_c1)

                loss = c1_loss + mse

                loss.backward()
                optimizer.step()

                running_loss_all.append(loss.item())
                running_loss_mse.append(mse.item())
                running_dice_c1_avg.append(torch.mean(score_mask_c1).item())
                
                # 更新进度条显示
                data_loader_with_progress.set_postfix({
                    'Loss All': f'{loss.item():.4f}',
                    'Loss MSE': f'{mse.item():.4f}',
                    'Dice': f'{torch.mean(score_mask_c1).item():.4f}'
                })


            epoch_loss = np.mean(running_loss_all)
            epoch_mse = np.mean(running_loss_mse)
            epoch_dice = np.mean(running_dice_c1_avg)
            current_lr = scheduler.get_last_lr()[0]
            epoch_time = time.time() - epoch_start_time

            # 记录训练历史
            train_history['loss'].append(epoch_loss)
            train_history['dice'].append(epoch_mse)
            train_history['mse'].append(epoch_dice)
            train_history['lr'].append(current_lr)
            train_history['epoch_time'].append(epoch_time)

            print('{} Loss ALL: {:.4f} '.format(phase, epoch_loss))
            print('{} Loss MSE: {:.4f} '.format(phase, epoch_mse))
            print('{} Dice Anatomical Structures: {:.4f}'.format(phase, epoch_dice))
            print(f'Epoch Time: {epoch_time:.2f}s')

            # save parameters
            if phase == 'train':
                if epoch_loss <= best_loss:
                    if epoch_loss <= best_loss and epoch > 50:
                        best_loss = epoch_loss
                        if args.continue_learn:
                            existing_weights = torch.load(args.checkpoints)
                            existing_weights['main'].update({k: v for k, v in model.state_dict().items() if not v.requires_grad})
                            existing_weights[f'p_{args.seg_channel}'].update({k: v for k, v in model.state_dict().items() if v.requires_grad})
                            torch.save(existing_weights, f'outputs/SurgicalSAM_{args.seg_channel}_{epoch}.pth')
                        else:
                            best_model_wts = {
                                'main': {k: v for k, v in model.state_dict().items() if not v.requires_grad},
                                f'p_{args.seg_channel}': {k: v for k, v in model.state_dict().items() if v.requires_grad}
                            }
                            torch.save(best_model_wts, f'outputs/SurgicalSAM_{args.seg_channel}_{epoch}.pth')

                scheduler.step()
                print(f"lr: {current_lr}")
                
                # 每10个epoch绘制一次训练曲线
                if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                    plot_training_curves(train_history, curves_dir)
        
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    
    # 保存最终的训练曲线和总结
    plot_training_curves(train_history, curves_dir)
    save_training_summary(train_history, vis_save_dir, args)
    
    return model, train_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='LapBench', help='')
    parser.add_argument('--jsonfile', type=str,default='data_split.json', help='')
    parser.add_argument('--batch', type=int, default=4, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=20, help='epoches')
    parser.add_argument('--seg_cls', type=int, default=0, help='number of classes (including background)')
    parser.add_argument('--seg_channel', type=int, default=1, help='number of classes (including background)')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--pretrain_weight', type=str, default='pretrains/sam2_hiera_large.pt', help='SAM_B_adapter, SAM_H_adapter, SAM_B, SAM_H')
    parser.add_argument('--checkpoints',default='outputs/SurgicalSAM_1_2.pth', type=str, help='')
    parser.add_argument('--continue_learn',default=False, type=bool, help='')
    
    args = parser.parse_args()

    os.makedirs('outputs/', exist_ok=True)

    jsonfile1 = f'dataset/channel_specific_meta_cholecseg8k_split.json'
    with open(jsonfile1, 'r') as f:
        df1 = json.load(f)

    jsonfile2 = f'dataset/channel_specific_meta_dresden_split.json'
    with open(jsonfile2, 'r') as f:
        df2 = json.load(f)

    jsonfile3 = f'dataset/channel_specific_meta_endo2023_split.json'
    with open(jsonfile3, 'r') as f:
        df3 = json.load(f)

    jsonfile4 = f'dataset/channel_specific_meta_m2caiseg_split.json'
    with open(jsonfile4, 'r') as f:
        df4 = json.load(f)
        
    jsonfile5 = f'dataset/channel_specific_meta_autolaparo_t3_split.json'
    with open(jsonfile5, 'r') as f:
        df5 = json.load(f)

    train_files = df1[f'channel_{args.seg_channel}']['train_frame_ids'] + df3[f'channel_{args.seg_channel}']['train_frame_ids'] + df4[f'channel_{args.seg_channel}']['train_frame_ids'] + df1[f'channel_{args.seg_channel}']['valid_frame_ids'] + df3[f'channel_{args.seg_channel}']['valid_frame_ids'] + df4[f'channel_{args.seg_channel}']['valid_frame_ids'] + df5[f'channel_{args.seg_channel}']['valid_frame_ids'] + df5[f'channel_{args.seg_channel}']['valid_frame_ids'] + df2[f'channel_{args.seg_channel}']['train_frame_ids'] + df2[f'channel_{args.seg_channel}']['valid_frame_ids']


    train_dataset = SurgLoaderP(args.seg_channel, args.seg_cls, train_files, A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ], 
        additional_targets={'mask2': 'mask','mask3': 'mask','mask4': 'mask','mask5': 'mask'}))

    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True,drop_last=True, num_workers=args.num_workers, pin_memory=True)

    dataloaders = {'train':train_loader}
    model_cfg = "sam2_hiera_l.yaml"
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module('sam2_configs', version_base='1.2')

    model = LapFM(build_sam2(model_cfg, args.pretrain_weight, mode='train'))


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    if args.continue_learn:
        model.load_state_dict(torch.load(args.checkpoints)['main'], strict=False)

    model = model.cuda()

    for n, value in model.model.sam_prompt_encoder.named_parameters():
        value.requires_grad = False
    for n, value in model.model.image_encoder.named_parameters():
        if f"train_p_" in n:
            value.requires_grad = True
            print(n)
        else:
            value.requires_grad = False
    for n, value in model.model.sam_mask_decoder.named_parameters():
        if f"train_p_" in n:
            value.requires_grad = True
            print(n)
        else:
            value.requires_grad = False

    trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad
    )
    print('Trainable Params = ' + str(trainable_params/1000**2) + 'M')
    total_params = sum(
	param.numel() for param in model.parameters()
    )
    print('Total Params = ' + str(total_params/1000**2) + 'M')
    print('Ratio = ' + str(trainable_params/total_params*100) + '%')

        
    # Loss, IoU and Optimizer
    dice_ce_loss_m = DiceCELoss(include_background=False, to_onehot_y=False, softmax=False, sigmoid=True)
    accuracy_metric_m = DiceEval(include_background=False, to_onehot_y=False, softmax=False, sigmoid=True)
    mse_loss = nn.MSELoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr = args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    
    trained_model, history = train_model(model, optimizer, exp_lr_scheduler, num_epochs=args.epoch, args=args)
