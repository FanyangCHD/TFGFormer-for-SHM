import os
import time
import argparse
import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
from Utils.dataset import MyDataset
from Utils.SignalProcessing import *
from Utils.utils import *
from Utils.TimeFrequencyLoss import JointTimeFreqLoss
from Network.TFGFormer import *

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--save_path', default='../TFGFormer/HardangerBridge/Experiment', type=str, help='save path')
args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
def seed_all(seed=42):    
    import random
    import torch.backends.cudnn as cudnn
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

total_start_time = time.time()

generator = TFGFormer()
generator.to(device)
generator.apply(weights_init_normal)

#####     Hardanger Bridge Dataset Path            #####
train_input = "D:\Fanyang\SHM_Data\Hardanger Bridge\\benchmark\S25\\train\\feature"
train_label = "D:\Fanyang\SHM_Data\Hardanger Bridge\\benchmark\S25\\train\\label"
val_input = "D:\Fanyang\SHM_Data\Hardanger Bridge\\benchmark\S25\\val\\feature"
val_label = "D:\Fanyang\SHM_Data\Hardanger Bridge\\benchmark\S25\\val\\label"
# train_input = "../SHM_Data/Hardanger Bridge/benchmark/train/feature"
# train_label = "../SHM_Data/Hardanger Bridge/benchmark/train/label"
# val_input = "../SHM_Data/Hardanger Bridge/benchmark/val/feature"
# val_label = "../SHM_Data/Hardanger Bridge/benchmark/val/label"

train_dataset = MyDataset(train_input, train_label)
val_dataset = MyDataset(val_input, val_label)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
optimizer = optim.Adam(generator.parameters(), lr=args.lr)

criterion = JointTimeFreqLoss()

temp_sets1 = []  
temp_sets2 = []  
temp_sets3 = []  
temp_sets4 = []
temp_sets5 = []   

best_val_loss = float("inf")
best_epoch = 0
best_metrics = {} 

gpu_info = setup_gpu_monitoring()
start_time = time.time()

for epoch in range(args.epochs):
    epoch_start_time = time.time()
    seed_all(42)
    lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    train_loss = 0.0
    generator = generator.train()
    
    for batch_idx1, (batch_x, batch_y) in enumerate(train_loader, 0):  
        batch_x = batch_x.to(device=device, dtype=torch.float32)
        batch_y = batch_y.to(device=device, dtype=torch.float32)

        gen_data = generator(batch_x)
        loss_train = criterion(gen_data, batch_y)
   
        train_loss += loss_train.item()
        optimizer.zero_grad()  
        loss_train.backward()
        optimizer.step()

    train_loss = train_loss / (batch_idx1 + 1)  

    generator = generator.eval()
    MAE_set = 0.0
    RMSE_set = 0.0
    RE_set = 0.0
    R2_set = 0.0
    val_loss = 0.0

    for batch_idx2, (val_x, val_y) in enumerate(val_loader, 0):
        val_x = val_x.to(device=device, dtype=torch.float32)
        val_y = val_y.to(device=device, dtype=torch.float32)

        with torch.no_grad(): 
            gen_data2 = generator(val_x)
            loss_val= criterion(gen_data2, val_y)
            MAE = calculate_mae(gen_data2, val_y)
            RMSE = calculate_rmse(gen_data2, val_y)
            RE = calculate_RE(gen_data2, val_y)  
            R2 = calculate_r2(gen_data2, val_y)  

            val_loss += loss_val.item()
            MAE_set += MAE
            RMSE_set += RMSE
            RE_set += RE 
            R2_set += R2 
    
    MAE_set = MAE_set / (batch_idx2 + 1)
    RMSE_set = RMSE_set / (batch_idx2 + 1)
    RE_set = RE_set / (batch_idx2 + 1)
    R2_set = R2_set / (batch_idx2 + 1)
    val_loss = val_loss / (batch_idx2 + 1)
    
    epoch_time = time.time() - epoch_start_time

    loss_set = [train_loss, val_loss]
    temp_sets1.append(loss_set)
    temp_sets2.append(MAE_set)
    temp_sets3.append(RMSE_set)
    temp_sets4.append(RE_set)
    temp_sets5.append(R2_set)

    current_gpu_usage = get_gpu_usage()
    
    print(
            "[Epoch %d/%d] [train_loss: %6f] [val_loss: %6f] [MAE: %6f] [RMSE: %6f] [RE: %6f] [R2: %6f]"
            % (epoch, args.epochs, train_loss, val_loss, MAE_set, RMSE_set, RE_set, R2_set)
        )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        
        best_model_path = os.path.join(args.save_path, 'best_model.pth')
        torch.save(generator, best_model_path)
        
        best_metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'MAE': MAE_set,
            'RMSE': RMSE_set,
            'RE': RE_set,
            'R2': R2_set,
            'epoch_time': epoch_time,
            'gpu_memory': current_gpu_usage.get('memory_allocated', 0)
        }
        print(f"✅ New Best Model! Epoch: {epoch}, Val Loss: {val_loss:.6f}")

final_model_path = os.path.join(args.save_path, 'final_model.pth')
torch.save(generator, final_model_path)

final_metrics = {
    'epoch': args.epochs - 1,
    'train_loss': train_loss,
    'val_loss': val_loss,
    'MAE': MAE_set,
    'RMSE': RMSE_set,
    'RE': RE_set,
    'R2': R2_set
}

total_training_time = time.time() - total_start_time
training_time_minutes = total_training_time / 60
training_time_hours = training_time_minutes / 60

final_gpu_usage = get_gpu_usage()

training_info_path = save_training_info(args.save_path, total_start_time, total_training_time, training_time_minutes, 
                                      training_time_hours, gpu_info, final_gpu_usage, args, 
                                      best_metrics, final_metrics)

def save_metrics_files(save_path, metrics_data, file_names):
    for data, name in zip(metrics_data, file_names):
        file_path = os.path.join(save_path, name)
        np.savetxt(file_path, data, fmt='%.8f')

metrics_data = [temp_sets1, temp_sets2, temp_sets3, temp_sets4, temp_sets5]
file_names = ['loss_sets.txt', 'MAE_sets.txt', 'RMSE_sets.txt', 'RE_sets.txt', 'R2_sets.txt']

save_metrics_files(args.save_path, metrics_data, file_names)

def load_metrics_data(save_path, file_names):
    metrics = {}
    for name in file_names:
        file_path = os.path.join(save_path, name)
        if os.path.exists(file_path):
            metrics[name.replace('.txt', '')] = np.loadtxt(file_path)
    return metrics

metrics_data = load_metrics_data(args.save_path, file_names)

def plot_training_curves(save_path, metrics_data):
    if 'loss_sets' in metrics_data:
        plt.figure(figsize=(10, 6))
        loss_data = metrics_data['loss_sets']
        if loss_data.ndim == 2 and loss_data.shape[1] >= 2:
            plt.plot(loss_data[:, 0], label='Train Loss')
            plt.plot(loss_data[:, 1], label='Val Loss')
        elif loss_data.ndim == 1:
            plt.plot(loss_data, label='Train Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        plt.savefig(os.path.join(save_path, 'Loss.png'), dpi=300, bbox_inches='tight')
        plt.close()

    metric_names = ['MAE_sets', 'RMSE_sets', 'RE_sets', 'R2_sets']
    titles = ['MAE', 'RMSE', 'Relative Error', 'R² Score']
    ylabels = ['MAE', 'RMSE', 'Relative Error', 'R²']
    
    for metric, title, ylabel in zip(metric_names, titles, ylabels):
        if metric in metrics_data:
            plt.figure(figsize=(10, 6))
            data = metrics_data[metric]
            if data.ndim == 1:
                plt.plot(data, label=title)
            elif data.ndim == 2 and data.shape[1] >= 1:
                plt.plot(data[:, 0], label=title)
            plt.xlabel("Epoch")
            plt.ylabel(ylabel)
            plt.legend()
            plt.title(f"{title} over Training")
            plt.savefig(os.path.join(save_path, f'{title.replace(" ", "").replace("²", "2")}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()

plot_training_curves(args.save_path, metrics_data)

if best_metrics:
    print("📊 Best Model Metric (Epoch {}):".format(best_metrics.get('epoch', '未知')))
    print("  - Train Loss (Train Loss): {:.6f}".format(best_metrics.get('train_loss', 0)))
    print("  - Validation loss (Val Loss): {:.6f}".format(best_metrics.get('val_loss', 0)))
    print("  - MAE: {:.6f}".format(best_metrics.get('MAE', 0)))
    print("  - RMSE: {:.6f}".format(best_metrics.get('RMSE', 0)))
    print("  - RE: {:.6f}".format(best_metrics.get('RE', 0)))
    print("  - R²: {:.6f}".format(best_metrics.get('R2', 0)))