import torch
import time
import os

def setup_gpu_monitoring():
    gpu_info = {}
    if torch.cuda.is_available():
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_info = {
                'device_name': pynvml.nvmlDeviceGetName(handle),
                'total_memory': pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3),  # GB
                'monitor_available': True
            }
            pynvml.nvmlShutdown()
        except ImportError:
            gpu_info = {'monitor_available': False, 'device_name': torch.cuda.get_device_name(0)}
    return gpu_info

def get_gpu_usage():
    if torch.cuda.is_available():
        return {
            'memory_allocated': torch.cuda.memory_allocated() / (1024**3),  # GB
            'memory_reserved': torch.cuda.memory_reserved() / (1024**3),    # GB
            'max_memory_allocated': torch.cuda.max_memory_allocated() / (1024**3)  # GB
        }
    return {}


def save_best_metrics(save_path, metrics, filename="best_metrics.txt"):
    
    file_path = os.path.join(save_path, filename)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("🎯 Best Model Evaluation Metrics\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"📅 Training Information:\n")
        f.write(f"  - Best epoch: {metrics['epoch']}\n")
        f.write(f"  - Saved at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("📊 Evaluation Metrics:\n")
        f.write(f"  - Train Loss: {metrics['train_loss']:.8f}\n")
        f.write(f"  - Validation Loss: {metrics['val_loss']:.8f}\n")
        f.write(f"  - MAE: {metrics['MAE']:.8f}\n")
        f.write(f"  - RMSE: {metrics['RMSE']:.8f}\n")
        f.write(f"  - RE: {metrics['RE']:.8f}\n")
        f.write(f"  - R²: {metrics['R2']:.8f}\n\n")
        
        f.write("="*60 + "\n")
        f.write("📁 Related Files:\n")
        f.write(f"  - Model file: best_model.pth\n")
        f.write(f"  - Training loss: loss_sets.txt\n")
        f.write(f"  - MAE records: MAE_sets.txt\n")
        f.write(f"  - RMSE records: RMSE_sets.txt\n")
        f.write(f"  - RE records: RE_sets.txt\n")
        f.write(f"  - R² records: R2_sets.txt\n")
        f.write("="*60)
    
    print(f"💾 Best model metrics have been saved to: {file_path}")



# Function to save training information
def save_training_info(save_path, total_start_time, total_time, total_minutes, total_hours, gpu_info, final_gpu_usage, args, best_metrics, final_metrics):
    """Save training information to a file"""
    file_path = os.path.join(save_path, 'training_info.txt')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("📊 Training Information Summary\n")
        f.write("="*60 + "\n\n")
        
        f.write("📅 Time Information:\n")
        f.write(f"  - Training start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_start_time))}\n")
        f.write(f"  - Training end time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  - Total training time: {total_time:.2f} seconds\n")
        f.write(f"  - Total training time: {total_minutes:.2f} minutes\n")
        f.write(f"  - Total training time: {total_hours:.2f} hours\n\n")
        
        f.write("🖥️ Hardware Information:\n")
        if torch.cuda.is_available():
            f.write(f"  - GPU device: {gpu_info.get('device_name', 'Unknown')}\n")
            f.write(f"  - GPU total memory: {gpu_info.get('total_memory', 0):.2f} GB\n")
            f.write(f"  - Final GPU memory allocated: {final_gpu_usage.get('memory_allocated', 0):.3f} GB\n")
            f.write(f"  - Peak GPU memory allocated: {final_gpu_usage.get('max_memory_allocated', 0):.3f} GB\n")
        f.write("\n")
        
        f.write("⚙️ Training Parameters:\n")
        f.write(f"  - Total epochs: {args.epochs}\n")
        f.write(f"  - Batch size: {args.batch_size}\n")
        f.write(f"  - Learning rate: {args.lr}\n")
        f.write(f"  - Beta1: {args.b1}\n")
        f.write(f"  - Beta2: {args.b2}\n\n")
        
        f.write("🏆 Best Model Performance (Epoch {}):\n".format(best_metrics.get('epoch', 'Unknown')))
        f.write(f"  - Train loss: {best_metrics.get('train_loss', 0):.6f}\n")
        f.write(f"  - Validation loss: {best_metrics.get('val_loss', 0):.6f}\n")
        f.write(f"  - MAE: {best_metrics.get('MAE', 0):.6f}\n")
        f.write(f"  - RMSE: {best_metrics.get('RMSE', 0):.6f}\n")
        f.write(f"  - RE: {best_metrics.get('RE', 0):.6f}\n")
        f.write(f"  - R²: {best_metrics.get('R2', 0):.6f}\n")
        f.write(f"  - Epoch training time: {best_metrics.get('epoch_time', 0):.2f} seconds\n")
        f.write(f"  - GPU memory allocated: {best_metrics.get('gpu_memory', 0):.3f} GB\n\n")
        
        f.write("📈 Final Model Performance (Epoch {}):\n".format(final_metrics.get('epoch', 'Unknown')))
        f.write(f"  - Train loss: {final_metrics.get('train_loss', 0):.6f}\n")
        f.write(f"  - Validation loss: {final_metrics.get('val_loss', 0):.6f}\n")
        f.write(f"  - MAE: {final_metrics.get('MAE', 0):.6f}\n")
        f.write(f"  - RMSE: {final_metrics.get('RMSE', 0):.6f}\n")
        f.write(f"  - RE: {final_metrics.get('RE', 0):.6f}\n")
        f.write(f"  - R²: {final_metrics.get('R2', 0):.6f}\n\n")
        
        f.write("📁 Generated Files:\n")
        f.write("  - Model files:\n")
        f.write(f"    * best_model.pth (best model)\n")
        f.write(f"    * final_model.pth (final model)\n")
        f.write("  - Metric files:\n")
        f.write("    * loss_sets.txt\n")
        f.write("    * MAE_sets.txt\n")
        f.write("    * RMSE_sets.txt\n")
        f.write("    * RE_sets.txt\n")
        f.write("    * R2_sets.txt\n")
        f.write("    * training_info.txt\n")
        f.write("  - Visualization plots:\n")
        f.write("    * Loss.png\n")
        f.write("    * MAE.png\n")
        f.write("    * RMSE.png\n")
        f.write("    * RE.png\n")
        f.write("    * R2.png\n")
        f.write("\n" + "="*60)    
    return file_path