import torch

def save_checkpoint(model, optimizer, epoch, loss, accuracy, filename, t_losses, v_losses):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        't_losses': t_losses,
        'v_losses': v_losses
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")