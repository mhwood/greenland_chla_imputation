
import torch
import torch.nn as nn
import torch.nn.functional as F



def masked_huber_loss(pred, target, target_mask, ocean_mask, delta=1.0):
    """
    target_mask: 1 where observed chl was hidden for training
    ocean_mask:  1 over ocean, 0 over land
    """

    valid_mask = target_mask * ocean_mask
    train_mask = (1-target_mask) * ocean_mask

    test_loss = F.huber_loss(pred, target, reduction="none", delta=delta)
    test_loss = test_loss * valid_mask
    test_loss = test_loss.sum() / valid_mask.sum().clamp_min(1.0)

    train_loss = F.huber_loss(pred, target, reduction="none", delta=delta)
    train_loss = train_loss * train_mask
    train_loss = train_loss.sum() / train_mask.sum().clamp_min(1.0)

    return train_loss, test_loss



def Unet_model_training_iteration(model, optimizer, batch):

    model.train()

    x = batch["x"]                              # [B, C, 184, 103]
    y = batch["target_log_chl"]                 # [B, 1, 184, 103]
    target_mask = batch["target_mask"]          # [B, 1, 184, 103]
    ocean_mask = batch["ocean_mask"]            # [B, 1, 184, 103]

    pred = model(x, ocean_mask)

    train_loss, test_loss = masked_huber_loss(
        pred=pred,
        target=y,
        target_mask=target_mask,
        ocean_mask=ocean_mask,
    )

    optimizer.zero_grad()
    test_loss.backward()
    optimizer.step()

    return train_loss.item(), test_loss.item()

def Unet_model_evaluation_iteration(model, batch):

    model.eval()

    x = batch["x"]                              # [B, C, 184, 103]
    y = batch["target_log_chl"]                 # [B, 1, 184, 103]
    target_mask = batch["target_mask"]          # [B, 1, 184, 103]
    ocean_mask = batch["ocean_mask"]            # [B, 1, 184, 103]

    with torch.no_grad():
        pred = model(x, ocean_mask)

        loss = masked_huber_loss(
            pred=pred,
            target=y,
            target_mask=target_mask,
            ocean_mask=ocean_mask,
        )

    return loss.item()