import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=100, patience=10):
    logger = logging.getLogger("Training")
    logger.info("Starting training...")
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    val_rmses = []
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for i, (images, speeds) in enumerate(train_loader):
            images, speeds = images.to(device), speeds.to(device).float()
            optimizer.zero_grad()
            predicted_speeds = model(images)
            loss = criterion(predicted_speeds, speeds)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i+1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_loss, val_rmse = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_rmses.append(val_rmse)

        scheduler.step(val_loss)

        logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
            logger.info(f'New best model saved with validation loss: {best_val_loss:.4f}')
        else:
            epochs_without_improvement += 1
            logger.info(f'No improvement in validation loss for {epochs_without_improvement} epochs.')
            if epochs_without_improvement >= patience:
                logger.info(f'Early stopping after {epoch+1} epochs')
                break

    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses, val_rmses

def validate(model, dataloader, criterion, device):
    logger = logging.getLogger("Validation")
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, speeds in dataloader:
            images, speeds = images.to(device), speeds.to(device).float()
            predicted_speeds = model(images)
            loss = criterion(predicted_speeds, speeds)
            total_loss += loss.item()
            all_preds.extend(predicted_speeds.cpu().numpy())
            all_labels.extend(speeds.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    logger.info(f"Validation Loss: {avg_loss:.4f}, RMSE: {rmse:.4f}")
    return avg_loss, rmse

def test(model, test_loader, criterion, device):
    logger = logging.getLogger("Testing")
    logger.info("Starting test...")
    model.eval()
    all_preds = []
    all_true = []
    total_loss = 0.0

    with torch.no_grad():
        for images, speeds in test_loader:
            images, speeds = images.to(device), speeds.to(device).float()
            predicted_speeds = model(images)
            loss = criterion(predicted_speeds, speeds)
            total_loss += loss.item()
            all_preds.extend(predicted_speeds.cpu().numpy())
            all_true.extend(speeds.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    rmse = np.sqrt(mean_squared_error(all_true, all_preds))
    mae = mean_absolute_error(all_true, all_preds)
    r2 = r2_score(all_true, all_preds)

    logger.info(f"Test Results - Loss: {avg_loss:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    return avg_loss, rmse, mae, r2
