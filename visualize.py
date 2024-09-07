import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from utils import get_transforms, save_data
from dataset import CycloneDataset
import datetime
import re
from PIL import Image
import logging

RESULTS_DIR = "/Users/soumilhooda/Desktop/Indian-Tropical-Cyclones/results"

# Ensure RESULTS_DIR exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_loss_curves(train_losses, val_losses, dataset_name):
    try:
        logging.info(f"Plotting loss curves for {dataset_name}...")
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss - {dataset_name}')
        plt.legend()
        save_path = os.path.join(RESULTS_DIR, f'{dataset_name}_loss_curves.png')
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Loss curves saved to {save_path}")
    except Exception as e:
        logging.error(f"Error in plot_loss_curves: {str(e)}")

def plot_actual_vs_predicted(dataset, model, set_name, dataset_name, device, batch_size=32):
    try:
        logging.info(f"Plotting actual vs predicted for {set_name} set ({dataset_name})...")
        model.eval()
        all_actual = []
        all_predicted = []
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for images, speeds in dataloader:
                images = images.to(device)
                predicted = model(images).cpu().numpy()
                all_actual.extend(speeds.numpy())
                all_predicted.extend(predicted)

        plt.figure(figsize=(10, 10))
        plt.scatter(all_actual, all_predicted, alpha=0.5)
        plt.plot([min(all_actual), max(all_actual)], [min(all_actual), max(all_actual)], 'r--')
        plt.xlabel('Actual Speed Intensity')
        plt.ylabel('Predicted Speed Intensity')
        plt.title(f'Actual vs Predicted Speed Intensity - {set_name} ({dataset_name})')
        save_path = os.path.join(RESULTS_DIR, f'{dataset_name}_{set_name}_actual_vs_predicted.png')
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Actual vs Predicted plot saved to {save_path}")

        return all_actual, all_predicted
    except Exception as e:
        logging.error(f"Error in plot_actual_vs_predicted: {str(e)}")
        return [], []

def parse_filename(filename):
    try:
        match = re.match(r'(\d{2}[A-Z]{3}\d{4})_(\d{4})_(\d{3})_(\d)\.jpg', filename)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            speed = int(match.group(3))
            date_time_str = date_str + time_str
            date_time = datetime.datetime.strptime(date_time_str, '%d%b%Y%H%M')
            return date_time, speed
        else:
            return None, None
    except Exception as e:
        logging.error(f"Error in parse_filename: {str(e)}")
        return None, None
    
def plot_cyclone_results(model, cyclone_info, dataset_name, device, batch_size=32):
    try:
        logging.info(f"Plotting cyclone results for {dataset_name}...")
        model.eval()
        transform = get_transforms()

        for cyclone, info in cyclone_info.items():
            try:
                if info['year'] != '2023':
                    continue  # Skip cyclones not from 2023

                images = info['images']
                true_speeds = info['speeds']
                dates = [parse_filename(os.path.basename(img))[0] for img in images]

                dataset = CycloneDataset(images, true_speeds, transform)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

                predicted_speeds = []
                with torch.no_grad():
                    for batch_images, _ in dataloader:
                        batch_images = batch_images.to(device)
                        batch_predictions = model(batch_images)
                        predicted_speeds.extend(batch_predictions.cpu().numpy())

                plt.figure(figsize=(12, 6))
                plt.plot(dates, true_speeds, 'bo-', label='Actual', markersize=4)
                plt.plot(dates, predicted_speeds, 'ro-', label='Predicted', markersize=4)
                plt.xlabel('Date and Time')
                plt.ylabel('Speed Intensity')
                plt.title(f'Cyclone {cyclone} - {dataset_name}')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                save_path = os.path.join(RESULTS_DIR, f'{dataset_name}_{cyclone}_time_series.png')
                plt.savefig(save_path)
                plt.close()
                
                logging.info(f"Saved plot for cyclone {cyclone} at {save_path}")

                # Save the data for this cyclone
                cyclone_data = {
                    'dates': [d.strftime('%Y-%m-%d %H:%M') for d in dates],
                    'true_speeds': true_speeds,
                    'predicted_speeds': predicted_speeds
                }
                save_data(cyclone_data, f"{dataset_name}_{cyclone}_data")

                # Update the cyclone_info dictionary with predicted speeds
                info['predicted_speeds'] = predicted_speeds

            except Exception as e:
                logging.error(f"Error processing cyclone {cyclone}: {str(e)}")

        return cyclone_info  # Return the updated cyclone_info
    except Exception as e:
        logging.error(f"Error in plot_cyclone_results: {str(e)}")
        return cyclone_info

def plot_all_2023_cyclones(cyclone_info, dataset_name):
    try:
        logging.info(f"Plotting all 2023 cyclones for {dataset_name}...")
        
        plt.figure(figsize=(15, 10))
        
        for cyclone, info in cyclone_info.items():
            if info['year'] == '2023':
                dates = [parse_filename(os.path.basename(img))[0] for img in info['images']]
                true_speeds = info['speeds']
                predicted_speeds = info['predicted_speeds']
                
                plt.plot(dates, true_speeds, '-', label=f'{cyclone} (Actual)', markersize=4)
                plt.plot(dates, predicted_speeds, '--', label=f'{cyclone} (Predicted)', markersize=4)
        
        plt.xlabel('Date and Time')
        plt.ylabel('Speed Intensity')
        plt.title(f'All 2023 Cyclones - {dataset_name}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        save_path = os.path.join(RESULTS_DIR, f'{dataset_name}_all_2023_cyclones.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Saved plot for all 2023 cyclones at {save_path}")
    
    except Exception as e:
        logging.error(f"Error in plot_all_2023_cyclones: {str(e)}")

# Add this function call in your main script after plot_cyclone_results
# plot_all_2023_cyclones(updated_cyclone_info, split_type)