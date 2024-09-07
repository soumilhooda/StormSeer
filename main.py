import os
import logging
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau  
from torch.utils.data import DataLoader
import numpy as np
from model import AdvancedCycloneModel
from dataset import CycloneDataset
from utils import get_transforms, setup_logging
from train import train, test

def setup_directories(args):
    global ROOT_DIR, RESULTS_DIR
    ROOT_DIR = args.root_dir
    RESULTS_DIR = args.results_dir if args.results_dir is not None else os.path.join(ROOT_DIR, 'results')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logging.info(f"ROOT_DIR: {ROOT_DIR}")
    logging.info(f"RESULTS_DIR: {RESULTS_DIR}")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_data(root_dir):
    logging.info("Loading data...")
    image_paths, speed_labels = [], []
    for year in os.listdir(root_dir):
        year_path = os.path.join(root_dir, year)
        if os.path.isdir(year_path):
            for cyclone in os.listdir(year_path):
                cyclone_path = os.path.join(year_path, cyclone)
                cyclone_jpg_path = os.path.join(cyclone_path, cyclone + '_JPG')
                if os.path.isdir(cyclone_jpg_path):
                    for img_file in os.listdir(cyclone_jpg_path):
                        if img_file.endswith('.jpg'):
                            img_path = os.path.join(cyclone_jpg_path, img_file)
                            speed = int(img_file.split('_')[2])
                            image_paths.append(img_path)
                            speed_labels.append(speed)
    logging.info(f"Loaded {len(image_paths)} images")
    return image_paths, speed_labels

def prepare_data_loaders(image_paths, speed_labels, split_type):
    logging.info(f"Preparing data loaders for {split_type} split...")
    
    def get_year_from_path(img_path):
        parts = img_path.split(os.sep)
        for part in reversed(parts):
            if part.isdigit() and len(part) == 4:
                return int(part)
        logging.warning(f"Could not extract year from path: {img_path}")
        return None

    if split_type == 'year_wise':
        train_img, val_img, test_img = [], [], []
        train_speed, val_speed, test_speed = [], [], []
        for img, speed in zip(image_paths, speed_labels):
            year = get_year_from_path(img)
            if year is None:
                continue
            if year <= 2020:
                train_img.append(img)
                train_speed.append(speed)
            elif year <= 2022:
                val_img.append(img)
                val_speed.append(speed)
            else:
                test_img.append(img)
                test_speed.append(speed)
    else:
        raise ValueError(f"Invalid split type: {split_type}")

    logging.info(f"Split data: Train: {len(train_img)}, Val: {len(val_img)}, Test: {len(test_img)}")

    transform = get_transforms()
    train_data = CycloneDataset(train_img, train_speed, transform, augment=True)
    val_data = CycloneDataset(val_img, val_speed, transform)
    test_data = CycloneDataset(test_img, test_speed, transform)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

def train_and_evaluate_model(train_loader, val_loader, test_loader, split_type):
    logging.info(f"Training and evaluating model for {split_type} split...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdvancedCycloneModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)  
    criterion = nn.MSELoss()

    model, train_losses, val_losses, val_rmses = train(model, train_loader, val_loader, criterion, optimizer, scheduler, device)

    logging.info(f"Saving {split_type} model...")
    torch.save(model.state_dict(), os.path.join(args.results_dir, f'{split_type}_model.pth'))

    logging.info(f"Testing {split_type} model...")
    test_loss, test_rmse, test_mae, test_r2 = test(model, test_loader, criterion, device)

    logging.info(f"Test Results - Loss: {test_loss:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R2: {test_r2:.4f}")

    return model, (train_losses, val_losses, val_rmses)

def main(args):
    print("Starting run.")
    setup_directories(args)
    setup_logging(os.path.join(args.results_dir, 'experiment_log.txt'))
    set_seed(args.seed)
    
    image_paths, speed_labels = load_data(ROOT_DIR)

    for split_type in ['year_wise']:
        train_loader, val_loader, test_loader = prepare_data_loaders(image_paths, speed_labels, split_type)
        model, losses = train_and_evaluate_model(train_loader, val_loader, test_loader, split_type)

    logging.info("All processes completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tropical Cyclone Intensity Estimation")
    parser.add_argument("--root_dir", type=str, default="/path/to/dataset", help="Root directory of the dataset")
    parser.add_argument("--results_dir", type=str, default="/path/to/results", help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)
