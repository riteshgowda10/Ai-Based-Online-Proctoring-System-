"""
ExamGuard AI (PRJ 137) - Object Detection Training Pipeline
Fine-tunes YOLOv8n detector on proctoring-specific classes (person, cell_phone, laptop, book, earphone).
Ensures full reproducibility, dataset verification, hyperparameter logging, and model export.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import yaml
import torch
from datetime import datetime
from ultralytics import YOLO

# Local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ml.object_detection.validate_dataset import run_dataset_validation

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
CONFIG_PATH = os.path.join(BASE_DIR, 'configs', 'training.yaml')
DATASET_YAML = os.path.join(BASE_DIR, 'ml', 'object_detection', 'dataset.yaml')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'object_detection')
os.makedirs(MODELS_DIR, exist_ok=True)

def train_object_detector(config_path=CONFIG_PATH, epochs_override=None):
    print("=" * 70)
    print("EXAMGUARD AI — OBJECT DETECTOR TRAINING PIPELINE")
    print("=" * 70)
    
    # 1. Dataset Verification Pre-flight Check
    print("[1/5] Verifying dataset integrity...")
    dataset_dir = os.path.join(BASE_DIR, 'dataset')
    if not run_dataset_validation(dataset_dir):
        raise RuntimeError("Dataset verification failed! Aborting training.")
        
    # 2. Load Configuration
    print("\n[2/5] Loading reproducible training configuration...")
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    obj_cfg = cfg.get('object_detection', {})
    epochs = epochs_override if epochs_override is not None else obj_cfg.get('epochs', 5)
    batch_size = obj_cfg.get('batch_size', 8)
    img_size = obj_cfg.get('image_size', 640)
    lr = obj_cfg.get('learning_rate', 0.005)
    optimizer = obj_cfg.get('optimizer', 'SGD')
    seed = cfg.get('experiment', {}).get('seed', 42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model_path = os.path.join(BASE_DIR, obj_cfg.get('base_model', 'models/yolov8n.pt'))
    
    print(f"  Framework Version : PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()}")
    print(f"  Target Device     : {device.upper()}")
    print(f"  Base Model        : {base_model_path}")
    print(f"  Epochs            : {epochs}")
    print(f"  Batch Size        : {batch_size}")
    print(f"  Image Size        : {img_size}")
    print(f"  Initial LR        : {lr}")
    print(f"  Optimizer         : {optimizer}")
    print(f"  Random Seed       : {seed}")
    
    # 3. Model Initialization
    print("\n[3/5] Initializing base YOLO architecture...")
    model = YOLO(base_model_path)
    
    # 4. Training Execution
    print("\n[4/5] Executing fine-tuning training loop...")
    start_time = time.time()
    
    results = model.train(
        data=DATASET_YAML,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=lr,
        optimizer=optimizer,
        seed=seed,
        device=device,
        project=MODELS_DIR,
        name="proctoring_run",
        exist_ok=True,
        verbose=True,
        save=True
    )
    
    training_duration = time.time() - start_time
    print(f"\n[OK] Training completed in {training_duration:.2f} seconds ({training_duration/60.0:.2f} mins).")
    
    # 5. Checkpoint Management & Best Model Preservation
    print("\n[5/5] Managing model checkpoints...")
    trained_weights_dir = os.path.join(MODELS_DIR, "proctoring_run", "weights")
    best_pt = os.path.join(trained_weights_dir, "best.pt")
    target_best = os.path.join(MODELS_DIR, "best.pt")
    
    if os.path.exists(best_pt):
        # Save best model to canonical location
        import shutil
        shutil.copy2(best_pt, target_best)
        print(f"  Preserved Best Checkpoint: {target_best}")
    else:
        # Fallback to last.pt
        last_pt = os.path.join(trained_weights_dir, "last.pt")
        if os.path.exists(last_pt):
            import shutil
            shutil.copy2(last_pt, target_best)
            print(f"  Preserved Last Checkpoint: {target_best}")
            
    # Record metadata
    meta = {
        'model_name': 'ExamGuard-YOLOv8n-Proctoring',
        'training_date': datetime.now().isoformat(),
        'base_weights': base_model_path,
        'epochs': epochs,
        'batch_size': batch_size,
        'image_size': img_size,
        'learning_rate': lr,
        'optimizer': optimizer,
        'device': device,
        'torch_version': torch.__version__,
        'training_time_seconds': round(training_duration, 2),
        'best_model_path': target_best
    }
    
    meta_path = os.path.join(MODELS_DIR, 'training_metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        yaml.dump(meta, f)
        
    print(f"  Training metadata exported to: {meta_path}")
    print("=" * 70)
    return target_best

if __name__ == '__main__':
    train_object_detector(epochs_override=3) # 3 epochs for fast, verified execution
