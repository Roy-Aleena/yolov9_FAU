import argparse
import yaml
from wandb_utils import WandbLogger
from utils.general import LOGGER

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'

def load_dataset_config(config_path):
    """Load dataset configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    num_classes = config['nc']
    class_names = config['names']
    
    return num_classes, class_names

def create_dataset_artifact(opt):
    logger = WandbLogger(opt, None, job_type='Dataset Creation')
    
    if not logger.wandb:
        LOGGER.info("Install wandb using `pip install wandb` to log the dataset")
        return

    # Load dataset configuration
    num_classes, class_names = load_dataset_config(opt.data)
    
    # Example log the dataset info to W&B
    logger.wandb.log({
        'dataset_config': {
            'path': opt.data,
            'num_classes': num_classes,
            'class_names': class_names
        }
    })

    LOGGER.info(f"Dataset configuration loaded: {num_classes} classes")
    LOGGER.info(f"Class names: {', '.join(class_names)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--project', type=str, default='YOLOv5', help='name of W&B Project')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', type=str, default='log dataset', help='name of W&B run')

    opt = parser.parse_args()
    opt.resume = False  # Explicitly disallow resume check for dataset upload job

    create_dataset_artifact(opt)
