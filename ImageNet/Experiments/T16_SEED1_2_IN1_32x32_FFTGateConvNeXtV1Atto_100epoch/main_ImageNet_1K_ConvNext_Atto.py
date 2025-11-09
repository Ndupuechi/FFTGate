



# %% Imports and Setup


########################################################################################################################
####-------| NOTE 1.A. IMPORTS LIBRARIES | XXX -----------------------------------------------------####################
########################################################################################################################


"""Train ConvNeXt with PyTorch."""




# ✅ === Enable flexible CUDA memory allocation to reduce fragmentation ===
# Must be set before importing torch!
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



# ✅ === Define currect working directory to ensure on right directory ===
import sys
ConvNeXt_PATH = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\ImageNet_1k\ConvNeXt_V1_Atto"
if os.getcwd() != ConvNeXt_PATH:
    os.chdir(ConvNeXt_PATH)
print(f"✅ Current working directory: {os.getcwd()}")

# ✅ Define absolute paths
PROJECT_PATH = ConvNeXt_PATH
MODELS_PATH = os.path.join(ConvNeXt_PATH, "models")
ACTIVATION_PATH = os.path.join(ConvNeXt_PATH, "activation")


# ✅ Ensure necessary paths are in sys.path
for path in [PROJECT_PATH, MODELS_PATH, ACTIVATION_PATH]:
    if path not in sys.path:
        sys.path.append(path)

# ✅ Print updated sys.path for debugging
print("✅ sys.path updated:")
for path in sys.path:
    print("   📂", path)





# ✅ === Standard libraries === 
# import os
import argparse
from tqdm import tqdm
import math
import random
import numpy as np
import time
import psutil   
import platform


# ✅ === PyTorch and related modules === 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# ✅ === torchvision for datasets and transforms
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD



# ✅ === Schedular, Optimizer and loss === 
import torch_optimizer as torch_opt  # Use 'torch_opt' for torch_optimizer
from timm.scheduler import CosineLRScheduler 
from torch.optim.lr_scheduler import OneCycleLR
import utils
from utils import cosine_scheduler
from utils import NativeScalerWithGradNormCount as NativeScaler
from timm.loss import LabelSmoothingCrossEntropy  # 🟡 ADDED to match ConvNeXt paper
from timm.loss import SoftTargetCrossEntropy



# ✅ === Regularization / Augmentations ===
from timm.data import Mixup, FastCollateMixup






########################################################################################################################
####-------| NOTE 1.C. GET DEVICE SPECIFICATION | XXX ----------------------------------------------####################
########################################################################################################################


# ✅ --- Define path to store Device Specifications ---
Device_Specifications_save_paths = {
    "summaryreport_log_history": r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\ImageNet_1k\ConvNeXt_V1_Atto\Results\FFTGate\Logs\Device_Specifications_logs.txt"
}

# ✅ --- Helper to write logs ---
def write_logs_to_file(logs, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(logs) + "\n")
    print(f"📄 Device specs saved to {path}")

# ✅ --- Collect logs here ---
device_logs = []



# ✅ === Get total RAM in bytes, convert to GB ===
total_ram = psutil.virtual_memory().total / (1024 ** 3)
ram_log = f"💻 Total system RAM: {total_ram:.2f} GB"
print(ram_log); device_logs.append(ram_log)

# ✅ === Get total Cores and GPU ===
cpu_log = f"🧠 Logical CPU cores: {os.cpu_count()}"
gpu_count_log = f"🖥️  Detected GPU devices: {torch.cuda.device_count()}"
print(cpu_log); device_logs.append(cpu_log)
print(gpu_count_log); device_logs.append(gpu_count_log)

# ✅ === PyTorch Code to Get GPU Info + Estimate Batch Size (64×64) ===
def get_gpu_info_and_batch_suggestion(resolution=(64, 64)):
    if not torch.cuda.is_available():
        msg = "❌ CUDA is not available. No GPU detected."
        print(msg); device_logs.append(msg)
        return

    device_id = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_id)
    total_mem_MB = torch.cuda.get_device_properties(device_id).total_memory // (1024 ** 2)

    gpu_msg = f"✅ GPU Detected: {device_name}"
    mem_msg = f"✅ Total VRAM: {total_mem_MB} MB"
    print(gpu_msg); device_logs.append(gpu_msg)
    print(mem_msg); device_logs.append(mem_msg)

    # Estimate safe batch size based on resolution and memory
    pixels = resolution[0] * resolution[1]
    memory_per_sample_MB = 0.0003 * pixels  # Empirical
    usable_mem_MB = total_mem_MB * 0.9      # Leave 10% headroom
    est_batch_size = int(usable_mem_MB // memory_per_sample_MB)

    batch_msg = f"✅ Estimated safe batch size for input {resolution[0]}x{resolution[1]}: {est_batch_size}"
    print(batch_msg); device_logs.append(batch_msg)

# Run it
get_gpu_info_and_batch_suggestion(resolution=(64, 64))

# ✅ === environment_line ===
props = torch.cuda.get_device_properties(0)
env_summary = (
    f"🖥️  GPU: {torch.cuda.get_device_name(0)} ({props.total_memory/1e9:.2f} GB)\n"
    f"🔧 PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | cuDNN: {torch.backends.cudnn.version()}\n"
    f"🐍 Python: {sys.version.split()[0]} | OS: {platform.system()} {platform.release()}"
)
print(env_summary)
device_logs.append(env_summary)

# ✅ Write to file
write_logs_to_file(device_logs, Device_Specifications_save_paths["summaryreport_log_history"])








########################################################################################################################
####-------| NOTE 1.C. IMPORTS MODEL AND ACTIVATION FUCNTION (s) | XXX -----------------------------####################
########################################################################################################################




# ✅ Import FFTGate (Check if the module exists)
try:
    from activation.FFTGate import FFTGate  # type: ignore
    print("✅ FFTGate imported successfully!")
except ModuleNotFoundError as e:
    print(f"❌ Import failed: {e}")
    print(f"🔍 Check that 'FFTGate.py' exists inside: {ACTIVATION_PATH}")

# ✅ Test if FFTGate is callable
try:
    _ = FFTGate(num_channels=64)  # simple smoke test
    print("✅ FFTGate instance created successfully!")
except Exception as e:
    print(f"❌ Error while initializing FFTGate: {e}")








# ✅ Now import FFTGate_ConvNeXt (Ensure module exists inside models/)
try:
    from models.FFTGate_ConvNeXt import convnextv1_atto  # <- the file you pasted "instantiate convnextv1_atto"
    print("✅ FFTGate_ConvNeXt imported successfully!")
except ModuleNotFoundError as e:
    print(f"❌ FFTGate_ConvNeXt import failed: {e}")
    print(f"🔍 Check that 'FFTGate_ConvNeXt.py' exists inside: {MODELS_PATH}")





# ✅ Now import FFTGateAdapter (Ensure module exists inside models/)
try:
    from models.FFTGate_ConvNeXt import FFTGateAdapter
    print("✅ FFTGateAdapter imported successfully from FFTGate_ConvNeXt!")
except ModuleNotFoundError as e:
    print(f"❌ FFTGateAdapter import failed: {e}")
    print("🔍 Check that 'FFTGate_ConvNeXt.py' exists inside the 'models/' directory and that it defines FFTGateAdapter.")















########################################################################################################################
####-------| NOTE 1.B. SEEDING FOR REPRODUCIBILITY | XXX -------------------------------------------####################
########################################################################################################################

def set_seed_torch(seed):
    torch.manual_seed(seed)                          



# def set_seed_main(seed):
#     random.seed(seed)                                ## Python's random module
#     np.random.seed(seed)                             ## NumPy's random module
#     torch.cuda.manual_seed(seed)                     ## PyTorch's random module for CUDA
#     torch.cuda.manual_seed_all(seed)                 ## Seed for all CUDA devices
#     torch.backends.cudnn.deterministic = True        ## Ensure deterministic behavior for CuDNN
#     torch.backends.cudnn.benchmark = False           ## Disable CuDNN's autotuning for reproducibility


def set_seed_main(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False   # ✅ Allow faster (but non-deterministic) kernels
    torch.backends.cudnn.benchmark = True        # ✅ Use cuDNN autotuner for best performance







# ✅ ============= Define Seed (Seed1=Seed2=4 is Best seed for CIFAR100 (78.46%)) =============
seed1 = 1
seed2 = 2

# Variable seed for DataLoader shuffling
set_seed_torch(seed1)  

# Variable main seed (model, CUDA, etc.)
set_seed_main(seed2)  





########################################################################################################################
####-------| NOTE 1.G. OUTPUT FILE NAMING CONVENTION| XXX ------------------------------------------####################
########################################################################################################################


# ✅ =============  Used for naming files ============= 
mode_name = "T12_Seed1_2"                           # Options: "Seed1", "Seed2", "Seed1_2" | # Options: "X1", "X2" , "X3"   ✅✅==> ABLATION OR SEED
dataset_name = "ImageNet_1K"               # Options: "ImageNet_100" or "ImageNet_1k" or "ImageNet_1K_DS" or " "    
act_name = "FFTGate"                       # Options: "FFTGate", "GELU", "RELU","TanhExp", "Swish" 















########################################################################################################################
####-------| NOTE 2.1. ARGUMENT PARSER TO GET USER INPUTS | XXX ------------------------------------####################
########################################################################################################################

# Main Execution (Placeholder)
if __name__ == "__main__":
    print("ConvNeXtV1-Atto Training Script Initialized...")
    # Add your training pipeline here






# ✅ ===== String to Boolean Helper for parser uses =====
def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')










parser = argparse.ArgumentParser('ConvNeXt-T training and evaluation script for image classification', add_help=False)
parser.add_argument('--batch_size', default=256, type=int,           # ✅ default in paper "(('--batch_size', default=64, type=int)" | ✅ ✅  default for Test: 256 
                        help='Per GPU batch size')
parser.add_argument('--epochs', default=100, type=int)                # ✅ default in paper "('--epochs', default=300, type=int)"
parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

# ✅ ===== Model parameters =====
parser.add_argument('--model', default='convnextv1_Atto', type=str, metavar='MODEL',
                        help='Name of model to train')
parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',    # ✅ default in paper "('--drop_path', type=float, default=0, metavar='PCT'"" | Tiny → 0.1, Small → 0.4 | Base / Large → 0.5
                        help='Drop path rate (default: 0.0)')

parser.add_argument('--input_size', default=32, type=int,          # ✅ default in paper "('--input_size', default=224, type=int)" | ✅ ✅  default for Test:32
                        help='image input size')


parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")

# ✅ ===== EMA related parameters =====
parser.add_argument('--model_ema', type=str2bool, default=False)
parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

# ✅ ===== Optimization parameters =====
parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')

parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')


parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
    weight decay. We use a cosine schedule for WD and using a larger decay by
    the end of training improves performance for ViTs.""")

parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',             # ✅ default in paper "('--lr', type=float, default=4e-3, metavar='LR')"         
                    help='learning rate (default: 4e-3), with total batch size 4096')
parser.add_argument('--layer_decay', type=float, default=1.0)
parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',      # ✅ default in paper "('--warmup_epochs', type=int, default=20, metavar='N')"  | ✅ ✅ default for Test:5
                    help='epochs to warmup LR, if scheduler supports')                   
parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                    help='num of steps to warmup LR, will overload warmup_epochs if set > 0')



# ✅ ===== Activation Optimization parameters =====
parser.add_argument('--min_act_lr', type=float, default=1e-6)         # 🟡 UPDATED (was default=1e-6))❗️❗️
parser.add_argument('--act_weight_decay', type=float, default=0.0)    # 🟡 UPDATED (was default=1e-4)❗️❗️
parser.add_argument('--unfreeze_act_epochs', default=0, type=int)
parser.add_argument('--Warm_act_epochs', default=0, type=int)   # default=0
parser.add_argument('--act_weight_decay_end', type=float, default=0.0, help="""keep end the same so it's constant""")  # 🟡 UPDATED (was default=1e-4)❗️❗️




# ✅ ===== Augmentation parameters =====
parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')

parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')

parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train_interpolation', type=str, default='bicubic',
                    help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

# ✅ ===== Evaluation parameters =====
parser.add_argument('--crop_pct', type=float, default=None)

# ✅ ===== * Random Erase params =====
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', type=str2bool, default=False,
                    help='Do not random erase first (clean) augmentation split')

# ✅ ===== * Mixup params =====
parser.add_argument('--mixup', type=float, default=0.0,                    # ✅ default in paper: ('--mixup', type=float, default=0.0 )  | ✅ ✅ Default_Latest=0.4 
                    help='mixup alpha, mixup enabled if > 0.')
parser.add_argument('--cutmix', type=float, default=0.0,                   # ✅ default in paper:  ('--mixup', type=float, default=0.0)  | ✅ ✅ Default_Latest=0.4 
                    help='cutmix alpha, cutmix enabled if > 0.')
parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup_prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup_mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

parser.add_argument('--mixup_off_epoch', type=int, default=0,             # ✅ default in paper: mixup stays enabled throughout training when activated | ✅ ✅ Default_Latest=80 
                    help='Epoch after which Mixup/CutMix is disabled')     # ✅ default in paper, if mixup enabled: mixup_off_epoch = epochs 



# ✅ ===== * Finetuning params =====
parser.add_argument('--finetune', default='',
                    help='finetune from checkpoint')

parser.add_argument('--head_init_scale', default=0.001, type=float,         # ✅ default in paper: 0.001
                    help='classifier head initial scale, typically adjusted in fine-tuning')

parser.add_argument('--model_key', default='model|module', type=str,
                    help='which key to load from saved state dict, usually model or model_ema')
parser.add_argument('--model_prefix', default='', type=str)

# ✅ ===== Dataset parameters =====
parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                    help='dataset path')
parser.add_argument('--eval_data_path', default=None, type=str,
                    help='dataset path for evaluation')
parser.add_argument('--nb_classes', default=1000, type=int,
                    help='number of the classification types (set to 100 for ImageNet-100 and 1000 for ImageNet-1K)')    # ✅ default in paper: ('--nb_classes', default=1000, type=int)

parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)

parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'image_folder'],
                    type=str, help='ImageNet dataset path')
parser.add_argument('--output_dir', default='',
                    help='path where to save, empty for no saving')
parser.add_argument('--log_dir', default=None,
                    help='path where to tensorboard log')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--seed', default=0, type=int)

parser.add_argument('--resume', default='',
                    help='resume from checkpoint')
parser.add_argument('--auto_resume', type=str2bool, default=True)
parser.add_argument('--save_ckpt', type=str2bool, default=True)
parser.add_argument('--save_ckpt_freq', default=1, type=int)
parser.add_argument('--save_ckpt_num', default=3, type=int)

parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--eval', type=str2bool, default=False,
                    help='Perform evaluation only')

parser.add_argument('--dist_eval', type=str2bool, default=False,        # ✅ CHANGE FROM "TRUE" to "FALSE": Explicitly disable distributed evaluation for single GPU usage:
                    help='Enabling distributed evaluation')

parser.add_argument('--disable_eval', type=str2bool, default=False,
                    help='Disabling evaluation during training')
# parser.add_argument('--num_workers', default=10, type=int)
parser.add_argument('--num_workers', default=12, type=int)              # ✅ works well:('--num_workers', default=10, type=int) | ✅ ✅  default for Test: 12
parser.add_argument('--pin_mem', type=str2bool, default=True,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

parser.add_argument('--prefetch_factor', default=2, type=int,           # ✅ ---- | ✅ ✅  default for Test: 2
                    help='Number of batches loaded in advance by each worker. Default = 4 (recommended for GPUs with high VRAM).')





# ✅ ===== distributed training parameters =====
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', type=str2bool, default=False)
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')

parser.add_argument('--use_amp', type=str2bool, default=True,           # ✅ default in paper:  ('--use_amp', type=str2bool, default=False)
                    help="Use PyTorch's AMP (Automatic Mixed Precision) or not")

# ✅ ===== Weights and Biases arguments =====
parser.add_argument('--enable_wandb', type=str2bool, default=False,
                    help="enable logging to Weights and Biases")
parser.add_argument('--project', default='convnext', type=str,
                    help="The name of the W&B project where you're sending the new run.")
parser.add_argument('--wandb_ckpt', type=str2bool, default=False,
                    help="Save model checkpoints as W&B Artifacts.")






# ✅ =====  Parse arguments FIRST ===== 
args, unknown = parser.parse_known_args()  # Avoids Jupyter argument issues









# ########################################################################################################################
# ####-------| NOTE 2.2. INITIALIZATION of device = 'cuda' | XXX -------------------------------------####################
# ########################################################################################################################

# device = 'cuda' if torch.cuda.is_available() else 'cpu'



########################################################################################################################
####-------| NOTE 2.2. INITIALIZATION of device = 'cuda' | XXX -------------------------------------####################
########################################################################################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ✅ Enable TF32 for faster training on Ampere/Ada GPUs (safe for ConvNeXt)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True






########################################################################################################################
####-------| NOTE 3. MIX-UP & CUTMIX| XXX ----------------------------------------------------------####################
########################################################################################################################

"""
🟢 mixup + aug_splits = 0 → ✅ works.
🔴 mixup + aug_splits > 0 → ❌ triggers this assert to avoid bugs.
"""


# Mixup and CutMix setup
mixup_fn = None
mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
if mixup_active:
    print(f"✅ Mixup/CutMix activated → mixup={args.mixup}, cutmix={args.cutmix}, minmax={args.cutmix_minmax}")
    mixup_fn = Mixup(
        mixup_alpha=args.mixup, 
        cutmix_alpha=args.cutmix, 
        cutmix_minmax=args.cutmix_minmax,
        prob=args.mixup_prob, 
        switch_prob=args.mixup_switch_prob, 
        mode=args.mixup_mode,
        label_smoothing=args.smoothing, 
        num_classes=args.nb_classes)
else:
    print(f"🔕 Mixup/CutMix disabled → mixup={args.mixup}, cutmix={args.cutmix}, minmax={args.cutmix_minmax}")




# # %%


# ########################################################################################################################
# ####-------| NOTE 3.1. ORGANIZE VAL / into 1000 Class Folders| XXX ---------------------------------####################
# ########################################################################################################################

# # ✅ =========== ORGANIZE VAL ===========

# import os
# import shutil
# import pandas as pd
# from tqdm import tqdm

# # === Path to the CSV file with mappings ===
# csv_path = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\ImageNet_100\ConvNeXt_T\datasets\ILSVRC\LOC_val_solution.csv"

# # === Path to the flat validation images folder ===
# val_dir = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\ImageNet_100\ConvNeXt_T\datasets\ILSVRC\ILSVRC\Data\CLS-LOC\val"

# # === Load mapping from CSV ===
# df = pd.read_csv(csv_path, header=None, names=["ImageID", "ClassInfo"])

# # === Extract only the synset/class ID (e.g., 'n02102040') ===
# df["ClassID"] = df["ClassInfo"].str.split().str[0]

# # === Move each image into its class folder ===
# for _, row in tqdm(df.iterrows(), total=len(df)):
#     image_filename = row['ImageID'] + ".JPEG"
#     class_folder = os.path.join(val_dir, row['ClassID'])

#     src = os.path.join(val_dir, image_filename)
#     dst = os.path.join(class_folder, image_filename)

#     os.makedirs(class_folder, exist_ok=True)

#     if os.path.exists(src):
#         shutil.move(src, dst)
#     else:
#         print(f"⚠️ Missing file: {src}")


# # %%
# # ✅ =========== Clean with script (safe) ===========

# import os
# import re
# import shutil

# val_dir = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\ImageNet_100\ConvNeXt_T\datasets\ILSVRC\ILSVRC\Data\CLS-LOC\val"

# wnid_pattern = re.compile(r"^n\d{8}$")

# for folder in os.listdir(val_dir):
#     full_path = os.path.join(val_dir, folder)
#     if os.path.isdir(full_path) and not wnid_pattern.match(folder):
#         print(f"❌ Deleting invalid folder: {folder}")
#         shutil.rmtree(full_path)








# # %%


# ########################################################################################################################
# ####-------| NOTE 3.1.A. CREATE IMAGNET-100 from IMAGENET-1K| XXX ----------------------------------####################
# ########################################################################################################################

# import os
# import shutil

# # === 1. Configure Paths ===
# imagenet100_txt = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\ImageNet_100\ConvNeXt_T\datasets\imagenet100.txt"

# src_train_dir = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\ImageNet_1k\ConvNeXt_T\datasets\ILSVRC\ILSVRC\Data\CLS-LOC\train"
# src_val_dir   = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\ImageNet_1k\ConvNeXt_T\datasets\ILSVRC\ILSVRC\Data\CLS-LOC\val"

# dst_train_dir = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\ImageNet_100\ConvNeXt_T\datasets\train"
# dst_val_dir   = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\ImageNet_100\ConvNeXt_T\datasets\val"

# # === 2. Create destination folders
# os.makedirs(dst_train_dir, exist_ok=True)
# os.makedirs(dst_val_dir, exist_ok=True)

# # === 3. Read WNIDs
# with open(imagenet100_txt, 'r') as f:
#     classes = [line.strip() for line in f if line.strip()]

# # === 4. Copy folders
# for cls in classes:
#     src_train = os.path.join(src_train_dir, cls)
#     src_val = os.path.join(src_val_dir, cls)
#     dst_train = os.path.join(dst_train_dir, cls)
#     dst_val = os.path.join(dst_val_dir, cls)

#     if os.path.exists(src_train):
#         print(f"Copying train class: {cls}")
#         shutil.copytree(src_train, dst_train, dirs_exist_ok=True)
#     else:
#         print(f"[WARNING] Train folder not found: {cls}")

#     if os.path.exists(src_val):
#         print(f"Copying val class: {cls}")
#         shutil.copytree(src_val, dst_val, dirs_exist_ok=True)
#     else:
#         print(f"[WARNING] Val folder not found: {cls}")







########################################################################################################################
####-------| NOTE 3.2.B. LOAD DATASET: TRAIN AND VALIDATION | XXX ----------------------------------####################
########################################################################################################################




# # ✅ Set your actual path: ⚠️ ImageNet_100
# train_dir = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\ImageNet_1k\ConvNeXt_V1_Atto\datasets\train"
# val_dir = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\ImageNet_1k\ConvNeXt_V1_Atto\datasets\val"


# ✅ Set your actual path: ⚠️ ImageNet_1K
train_dir = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\ImageNet_1k\ConvNeXt_T\datasets\ILSVRC\ILSVRC\Data\CLS-LOC\train"
val_dir = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\ImageNet_1k\ConvNeXt_T\datasets\ILSVRC\ILSVRC\Data\CLS-LOC\val"






# ✅ Define mean and std for transfrom
imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
mean_Transform = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
std_Transform = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD


# ✅ TRAIN TRANSFORMS using timm (matches the paper)
train_transform = create_transform(
    input_size=args.input_size,                   # usually 224
    is_training=True,
    color_jitter=args.color_jitter,               # 0.4 
    auto_augment=args.aa,                         # 'rand-m9-mstd0.5-inc1'
    interpolation=args.train_interpolation,       # 'bicubic'
    re_prob=args.reprob,                          # 0.25
    re_mode=args.remode,                          # 'pixel'
    re_count=args.recount,                        # 1
    mean=mean_Transform,
    std=std_Transform,
)



# ✅ VAL TRANSFORMS (standard ImageNet val preprocessing, matches authors)
if args.crop_pct is None:
    args.crop_pct = 224 / 256
resize_size = int(args.input_size / args.crop_pct)       #  resize_size = int(32 / 0.875) = 36 | Resize → 36×36 → CenterCrop 32×32 → Normalize.

val_transform = transforms.Compose([
    transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(args.input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_Transform, std=std_Transform)
])










# ✅ Create datasets
train_dataset = ImageFolder(train_dir, transform=train_transform)
val_dataset = ImageFolder(val_dir, transform=val_transform)









# ✅ Training loader (shuffle ON)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,               # ✅ try 16, 24, or 32 on your 32-core CPU
    pin_memory=args.pin_mem,                    # ✅ async CPU→GPU copy
    persistent_workers=True,                    # ✅ keep workers alive between epochs
    prefetch_factor=args.prefetch_factor,                   # ✅ keep GPU fed | keeps RAM use sane with many workers | ✅ ✅ default for Test:2
    drop_last=True,
)


# ✅ Validation loader (shuffle OFF)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,               # ✅ try 16, 24, or 32 on your 32-core CPU
    pin_memory=args.pin_mem,                    # ✅ async CPU→GPU copy
    persistent_workers=True,                    # ✅ keep workers alive between epochs
    prefetch_factor=args.prefetch_factor,       # ✅ keep GPU fed | keeps RAM use sane with many workers | ✅ ✅ default for Test:2
    drop_last=False,
)














# ✅ Log info
print(f"✅ Loaded {len(train_dataset)} training images from: {train_dir}")
print(f"✅ Loaded {len(val_dataset)} validation images from: {val_dir}")
print(f"✅ Number of classes: {args.nb_classes}")

# ✅ Define steps per epoch (needed for learning rate schedule)
# num_training_steps_per_epoch = len(train_loader)




# ✅ Length of training and validation datasets
len_train = len(train_dataset)
len_val = len(val_dataset)
print(f"Length of training dataset: {len_train}")
print(f"Length of validation dataset: {len_val}")

# ✅ Number of classes
num_classesx = len(train_dataset.classes)
print(f"Number of classes in ImageNet-{num_classesx}: {num_classesx}")



# ✅ Mean and std used for tranform
print(f"Mean used for train & Test tranform: {mean_Transform}")
print(f"std used for train & Test tranform: {std_Transform}")











########################################################################################################################
####-------| NOTE 3.3. DEBUGGING ARGPARSE PARAMETERS AND USED FOR NAMING FILE| XXX -----------------####################
########################################################################################################################


# Ensure lr and batch size are correctly parsed
lr = args.lr  # Get learning rate from argparse
lr_str = str(lr).replace('.', '_')  # Convert to string and replace '.' for filenames

bs = args.batch_size  # Get batch size from argparse

input_size_str =str(args.input_size) # Get input_size from argparse and convert to string for naming files




# ✅ ===== Debugging prints  =====
print(f"Using device: {device}")
print(f"Parsed learning rate: {lr} (type: {type(lr)})")
print(f"Formatted learning rate for filenames: {lr_str}")
print(f"Parsed batch size: {bs} (type: {type(bs)})")
print(f"Parsed input_size_str: {input_size_str} (type: {type(input_size_str)})")
print(f"Total epochs (args.epochs): {args.epochs} (type: {type(args.epochs)})")
print(f"Starting epoch (args.start_epoch): {args.start_epoch} (type: {type(args.start_epoch)})")


if args.use_amp:
    print("✅ AMP is ENABLED with GradScaler")
else:
    print("⚠️ AMP is DISABLED")




# ✅ Optional: Print confirmation
print(f"✅ Using {args.num_workers} DataLoader workers")
print(f"✅ persistent_workers is {'ENABLED' if args.num_workers > 0 else 'DISABLED'}")







########################################################################################################################
####-------| NOTE 3.4. INITIALIZE TRAINING PARAMETERS | XXX ----------------------------------------####################
########################################################################################################################

# ✅ ===== Initialize training variables  =====
best_acc = 0  # Best test accuracy
start_epoch = args.start_epoch  # ✅ Use parsed start_epoch value (from checkpoint or default 0)






















########################################################################################################################
####-------| NOTE 4. DYNAMIC REGULARIZATION | XXX --------------------------------------------------####################
########################################################################################################################

# 🟢 Global cache variables (initialized once)
cached_batch_std = None
cached_mags = None
"""
If cache_interval=10 → recompute every 10 batches.
If cache_interval=0 → recompute every batch.
"""

# 🔴 TOGGLE FLAGS (set to True/False)
ENABLE_ADAPTIVE_TARGET_REG = True    # 🔴 enable/disable Adaptive Target Regularization
ENABLE_ADAPTIVE_NOISE_REG  = False   # 🔴 enable/disable Adaptive Noise Regularization
ENABLE_GATE_ENTROPY_REG    = False   # 🔴 new: regularize gate activations | turn this OFF for channel-wise | UPDATED (was "True"))❗️❗️

def apply_dynamic_regularization(
    inputs,
    feature_activations,
    epoch,
    prev_params,
    layer_index_map,
    batch_idx,
    num_batches=None,
    cache_interval=0,
    enable_target_reg=ENABLE_ADAPTIVE_TARGET_REG,
    enable_noise_reg=ENABLE_ADAPTIVE_NOISE_REG,
    enable_gate_entropy=ENABLE_GATE_ENTROPY_REG
):
    global activation_layers, cached_batch_std, cached_mags

    epoch_AddNoise = 50
    fftgateparam_reg = 0.0








    # 🔴 Regularization toggles status (print only once per epoch)
    if batch_idx == 0 and (epoch % 10 == 0 or epoch == (unfreeze_activation_epoch + 1)):
        print(
            f"[REG_STATUS] Epoch {epoch} | "
            f"Target Regularization={enable_target_reg} | "
            f"Noise Regularization={enable_noise_reg} | "
            f"Gate Entropy={enable_gate_entropy}"
        )






    # === Batch std (cached) ===
    if cache_interval == 0 or batch_idx % cache_interval == 0 or cached_batch_std is None:
        cached_batch_std = torch.std(inputs.detach()) + 1e-6
        if epoch <= 5 and (batch_idx == 0 or (num_batches is not None and batch_idx == num_batches - 1)):
            print(f"🔄 Recomputed batch_std (Epoch {epoch}, Batch {batch_idx})")

    batch_std = cached_batch_std

    # ✅ === Adaptive Regularization: strength schedule ===
    if epoch < args.warmup_epochs:   # Early warmup phase
        reg_strength = 0.001 
    elif epoch < args.epochs * 0.5: # Mid training
        reg_strength = 0.0005
    else:                           # Final stabilization
        reg_strength = 0.001

    noisy_layers = []
    for idx, layer in enumerate(activation_layers):
        if idx not in layer_index_map:
            continue
        if not (hasattr(layer, "gamma1_raw") and hasattr(layer, "freq_raw")):
            continue

        prev_layer_params = prev_params[layer_index_map[idx]]

        # 1) raw params (trainable, for optimizer & temporal smoothness)
        gamma1_raw = layer.gamma1_raw
        freq_raw   = layer.freq_raw
        

        # 🔒 === Adaptive target regularization === 
        if enable_target_reg:
            reg_target = 0.0
            # 🔵 Smooth temporal consistency (raw params)
            reg_target += compute_target("gamma1", batch_std, gamma1_raw, prev_layer_params["gamma1"], reg_strength)
            reg_target += compute_target("freq_factor", batch_std, freq_raw, prev_layer_params["freq_factor"], reg_strength * 0.5)
            


        

            fftgateparam_reg += reg_target




        # 🔒 === Optional gate entropy regularization ===
        if enable_gate_entropy and feature_activations is not None:
            with torch.no_grad():
                if feature_activations.ndim == 4:
                    feat_mean = feature_activations.mean(dim=(0, 2, 3), keepdim=True)
                elif feature_activations.ndim == 2:
                    feat_mean = feature_activations.mean(dim=0, keepdim=True)
                else:
                    feat_mean = feature_activations.mean()


                gate_snapshot = torch.sigmoid(
                    gamma1_raw.detach().mean() * feat_mean -
                    freq_raw.detach().mean() * (cached_mags if cached_mags is not None else 1.0)
                )
                gate_entropy = -(gate_snapshot * torch.log(gate_snapshot + 1e-6) +
                                 (1 - gate_snapshot) * torch.log(1 - gate_snapshot + 1e-6))
                fftgateparam_reg += 0.001 * gate_entropy.mean()




        # 🔒 === Optional noise regularization ===
        if enable_noise_reg and epoch > epoch_AddNoise:
            current_gamma1 = gamma1_raw
            prev_gamma1    = prev_layer_params["gamma1"]
            param_variation = torch.abs(current_gamma1.detach() - prev_gamma1).mean()
            if param_variation < 0.015:
                noise = (0.0005 + 0.0002 * batch_std.item()) * torch.randn_like(current_gamma1)
                penalty = (current_gamma1 - (prev_gamma1 + noise)).pow(2).mean()
                fftgateparam_reg += 0.00015 * penalty
                noisy_layers.append(f"{idx} (Δ={param_variation.item():.5f})")

    if batch_idx == 0 and epoch <= (epoch_AddNoise + 4) and noisy_layers:
        print(f"🔥 Stable Noise Injected | Epoch {epoch} | Batch {batch_idx} | Layers: " + ", ".join(noisy_layers), flush=True)

    # # === Cache mags ===
    # if cache_interval == 0 or batch_idx % cache_interval == 0 or cached_mags is None:
    #     if feature_activations is None:
    #         cached_mags = None
    #     elif feature_activations.ndim == 4:
    #         cached_mags = feature_activations.detach().abs().mean(dim=(0, 2, 3))
    #     elif feature_activations.ndim == 2:
    #         cached_mags = feature_activations.detach().abs().mean(dim=0)
    #     else:
    #         raise ValueError(f"Unsupported shape for feature_activations: {feature_activations.shape}")

    #     if epoch <= 5 and (batch_idx == 0 or (num_batches is not None and batch_idx == num_batches - 1)):
    #         print(f"🔄 Recomputed mags (Epoch {epoch}, Batch {batch_idx})")

    # if cached_mags is not None and epoch >= 20:                              # UPDATED: Disable mag entropy reg for channel-wise (was "True")❗️❗️ 
    #     m = cached_mags / (cached_mags.sum() + 1e-6)
    #     fftgateparam_reg += 0.001 * (-(m * torch.log(m + 1e-6)).sum())

    return fftgateparam_reg


# ✅ === 🔒 Adaptive Target Regularization ===
def compute_target(param_name, batch_std, param_raw, prev_param_raw, reg_strength):
    prev_param_raw = prev_param_raw.detach()   # ✅ snapshot only
    return reg_strength * (param_raw - prev_param_raw).pow(2).mean()













########################################################################################################################
####-------| NOTE 5.1. INITIALIZE MODEL | XXX ------------------------------------------------------####################
########################################################################################################################



# ✅ Build model from scratch (Table 1 style)
print('==> Building model..')



# # ✅ ======  ConvNeXt V1 Atto Model ====== 
# # From scratch (Table 1 upper style): ⚠️ ImageNet-1K
# net = convnextv1_atto(pretrained=False, num_classes=1000); net1 = 'ConvNeXtV1_ATTO'  # from scratch
# print(f"✅ Model head: {net.head}")  # <--- ✅ This line shows output layer






# ✅ ====== ConvNeXt V1 Atto Model ======
# From scratch (Table 1 style): ⚠️ ImageNet-1K
net = convnextv1_atto(
    pretrained=False,
    num_classes=args.nb_classes,             # ✅ parser: default = 1000 (ImageNet-1K)
    drop_path_rate=args.drop_path,           # ✅ parser: default = 0.1 for Atto
    layer_scale_init_value=args.layer_scale_init_value,  # ✅ parser: default = 1e-6 (stabilizes training)
    head_init_scale=args.head_init_scale     # ✅ parser: default = 0.001 per paper
)
net1 = 'ConvNeXtV1_ATTO'
print(f"✅ Model head: {net.head}")














# # ✅====== For Single GPU ======
# net = net.to(device)
# cudnn.benchmark = True



# ✅====== For Single GPU ======
# Move model to GPU with channels_last memory format for faster conv kernels
net = net.to(device, memory_format=torch.channels_last)
cudnn.benchmark = True






########################################################################################################################
####-------| NOTE 5.2. INITIALIZE LOSS FUNCTION | XXX ----------------------------------------------####################
########################################################################################################################

# ✅ Initialize AMP GradScaler once globally
scaler = torch.cuda.amp.GradScaler()




# ✅ ====== Loss and optimizer ====== 

"""
🧠 Loss Function Selection Summary:

- SoftTargetCrossEntropy():
    🔹 Used when Mixup or CutMix is active.
    🔹 Targets are soft probability distributions (e.g., [0.1, 0.0, 0.9]).
    🔹 Handles soft labels from Mixup/CutMix properly.

- LabelSmoothingCrossEntropy():
    🔹 Used when label smoothing is enabled (args.smoothing > 0.0).
    🔹 Targets are hard labels, but smoothing prevents overconfidence.
    🔹 Suitable for standard training without Mixup.

- nn.CrossEntropyLoss():
    🔹 Standard cross-entropy for hard labels.
    🔹 Used when both Mixup/CutMix and label smoothing are OFF.
"""


if mixup_fn is not None:
    # smoothing is handled with mixup label transform
    criterion = SoftTargetCrossEntropy()
elif args.smoothing > 0.:
    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
else:
    criterion = nn.CrossEntropyLoss()

print("criterion = %s" % str(criterion))














########################################################################################################################
####-------| NOTE 6. MODEL CHECK POINT | XXX -------------------------------------------------------####################
########################################################################################################################


# Ensure directories exist
if not os.path.exists('checkpoint'):
    os.makedirs('checkpoint')

if not os.path.exists('Results'):
    os.makedirs('Results')

# Construct checkpoint path
checkpoint_path = f'./checkpoint/{act_name}_{net1}_{dataset_name}_B{bs}_LR{lr}_{mode_name}_INs{input_size_str}.t7'




# ✅ Suggested correction for resume block
if args.resume:
    print('==> Resuming from checkpoint..')

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint.get('acc', 0.0)
        start_epoch = checkpoint.get('epoch', -1) + 1  # ✅ increment to resume from next epoch
        args.start_epoch = start_epoch  # ✅ sync args.start_epoch for scheduler & loop
        print(f"✅ Checkpoint loaded: {checkpoint_path} | Resuming at epoch {args.start_epoch}")
    else:
        print(f"❌ Error: Checkpoint file not found: {checkpoint_path}")
        start_epoch = 0
        args.start_epoch = 0


















########################################################################################################################
####-------| NOTE 7.1. INITIALIZE MAIN OPTIMIZER SCHEDULER | XXX -----------------------------------####################
########################################################################################################################

# 🟡 NEW: compute ConvNeXt-style per-step schedule (LR & WD) just like upstream repo
total_batch_size = args.batch_size  # (per process). If you use DDP, multiply by world size if needed.
num_training_steps_per_epoch = len(train_loader)  # 🟡 per-iteration schedule uses number of batches

# 🟡 Main scheduler | ConvNeXt cosine LR schedule (vector of length epochs * steps_per_epoch)
print("Use Cosine LR scheduler")
lr_schedule_values = utils.cosine_scheduler(
    args.lr,               # base lr
    args.min_lr,           # final lr
    args.epochs,           # total epochs
    num_training_steps_per_epoch,  # steps per epoch
    warmup_epochs=args.warmup_epochs,
    warmup_steps=args.warmup_steps,
)

# 🟡 ConvNeXt cosine weight-decay schedule; falls back to constant if weight_decay_end is None
if args.weight_decay_end is None:
    args.weight_decay_end = args.weight_decay

wd_schedule_values = utils.cosine_scheduler(
    args.weight_decay,
    args.weight_decay_end,
    args.epochs,
    num_training_steps_per_epoch
)






# 🟡 FFTGate scheduler — exact duplicate of main, just with scaled base LR
# 🟡 Separate FFTGate schedulers for gamma1, freq_factor
print("Use Cosine LR schedulers for gamma1, freq_factor")

# Gamma1 → learns fastest
gamma1_lr_schedule_values = utils.cosine_scheduler(
    args.lr * 2.0,              # 🟡 gamma1 faster (2× base lr)
    args.min_act_lr,
    args.epochs,
    num_training_steps_per_epoch,
    warmup_epochs=args.Warm_act_epochs,
    warmup_steps=args.warmup_steps,
)

# Freq → slower
freq_lr_schedule_values = utils.cosine_scheduler(
    args.lr * 1.0,              # 🟡 freq = base lr
    args.min_act_lr,
    args.epochs,
    num_training_steps_per_epoch,
    warmup_epochs=args.Warm_act_epochs,
    warmup_steps=args.warmup_steps,
)



# Weight-decay schedules (all constant = args.act_weight_decay since you set it to 0.0)
gamma1_wd_schedule_values = utils.cosine_scheduler(
    args.act_weight_decay,
    args.act_weight_decay_end or args.act_weight_decay,
    args.epochs,
    num_training_steps_per_epoch
)

freq_wd_schedule_values = utils.cosine_scheduler(
    args.act_weight_decay,
    args.act_weight_decay_end or args.act_weight_decay,
    args.epochs,
    num_training_steps_per_epoch
)



# 🔎 Debug print
print(f"🟡 Using per-step cosine schedules → "
      f"MainLR[{len(lr_schedule_values)}], MainWD[{len(wd_schedule_values)}], "
      f"Gamma1LR[{len(gamma1_lr_schedule_values)}], FreqLR[{len(freq_lr_schedule_values)}] ")

print(f"🟡 Max/Min MainLR: {max(lr_schedule_values):.6f}/{min(lr_schedule_values):.6f} | "
      f"Max/Min Gamma1LR: {max(gamma1_lr_schedule_values):.6f}/{min(gamma1_lr_schedule_values):.6f} | "
      f"Max/Min FreqLR: {max(freq_lr_schedule_values):.6f}/{min(freq_lr_schedule_values):.6f} ")












########################################################################################################################
####-------| NOTE 7.2. SEARCH FOR FFTGate LAYERS| XXX ----------------------------------------------####################
########################################################################################################################

def find_activations_split(module, activation_layers, gamma1_params, freq_params):
    for layer in module.children():
        if isinstance(layer, FFTGateAdapter) and isinstance(layer.core, FFTGate):
            activation_layers.append(layer.core)                 # 👈 append FFTGate object
            # ⬇️ collect raw trainable params
            gamma1_params.append(layer.core.gamma1_raw)
            freq_params.append(layer.core.freq_raw)
        elif isinstance(layer, nn.Module):
            find_activations_split(layer, activation_layers, gamma1_params, freq_params)








########################################################################################################################
####-------| NOTE 7.3. INITIALIZE MAIN / ACTIVATION PARAMETERS & OPTIMIZERS | XXX ------------------####################
########################################################################################################################

# ---- 1) Collect FFTGate activation parameters (gamma1, freq_factor) from ConvNeXt model (via FFTGateAdapter.core) ----
gamma1_params: list = []
freq_params: list = []
activation_layers: list = []

find_activations_split(net, activation_layers, gamma1_params, freq_params)

# ---- 2) Build MAIN optimizer that EXCLUDES gamma1, freq_factor ----
if gamma1_params or freq_params:
    act_id_set = {id(p) for p in (gamma1_params + freq_params)}  
    main_params = [p for p in net.parameters() if id(p) not in act_id_set]
    optimizer = torch.optim.AdamW(
        main_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=args.opt_eps,
        weight_decay=args.weight_decay
    ); optimizer1 = 'AdamW'
else:
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=args.opt_eps,
        weight_decay=args.weight_decay
    ); optimizer1 = 'AdamW'

# ✅ Sanity check
if gamma1_params or freq_params:
    main_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
    act_ids  = {id(p) for p in (gamma1_params + freq_params)}
    assert main_ids.isdisjoint(act_ids), "gamma1/freq_factor still inside MAIN optimizer!"

# ---- 3) Initialize ACTIVATION optimizers & schedulers (start frozen) ----
activation_optimizers = {}
gamma1_lr_schedule, freq_lr_schedule = None, None  

if gamma1_params or freq_params:
    unfreeze_activation_epoch = args.unfreeze_act_epochs
    WARMUP_ACTIVATION_EPOCHS = args.Warm_act_epochs

    # freeze initially
    for p in gamma1_params + freq_params:
        p.requires_grad = False

    # Separate optimizers only if params exist
    if gamma1_params:
        activation_optimizers["gamma1"] = torch.optim.AdamW(
            gamma1_params, 
            lr=args.lr * 2.0,             # 🟡 gamma1 learns faster
            weight_decay=args.act_weight_decay
        )
        gamma1_lr_schedule = utils.cosine_scheduler(
            args.lr * 2.0,
            args.min_act_lr,
            args.epochs,
            len(train_loader),
            warmup_epochs=WARMUP_ACTIVATION_EPOCHS,
            warmup_steps=args.warmup_steps
        )

    if freq_params:
        activation_optimizers["freq_factor"] = torch.optim.AdamW(
            freq_params, 
            lr=args.lr * 1.0,             # 🟡 freq slower than gamma1
            weight_decay=args.act_weight_decay
        )
        freq_lr_schedule = utils.cosine_scheduler(
            args.lr * 1.0,
            args.min_act_lr,
            args.epochs,
            len(train_loader),
            warmup_epochs=WARMUP_ACTIVATION_EPOCHS,
            warmup_steps=args.warmup_steps
        )



# Final fallback if nothing was found
if not activation_optimizers:
    activation_optimizers = None


# ---- 4) Logging ----
if activation_layers and (gamma1_params or freq_params):
    print(f"✅ Found {len(activation_layers)} FFTGate layers.")
    print(f"✅ Collected {len(gamma1_params)} gamma1 params, "
          f"{len(freq_params)} freq_factor params"
          )
    for idx, layer in enumerate(activation_layers):
        print(f"   🔹 Layer {idx}: {layer}")
elif activation_layers:
    print(f"⚠ Warning: Found {len(activation_layers)} FFTGate layers, but no parameters.")
else:
    print("⚠ Warning: No FFTGate layers found.")

# ---- Main optimizer groups
print("Main Optimizer Groups:")
for i, pg in enumerate(optimizer.param_groups):
    print(f" Group {i} → {len(pg['params'])} params, "
          f"lr={pg['lr']:.6e}, wd={pg['weight_decay']:.6e}")

# ---- Activation optimizer groups
if activation_optimizers is not None:
    for name, opt in activation_optimizers.items():
        print(f"{name.capitalize()} Optimizer Groups:")
        for i, pg in enumerate(opt.param_groups):
            print(f" Group {i} → {len(pg['params'])} {name} params, "
                  f"lr={pg['lr']:.6e}, wd={pg['weight_decay']:.6e}")
else:
    print("⚠ No activation optimizers built yet (will be created at unfreeze).")

















########################################################################################################################
####-------| NOTE 7.4. FINALIZE MAIN AND ACT SCHEDULERs | XXX --------------------------------------####################
########################################################################################################################


# 🟡 Compute starting global step so schedules line up perfectly after resume
global_start_step = args.start_epoch * num_training_steps_per_epoch  # 🟡 aligns with ConvNeXt
print(f"🟡 global_start_step = {global_start_step}")








# 🟡 Helper to set per-step LR and WD on the main optimizer (ConvNeXt-style)
def _apply_lr_wd_at_step(optimizer, step_idx):
    # Clamp index
    if step_idx >= len(lr_schedule_values):
        step_idx = len(lr_schedule_values) - 1

    lr_this_step = lr_schedule_values[step_idx]
    wd_this_step = wd_schedule_values[step_idx]

    for pg in optimizer.param_groups:
        pg["lr"] = lr_this_step
        if "weight_decay" in pg and pg["weight_decay"] is not None:
            pg["weight_decay"] = wd_this_step

    return lr_this_step, wd_this_step


# 🟡 Helper to set per-step LR/WD on gamma1 & freq_factor optimizers
def _apply_single_lr_at_step(key, optimizers, schedule, step_idx):
    if not optimizers or key not in optimizers or schedule is None:
        return None, None

    if step_idx >= len(schedule):
        step_idx = len(schedule) - 1

    lr_this_step = schedule[step_idx]
    wd_this_step = args.act_weight_decay


    for pg in optimizers[key].param_groups:
        pg["lr"] = lr_this_step
        pg["weight_decay"] = wd_this_step
        # keep explicit keys for logging/debugging
        pg["act_lr"] = lr_this_step
        pg["act_weight_decay"] = wd_this_step

    return lr_this_step, wd_this_step



########################################################################################################################
####-------| NOTE 8. DEFINE TRAIN LOOP | XXX -------------------------------------------------------####################
########################################################################################################################



# ✅ === Define path to store LR log === 
LR_save_paths = {
       
    "LR_history": f"C:\\Users\\emeka\\Research\\ModelCUDA\\Big_Data_Journal\\Comparison\\Code\\Paper\\github3\\ImageNet_1k\\ConvNeXt_V1_Atto\\Results\\FFTGate\\{act_name}_{net1}_{dataset_name}_B{bs}_LR{lr}_{optimizer1}_{mode_name}_INs{input_size_str}_LR_log.txt"  # ✅ Training log_history 
}



# ✅ === Define path to store Training log === 
save_paths = {
       
    "log_history": f"C:\\Users\\emeka\\Research\\ModelCUDA\\Big_Data_Journal\\Comparison\\Code\\Paper\\github3\\ImageNet_1k\\ConvNeXt_V1_Atto\\Results\\FFTGate\\{act_name}_{net1}_{dataset_name}_B{bs}_LR{lr}_{optimizer1}_{mode_name}_INs{input_size_str}_training_logs.txt"  # ✅ Training log_history 
}




# ✅ === Training  ===
def train(epoch, net, train_loader, device, criterion, optimizer, activation_optimizers, unfreeze_activation_epoch, global_start_step):
    global train_loss_history, best_train_acc, prev_params, recent_test_acc, gamma1_history, activation_layers, activation_params, test_acc_history, gamma1_params, freq_params   # 🟡 bring the param lists into this scope




    epoch_start_time = time.time()  # ⏱️ Start epoch timer

    # if epoch == 0:
    if epoch == args.start_epoch:      # 🟡 initialize histories on the *first* actual epoch we run
        train_loss_history = []
        best_train_acc = 0.0
        recent_test_acc = 0.0
        gamma1_history = {}        # ✅ Initialize history 
        test_acc_history = []      # ✅ test accuracy history




    # 🟡 Initialize prev_params once, with detached snapshots | BEGINNING OF TRAINING (before epochs start) 
    prev_params = {}
    layer_index_map = {idx: idx for idx in range(len(activation_layers))}
    
    for idx, layer in enumerate(activation_layers):
        prev_params[idx] = {
            "gamma1": layer.gamma1_raw.clone().detach(),  # clone THEN detach
            "freq_factor": layer.freq_raw.clone().detach()
        }












    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_accuracy = 0.0

    # ✅ Initialize log history
    log_history = []

    # ✅ Initialize LR log list for per-batch LR tracking
    lr_log_history = []






    # 🟡 Compute this epoch's base step index for per-step schedule (ConvNeXt-style)
    epoch_base_step = global_start_step + (epoch - args.start_epoch) * num_training_steps_per_epoch



    # ✅ Unfreeze at specified epoch
    if epoch == unfreeze_activation_epoch:
        print("\n🔓 Unfreezing FFTGate Parameters 🔓")
        for p in (gamma1_params + freq_params):   
            p.requires_grad = True

        # rebuild optimizers if frozen
        if gamma1_params:
            activation_optimizers["gamma1"] = torch.optim.AdamW(
                gamma1_params, lr=0.0, weight_decay=args.act_weight_decay
            )
        if freq_params:
            activation_optimizers["freq_factor"] = torch.optim.AdamW(
                freq_params, lr=0.0, weight_decay=args.act_weight_decay
            )


        # ✅ Sync schedulers immediately
        step_idx = epoch_base_step
        g_lr, g_wd = _apply_single_lr_at_step("gamma1", activation_optimizers, gamma1_lr_schedule_values, step_idx)
        f_lr, f_wd = _apply_single_lr_at_step("freq_factor", activation_optimizers, freq_lr_schedule_values, step_idx)


        print(f"✅ Gamma1 Optimizer Synced! LR={g_lr if g_lr else 0:.6e}, WD={g_wd if g_wd else 0:.6f}")
        print(f"✅ FreqFactor Optimizer Synced! LR={f_lr if f_lr else 0:.6e}, WD={f_wd if f_wd else 0:.6f}")






    activation_history = []  # 🔴 Initialize empty history at start of epoch (outside batch loop)







    # ✅ Training Loop
    with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}") as progress:
        for batch_idx, (inputs, targets) in progress:


            # ✅ Compute step
            step_idx = epoch_base_step + batch_idx



            # ✅ Apply per-step LR/WD (main + FFTGate)
            lr_now, wd_now = _apply_lr_wd_at_step(optimizer, step_idx)
            g_lr, g_wd = _apply_single_lr_at_step("gamma1", activation_optimizers, gamma1_lr_schedule_values, step_idx)
            f_lr, f_wd = _apply_single_lr_at_step("freq_factor", activation_optimizers, freq_lr_schedule_values, step_idx)


            # ✅ Log LR/WD/FFTGate every 1000 steps (or last batch)
            if batch_idx % 1000 == 0 or batch_idx == len(train_loader) - 1:
                lr_log_msg = (
                    f"[Epoch {epoch} | Batch {batch_idx} | Global Step {step_idx}] "
                    f"Main LR: {lr_now:.8f} | WD: {wd_now:.8f} | "
                    f"Gamma1 LR: {g_lr if g_lr is not None else 'FROZEN'} | WD: {g_wd if g_wd is not None else 'FROZEN'} | "
                    f"Freq LR: {f_lr if f_lr is not None else 'FROZEN'} | WD: {f_wd if f_wd is not None else 'FROZEN'} "
                )
                lr_log_history.append(lr_log_msg)










            # ✅ Use channels_last layout for inputs to match model
            inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)






            # ✅ Apply Mixup/CutMix only before mixup_off_epoch
            if mixup_fn is not None and epoch < args.mixup_off_epoch:  # 🟢 Apply Mixup/CutMix here
                if inputs.size(0) % 2 == 0:
                    inputs, targets = mixup_fn(inputs, targets)
                else:
                    print(f"⚠️ Skipping Mixup: Batch size {inputs.size(0)} is not even.")




            # ✅ Log only once when mixup is disabled
            if epoch == args.mixup_off_epoch and batch_idx == 0:       # Log when mixup is turned off (only once)
                # print(f"{epoch} -- 🔕 Mixup/CutMix disabled after epoch ")   
                log_msg = f"{epoch} -- 🔕 Mixup/CutMix disabled after epoch"
                print(log_msg)
                log_history.append(log_msg)  # ✅ Save to history






            # ✅ ⬇️⬇️ NEW: Dynamically select loss based on mixup and smoothing ⬇️⬇️
            if epoch >= args.mixup_off_epoch:
                if args.smoothing > 0:
                    loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
                else:
                    loss_fn = nn.CrossEntropyLoss()
            else:
                loss_fn = criterion  # likely SoftTargetCrossEntropy

            # 🧪 Sanity check (optional, for debugging)
            if isinstance(loss_fn, SoftTargetCrossEntropy):
                assert targets.ndim == 2, "SoftTargetCrossEntropy expects soft labels with shape [B, num_classes]"
            else:
                assert targets.ndim == 1, "CrossEntropyLoss expects hard labels (class indices)"
            # ✅ ⬆️⬆️ END of new block ⬆️⬆️







            optimizer.zero_grad()

            # zero_grad activation parameter
            for act_opt in activation_optimizers.values():
                act_opt.zero_grad()








            # ✅ Forward Pass + Loss (⚠️ 🔄 AMP-friendly; scaler handles mixed precision internally)
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs, feature_activations = net(inputs, epoch=epoch)
                    loss = loss_fn(outputs, targets)
            else:
                outputs, feature_activations = net(inputs, epoch=epoch)
                loss = loss_fn(outputs, targets)











            # ✅ Compute Training Accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)


            # ✅ Convert soft labels to hard class indices if needed (e.g., Mixup active) | 🔄 Handle both hard and soft targets
            if targets.ndim == 2:
                # 🧪 Mixup/CutMix targets: convert to hard labels
                targets = targets.argmax(dim=1)


            correct += predicted.eq(targets).sum().item()


            train_accuracy = 100. * correct / total if total > 0 else 0.0  # Compute training accuracy






            # 🔍 Debug before regularization
            if batch_idx % 1500 == 0 and epoch <= 1:
                print(f"[Epoch {epoch} | Batch {batch_idx}] BEFORE reg")
                for name, p in net.named_parameters():
                    if name.endswith("act.core.gamma1_raw") or name.endswith("act.core.freq_raw"):
                        print(f"  {name}: value={p.data.item():.4f}")





            # ✅ Call Regularization Function for the Activation Parameter (gamma1 + freq_factor)
            if epoch >= unfreeze_activation_epoch:                
                fftgate_reg = apply_dynamic_regularization(
                    inputs, feature_activations, epoch,
                    prev_params, layer_index_map, batch_idx,
                    num_batches=len(train_loader)   # 🟡 pass the total #batches for safe prints
                )
                loss += fftgate_reg





                # 🔍 Debug after regularization
                if batch_idx % 1500 == 0 and epoch <= 1:
                    print(f"[Epoch {epoch} | Batch {batch_idx}] AFTER reg | fftgate_reg={fftgate_reg.item():.6f}")
                    for name, p in net.named_parameters():
                        if name.endswith("act.core.gamma1_raw") or name.endswith("act.core.freq_raw"):
                            print(f"  {name}: value={p.data.item():.4f}")







            # # ✅ Backward pass - ⚠️ NO AMP
            # loss.backward()


            # ✅ Backward pass  === ⚠️ 🔄 Use AMP if enabled ===
            if args.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()







            # 🔒 Gradient Clipping for FFTGate params
            if epoch >= unfreeze_activation_epoch:
                for layer in activation_layers:
                    if isinstance(layer, FFTGate):
                        torch.nn.utils.clip_grad_value_(
                            [layer.gamma1_raw, layer.freq_raw],
                            clip_value=0.05   # ✅ safer threshold for stability
                        )









            # 🔍 Debug gradients after backward
            if batch_idx % 1500 == 0 and epoch <= 1:
                print(f"[Epoch {epoch} | Batch {batch_idx}] AFTER backward (gradients)")
                for name, p in net.named_parameters():
                    if name.endswith("act.core.gamma1_raw") or name.endswith("act.core.freq_raw"):
                        if p.grad is not None:
                            print(f"  [Grad] {name}: mean={p.grad.mean().item():.6e}, "
                                  f"|g|mean={p.grad.abs().mean().item():.6e}, |g|max={p.grad.abs().max().item():.6e}")
                        else:
                            print(f"  [Grad] {name}: grad=None")








            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # === Apply Optimizer Step for Model Parameters   === ⚠️ 🔄 Use AMP if enabled===
            # model params
            if args.use_amp:
                scaler.step(optimizer)
            else:
                optimizer.step()

            # activation params (only when unfrozen)
            if epoch >= unfreeze_activation_epoch:
                if "gamma1" in activation_optimizers:
                    if args.use_amp:
                        scaler.step(activation_optimizers["gamma1"])
                    else:
                        activation_optimizers["gamma1"].step()

                if "freq_factor" in activation_optimizers:
                    if args.use_amp:
                        scaler.step(activation_optimizers["freq_factor"])
                    else:
                        activation_optimizers["freq_factor"].step()


            # finally update the scaler once per iteration
            if args.use_amp:
                scaler.update()            
            # ─────────────────────────────────────────────────────────────────────────────────────────────────






       



            # ✅ Accumulate loss
            train_loss += loss.item()



            # # ✅ Clamping of gamma1_raw and freq_raw (Applied AFTER Optimizer Step) | NOTE: Hard clamp = zero gradients once the value hits the bound → optimizer can’t learn smoothly near the boundary.
            # with torch.no_grad():
            #     for layer in activation_layers:
            #         layer.gamma1_raw.clamp_(0.5, 6.0)
            #         layer.freq_raw.clamp_(0.01, 3.0)










            # ✅ Update progress bar live
            progress.set_postfix(
                Train_loss=round(train_loss / (batch_idx + 1), 3),
                Train_acc=train_accuracy,
                LR=f"{lr_now:.8f}",
                WD=f"{wd_now:.8f}",
                Gamma1_LR=f"{g_lr:.8f}" if g_lr is not None else "FROZEN",
                Freq_LR=f"{f_lr:.8f}" if f_lr is not None else "FROZEN"
            )









 


    # ✅  Update prev_params here AFTER all updates | END OF EACH EPOCH (AFTER optimizer.step())
    for idx, layer in enumerate(activation_layers):
        if isinstance(layer, FFTGate):
            prev_params[idx] = {
                "gamma1": layer.gamma1_raw.clone().detach(),   # ✅ detach snapshot only
                "freq_factor": layer.freq_raw.clone().detach()
            }











    # ⏱️ Timing/logging for this epoch
    epoch_end_time = time.time()
    duration = epoch_end_time - epoch_start_time
    mins, secs = divmod(duration, 60)
    print(f"⏱ Epoch {epoch} Training time {act_name}_{net1}: {int(mins)} min {secs:.2f} sec")
















    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Logging Parameters & Gradients (RAW only, no "effective")
    gamma1_param_values = []
    gamma1_grad_values  = []
    freq_param_values   = []
    freq_grad_values    = []


    # Log RAW parameters by name from the module tree
    for name, p in net.named_parameters():
        # γ1_raw logging
        if name.endswith('act.core.gamma1_raw'):
            v = p.data
            gamma1_param_values.append(f"{name}[val={v.item():.6f}]")

            if p.grad is None:
                gamma1_grad_values.append(f"{name}[grad=None]")
            else:
                g = p.grad
                grad_str = f"val={g.item():.6e}, |g|={g.abs().item():.6e}"
                gamma1_grad_values.append(f"{name}[{grad_str}]")

        # freq_raw logging
        if name.endswith('act.core.freq_raw'):
            v = p.data
            freq_param_values.append(f"{name}[val={v.item():.6f}]")

            if p.grad is None:
                freq_grad_values.append(f"{name}[grad=None]")
            else:
                g = p.grad
                grad_str = f"val={g.item():.6e}, |g|={g.abs().item():.6e}"
                freq_grad_values.append(f"{name}[{grad_str}]")



    # grab current LRs/WDs for logging
    current_lr = optimizer.param_groups[0]["lr"]
    current_wd = optimizer.param_groups[0].get("weight_decay", 0.0)

    current_g_lr = activation_optimizers["gamma1"].param_groups[0]["lr"]
    current_g_wd = activation_optimizers["gamma1"].param_groups[0].get("weight_decay", 0.0)

    current_f_lr = activation_optimizers["freq_factor"].param_groups[0]["lr"]
    current_f_wd = activation_optimizers["freq_factor"].param_groups[0].get("weight_decay", 0.0)



    # 🟡 Log message (bias integrated)
    log_msg = (
        f"Epoch {epoch}: M_Optimizer LR => {current_lr:.6f} | WD => {current_wd:.6f} | "
        f"Gamma1 LR => {current_g_lr:.6f} | Gamma1 WD => {current_g_wd:.6f} | "
        f"FreqFactor LR => {current_f_lr:.6f} | FreqFactor WD => {current_f_wd:.6f} | "

        f"Gamma1 RAW: [{', '.join(gamma1_param_values)}] | "
        f"Gamma1 Grad: [{', '.join(gamma1_grad_values)}] | "
        f"Freq RAW: [{', '.join(freq_param_values)}] | "
        f"Freq Grad: [{', '.join(freq_grad_values)}] | "

        f"⏱ Training time => {act_name}_{net1}: {int(mins)} min {secs:.2f} sec"
    )

    log_history.append(log_msg)
    print(log_msg)  # ✅ Prints only once per epoch
    # ────────────────────────────────────────────────────────────────────────────────────────────────






















    # ✅ Initialize log file at the beginning of training (Clear old logs)
    if epoch == args.start_epoch:  # ✅ Only clear at the start of training
        with open(save_paths["log_history"], "w", encoding="utf-8") as log_file:
            log_file.write("")  # ✅ Clears previous logs

    # ✅ Save logs once per epoch (Append new logs)
    if log_history:
        with open(save_paths["log_history"], "a", encoding="utf-8") as log_file:
            log_file.write("\n".join(log_history) + "\n")         # ✅ Ensure each entry is on a new line
        print(f"📜 Logs saved to {save_paths['log_history']}!")  # ✅ Only prints once per epoch
    else:
        print("⚠ No logs to save!")




    # ✅ Save LR log history to file (once per epoch)
    if epoch == args.start_epoch:
        with open(LR_save_paths["LR_history"], "w", encoding="utf-8") as f:
            f.write("")  # Clear previous content on first epoch

    if lr_log_history:
        with open(LR_save_paths["LR_history"], "a", encoding="utf-8") as f:
            f.write("\n".join(lr_log_history) + "\n")
        print(f"📈 LR logs saved to {LR_save_paths['LR_history']}!")
    else:
        print("⚠ No LR logs to save.")







    # ✅ Compute final training accuracy for the epoch
    final_train_loss = train_loss / len(train_loader)
    final_train_acc = 100. * correct / total

    # Append to history
    train_loss_history.append(final_train_loss)
    test_acc_history.append(final_train_acc)  # Track test/train accuracy across epochs






    # ✅ Save training results (without affecting best accuracy tracking)
    train_results_path = f'./Results/Train_{act_name}_{net1}_{dataset_name}_B{bs}_LR{lr}_{optimizer1}_{mode_name}_INs{input_size_str}.txt'

    # ✅ Clear the log file at the start of training (Epoch 0)
    if epoch == args.start_epoch and os.path.exists(train_results_path):
        with open(train_results_path, 'w') as f:
            f.write("")  # ✅ Clears previous logs only once

    # ✅ Append new training results for each epoch
    with open(train_results_path, 'a') as f:
        f.write(f"Epoch {epoch} | Train Loss: {final_train_loss:.3f} | Train Acc: {final_train_acc:.3f}%\n")

    if final_train_acc > best_train_acc:
        best_train_acc = final_train_acc  # ✅ Update best training accuracy
        print(f"🏆 New Best Training Accuracy: {best_train_acc:.3f}% (Updated)")

    # ✅ Append the best training accuracy **only once at the end of training**
    if epoch == (args.epochs - 1):  # Only log once at the final epoch
        with open(train_results_path, 'a') as f:
            f.write(f"\n🏆 Best Training Accuracy: {best_train_acc:.3f}%\n")  

    # ✅ Print both Final and Best Training Accuracy
    print(f"📊 Train Accuracy: {final_train_acc:.3f}% | 🏆 Best Train Accuracy: {best_train_acc:.3f}%")




    print(f"📜 Training logs saved to {train_results_path}!")
    print(f"🏆 Best Training Accuracy: {best_train_acc:.3f}% (Updated)")



    # if epoch % 10 == 0 or epoch == (unfreeze_activation_epoch + 1):    
    #     print(f"📏 Epoch {epoch}: Sizes → ActivationHist: {len(activation_history)} | TestAccHist: {len(test_acc_history)} | TrainLossHist: {len(train_loss_history)}")





    # return final_train_loss, final_train_acc, feature_activations







########################################################################################################################
####-------| NOTE 9. DEFINE TEST LOOP | XXX --------------------------------------------------------####################
########################################################################################################################

# ✅ Initialize total test duration only once (outside the loop)
total_test_duration = 0.0  # 🟡 Accumulate test time across epochs



# ✅ === Define path to store Training log === 
Test_save_paths = {
       
    "Test_log_history": f"C:\\Users\\emeka\\Research\\ModelCUDA\\Big_Data_Journal\\Comparison\\Code\\Paper\\github3\\ImageNet_1k\\ConvNeXt_V1_Atto\\Results\\FFTGate\\{act_name}_{net1}_{dataset_name}_B{bs}_LR{lr}_{optimizer1}_{mode_name}_INs{input_size_str}_test_logs.txt"  # ✅ Training log_history 
}



# ✅ === Testing  ===

def test(epoch, net, val_loader, device, save_results=True):
    """
    Evaluates the model on the test set and optionally saves the results.
    
    Args:
    - epoch (int): The current epoch number.
    - save_results (bool): Whether to save results to a file.

    Returns:
    - acc (float): Test accuracy percentage.
    """
    global best_acc, val_accuracy, test_results_path, total_test_duration  # ✅ This allows accumulation across calls  



    Test_epoch_start_time = time.time()  # ⏱️ Test Start epoch timer


    net.eval()
    test_loss = 0
    correct = 0
    total = 0


    # ✅ Initialize log history
    Test_log_history = []



    # ✅ Select the correct evaluation loss
    """
    The block you pasted was there to handle the case where your training criterion was SoftTargetCrossEntropy (for Mixup/CutMix). 
    But during validation you’re not using Mixup or smoothed soft labels — so you don’t need that conditional.
    """
    test_criterion = nn.CrossEntropyLoss()





    # ✅ Ensure activation function parameters are clamped before evaluation
    with torch.no_grad():
        with tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Testing Epoch {epoch}") as progress:
            for batch_idx, (inputs, targets) in progress:

                # inputs, targets = inputs.to(device), targets.to(device)



                inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
                targets = targets.to(device, non_blocking=True)



                # outputs = net(inputs)

                # # ✅ ⚠️ NO AMP
                # outputs = net(inputs, epoch=0)  # Pass epoch=0 to fix TypeError!



                # ✅ Forward Pass  | ===⚠️ 🔄 Use AMP if enabled ===
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _ = net(inputs, epoch=0)  # ✅ Only use logits for loss
                else:
                    outputs, _ = net(inputs, epoch=0)






                # loss = criterion(outputs, targets)
                # ✅ Use standard classification loss here
                loss = test_criterion(outputs, targets)


                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # ✅ Pass validation accuracy to activation function
                val_accuracy = 100. * correct / total if total > 0 else 0


                # ✅ Update progress bar with loss & accuracy
                progress.set_postfix(Test_loss=round(test_loss / (batch_idx + 1), 3),
                                     Test_acc=round(val_accuracy, 3))










    # ✅ Compute final test accuracy
    final_test_loss = test_loss / len(val_loader)
    final_test_acc = 100. * correct / total







    # ⏱️ Timing/logging for this epoch
    Test_epoch_end_time = time.time()
    Test_duration = Test_epoch_end_time - Test_epoch_start_time

    total_test_duration += Test_duration  # 🟡 Accumulate test time

    mins, secs = divmod(Test_duration, 60)

    # ✅ Create and print test timing log line
    # Test_log_line = f"Epoch {epoch} | ⏱ Test time => {net1}: {int(mins)} min {secs:.2f} sec"

    Test_log_line = (
        f"Epoch {epoch} | Test Loss: {final_test_loss:.3f} | Test Acc: {final_test_acc:.3f}% | "
        f"⏱ Test time => {act_name}_{net1}: {int(mins)} min {secs:.2f} sec"
    )
    
    print(Test_log_line)

    # ✅ Insert test log at the beginning of the list
    Test_log_history.insert(0, Test_log_line)

    # ✅ Initialize log file at the beginning of training (Clear old logs)
    if epoch == args.start_epoch:
        with open(Test_save_paths["Test_log_history"], "w", encoding="utf-8") as Test_log_file:
            Test_log_file.write("")  # Clears previous logs

    # ✅ Save logs once per epoch (Append new logs)
    if Test_log_history:
        with open(Test_save_paths["Test_log_history"], "a", encoding="utf-8") as Test_log_file:
            Test_log_file.write("\n".join(Test_log_history) + "\n")
        print(f"📜 Logs saved to {Test_save_paths['Test_log_history']}!")  # Only prints once per epoch
    else:
        print("⚠ No logs to save!")








    # # ✅ Compute final test accuracy
    # final_test_loss = test_loss / len(val_loader)
    # final_test_acc = 100. * correct / total









    # ✅ Ensure "Results" folder exists (just like training logs)
    results_dir = os.path.join(PROJECT_PATH, "Results")
    os.makedirs(results_dir, exist_ok=True)

    # ✅ Define log file path for test results
    test_results_path = os.path.join(results_dir, f'Test_{act_name}_{net1}_{dataset_name}_B{bs}_LR{lr}_{optimizer1}_{mode_name}_INs{input_size_str}.txt')

    # ✅ Initialize log file at the beginning of training (clear old logs)
    if epoch == args.start_epoch:
        with open(test_results_path, 'w', encoding="utf-8") as f:
            f.write("")  # ✅ Clears previous logs

    # ✅ Append new test results for each epoch (same style as training)
    with open(test_results_path, 'a', encoding="utf-8") as f:
        f.write(f"Epoch {epoch} | Test Loss: {final_test_loss:.3f} | Test Acc: {final_test_acc:.3f}%\n")







    # ✅ Track Best Accuracy for check point saving
    improved = final_test_acc > best_acc

    # ✅ Track Best Accuracy to get Best Test Accuracy
    if final_test_acc > best_acc:
        best_acc = final_test_acc  # ✅ Update best accuracy



    
    # ✅ Save only in the last N epochs AND when accuracy improved
    LAST_N = 20   #✅ Default save: use 20 for normal runs if you prefer (save check point from 80-args.epochs where improvement occurs)
    if improved and epoch >= (args.epochs - LAST_N):    
        print('🏆 Saving best model...')
        state = {
            'net': net.state_dict(),
            'acc': final_test_acc,  # ✅ Ensures the best test accuracy is saved in checkpoint
            'epoch': epoch,
        }




        # Ensure checkpoint directory exists
        checkpoint_dir = "checkpoint"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)


        # ✅ Format learning rate properly before saving filename
        lr_str = str(lr).replace('.', '_')
        checkpoint_path = f'./checkpoint/{act_name}_{net1}_{dataset_name}_B{bs}_LR{lr_str}_{optimizer1}_{mode_name}_INs{input_size_str}.t7'
        torch.save(state, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")








    # ✅ Append the best test accuracy **only once at the end of training**
    if epoch == (args.epochs - 1):
        with open(test_results_path, 'a', encoding="utf-8") as f:
            f.write(f"\n🏆 Best Test Accuracy: {best_acc:.3f}%\n")

            # 🟡 Log total test time at the END
            total_mins, total_secs = divmod(total_test_duration, 60)
            f.write(f"\n🕒 Total Test Time => {act_name}_{net1}: {int(total_mins)} min {total_secs:.2f} sec\n")
            print(f"🕒 Total Test Time => {act_name}_{net1}: {int(total_mins)} min {total_secs:.2f} sec")


    # ✅ Print both Final and Best Test Accuracy (always executed)
    print(f"📊 Test Accuracy: {final_test_acc:.3f}% | 🏆 Best Test Accuracy: {best_acc:.3f}%")
    print(f"📜 Test logs saved to {test_results_path}!")

    


    global recent_test_acc
    recent_test_acc = final_test_acc  # Capture latest test accuracy for next train() call | Store latest test accuracy

    return final_test_acc  # ✅ Return the test accuracy



# %% 

########################################################################################################################
####-------| NOTE 10. TRAIN MODEL WITH SHEDULAR | XXX ----------------------------------------------####################
########################################################################################################################

# ✅ Force pythin to use 'spawn'
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)


    # ✅ Set Seed for Reproducibility BEFORE training starts

    # Variable seed for DataLoader shuffling
    set_seed_torch(seed1)  

    # Variable main seed (model, CUDA, etc.)
    set_seed_main(seed2)  



    # ✅ Optional: Free unused GPU memory BEFORE training starts
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


    # ⏱️ Track total training time outside loop
    training_total_start = time.time()




    # ✅ Training Loop
    for epoch in range(args.start_epoch, args.epochs):   # Runs training for 100 epochs


        # # ✅ Optional: Free unused GPU memory BEFORE training starts | Free leftover memory from previous epoch
        # torch.cuda.empty_cache()
        # torch.cuda.reset_peak_memory_stats()


        # train(epoch, net, train_loader, device, criterion, optimizer, activation_optimizers, activation_schedulers, unfreeze_activation_epoch, main_scheduler, WARMUP_ACTIVATION_EPOCHS) # ✅ Pass required arguments
        train(epoch, net, train_loader, device, criterion, optimizer, activation_optimizers, unfreeze_activation_epoch, global_start_step ) # ✅ Pass required arguments

        test(epoch, net, val_loader, device)  # ✅ Test the model
        tqdm.write("")  # ✅ Clear leftover progress bar from test()


    print("Best Test Accuracy: ", best_acc)


    # ✅ Compute training time 
    training_total_end = time.time()
    total_mins, total_secs = divmod(training_total_end - training_total_start, 60)



    # ✅ Log to training log file (unchanged)
    with open(save_paths["log_history"], "a", encoding="utf-8") as log_file:
        log_file.write(f"\n🕒 Total Training Time => {act_name}_{net1}: {int(total_mins)} min {total_secs:.2f} sec\n")

    # ✅ Log to test results file 
    with open(test_results_path, 'a', encoding="utf-8") as f:
        f.write(f"\n🕒 Total Training Time => {act_name}_{net1}: {int(total_mins)} min {total_secs:.2f} sec\n")
        



    print(f"\n🕒 Total Training Time_{act_name}_{net1}: {int(total_mins)} min {total_secs:.2f} sec")








 # %%
# ########################################################################################################################
# ####-------| NOTE 12. LOAD AND TEST MODEL ACCURACY | XXX -------------------------------------------####################
# ########################################################################################################################


# checkpoint_path = f'./checkpoint/{act_name}_{net1}_{dataset_name}_B{bs}_LR{lr}_{optimizer1}_{mode_name}_INs{input_size_str}.t7'

# # 🔹 Load Checkpoint
# checkpoint = torch.load(checkpoint_path)

# # 🔹 Restore Model Weights
# net.load_state_dict(checkpoint['net'])  
# best_acc = checkpoint['acc']  
# start_epoch = checkpoint['epoch']  

# print("\n✅ Checkpoint successfully loaded!")
# print(f"🔹 Best Accuracy (Saved in Checkpoint): {best_acc:.3f}%")
# print(f"🔹 Last Training Epoch: {start_epoch}")

# # 🔹 Restore Optimizers & Schedulers
# if 'optimizer' in checkpoint:
#     optimizer.load_state_dict(checkpoint['optimizer'])
#     print("🔹 Main Optimizer state restored!")

# if 'scheduler' in checkpoint:
#     main_scheduler.load_state_dict(checkpoint['scheduler'])
#     print("🔹 Main Scheduler state restored!")

# if 'activation_optimizer' in checkpoint:
#     activation_optimizers["gamma1"].load_state_dict(checkpoint['activation_optimizer'])
#     print("🔹 Activation Optimizer restored!")

# if 'activation_scheduler' in checkpoint:
#     activation_schedulers["gamma1"].load_state_dict(checkpoint['activation_scheduler'])
#     print("🔹 Activation Scheduler restored!")

# # 🔹 Run Test After Checkpoint Load
# test_accuracy = test(0)  # Call test function with epoch=0

# # ✅ Compare Results for Debugging
# print("\n🎯 **Checkpoint Test Run Completed**")
# print(f"🔹 Test Accuracy from `test(0)`: {test_accuracy:.3f}%")
# print(f"🔹 **Best Accuracy Saved in Checkpoint**: {best_acc:.3f}%")

# # ⚠ Check for Mismatch in Accuracy
# if abs(test_accuracy - best_acc) > 0.01:  # Small tolerance for floating point errors
#     print(f"⚠ WARNING: Test accuracy mismatch! (Saved: {best_acc:.3f}%, Current Run: {test_accuracy:.3f}%)")
# else:
#     print("✅ No mismatch detected. Checkpoint and test accuracy match!")


# %%    

