



# %% Imports and Setup


########################################################################################################################
####-------| NOTE 1.A. IMPORTS LIBRARIES | XXX -----------------------------------------------------####################
########################################################################################################################


"""Train CIFAR100 with PyTorch."""

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === Enable flexible CUDA memory allocation to reduce fragmentation ===
# Must be set before importing torch!
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Alternative for memory split limits:
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"


# ✅ === Ensure correct working directory ===
import sys
Project_PATH = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\CIFAR100\VGG16"
if os.getcwd() != Project_PATH:
    os.chdir(Project_PATH)
print(f"✅ Current working directory: {os.getcwd()}")


# ✅ === Define core project paths ===
PROJECT_PATH = Project_PATH
MODELS_PATH = os.path.join(Project_PATH, "models")
ACTIVATION_PATH = os.path.join(Project_PATH, "activation")


# ✅ === Add essential paths to sys.path ===
for path in [PROJECT_PATH, MODELS_PATH, ACTIVATION_PATH]:
    if path not in sys.path:
        sys.path.append(path)

print("✅ sys.path updated:")
for path in sys.path:
    print("   📂", path)
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ======================================================================================================
# ✅ === Core Libraries ===
# ======================================================================================================
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import time
import random
import math
import re
import glob
from copy import deepcopy



# ======================================================================================================
# ✅ === torchvision for datasets and transforms ===
# ======================================================================================================
import torchvision
import torchvision.transforms as transforms


# ======================================================================================================
# ✅ === From utils.py in project folder import cosine_scheduler ===
# ======================================================================================================
import utils as my_utils


# ======================================================================================================
# ✅ === Optimizer and schedulars ===
# ======================================================================================================


import torch_optimizer as torch_opt     # Use 'torch_opt' for torch_optimizer
from timm.scheduler import CosineLRScheduler 
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from timm.loss import LabelSmoothingCrossEntropy










# ======================================================================================================
# ✅ === Import FFTGate and Model ===
# ======================================================================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Import FFTGate (Check if the module exists)
try:
    from activation.FFTGate import FFTGate  # type: ignore
    print("✅ FFTGate imported successfully!")
except ModuleNotFoundError as e:
    print(f"❌ Import failed: {e}")
    print(f"🔍 Check that 'FFTGate.py' exists inside: {ACTIVATION_PATH}")

# ✅ Test if FFTGate is callable
try:
    activation_test = FFTGate()
    print("✅ FFTGate instance created successfully!")
except Exception as e:
    print(f"❌ Error while initializing FFTGate: {e}")

# ✅ Now import FFTGate_VGG (Ensure module exists inside models/)
try:
    from models.FFTGate_VGG import FFTGate_VGG  # type: ignore
    print("✅ FFTGate_VGG imported successfully!")
except ModuleNotFoundError as e:
    print(f"❌ FFTGate_VGG import failed: {e}")
    print(f"🔍 Check that 'FFTGate_VGG.py' exists inside: {MODELS_PATH}")
# ─────────────────────────────────────────────────────────────────────────────────────────────────










########################################################################################################################
####-------| NOTE 1.B. SEEDING FOR REPRODUCIBILITY | XXX -------------------------------------------####################
########################################################################################################################


def set_seed_torch(seed):
    torch.manual_seed(seed)                          ## Controls DataLoader shuffling (torch's RNG)



def set_seed_main(seed):
    random.seed(seed)                                ## Python's random module
    np.random.seed(seed)                             ## NumPy's random module
    torch.cuda.manual_seed(seed)                     ## PyTorch's random module for CUDA
    torch.cuda.manual_seed_all(seed)                 ## Seed for all CUDA devices
    torch.backends.cudnn.deterministic = True        ## Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.benchmark = False           ## Disable CuDNN's autotuning for reproducibility
    torch.backends.cuda.matmul.allow_tf32 = False    # Disable TF32 (strict reproducibility)
    torch.backends.cudnn.allow_tf32 = False          # Disable TF32 (strict reproducibility)



# ✅ ============= Define Seed =============
seed1, seed2 = 1, 2
set_seed_torch(seed1)  
set_seed_main(seed2)  











########################################################################################################################
####-------| NOTE 1.C. DYNAMIC REGULARIZATION | XXX ---------------⚡-------------------------------####################
########################################################################################################################

# 🟢 Global cache variables initailization
cached_batch_std = None
cached_mags = None
"""
If cache_interval=10 → recompute every 10 batches.
If cache_interval=0 → recompute every batch.
"""

# 🔒 TOGGLE FLAGS
ENABLE_ADAPTIVE_TARGET_REG   = True    # enable/disable Adaptive Target Regularization
ENABLE_ADAPTIVE_NOISE_REG    = False   # enable/disable Adaptive Noise Regularization
ENABLE_FFTGATE_ACT_REG       = False   # enable/disable activation-based Reg 

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
    enable_act_reg=ENABLE_FFTGATE_ACT_REG,
):
    global activation_layers, cached_batch_std, cached_mags
    epoch_AddNoise = 50
    fftgateparam_reg = 0.0


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔴 Regularization toggles status (print only once per epoch)
    if batch_idx == 0 and (epoch % 10 == 0 or epoch == (args.unfreeze_act_epochs + 1)):
        print(
            f"[REG_STATUS] Epoch {epoch} | "
            f"Target Regularization={enable_target_reg} | "
            f"Noise Regularization={enable_noise_reg} | "
            f"ActReg={enable_act_reg}"
        )


    # === Batch std (cached) ===
    if cache_interval == 0 or batch_idx % cache_interval == 0 or cached_batch_std is None:
        cached_batch_std = torch.std(inputs.detach()) + 1e-6
        if epoch <= 5 and (batch_idx == 0 or (num_batches is not None and batch_idx == num_batches - 1)):
            print(f"🔄 Recomputed batch_std (Epoch {epoch}, Batch {batch_idx})")

    batch_std = cached_batch_std
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Adaptive Regularization: strength schedule ===
    if epoch < args.warmup_epochs_reg:   
        reg_strength = 0.001         # Early warmup phase                   
    elif epoch < args.epochs * 0.5: 
        reg_strength = 0.0005        # Mid training         
    else:                           
        reg_strength = 0.001         # Final stabilization   

    noisy_layers = []
    for idx, layer in enumerate(activation_layers):
        if idx not in layer_index_map:
            continue
        if not (hasattr(layer, "gamma1_raw") and hasattr(layer, "freq_raw")):
            continue

        prev_layer_params = prev_params[layer_index_map[idx]]

        # ⚙️ raw params (trainable, for optimizer & temporal smoothness)
        gamma1_raw = layer.gamma1_raw
        freq_raw   = layer.freq_raw
        

        # 🔒 === Adaptive target regularization === 
        if enable_target_reg:
            reg_target = 0.0
            # 🔵 Smooth temporal consistency (raw params)
            reg_target += compute_target("gamma1", batch_std, gamma1_raw, prev_layer_params["gamma1"], reg_strength * 1.0)          
            reg_target += compute_target("freq_factor", batch_std, freq_raw, prev_layer_params["freq_factor"], reg_strength * 0.5)  
                    
            fftgateparam_reg += reg_target
        # ─────────────────────────────────────────────────────────────────────────────────────────────────

        # ─────────────────────────────────────────────────────────────────────────────────────────────────
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
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # === Cache mags + activation reg (optional) ===
    if enable_act_reg:
        if cache_interval == 0 or batch_idx % cache_interval == 0 or cached_mags is None:
            if feature_activations is None:
                cached_mags = None
            elif feature_activations.ndim == 4:
                cached_mags = feature_activations.detach().abs().mean(dim=(0, 2, 3))
            elif feature_activations.ndim == 2:
                cached_mags = feature_activations.detach().abs().mean(dim=0)
            else:
                raise ValueError(f"Unsupported shape for feature_activations: {feature_activations.shape}")

            if epoch <= 5 and (batch_idx == 0 or (num_batches is not None and batch_idx == num_batches - 1)):
                print(f"🔄 Recomputed mags (Epoch {epoch}, Batch {batch_idx})")

        if cached_mags is not None and epoch >= 20:                               
            m = cached_mags / (cached_mags.sum() + 1e-6)
            fftgateparam_reg += 0.001 * (-(m * torch.log(m + 1e-6)).sum())

    return fftgateparam_reg
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ✅ === 🔒 Adaptive Target Regularization ===
def compute_target(param_name, batch_std, param_raw, prev_param_raw, reg_strength):
    prev_param_raw = prev_param_raw.detach()   # ✅ snapshot only
    return reg_strength * (param_raw - prev_param_raw).pow(2).mean()
# ─────────────────────────────────────────────────────────────────────────────────────────────────







########################################################################################################################
####-------| NOTE 1. ARGUMENT PARSER | ----------------------------------------------------------------#################
########################################################################################################################

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')

# ✅=== Training parameters ===
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--use_amp', type=bool, default=False, help="Use PyTorch's AMP (Automatic Mixed Precision) or not") # default: True


# ✅=== Data parameters ===
parser.add_argument('--crop_size', type=int, default=32, help='RandomCrop size (default: 32)')
parser.add_argument('--padding', type=int, default=4, help='Padding for RandomCrop (default: 4)')
parser.add_argument('--batch_size', type=int,  default=64, help='Batch size (default: 64)')

# ✅ === DataLoader performance parameters ===
parser.add_argument('--num_workers', type=int, default=5, help='Number of data loading workers (default: 5). Set 0 for debugging.')
parser.add_argument('--pin_mem', type=bool, default=True, help='Use pinned memory for faster host→GPU transfer (default: True).')
parser.add_argument('--prefetch_factor', type=int, default=2, help='Number of batches loaded in advance per worker (default: 2).')
parser.add_argument('--persistent_workers', type=bool, default=True, help='Keep data loader workers alive between epochs for speed (default: True).')
parser.add_argument('--drop_last_trainL', type=bool, default=True, help='Drop last incomplete batch during training (default: True).')
parser.add_argument('--drop_last_testL', type=bool, default=False, help=' (default: False).')

# ✅=== Training state parameters (for checkpointing / resume) ===
parser.add_argument('--epochs', default=100, type=int)             # Default: 100   
parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch number (default: 0)')
parser.add_argument('--best_acc', default=0.0, type=float, help='Best test accuracy so far (default: 0.0)')

# ✅=== Naming Covention / path defination params ===
parser.add_argument('--dataset_name', default="CIFAR100", type=str)
parser.add_argument('--mode_name', default="Seed1_2_EXP2_T512", type=str)
parser.add_argument('--act_name', default="FFTGate", type=str)
parser.add_argument('--model_name', default="VGG16", type=str)
parser.add_argument('--main_opt_name', default="Adam", type=str)

# ✅ ===== Main Optimization parameters =====
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')                           
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay (default: 0.05)')
parser.add_argument('--opt_eps', type=float, default=1e-8, help='Adam optimizer epsilon (default: 1e-8)')
parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
parser.add_argument('--warmup_epochs_reg', default=5, type=int, help='Number of epochs for warmup phase (used in regularization)')

# ✅ ===== Activation Optimization parameters =====
parser.add_argument('--min_act_lr', type=float, default=1e-6)
parser.add_argument('--actbase_lr', type=float, default=1e-2, help="Base learning rate for activation parameters (gamma1, freq)")

# Separate WD for gamma1 and freq (renamed to avoid Jupyter --f conflict)
parser.add_argument('--gamma1_wd', type=float, default=0.0)   # was 0
parser.add_argument('--wd_freq', type=float, default=0.0)     # was 0

parser.add_argument('--unfreeze_act_epochs', default=0, type=int)
parser.add_argument('--Warm_act_epochs', default=0, type=int)

# Constant decay schedule
parser.add_argument('--gamma1_wd_end', type=float, default=0.0)
parser.add_argument('--wd_end_freq', type=float, default=0.0)
parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N', help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

args, unknown = parser.parse_known_args()  # Avoid Jupyter issues




########################################################################################################################
####-------| NOTE 2. DEVICE & TRAINING VARS | ---------------------------------------------------------#################
########################################################################################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ✅ Debugging prints
print(f"Using device: {device}")
print(f"Parsed learning rate: {args.lr}")
print(f"Batch size: {args.batch_size}, Num workers: {args.num_workers}")
print(f"Crop size: {args.crop_size}, Padding: {args.padding}")
print(f"Start epoch: {args.start_epoch}, Best acc: {args.best_acc}")

# ✅ Initialize training variables
best_acc = args.best_acc
start_epoch = args.start_epoch








########################################################################################################################
####-------| NOTE 3. LOAD DATASET | XXX ------------------------------------------------------------####################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Dataset
print('==> Preparing data....')
transform_train = transforms.Compose([
    transforms.RandomCrop(args.crop_size, padding=args.padding),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=args.batch_size, 
    shuffle=True, 
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    persistent_workers=args.persistent_workers,
    prefetch_factor=args.prefetch_factor,
    drop_last=args.drop_last_trainL 
    )


testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, 
    batch_size=args.batch_size, 
    shuffle=False, 
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    persistent_workers=args.persistent_workers,
    prefetch_factor=args.prefetch_factor,
    drop_last=args.drop_last_testL 
    )
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Length of train and test datasets
len_train = len(trainset)
len_test = len(testset)
print(f"Length of training dataset: {len_train} | Length of testing dataset: {len_test}")

# ✅ Print number of classes
num_classes_Print = len(trainset.classes)
print(f"Number of classes in CIFAR-{args.dataset_name}: {num_classes_Print}")
# ─────────────────────────────────────────────────────────────────────────────────────────────────
 







########################################################################################################################
####-------| NOTE 4.1. INITIALIZE MODEL | XXX ------------------------------------------------------####################
########################################################################################################################

print('==> Building model..')

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅  =========== Check GPU availability (raise error if none)  ===========
if not torch.cuda.is_available():
    raise RuntimeError("❌ No GPU detected! CUDA is required for this experiment.")

device = torch.device("cuda")
print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
print(f"   CUDA Device Count: {torch.cuda.device_count()}")
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅  =========== Initialize model only once  ===========
net = FFTGate_VGG('VGG16')
net = net.to(device, memory_format=torch.channels_last)   # Channels-last = faster
torch.backends.cudnn.benchmark = False                    # Fast but reduced reproducibility when "True"
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ ===========  Initialize AMP GradScaler once globall =========== 
scaler = torch.cuda.amp.GradScaler()


# ✅  =========== Define loss  ===========
criterion = nn.CrossEntropyLoss()
# criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)

print(f"✅ Model 'VGG16' initialized on device: {device}")
# ─────────────────────────────────────────────────────────────────────────────────────────────────






########################################################################################################################
####-------| NOTE 4.2. INITAILIZE COSINE SCHEDULERS | XXX ------------------------------------------####################
########################################################################################################################

total_batch_size = args.batch_size
num_training_steps_per_epoch = len(trainloader)
print("🟡 Using cosine LR + WD schedules for for gamma1, freq_factor...")


########################################################################################################################
# === FFTGate sub-optimizers for gamma1 and freq ===
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔸 Learning rate multipliers
lr_gamma1 = args.actbase_lr * 2.0  
lr_freq   = args.actbase_lr * 1.0   

print(f"Gamma1 LR={lr_gamma1}, WD={args.gamma1_wd}")
print(f"Freq   LR={lr_freq}, WD={args.wd_freq}")
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Per-step cosine LR schedules
gamma1_lr_schedule_values = my_utils.cosine_scheduler(
    lr_gamma1,             # 🟡 gamma1 faster (2× base lr)
    args.min_act_lr, 
    args.epochs,
    num_training_steps_per_epoch,
    warmup_epochs=args.Warm_act_epochs,
    warmup_steps=args.warmup_steps
)

freq_lr_schedule_values = my_utils.cosine_scheduler(
    lr_freq,               # 🟡 freq = base lr
    args.min_act_lr, 
    args.epochs,
    num_training_steps_per_epoch,
    warmup_epochs=args.Warm_act_epochs,
    warmup_steps=args.warmup_steps
)

gamma1_wd_schedule_values = my_utils.cosine_scheduler(
    args.gamma1_wd, 
    args.gamma1_wd_end or args.gamma1_wd,
    args.epochs, 
    num_training_steps_per_epoch
)

freq_wd_schedule_values = my_utils.cosine_scheduler(
    args.wd_freq, 
    args.wd_end_freq or args.wd_freq,
    args.epochs, 
    num_training_steps_per_epoch
)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔎 Debug print
print(f"🟡 Using per-step cosine schedules → "
      f"Gamma1LR[{len(gamma1_lr_schedule_values)}], FreqLR[{len(freq_lr_schedule_values)}] ")
print(
      f"Max/Min Gamma1LR: {max(gamma1_lr_schedule_values):.6f}/{min(gamma1_lr_schedule_values):.6f} | "
      f"Max/Min FreqLR: {max(freq_lr_schedule_values):.6f}/{min(freq_lr_schedule_values):.6f} ")
# ─────────────────────────────────────────────────────────────────────────────────────────────────




########################################################################################################################
####-------| NOTE 5. INITIALIZE ACTIVATION PARAMETERS, OPTIMIZERS & SCHEDULERS | XXX ---------------####################
########################################################################################################################

# 1️⃣ =========== Helper: recursively find FFTGate layers and collect trainable params ===========
def find_activations_split(module, activation_layers, gamma1_params, freq_params):
    for layer in module.children():
        # ✅ Direct FFTGate check
        if isinstance(layer, FFTGate):
            activation_layers.append(layer)            # ✅ append FFTGate object directly
            gamma1_params.append(layer.gamma1_raw)     # ✅ collect trainable γ₁ param
            freq_params.append(layer.freq_raw)         # ✅ collect trainable freq param

        # ✅ recurse deeper into children
        elif isinstance(layer, nn.Module):
            find_activations_split(layer, activation_layers, gamma1_params, freq_params)



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣ ===========  Collect all FFTGate layers and params =========== 

# ─────────────────────────────────────────────────────────────────────────────────────────────────
gamma1_params: list = []
freq_params: list = []
activation_layers: list = []


# ✅ ===========  Support both plain and DataParallel model =========== 
find_activations_split(net, activation_layers, gamma1_params, freq_params)

# ✅ Give every FFTGate a compact id so log_params prints Layer1..LayerN
for idx, layer in enumerate(activation_layers, start=1):
    layer.layer_id = f"Layer{idx}"


# ✅ Assign per-layer depth info for Transformer-safe FFTGate
total_layers = len(activation_layers)
for idx, layer in enumerate(activation_layers, start=1):
    layer.layer_depth = idx
    layer.total_layers = total_layers
    print(f"🧩 Assigned FFTGate depth {idx}/{total_layers}")
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 3️⃣===========  Build MAIN optimizer excluding gamma1/freq params =========== 
if gamma1_params or freq_params:
    act_id_set = {id(p) for p in (gamma1_params + freq_params)}  
    main_params = [p for p in net.parameters() if id(p) not in act_id_set]
    optimizer = torch.optim.Adam(
        main_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=args.opt_eps,
        weight_decay=args.weight_decay
    )
else:
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=args.opt_eps,
        weight_decay=args.weight_decay
    )


# ✅ Sanity check
if gamma1_params or freq_params:
    main_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
    act_ids  = {id(p) for p in (gamma1_params + freq_params)}
    assert main_ids.isdisjoint(act_ids), "gamma1/freq_factor still inside MAIN optimizer!"
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣=========== Initialize ACTIVATION optimizers & schedulers (start frozen) =========== 
activation_optimizers = {}
gamma1_lr_schedule, freq_lr_schedule = None, None  

if gamma1_params or freq_params:
    unfreeze_activation_epoch = args.unfreeze_act_epochs
    WARMUP_ACTIVATION_EPOCHS = args.Warm_act_epochs

    # ✅ freeze initially
    for p in gamma1_params + freq_params:
        p.requires_grad = False       # 🚫 Keep frozen before the unfreeze epoch

    # ✅ Separate optimizers only if params exist
    if gamma1_params:
        activation_optimizers["gamma1"] = torch.optim.AdamW(
            gamma1_params, 
            lr=lr_gamma1,             # 🟡 gamma1 learns faster
            weight_decay=args.gamma1_wd
        )
        gamma1_lr_schedule = my_utils.cosine_scheduler(
            lr_gamma1,
            args.min_act_lr,
            args.epochs,
            num_training_steps_per_epoch,
            warmup_epochs=WARMUP_ACTIVATION_EPOCHS,
            warmup_steps=args.warmup_steps
        )

    if freq_params:
        activation_optimizers["freq_factor"] = torch.optim.AdamW(
            freq_params, 
            lr=lr_freq,             # 🟡 freq slower than gamma1
            weight_decay=args.wd_freq
        )
        freq_lr_schedule = my_utils.cosine_scheduler(
            lr_freq,
            args.min_act_lr,
            args.epochs,
            num_training_steps_per_epoch,
            warmup_epochs=WARMUP_ACTIVATION_EPOCHS,
            warmup_steps=args.warmup_steps
        )
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ✅ Final fallback if nothing was found
if not activation_optimizers:
    activation_optimizers = None


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 5️⃣ =========== Logging =========== 
if activation_layers and (gamma1_params or freq_params):
    print(f"✅ Found {len(activation_layers)} FFTGate layers.")
    print(f"✅ Collected {len(gamma1_params)} gamma1 params, "
          f"{len(freq_params)} freq_factor params")
    for layer in activation_layers:
        print(f"   🔹 {layer.layer_id}")
elif activation_layers:
    print(f"⚠ Warning: Found {len(activation_layers)} FFTGate layers, but no parameters.")
else:
    print("⚠ Warning: No FFTGate layers found.")

# ---- Main optimizer groups
print("Main Optimizer Groups:")
for i, pg in enumerate(optimizer.param_groups):
    print(f" Group {i} → {len(pg['params'])} params, "
          f"lr={pg['lr']:.6f}, wd={pg['weight_decay']:.6f}")

# ---- Activation optimizer groups
if activation_optimizers is not None:
    for name, opt in activation_optimizers.items():
        print(f"{name.capitalize()} Optimizer Groups:")
        for i, pg in enumerate(opt.param_groups):
            print(f" Group {i} → {len(pg['params'])} {name} params, "
                  f"lr={pg['lr']:.6f}, wd={pg['weight_decay']:.6f}")
else:
    print("⚠ No activation optimizers built yet (will be created at unfreeze).")
# ─────────────────────────────────────────────────────────────────────────────────────────────────







########################################################################################################################
####-------| NOTE 6. INITIALIZE MAIN OPTIMIZER SCHEDULER | XXX --------------------------------------####################
########################################################################################################################

# ✅ Define MultiStepLR for Main Optimizer
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80], gamma=0.1, last_epoch=-1)

main_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80], gamma=0.1, last_epoch=-1)






########################################################################################################################
####-------| NOTE 7. FINALIZE ACT SCHEDULERs | XXX ------------------------------------------------####################
########################################################################################################################

# ✅ Compute starting global step so schedules line up perfectly after resume
global_start_step = args.start_epoch * num_training_steps_per_epoch  # ✅ aligns with ConvNeXt
print(f"🌍 global_start_step = {global_start_step}")


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Helper to set per-step LR/WD on gamma1 & freq_factor optimizers
def _apply_single_lr_at_step(key, optimizers, schedule, step_idx):
    if not optimizers or key not in optimizers or schedule is None:
        return None, None

    if step_idx >= len(schedule):
        step_idx = len(schedule) - 1

    lr_this_step = schedule[step_idx]
    # wd_this_step = args.act_weight_decay

    # 🔑 choose weight decay depending on which optimizer we're updating
    if key == "gamma1":
        wd_this_step = args.gamma1_wd
    elif key == "freq_factor":
        wd_this_step = args.wd_freq
    else:
        raise ValueError(f"Unexpected optimizer key: {key}")


    for pg in optimizers[key].param_groups:
        pg["lr"] = lr_this_step
        pg["weight_decay"] = wd_this_step
        # keep explicit keys for logging/debugging
        pg["act_lr"] = lr_this_step
        pg["act_weight_decay"] = wd_this_step

    return lr_this_step, wd_this_step
# ─────────────────────────────────────────────────────────────────────────────────────────────────










########################################################################################################################
####-------| NOTE 8. MODEL CHECK POINT | XXX -------------------------------------------------------####################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# Ensure directories exist
if not os.path.exists('checkpoint'):
    os.makedirs('checkpoint')

if not os.path.exists('Results'):
    os.makedirs('Results')

# Construct checkpoint path
checkpoint_path = f'./checkpoint/{args.act_name}_{args.model_name}_{args.dataset_name}_{args.main_opt_name}_{args.mode_name}.t7'

# Resume checkpoint only if file exists
if args.resume:
    print('==> Resuming from checkpoint....')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        print(f"Checkpoint loaded: {checkpoint_path}")
    else:
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
# ─────────────────────────────────────────────────────────────────────────────────────────────────








########################################################################################################################
####-------| NOTE 9. PATH DEFINATION AND GLOBAL INITAILIZATION | XXX ------------------------------#####################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅  === Define Paths for test and train logs  === 
train_results_path = f'./Results/Train_{args.act_name}_{args.model_name}_{args.dataset_name}_{args.main_opt_name}_{args.mode_name}.txt'
test_results_path = f'./Results/Test_{args.act_name}_{args.model_name}_{args.dataset_name}_{args.main_opt_name}_{args.mode_name}.txt'

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === Define path to store FFTGate Params log === 
LR_save_paths = {
    "LR_history": f"./Results/{args.act_name}/{args.act_name}_{args.model_name}_{args.dataset_name}_{args.main_opt_name}_{args.mode_name}_LR_log.txt"
}
save_paths = {
    "log_history": f"./Results/{args.act_name}/{args.act_name}_{args.model_name}_{args.dataset_name}_{args.main_opt_name}_{args.mode_name}_training_logs.txt"
}
FFTGate_param_save_paths = {       
    "FFTGate_param_history": f"./Results/{args.act_name}/{args.act_name}_{args.model_name}_{args.dataset_name}_{args.main_opt_name}_{args.mode_name}_FFTGate_param_log.txt"              # ✅ FFTGate Parameters log_history     
}
FFTGate_Grad_save_paths = {       
    "FFTGate_Grad_history": f"./Results/{args.act_name}/{args.act_name}_{args.model_name}_{args.dataset_name}_{args.main_opt_name}_{args.mode_name}_FFTGate_Grad_log.txt"                # ✅ FFTGate Parameters Gradiants log_history     
}
FFTGate_SummaryGrad_save_paths = {       
    "FFTGate_SummaryGrad_history": f"./Results/{args.act_name}/{args.act_name}_{args.model_name}_{args.dataset_name}_{args.main_opt_name}_{args.mode_name}_FFTGate_SummaryGrad_log.txt"  # ✅ Gradiants Summary log_history 1     
}
FFTGate_Main_SummaryGrad_save_paths = {       
    "FFTGate_Main_SummaryGrad_history": f"./Results/{args.act_name}/{args.act_name}_{args.model_name}_{args.dataset_name}_{args.main_opt_name}_{args.mode_name}_FFTGate_Main_SummaryGrad_log.txt"  # ✅ Gradiants Summary log_history 2       
}
# ─────────────────────────────────────────────────────────────────────────────────────────────────








########################################################################################################################
####-------| NOTE 10. DEFINE TRAIN LOOP | XXX ------------------------------------------------------####################
########################################################################################################################

# ✅ === Training ===

def train(epoch, net, device, criterion, optimizer, activation_optimizers, unfreeze_activation_epoch, global_start_step):
    #🌍
    global train_loss_history, best_train_acc, smoothed_skew, prev_params, recent_test_acc, gamma1_history, activation_layers, test_acc_history   # 🟢🟢🟢


    epoch_start_time = time.time()

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Initialize histories and activation params tracking ===
    if epoch == args.start_epoch:   
        train_loss_history, test_acc_history = [], []    # BLEU/test acc history
        best_train_acc, recent_test_acc = 0.0, 0.0
        gamma1_history, freq_history  = {}, {}           # per-layer gamma1/Freq tracking
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Initialize prev_params once, with detached snapshots
    prev_params = {}
    layer_index_map = {idx: idx for idx in range(len(activation_layers))}
    
    for idx, layer in enumerate(activation_layers):
        prev_params[idx] = {
            "gamma1": layer.gamma1_raw.clone().detach(),  # clone THEN detach
            "freq_factor": layer.freq_raw.clone().detach()
        }
    print(f"✅ Initialized tracking for {len(activation_layers)} FFTGate layers")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ✅ Train mode
    net.train()


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔖 Sync current epoch into all FFTGate layers before training batches
    #     • Updates each layer’s "current_epoch" for scheduling and logging.
    #     • Resets only the per-epoch batch counter (_epoch_step) → used for debug prints.
    #     • Keeps the global forward counter (_step) untouched → preserves rolling FFT window.
    # ────────────────────────────────────────────────────────────────────────────────────────────────   
    for layer in activation_layers:
        if isinstance(layer, FFTGate):
            layer.current_epoch.fill_(epoch)  # ✅ update current epoch buffer for scheduling & decay tracking
            layer._epoch_step = 0             # ✅ reset per-epoch batch counter (for debug / milestone logs)
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ──────────────────────────────────
    # ❌ GPU memory check at epoch start
    if torch.cuda.is_available():
        print(f"[❗Epoch {epoch} TRAIN START] GPU mem allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB | reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    # ──────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    train_loss, correct, total, train_accuracy = 0, 0, 0, 0.0

    # ✅ Initialize log histories
    log_history, lr_log_history, FFTGate_param_history, FFTGate_Grad_history, FFTGate_SummaryGrad_history, FFTGate_Main_SummaryGrad_history, activation_history = [], [], [], [], [], [], []  
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Compute this epoch's base step index for per-step schedule (ConvNeXt-style)
    epoch_base_step = global_start_step + (epoch - args.start_epoch) * num_training_steps_per_epoch


    # ✅ Unfreeze at specified epoch
    if epoch == unfreeze_activation_epoch:
        print("\n🔓 Unfreezing FFTGate Parameters 🔓")
        for p in (gamma1_params + freq_params):   
            p.requires_grad = True

        # Rebuild optimizers if frozen
        if gamma1_params:
            activation_optimizers["gamma1"] = torch.optim.AdamW(
                gamma1_params, lr=0.0, weight_decay=args.gamma1_wd
            )
        if freq_params:
            activation_optimizers["freq_factor"] = torch.optim.AdamW(
                freq_params, lr=0.0, weight_decay=args.wd_freq
            )

        # ✅ Sync schedulers immediately
        step_idx = epoch_base_step
        g_lr, g_wd = _apply_single_lr_at_step("gamma1", activation_optimizers, gamma1_lr_schedule_values, step_idx)
        f_lr, f_wd = _apply_single_lr_at_step("freq_factor", activation_optimizers, freq_lr_schedule_values, step_idx)

        print(f"✅ Gamma1 Optimizer Synced! LR={g_lr if g_lr else 0:.6f}, WD={g_wd if g_wd else 0:.6f}")
        print(f"✅ FreqFactor Optimizer Synced! LR={f_lr if f_lr else 0:.6f}, WD={f_wd if f_wd else 0:.6f}")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Training Loop
    with tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch}") as progress:
        for batch_idx, (inputs, targets) in progress:


            # ✅ Compute step
            step_idx = epoch_base_step + batch_idx

            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ Apply per-step LR/WD (FFTGate)
            g_lr, g_wd = _apply_single_lr_at_step("gamma1", activation_optimizers, gamma1_lr_schedule_values, step_idx)
            f_lr, f_wd = _apply_single_lr_at_step("freq_factor", activation_optimizers, freq_lr_schedule_values, step_idx)
            # ─────────────────────────────────────────────────────────────────────────────────────────────────

            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # 🔍 Debug LR
            lr_now = optimizer.param_groups[0]["lr"]
            wd_now = optimizer.param_groups[0]["weight_decay"]

            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # 🔍 Debug milestones
            detailed_steps = {0, 1, 2, 5, 10, 50, 70, 100, 200, 500, 700, 750}
            detailed_steps.add(num_training_steps_per_epoch - 1)
            milestone_epochs = {0, 1, 3, 5, 10, 20, 30, 50, 80, 95}
            # ─────────────────────────────────────────────────────────────────────────────────────────────────

            if (epoch in milestone_epochs) and (batch_idx in detailed_steps):
                lr_log_msg = (
                    f"[Epoch {epoch} | Batch {batch_idx} | Global Step {step_idx}] "
                    f"Main LR: {lr_now:.8f} | WD: {wd_now:.8f} | "
                    f"Gamma1 LR: {g_lr if g_lr is not None else 'FROZEN'} | WD: {g_wd if g_wd is not None else 'FROZEN'} | "
                    f"Freq LR: {f_lr if f_lr is not None else 'FROZEN'} | WD: {f_wd if f_wd is not None else 'FROZEN'} "                    
                )
                lr_log_history.append(lr_log_msg)
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ Use channels_last layout for inputs to match model
            inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


           # ─────────────────────────────────────────────────────────────────────────────────────────────────
           # ✅ zero_grad: Main & FFTGate
            optimizer.zero_grad()

            # zero_grad activation parameter
            for act_opt in activation_optimizers.values():
                act_opt.zero_grad()
           # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ Forward Pass + Loss (⚠️ 🔄 AMP-friendly; scaler handles mixed precision internally)
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs, feature_activations = net(inputs, epoch=epoch)
                    loss = criterion(outputs, targets)
            else:
                outputs, feature_activations = net(inputs, epoch=epoch)
                loss = criterion(outputs, targets)
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ Compute Training Accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            train_accuracy = 100. * correct / total if total > 0 else 0.0  # ✅ Compute training accuracy
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # 🧭 Call Regularization Function for the Activation Parameter (gamma1 + freq_factor)
            if epoch >= unfreeze_activation_epoch:                
                fftgate_reg = apply_dynamic_regularization(
                    inputs, feature_activations, epoch,
                    prev_params, layer_index_map, batch_idx,
                    num_batches=num_training_steps_per_epoch     # 🟡 pass the total batches
                )
                loss += fftgate_reg
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ Backward pass  === ⚠️ 🔄 Use AMP if enabled ===
            if args.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # # 🔒 Gradient Clipping for FFTGate params (scaled by LR-ratio × effective param)                       
            # # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # 🔍 Debug AFTER backward (PARAMS ONLY: raw + effective, no gradients)
            if (epoch in milestone_epochs) and (batch_idx in detailed_steps):
                gamma1_raw_vals, gamma1_eff_vals = [], []
                freq_raw_vals,  freq_eff_vals  = [], []

                gamma1_raw_strs, gamma1_eff_strs = [], []
                freq_raw_strs,  freq_eff_strs  = [], []

                for layer_idx, layer in enumerate(activation_layers, start=1):
                    # collect parameter values
                    g_raw = layer.gamma1_raw.item()
                    g_eff = layer.gamma1_eff().item()
                    f_raw = layer.freq_raw.item()
                    f_eff = layer.freq_factor_eff().item()

                    # append for min/max
                    gamma1_raw_vals.append(g_raw); gamma1_eff_vals.append(g_eff)
                    freq_raw_vals.append(f_raw);   freq_eff_vals.append(f_eff)

                    # build per-layer strings
                    gamma1_raw_strs.append(f"Layer{layer_idx}: {g_raw:+.8f}")
                    gamma1_eff_strs.append(f"Layer{layer_idx}: {g_eff:+.8f}")
                    freq_raw_strs.append(f"Layer{layer_idx}: {f_raw:+.8f}")
                    freq_eff_strs.append(f"Layer{layer_idx}: {f_eff:+.8f}")

                # Safe handling if lists are empty (avoid min/max on [])
                gamma1_raw_summary = (
                    f"(MIN: {min(gamma1_raw_vals):+.8f} / MAX: {max(gamma1_raw_vals):+.8f})"
                    if gamma1_raw_vals else "(no values)"
                )
                gamma1_eff_summary = (
                    f"(MIN: {min(gamma1_eff_vals):+.8f} / MAX: {max(gamma1_eff_vals):+.8f})"
                    if gamma1_eff_vals else "(no values)"
                )
                freq_raw_summary = (
                    f"(MIN: {min(freq_raw_vals):+.8f} / MAX: {max(freq_raw_vals):+.8f})"
                    if freq_raw_vals else "(no values)"
                )
                freq_eff_summary = (
                    f"(MIN: {min(freq_eff_vals):+.8f} / MAX: {max(freq_eff_vals):+.8f})"
                    if freq_eff_vals else "(no values)"
                )

                log_msg = (
                    f"Epoch {epoch} | Batch {batch_idx} | AFTER backward || "
                    f"Gamma1 RAW {gamma1_raw_summary}: [ {' | '.join(gamma1_raw_strs)} ] || "
                    f"Gamma1 EFF {gamma1_eff_summary}: [ {' | '.join(gamma1_eff_strs)} ] || "
                    f"Freq RAW {freq_raw_summary}: [ {' | '.join(freq_raw_strs)} ] || "
                    f"Freq EFF {freq_eff_summary}: [ {' | '.join(freq_eff_strs)} ]"
                )

                # print(log_msg)
                FFTGate_param_history.append(log_msg)  # ✅ Save to param history
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # 🔍 Debug gradients AFTER backward (GRAD-ONLY, with MIN/MAX summary)
            if (epoch in milestone_epochs) and (batch_idx in detailed_steps):

                # ───────────────────────────────────────────────────
                # 🎀 Correct AMP-scaled gradients using scaler factor
                if args.use_amp:
                    AMP_scale_value = scaler.get_scale() if 'scaler' in globals() else 1.0
                else:
                    AMP_scale_value = 1.0

                inv_scale = 1.0 / AMP_scale_value
                # ───────────────────────────────────────────────────

                gamma1_grad_vals, freq_grad_vals = [], []
                gamma1_grad_strs, freq_grad_strs = [], []
                # ───────────────────────────────────────────────────
                for layer_idx, layer in enumerate(activation_layers, start=1):
                    # --- Gamma1 grad ---
                    if layer.gamma1_raw.grad is None:
                        gamma1_grad_strs.append(f"Layer{layer_idx}: grad=None")
                    elif torch.all(layer.gamma1_raw.grad == 0):
                        gamma1_grad_strs.append(f"Layer{layer_idx}: grad=ZERO")
                    else:
                        # 🎀 divide by AMP_scale_value to get true gradient magnitude for gamma1
                        g_grad = layer.gamma1_raw.grad.item() * inv_scale
                        gamma1_grad_vals.append(g_grad)
                        gamma1_grad_strs.append(f"Layer{layer_idx}: {g_grad:+.8f}")

                    # --- Freq grad ---
                    if layer.freq_raw.grad is None:
                        freq_grad_strs.append(f"Layer{layer_idx}: grad=None")
                    elif torch.all(layer.freq_raw.grad == 0):
                        freq_grad_strs.append(f"Layer{layer_idx}: grad=ZERO")
                    else:
                        # 🎀 apply same unscale correction for freq gradients
                        f_grad = layer.freq_raw.grad.item() * inv_scale
                        freq_grad_vals.append(f_grad)
                        freq_grad_strs.append(f"Layer{layer_idx}: {f_grad:+.8f}")
                # ───────────────────────────────────────────────────

                # 🧩 Safe handling if lists are empty (avoid min/max on [])
                gamma1_grad_summary = (
                    f"(MIN: {min(gamma1_grad_vals):+.8f} / MAX: {max(gamma1_grad_vals):+.8f})"
                    if gamma1_grad_vals else "(no grads)"
                )
                freq_grad_summary = (
                    f"(MIN: {min(freq_grad_vals):+.8f} / MAX: {max(freq_grad_vals):+.8f})"
                    if freq_grad_vals else "(no grads)"
                )

                log_msg = (
                    f"Epoch {epoch} | Batch {batch_idx} | AFTER backward || "
                    f"Gamma1 Grad {gamma1_grad_summary}: [ {' | '.join(gamma1_grad_strs)} ] || "
                    f"Freq Grad {freq_grad_summary}: [ {' | '.join(freq_grad_strs)} ]"
                    f"| GradScale={AMP_scale_value:.1e}"
                )
                # print(log_msg)
                FFTGate_Grad_history.append(log_msg)   # ✅ Save for grad-only history
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # === Apply Optimizer Step for Model Parameters   === ⚠️ 🔄 Use AMP if enabled===
            # ✅ model params
            if args.use_amp:
                scaler.step(optimizer)
            else:
                optimizer.step()

            # ✅ activation params (only when unfrozen)
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


            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅7️⃣ Update progress bar
            progress.set_postfix(
                Train_loss=round(train_loss / (batch_idx + 1), 3),
                Train_acc=train_accuracy,
                LR=f"{lr_now:.8f}",
                WD=f"{wd_now:.8f}",
                Gamma1_LR=f"{g_lr:.8f}" if g_lr is not None else "FROZEN",
                Freq_LR=f"{f_lr:.8f}" if f_lr is not None else "FROZEN"
            )
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Step the main optimizer scheduler (ONLY for model parameters)
    # 🔖 FFTGate schedulers are updated manually via `_apply_single_lr_at_step()` above | batchwise
    main_scheduler.step()
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅  Update prev_params here AFTER all updates | END OF EACH EPOCH (AFTER optimizer.step())
    for idx, layer in enumerate(activation_layers):
        if isinstance(layer, FFTGate):
            prev_params[idx] = {
                "gamma1": layer.gamma1_raw.clone().detach(),   # ✅ detach snapshot only
                "freq_factor": layer.freq_raw.clone().detach()
            }
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Timing
    duration = time.time() - epoch_start_time
    mins, secs = divmod(duration, 60)
    print(f"⏱ Epoch {epoch} Training time {args.act_name}_{args.model_name}: {int(mins)} min {secs:.2f} sec")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────



    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Logging Parameters & Gradients (RAW + EFFECTIVE) with signs and MIN/MAX (fixed-point)

    gamma1_param_values, gamma1_eff_values = [], []
    freq_param_values,  freq_eff_values   = [], []
    gamma1_grad_values, freq_grad_values  = [], []

    gamma1_param_nums, gamma1_eff_nums = [], []
    freq_param_nums,  freq_eff_nums    = [], []
    gamma1_grad_nums, freq_grad_nums   = [], []

    layer_counter = 1

    # ───────────────────────────────────────────────────
    # 🎀 Correct AMP-scaled gradients using scaler factor
    if args.use_amp:
        AMP_scale_value = scaler.get_scale() if 'scaler' in globals() else 1.0
    else:
        AMP_scale_value = 1.0

    inv_scale = 1.0 / AMP_scale_value
    # ───────────────────────────────────────────────────
 
    # ────────────────────────────────────────────────────────────────────────────────────────────────
       # Log RAW + EFF + Grad for Gamma1
    for name, p in net.named_parameters():
        if name.endswith("gamma1_raw"):
            v = p.data.item()
            gamma1_param_values.append(f"{name}: {v:+.8f}")
            gamma1_param_nums.append(v)

            if p.grad is None:
                gamma1_grad_values.append(f"{name}: grad=None")
            else:
                # 🎀 Apply AMP unscale correction to log true gradient magnitude
                g = p.grad.item() * inv_scale
                gamma1_grad_values.append(f"{name}: {g:+.8f}")
                gamma1_grad_nums.append(g)

            module_name = name.rsplit(".", 1)[0]  # strip off ".gamma1_raw"
            module = dict(net.named_modules())[module_name]
            v_eff = module.gamma1_eff().item()
            gamma1_eff_values.append(f"{name}: {v_eff:+.8f}")
            gamma1_eff_nums.append(v_eff)

    for name, p in net.named_parameters():
        if name.endswith("freq_raw"):
            v = p.data.item()
            freq_param_values.append(f"{name}: {v:+.8f}")
            freq_param_nums.append(v)

            if p.grad is None:
                freq_grad_values.append(f"{name}: grad=None")
            else:
                # 🎀Apply AMP unscale correction to freq gradients
                g = p.grad.item() * inv_scale
                freq_grad_values.append(f"{name}: {g:+.8f}")
                freq_grad_nums.append(g)
        
            module_name = name.rsplit(".", 1)[0]  # strip off ".freq_raw"
            module = dict(net.named_modules())[module_name]
            v_eff = module.freq_factor_eff().item()
            freq_eff_values.append(f"{name}: {v_eff:+.8f}")
            freq_eff_nums.append(v_eff)
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # grab current LRs/WDs
    current_lr = optimizer.param_groups[0]["lr"]
    current_wd = optimizer.param_groups[0].get("weight_decay", 0.0)

    current_g_lr = activation_optimizers["gamma1"].param_groups[0]["lr"]
    current_g_wd = activation_optimizers["gamma1"].param_groups[0].get("weight_decay", 0.0)

    current_f_lr = activation_optimizers["freq_factor"].param_groups[0]["lr"]
    current_f_wd = activation_optimizers["freq_factor"].param_groups[0].get("weight_decay", 0.0)
    
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # build log message with MIN/MAX summaries (SAFE version with empty list guards)
    gamma1_raw_summary = (
        f"(MIN: {min(gamma1_param_nums):+.8f} / MAX: {max(gamma1_param_nums):+.8f})"
        if gamma1_param_nums else "(no values)"
    )
    gamma1_eff_summary = (
        f"(MIN: {min(gamma1_eff_nums):+.8f} / MAX: {max(gamma1_eff_nums):+.8f})"
        if gamma1_eff_nums else "(no values)"
    )
    freq_raw_summary = (
        f"(MIN: {min(freq_param_nums):+.8f} / MAX: {max(freq_param_nums):+.8f})"
        if freq_param_nums else "(no values)"
    )
    freq_eff_summary = (
        f"(MIN: {min(freq_eff_nums):+.8f} / MAX: {max(freq_eff_nums):+.8f})"
        if freq_eff_nums else "(no values)"
    )
    gamma1_grad_summary = (
        f"(MIN: {min(gamma1_grad_nums):+.8f} / MAX: {max(gamma1_grad_nums):+.8f})"
        if gamma1_grad_nums else "(no grads)"
    )
    freq_grad_summary = (
        f"(MIN: {min(freq_grad_nums):+.8f} / MAX: {max(freq_grad_nums):+.8f})"
        if freq_grad_nums else "(no grads)"
    )


    log_msg = (
        f"Epoch {epoch} | Batch {batch_idx} | "
        f"M_Optimizer LR: {current_lr:.6f} | WD: {current_wd:.6f} | "
        f"Gamma1 LR: {current_g_lr:.6f} | Gamma1 WD: {current_g_wd:.6f} | "
        f"FreqFactor LR: {current_f_lr:.6f} | FreqFactor WD: {current_f_wd:.6f} || "
        f"Gamma1 RAW {gamma1_raw_summary}: [ {' | '.join(gamma1_param_values)} ] || "
        f"Gamma1 EFF {gamma1_eff_summary}: [ {' | '.join(gamma1_eff_values)} ] || "
        f"Freq RAW {freq_raw_summary}: [ {' | '.join(freq_param_values)} ] || "
        f"Freq EFF {freq_eff_summary}: [ {' | '.join(freq_eff_values)} ] || "
        f"Gamma1 Grad {gamma1_grad_summary}: [ {' | '.join(gamma1_grad_values)} ] || "
        f"Freq Grad {freq_grad_summary}: [ {' | '.join(freq_grad_values)} ] || "
        f"GradScale={AMP_scale_value:.1e} || "
        f"⏱ Training time => {args.act_name}_{args.model_name}: {int(mins)} min {secs:.2f} sec"
    )

    log_history.append(log_msg)
    print(log_msg)
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────────────────────────



    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Compute final training accuracy for the epoch
    final_train_loss = train_loss / len(trainloader)
    final_train_acc = 100. * correct / total

    # Append to history (this was missing)
    train_loss_history.append(final_train_loss)
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ ============== Save Logs, Training and Test Results ========================================
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Save Test Results
    if epoch == args.start_epoch and os.path.exists(train_results_path):  # ✅ Clear the log file at the start of training (Epoch 0)
        with open(train_results_path, 'w') as f:
            f.write("")  # ✅ Clears previous logs only once

    # ✅ Append new training results for each epoch
    with open(train_results_path, 'a') as f:
        f.write(f"Epoch {epoch} | Train Loss: {final_train_loss:.3f} | Train Acc: {final_train_acc:.3f}%\n")

    if final_train_acc > best_train_acc:
        best_train_acc = final_train_acc  # ✅ Update best training accuracy
        print(f"🏆 New Best Training Accuracy: {best_train_acc:.3f}% (Updated)")

    # ✅ Append the best training accuracy **only once at the end of training**
    if epoch == (args.epochs - 1):  
        with open(train_results_path, 'a') as f:
            f.write(f"\n🏆 Best Training Accuracy: {best_train_acc:.3f}%\n")  

    # ✅ Print both Final and Best Training Accuracy
    print(f"📊 Train Accuracy: {final_train_acc:.3f}% | 🏆 Best Train Accuracy: {best_train_acc:.3f}%")
    # ────────────────────────────────────────────────────────────────────────────────────────────────

    # print(f"📜 Training logs saved to {train_results_path}!")
    print(f"🏆 Best Training Accuracy: {best_train_acc:.3f}% (Updated)")


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Save Training logs
    if epoch == args.start_epoch:  # ✅ Only clear at the start of training
        with open(save_paths["log_history"], "w", encoding="utf-8") as log_file:
            log_file.write("")  # ✅ Clears previous logs

    # ✅ Save logs once per epoch (Append new logs)
    if log_history:
        with open(save_paths["log_history"], "a", encoding="utf-8") as log_file:
            log_file.write("\n".join(log_history) + "\n")             # ✅ Ensure each entry is on a new line
        # print(f"📜 Logs saved to {save_paths['log_history']}!")     # ✅ Only prints once per epoch
    # else:
        # print("⚠ No logs to save!")


    # ✅ Save LR log history to file (once per epoch)
    if epoch == args.start_epoch:
        with open(LR_save_paths["LR_history"], "w", encoding="utf-8") as f:
            f.write("")  # Clear previous content on first epoch

    if lr_log_history:
        os.makedirs(os.path.dirname(LR_save_paths["LR_history"]), exist_ok=True)
        with open(LR_save_paths["LR_history"], "a", encoding="utf-8") as f:
            f.write("\n".join(lr_log_history) + "\n")
    #     print(f"📈 LR logs saved to {LR_save_paths['LR_history']}!")
    # else:
    #     print("⚠ No LR logs to save.")
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Save FFTGate param log history to file (once per epoch)
    if epoch == args.start_epoch:
        with open(FFTGate_param_save_paths["FFTGate_param_history"], "w", encoding="utf-8") as f:
            f.write("")  # Clear previous content on first epoch
    if FFTGate_param_history:
        os.makedirs(os.path.dirname(FFTGate_param_save_paths["FFTGate_param_history"]), exist_ok=True)
        with open(FFTGate_param_save_paths["FFTGate_param_history"], "a", encoding="utf-8") as f:
            f.write("\n".join(FFTGate_param_history) + "\n")
        # print(f"📈 FFTGate param logs saved to {FFTGate_param_save_paths['FFTGate_param_history']}!")

    # ✅ Save FFTGate Grad log history to file (once per epoch)
    if epoch == args.start_epoch:
        with open(FFTGate_Grad_save_paths["FFTGate_Grad_history"], "w", encoding="utf-8") as f:
            f.write("")  # Clear previous content on first epoch
    if FFTGate_Grad_history:
        os.makedirs(os.path.dirname(FFTGate_Grad_save_paths["FFTGate_Grad_history"]), exist_ok=True)
        with open(FFTGate_Grad_save_paths["FFTGate_Grad_history"], "a", encoding="utf-8") as f:
            f.write("\n".join(FFTGate_Grad_history) + "\n")

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Save FFTGate Summary Grad log history to file (once per epoch)
    if epoch == args.start_epoch:
        with open(FFTGate_SummaryGrad_save_paths["FFTGate_SummaryGrad_history"], "w", encoding="utf-8") as f:
            f.write("")  # Clear previous content on first epoch
    if FFTGate_SummaryGrad_history:
        os.makedirs(os.path.dirname(FFTGate_SummaryGrad_save_paths["FFTGate_SummaryGrad_history"]), exist_ok=True)
        with open(FFTGate_SummaryGrad_save_paths["FFTGate_SummaryGrad_history"], "a", encoding="utf-8") as f:
            f.write("\n".join(FFTGate_SummaryGrad_history) + "\n")

    # ✅ Save FFTGate _Main_Summary Grad log history to file (once per epoch)
    if epoch == args.start_epoch:
        with open(FFTGate_Main_SummaryGrad_save_paths["FFTGate_Main_SummaryGrad_history"], "w", encoding="utf-8") as f:
            f.write("")  # Clear previous content on first epoch
    if FFTGate_Main_SummaryGrad_history:
        os.makedirs(os.path.dirname(FFTGate_Main_SummaryGrad_save_paths["FFTGate_Main_SummaryGrad_history"]), exist_ok=True)
        with open(FFTGate_Main_SummaryGrad_save_paths["FFTGate_Main_SummaryGrad_history"], "a", encoding="utf-8") as f:
            f.write("\n".join(FFTGate_Main_SummaryGrad_history) + "\n")
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    if epoch % 10 == 0:
        print(f"📏 Sizes → ActivationHist: {len(activation_history)} | TestAccHist: {len(test_acc_history)} | TrainLossHist: {len(train_loss_history)}")



    # return final_train_loss, final_train_acc, feature_activations













########################################################################################################################
####-------| NOTE 11. DEFINE TEST LOOP | XXX -------------------------------------------------------####################
########################################################################################################################


# ✅ Initialize total test duration only once (outside the loop)
total_test_duration = 0.0  # 🟡 Accumulate test time across epochs


def test(epoch, net, device, save_results=True):
    """
    Evaluates the model on the test set and optionally saves the results.
    
    Args:
    - epoch (int): The current epoch number.
    - save_results (bool): Whether to save results to a file.

    Returns:
    - acc (float): Test accuracy percentage.
    """
    global best_acc, val_accuracy, test_results_path, total_test_duration   


    Test_epoch_start_time = time.time()  # ⏱️ Test Start epoch timer

    # ✅ Evaluation mode
    net.eval()


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔖 Sync current epoch into all FFTGate layers before test batches
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    for layer in activation_layers:
        if isinstance(layer, FFTGate):
            layer.current_epoch.fill_(epoch)
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    test_loss, correct, total = 0, 0, 0

    # ✅ Initialize log history
    Test_log_history = []
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Ensure activation function parameters are clamped before evaluation
    with torch.no_grad():
        with tqdm(enumerate(testloader), total=len(testloader), desc=f"Testing Epoch {epoch}") as progress:
            for batch_idx, (inputs, targets) in progress:

                
                # ────────────────────────────────────────────────────────────────────────────────────────────────
                inputs = inputs.to(device, non_blocking=True, memory_format=torch.channels_last)
                targets = targets.to(device, non_blocking=True)


                # ────────────────────────────────────────────────────────────────────────────────────────────────
                # ✅ Forward Pass  | ===⚠️ 🔄 Use AMP if enabled ===
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _ = net(inputs, epoch=0)  # ✅ Only use logits for loss
                else:
                    outputs, _ = net(inputs, epoch=0)
                # ────────────────────────────────────────────────────────────────────────────────────────────────


                # ────────────────────────────────────────────────────────────────────────────────────────────────
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # ✅ Pass validation accuracy to activation function
                val_accuracy = 100. * correct / total if total > 0 else 0
                # ────────────────────────────────────────────────────────────────────────────────────────────────


                # ✅ Update progress bar with loss & accuracy
                progress.set_postfix(Test_loss=round(test_loss / (batch_idx + 1), 3),
                                     Test_acc=round(val_accuracy, 3))
                # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ✅ Compute final test accuracy
    final_test_loss = test_loss / len(testloader)
    final_test_acc = 100. * correct / total


  # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ⏱️ Timing/logging for this epoch
    Test_epoch_end_time = time.time()
    Test_duration = Test_epoch_end_time - Test_epoch_start_time

    total_test_duration += Test_duration  # 🟡 Accumulate test time

    mins, secs = divmod(Test_duration, 60)

    # ✅ Create and print test timing log line
    Test_log_line = (
        f"Epoch {epoch} | Test Loss: {final_test_loss:.3f} | Test Acc: {final_test_acc:.3f}% | "
        f"⏱ Test time => {args.act_name}_{args.model_name}: {int(mins)} min {secs:.2f} sec"
    )
    
    print(Test_log_line)

    # ✅ Insert test log at the beginning of the list
    Test_log_history.insert(0, Test_log_line)
  # ────────────────────────────────────────────────────────────────────────────────────────────────


  # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Initialize log file at the beginning of training (clear old logs)
    if epoch == args.start_epoch:
        with open(test_results_path, 'w', encoding="utf-8") as f:
            f.write("")  # ✅ Clears previous logs

    # ✅ Append new test results for each epoch (same style as training)
    with open(test_results_path, 'a', encoding="utf-8") as f:
        f.write(f"Epoch {epoch} | Test Loss: {final_test_loss:.3f} | Test Acc: {final_test_acc:.3f}%\n")
  # ────────────────────────────────────────────────────────────────────────────────────────────────



    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Save only the BEST checkpoint (overwrite previous best)
    if final_test_acc > best_acc:
        best_acc = final_test_acc  # ✅ Update best accuracy
        # print(f"🏆 New best model found! Saving checkpoint... ({best_acc:.3f}%)")

        # ✅ Ensure checkpoint directory exists
        checkpoint_dir = "./checkpoint"
        os.makedirs(checkpoint_dir, exist_ok=True)

        # ✅ Always use the same file name — overwrites the previous best
        checkpoint_path = (
            f"{checkpoint_dir}/{args.act_name}_{args.model_name}_{args.dataset_name}_"
            f"{args.main_opt_name}_{args.mode_name}_BEST.pt"
        )

        # ✅ Full training state for resuming or re-evaluation
        state = {
            "epoch": epoch,
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),                 # Main model optimizer (Adam)
            "lr_scheduler": main_scheduler.state_dict(),         # Main LR scheduler (MultiStepLR)
            "activation_optimizers": {
                k: opt.state_dict() for k, opt in activation_optimizers.items()
            } if activation_optimizers else None,
            "gamma1_lr_schedule": gamma1_lr_schedule_values,
            "freq_lr_schedule": freq_lr_schedule_values,
            "best_acc": best_acc,
        }

        torch.save(state, checkpoint_path)
        # print(f"✅ Best checkpoint updated → {checkpoint_path}")
    # ────────────────────────────────────────────────────────────────────────────────────────────────


        best_acc = final_test_acc  # ✅ Update best accuracy


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Append the best test accuracy **only once at the end of training**
    if epoch == (args.epochs - 1):
        with open(test_results_path, 'a', encoding="utf-8") as f:
            f.write(f"\n🏆 Best Test Accuracy: {best_acc:.3f}%\n")


            # 🟡 Log total test time at the END
            total_mins, total_secs = divmod(total_test_duration, 60)
            f.write(f"\n🕒 Total Test Time => {args.act_name}_{args.model_name}: {int(total_mins)} min {total_secs:.2f} sec\n")
            print(f"🕒 Total Test Time => {args.act_name}_{args.model_name}: {int(total_mins)} min {total_secs:.2f} sec")


    # ✅ Print both Final and Best Test Accuracy (always executed)
    print(f"📊 Test Accuracy: {final_test_acc:.3f}% | 🏆 Best Test Accuracy: {best_acc:.3f}%")
    # print(f"📜 Test logs saved to {test_results_path}!")
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    global recent_test_acc
    recent_test_acc = final_test_acc  # ✅ Capture latest test accuracy for next train() call | Store latest test accuracy

    return final_test_acc             # ✅ Return the test accuracy



# %% 

########################################################################################################################
####-------| NOTE 12. TRAIN MODEL WITH SHEDULAR | XXX ----------------------------------------------####################
########################################################################################################################



# ✅ Force pythin to use 'spawn'
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Set Seed for Reproducibility BEFORE training starts
    set_seed_torch(seed1)  
    set_seed_main(seed2)  

    # ✅ Optional: Free unused GPU memory BEFORE training starts
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # ────────────────────────────────────────────────────────────────────────────────────────────────

    # ⏱️ Track total training time outside loop
    training_total_start = time.time()

    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Training Loop
    for epoch in range(args.start_epoch, args.epochs):   # Runs training for 100 epochs

        train(epoch, net, device, criterion, optimizer, activation_optimizers, args.unfreeze_act_epochs, global_start_step ) 

        test(epoch, net, device)  # ✅ Test the model
        tqdm.write("")  # ✅ Clear leftover progress bar from test()
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    print("Best Test Accuracy: ", best_acc)

    # ✅ Compute training time 
    training_total_end = time.time()
    total_mins, total_secs = divmod(training_total_end - training_total_start, 60)


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Log to training log file (unchanged)
    with open(save_paths["log_history"], "a", encoding="utf-8") as log_file:
        log_file.write(f"\n🕒 Total Training Time => {args.act_name}_{args.model_name}: {int(total_mins)} min {total_secs:.2f} sec\n")

    # ✅ Log to test results file 
    with open(test_results_path, 'a', encoding="utf-8") as f:
        f.write(f"\n🕒 Total Training Time => {args.act_name}_{args.model_name}: {int(total_mins)} min {total_secs:.2f} sec\n")
    # ────────────────────────────────────────────────────────────────────────────────────────────────        

    print(f"\n🕒 Total Training Time_{args.act_name}_{args.model_name}: {int(total_mins)} min {total_secs:.2f} sec")







# %% 

########################################################################################################################
####-------| NOTE 13. READ/EVALUATE SAVED CHECKPOINT  | XXX ----------------------------------------####################
########################################################################################################################
import datetime



def evaluate_checkpoint(checkpoint_dir, args, net, testloader, device, criterion):
    """
    Loads a checkpoint from disk, evaluates it on the test set,
    and writes results to ./Results/ directory.

    Args:
        checkpoint_dir (str): Directory containing checkpoint files.
        args: Namespace containing act_name, model_name, dataset_name, etc.
        net (nn.Module): Model definition (same architecture as during training).
        testloader (DataLoader): Dataloader for test set.
        device (torch.device): 'cuda' or 'cpu'.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).

    Returns:
        (float): Test accuracy (%) of the checkpoint.
    """

    # ────────────────────────────────────────────────────────────────────────────────
    # ✅ Automatically locate the latest/best checkpoint
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"❌ Checkpoint directory not found: {checkpoint_dir}")

    checkpoint_files = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt") or f.endswith(".t7")
    ]

    if not checkpoint_files:
        raise FileNotFoundError(f"❌ No checkpoints found in {checkpoint_dir}")

    # Sort by modification time → newest last
    checkpoint_files.sort(key=os.path.getmtime)
    checkpoint_path = checkpoint_files[-1]
    print(f"🟢 Loading checkpoint for evaluation: {checkpoint_path}")
    # ────────────────────────────────────────────────────────────────────────────────

    # ✅ Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["net"])
    net.to(device)
    net.eval()

    # ────────────────────────────────────────────────────────────────────────────────
    # ✅ Evaluate on test set
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device, memory_format=torch.channels_last)
            targets = targets.to(device)

            outputs, _ = net(inputs, epoch=0)  # FFTGate requires epoch input
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    avg_loss = test_loss / len(testloader)

    print(f"📊 Evaluation Complete | Test Loss: {avg_loss:.3f} | Accuracy: {acc:.3f}%")
    print(f"🏆 From checkpoint: {os.path.basename(checkpoint_path)}")

    # ────────────────────────────────────────────────────────────────────────────────
    # ✅ Write results to a new file under ./Results/
    results_dir = "./Results/Eval_Checkpoints"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_filename = (
        f"{results_dir}/Eval_{args.act_name}_{args.model_name}_{args.dataset_name}_"
        f"{args.main_opt_name}_{args.mode_name}_{timestamp}.txt"
    )

    with open(eval_filename, "w", encoding="utf-8") as f:
        f.write("=============================================================\n")
        f.write(f"📦 Checkpoint Evaluation Log\n")
        f.write("=============================================================\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Dataset: {args.dataset_name}\n")
        f.write(f"Activation: {args.act_name}\n")
        f.write(f"Optimizer: {args.main_opt_name}\n")
        f.write(f"Mode: {args.mode_name}\n")
        f.write(f"Checkpoint: {os.path.basename(checkpoint_path)}\n")
        f.write("-------------------------------------------------------------\n")
        f.write(f"Test Loss: {avg_loss:.4f}\n")
        f.write(f"Test Accuracy: {acc:.4f}%\n")
        f.write("-------------------------------------------------------------\n")
        if "epoch" in checkpoint:
            f.write(f"Epoch: {checkpoint['epoch']}\n")
        if "best_acc" in checkpoint:
            f.write(f"Best Accuracy Stored: {checkpoint['best_acc']:.4f}%\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=============================================================\n")

    print(f"✅ Evaluation results saved to: {eval_filename}")
    # ────────────────────────────────────────────────────────────────────────────────

    return acc



# ===============================================================
#  ✅ ----- Call evaluate_checkpoint ----------------------------
# ===============================================================
checkpoint_dir = f"./checkpoint"

final_acc = evaluate_checkpoint(
    checkpoint_dir=checkpoint_dir,
    args=args,
    net=net,
    testloader=testloader,
    device=device,
    criterion=criterion,
)



# %%
