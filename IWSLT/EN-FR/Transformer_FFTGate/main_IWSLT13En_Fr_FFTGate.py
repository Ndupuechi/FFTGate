



# %%  
########################################################################################################################
####-------| NOTE 1. IMPORTS & GLOBAL CONFIGURATION | XXX --------------------------------------------##################
########################################################################################################################

# ✅ === Enable flexible CUDA memory allocation to reduce fragmentation ===
# Must be set before importing torch!
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Alternative for memory split limits:
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"


# ✅ === Ensure correct working directory ===
import sys
Project_PATH = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\IWSLT\Transformer_FFTGate"
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


# ======================================================================================================
# ✅ === Core Libraries ===
# ======================================================================================================
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import random
import math
import re
import glob
from copy import deepcopy
import sacrebleu  # ✅ for BLEU evaluation
from timm.loss import LabelSmoothingCrossEntropy



# ======================================================================================================
# ✅ === From utils.py in project folder import cosine_scheduler ===
# ======================================================================================================
import utils as my_utils


# ======================================================================================================
# ✅ === Fairseq & Supporting Imports ===
# ======================================================================================================
import importlib.metadata as importlib_metadata
import fairseq                                                      # type: ignore
from fairseq.data import Dictionary, data_utils, iterators          # type: ignore
from fairseq.tasks.translation import TranslationTask               # type: ignore
from fairseq import options                                         # type: ignore
from fairseq.dataclass.utils import convert_namespace_to_omegaconf  # type: ignore                                             
from fairseq import utils                                           # type: ignore 
import hydra                                                        # type: ignore  





# ✅ Print environment summary for sanity check
print("Python:", sys.version)
print("Torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("Fairseq:", fairseq.__version__)
print("OmegaConf:", importlib_metadata.version("omegaconf"))
print("Hydra-Core:", importlib_metadata.version("hydra-core"))












# # %%  


# ########################################################################################################################
# ####-------| NOTE 2.1. PROCESSING DATASET 1 | XXX --------------------------------------------------####################
# ########################################################################################################################

# import sentencepiece as spm
# from sklearn.model_selection import train_test_split
# import os

# # Path to your dataset folder
# DATA_PATH = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\IWSLT\Transformer_FFTGate\Dataset\en-fr"

# # --------------------------
# # 2.1.a. Train SentencePiece
# # --------------------------
# print("Training SentencePiece model...")
# spm.SentencePieceTrainer.Train(
#     input=f"{DATA_PATH}\\TED2013.en-fr.en,{DATA_PATH}\\TED2013.en-fr.fr",
#     model_prefix="spm_enfr",
#     vocab_size=10000,
#     character_coverage=1.0,
#     model_type="bpe"
# )

# # --------------------------
# # 2.1.b. Encode files
# # --------------------------
# print("Encoding dataset...")
# sp = spm.SentencePieceProcessor(model_file="spm_enfr.model")

# def encode_file(in_file, out_file):
#     with open(in_file, encoding="utf-8") as fin, open(out_file, "w", encoding="utf-8") as fout:
#         for line in fin:
#             fout.write(" ".join(sp.encode(line.strip(), out_type=str)) + "\n")

# encode_file(f"{DATA_PATH}\\TED2013.en-fr.en", "all.spm.en")
# encode_file(f"{DATA_PATH}\\TED2013.en-fr.fr", "all.spm.fr")

# # --------------------------
# # 2.1.c. Split into train/test (95% / 5%)
# # --------------------------
# print("Splitting into train/test (95%/5%)...")
# with open("all.spm.en", encoding="utf-8") as f: en = f.readlines()
# with open("all.spm.fr", encoding="utf-8") as f: fr = f.readlines()

# train_en, test_en, train_fr, test_fr = train_test_split(
#     en, fr, test_size=0.05, random_state=42
# )

# def write(name, lines):
#     with open(name, "w", encoding="utf-8") as f: f.writelines(lines)

# write("train.en", train_en)
# write("train.fr", train_fr)
# write("test.en",  test_en)
# write("test.fr",  test_fr)

# print("✅ Done! Now run fairseq-preprocess to binarize.")







# # %%  
# ########################################################################################################################
# ####-------| NOTE 2.2. PROCESSING DATASET 2 | XXX --------------------------------------------------####################
# ########################################################################################################################



# import subprocess

# print("Running fairseq-preprocess...")

# cmd = [
#     "fairseq-preprocess",
#     "--source-lang", "en",
#     "--target-lang", "fr",
#     "--trainpref", "train",
#     "--testpref", "test",
#     "--destdir", "data-bin/iwslt13.en-fr",
#     "--workers", "4"
# ]

# subprocess.run(cmd, check=True)

# print("✅ Preprocessing complete! Binarized data saved in data-bin/iwslt13.en-fr")




# # %% 










########################################################################################################################
####-------| NOTE 3.2. SEEDING FOR REPRODUCIBILITY | XXX -------------------------------------------####################
########################################################################################################################

def set_seed_torch(seed):
    torch.manual_seed(seed)

def set_seed_main(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False    # ✅ Default: "False" for Faster, non-deterministic kernels | set "True" to Ensure deterministic behavior for CuDNN (Slower)
    torch.backends.cudnn.benchmark = True         # ✅ Default: "True" for Autotune kernels for performance | Disable CuDNN's autotuning for reproducibility (Slower)

# ✅ ============= Define Seed =============
seed1, seed2 = 1, 1
set_seed_torch(seed1)  
set_seed_main(seed2)  










########################################################################################################################
####-------| NOTE 4. DEVICE AND PARSER (INCLUDING ACTIVATIONS)| XXX --------------------------------####################
########################################################################################################################

# --------------------------
# 4.1. Define device
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# 4.2.a. Custom parser (experiment-level configs)
# --------------------------

exp_parser = argparse.ArgumentParser("IWSLT13 EN-FR Experiment Config")

exp_parser.add_argument('--mode_name', default="Seed1_1_EXP3", type=str)
exp_parser.add_argument(
    '--act_name', default="fftgate", type=str,
    help="Activation function (relu, gelu, tanh, sigmoid, swish, glu, tanhexp, fftgate)")
     

exp_parser.add_argument('--dataset_name', default="IWSLT13_En_Fr", type=str,
    help="Choose dataset direction: 'IWSLT13_En_Fr' = English→French, 'IWSLT13_Fr_En' = French→English")  # Option: IWSLT13_En_Fr or IWSLT13_Fr_En 

exp_parser.add_argument('--net', default="Transformer", type=str)
exp_parser.add_argument('--optimizer1', default="Adam", type=str)

# ✅ Paper defaults
exp_parser.add_argument('--epochs', default=50, type=int)      # Default: 50   

exp_parser.add_argument('--start_epoch', default=0, type=int)    

exp_parser.add_argument('--lr', default=1e-4, type=float)        # peak LR
exp_parser.add_argument('--weight_decay', default=1e-4, type=float)
exp_parser.add_argument('--eps', default=1e-9, type=float)
exp_parser.add_argument('--smoothing', default=0.1, type=float)  


# ✅ Learning rate schedule
exp_parser.add_argument('--warmup_updates', default=4000, type=int,
    help="Number of warmup steps before starting inverse square root decay")



# ✅ Data loader
exp_parser.add_argument('--max_tokens', default=4096, type=int,     
    help="Maximum number of tokens per batch (default: 4096, paper standard for IWSLT14)")
exp_parser.add_argument('--max_source_positions', default=256, type=int,   
    help="Maximum source sequence length (default: 1024)")
exp_parser.add_argument('--max_target_positions', default=256, type=int, 
    help="Maximum target sequence length (default: 1024)")
exp_parser.add_argument('--num_workers', default=0, type=int,    
    help="Number of dataloader workers (default: 4)")

exp_parser.add_argument('--num_shards', default=1, type=int,
    help="Number of dataset shards (default: 1 = no sharding)")
exp_parser.add_argument('--shard_id', default=0, type=int,
    help="ID of this shard (default: 0)")
exp_parser.add_argument('--ignore_invalid_inputs', type=bool, default=True,
    help="Ignore invalid input samples instead of crashing (default: True)")





# ✅ ===== Activation Optimization parameters (for FFTGate) =====

exp_parser.add_argument('--actbase_lr', type=float, default=1e-4,
    help="Base learning rate for activation parameters (gamma1, freq)")

exp_parser.add_argument('--min_act_lr', type=float, default=2e-5,         #🔧 Adjust scheduler floor : was 1e-6 → keep at ~10–20% of peak
    help="Minimum learning rate for activation parameters (gamma1, freq)")



# 🟡 Separate WD for gamma1 and freq (renamed to avoid Jupyter --f conflict)
exp_parser.add_argument('--gamma1_wd', type=float, default=0.0,
    help="Weight decay for gamma1 parameter")
exp_parser.add_argument('--wd_freq', type=float, default=0.0,
    help="Weight decay for frequency parameter")

# 🔓 Unfreeze schedule
exp_parser.add_argument('--unfreeze_act_epochs', type=int, default=0,
    help="Epoch to start unfreezing FFTGate parameters (0 = no delay)")
exp_parser.add_argument('--warm_act_epochs', type=int, default=0,
    help="Number of warmup epochs for activation parameters after unfreeze")
exp_parser.add_argument('--addnoise_act_epochs', default=50, type=int)

# 📉 Constant decay schedule for WD and update steps
exp_parser.add_argument('--gamma1_wd_end', type=float, default=0.0,
    help="Final gamma1 weight decay at end of training")
exp_parser.add_argument('--wd_end_freq', type=float, default=0.0,
    help="Final freq weight decay at end of training")

exp_parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                    help='num of steps to warmup LR, will overload warmup_epochs if set > 0')



exp_args = exp_parser.parse_args([])   # ← for naming/logging








# --------------------------
# 4.2.b. Patch Fairseq get_activation_fn
# --------------------------

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Ensure activation folder is in sys.path
if ACTIVATION_PATH not in sys.path:
    sys.path.append(ACTIVATION_PATH)

# ✅ Import Global FFTGate
from activation.FFTGate import FFTGate  
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Backup original get_activation_fn | ✅ utils from Fairseq
orig_get_activation_fn = utils.get_activation_fn

def custom_get_activation_fn(activation: str):
    activation = activation.lower()
    if activation == "tanhexp":
        from activation.TanhExp import TanhExp
        print("✅ Registered custom activation: TanhExp")
        return TanhExp()
    elif activation == "fftgate":
        from activation.FFTGate import FFTGate
        print("✅ Registered custom activation: FFTGate")
        return FFTGate()
    else:
        return orig_get_activation_fn(activation)

# ✅ Patch fairseq utils
utils.get_activation_fn = custom_get_activation_fn
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────────────────────────
print(f"⚡ Using activation function: {exp_args.act_name.lower()}")

# ✅ Test-call the patched activation to confirm it works
try:
    act_fn = utils.get_activation_fn(exp_args.act_name.lower())
    print(f"🔍 Activation function resolved: {act_fn}")
except Exception as e:
    print(f"❌ Failed to resolve activation '{exp_args.act_name.lower()}': {e}")
# ─────────────────────────────────────────────────────────────────────────────────────────────────





# --------------------------
# 4.2.c. Fairseq parser (model/task configs)
# --------------------------


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ use the patched parser

fs_parser = options.get_training_parser()



fairseq_args = options.parse_args_and_arch(
    fs_parser,
    input_args=[
        "data-bin/iwslt13.en-fr",   # ✅ EN→FR dataset directory
        "--arch", "transformer",    # ✅ standard transformer arch
        "--task", "translation",
        "--source-lang", "en",      # ✅ source language = English
        "--target-lang", "fr",      # ✅ target language = French
        "--encoder-layers", "6",
        "--decoder-layers", "6",
        "--encoder-embed-dim", "512",
        "--decoder-embed-dim", "512",
        "--dropout", "0.3",
        "--max-tokens", str(exp_args.max_tokens),     # ✅ keep as string
        
        # ✅ BLEU evaluation settings
        "--eval-bleu",
        "--eval-bleu-args", '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}',
    ],
)
cfg = convert_namespace_to_omegaconf(fairseq_args)
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Manually inject custom activation into config
cfg.model.activation_fn = exp_args.act_name.lower()
print(f"⚡ Overwrote cfg.model.activation_fn = {cfg.model.activation_fn}")
# ─────────────────────────────────────────────────────────────────────────────────────────────────






# --------------------------
# 4.2.d. Load dictionaries
# --------------------------
src_dict = Dictionary.load(os.path.join(fairseq_args.data, f"dict.{fairseq_args.source_lang}.txt"))
tgt_dict = Dictionary.load(os.path.join(fairseq_args.data, f"dict.{fairseq_args.target_lang}.txt"))

# --------------------------
# 4.2.f. Setup task 
# --------------------------

# ✅✅ Setup task
task = TranslationTask.setup_task(cfg.task)
task.load_dataset("train")
task.load_dataset("test")



# ✅ Print translation direction (FR→EN confirmation)
print(f"🌍 Translation direction: {fairseq_args.source_lang} → {fairseq_args.target_lang}")


# ✅ Inspect dataset lengths after loading
train_dataset = task.dataset("train")
test_dataset  = task.dataset("test")

print(f"📏 Train set - Max source length: {train_dataset.src_sizes.max()}, Max target length: {train_dataset.tgt_sizes.max()}")
print(f"📏 Test set  - Max source length: {test_dataset.src_sizes.max()}, Max target length: {test_dataset.tgt_sizes.max()}")


# ✅ Inspect data you’d lose if you lower max_positions
def check_percentage(dataset, N):
    src_long = (dataset.src_sizes > N).sum()
    tgt_long = (dataset.tgt_sizes > N).sum()
    total = len(dataset)

    print(f"Dataset size: {total}")
    print(f"  > {N} tokens (source): {src_long} ({100*src_long/total:.2f}%)")
    print(f"  > {N} tokens (target): {tgt_long} ({100*tgt_long/total:.2f}%)")
    print()

# ✅ Check on train + test
check_percentage(train_dataset, exp_args.max_source_positions)
check_percentage(test_dataset, exp_args.max_source_positions)







# --------------------------
# 4.2.g. model
# --------------------------

model = task.build_model(cfg.model)

model = model.to(DEVICE)



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🔍 Print custom activations to confirm they are used
for name, module in model.named_modules():
    if isinstance(module, nn.Module) and module.__class__.__name__ in ["TanhExp", "FFTGate"]:
        print(f"🔎 Found custom activation in {name}: {module}")



# ✅ Sanity check Device
if DEVICE.type == "cuda":
    print(f"🚀 Model moved to CUDA: {torch.cuda.get_device_name(DEVICE)}")
else:
    print("⚠️ Using CPU (CUDA not available)")
# ─────────────────────────────────────────────────────────────────────────────────────────────────










########################################################################################################################
####-------| NOTE 5. DYNAMIC REGULARIZATION OF FFTGATE PARAMS | XXX ---------------⚡---------------####################
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
    
    fftgateparam_reg = 0.0


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔴 Regularization toggles status (print only once per epoch)
    if batch_idx == 0 and (epoch % 10 == 0 or epoch == (exp_args.unfreeze_act_epochs + 1)):
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
    # ✅ === Adaptive Regularization strength schedule (epoch-based) ===
    warmup_epochs_main = round(exp_args.warmup_updates / num_training_steps_per_epoch)


    if epoch < warmup_epochs_main:   
        reg_strength = 0.001         # Early warmup phase                   
    elif epoch < exp_args.epochs * 0.5: 
        reg_strength = 0.0005        # Mid training         
    else:                           
        reg_strength = 0.001         # Final stabilization   

    # ✅ Debug print for sanity
    if batch_idx == 0 and epoch % 10 == 0:
        print(f"🔧 Reg strength at Epoch {epoch} | Batch {batch_idx}: {reg_strength:.5f} (warmup_epochs_main={warmup_epochs_main})")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Loop over activation layers
    noisy_layers = []
    for idx, layer in enumerate(activation_layers):
        if idx not in layer_index_map:
            continue
        if not (hasattr(layer, "gamma1_raw") and hasattr(layer, "freq_raw")):
            continue

        prev_layer_params = prev_params[layer_index_map[idx]]


        # ✅ Print layer state every 50 batches for sanity
        if  batch_idx == 0 and epoch % 10 == 0:
            print(f"[Layer {idx} at Epoch {epoch} | Batch {batch_idx}] γ₁={layer.gamma1_raw.item():+.4f} | f={layer.freq_raw.item():+.4f}")

        # 📉 raw FFTGate params
        gamma1_raw = layer.gamma1_raw
        freq_raw   = layer.freq_raw
        # ─────────────────────────────────────────────────────────────────────────────────────────────────


        # ─────────────────────────────────────────────────────────────────────────────────────────────────
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
        if enable_noise_reg and epoch > exp_args.addnoise_act_epochs:
            current_gamma1 = gamma1_raw
            prev_gamma1    = prev_layer_params["gamma1"]
            param_variation = torch.abs(current_gamma1.detach() - prev_gamma1).mean()
            if param_variation < 0.015:
                noise = (0.0005 + 0.0002 * batch_std.item()) * torch.randn_like(current_gamma1)
                penalty = (current_gamma1 - (prev_gamma1 + noise)).pow(2).mean()
                fftgateparam_reg += 0.00015 * penalty
                noisy_layers.append(f"{idx} (Δ={param_variation.item():.5f})")

    if batch_idx == 0 and epoch <= (exp_args.addnoise_act_epochs + 4) and noisy_layers:
        print(f"🔥 Stable Noise Injected | Epoch {epoch} | Batch {batch_idx} | Layers: " + ", ".join(noisy_layers), flush=True)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # === Cache mags + activation reg (optional) ===
    if enable_act_reg:
        if cache_interval == 0 or batch_idx % cache_interval == 0 or cached_mags is None:
            if feature_activations is None:
                cached_mags = None
            elif feature_activations.ndim == 4:
                cached_mags = feature_activations.abs().mean(dim=(0, 2, 3))  # ⬅️ no detach()
            elif feature_activations.ndim == 2:
                cached_mags = feature_activations.abs().mean(dim=0)
            else:
                raise ValueError(f"Unsupported shape for feature_activations: {feature_activations.shape}")

            if epoch <= 15 and (batch_idx == 0 or (num_batches is not None and batch_idx == num_batches - 1)):
                print(f"🔄 Recomputed mags (Epoch {epoch}, Batch {batch_idx})")

        if cached_mags is not None and epoch >= 10:
            m = cached_mags / (cached_mags.sum() + 1e-6)
            fftgateparam_reg += 0.003 * (-(m * torch.log(m + 1e-6)).sum())
    return fftgateparam_reg
# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ✅ === 🔒 Adaptive Target Regularization ===
def compute_target(param_name, batch_std, param_raw, prev_param_raw, reg_strength):
    prev_param_raw = prev_param_raw.detach()   # ✅ snapshot only
    return reg_strength * (param_raw - prev_param_raw).pow(2).mean()
# ─────────────────────────────────────────────────────────────────────────────────────────────────











########################################################################################################################
####-------| NOTE 6. INITIALIZATION AMP AND LOSS FUNCTION | XXX -----------------------------------#####################
########################################################################################################################



# ✅ Initialize AMP GradScaler once globall
scaler = torch.cuda.amp.GradScaler()




# --------------------------
# 6.a. Criterion
# --------------------------

criterion = LabelSmoothingCrossEntropy(smoothing=exp_args.smoothing)






########################################################################################################################
####-------| NOTE 7. INITIALIZATION FFTGATE OPTIMZER | XXX ----------------------------------------#####################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Effective batch size (your code uses max_tokens, so approximate here)
total_batch_size = exp_args.max_tokens   

# ✅ Per-epoch number of steps (matches your training iterator length)
num_training_steps_per_epoch = len(
    task.get_batch_iterator(
        dataset=task.dataset("train"),
        max_tokens=fairseq_args.max_tokens,
        max_positions=(exp_args.max_source_positions, exp_args.max_target_positions),
        # seed=1,        
        num_shards=exp_args.num_shards,                 
        shard_id=exp_args.shard_id,                     
        num_workers=exp_args.num_workers,
        ignore_invalid_inputs=exp_args.ignore_invalid_inputs
    ).next_epoch_itr(shuffle=True)
)


print(f"📏📏 Per-epoch training steps = {num_training_steps_per_epoch} | "
      f"Total batch size ≈ {total_batch_size}")
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ FFTGate scheduler — duplicate of main, but with scaled base LR | 🟡 Best setting so far
print("✅ Using Cosine LR schedulers for gamma1 and freq_factor (FFTGate)")

# 🔄 Define base learning rates for FFTGate params
lr_gamma1 = exp_args.actbase_lr * 2.0   # gamma1 → learns faster
lr_freq   = exp_args.actbase_lr * 1.0   # freq   → same as base LR

print(f"Gamma1 LR = {lr_gamma1:.6e} | WD = {exp_args.gamma1_wd}")
print(f"FreqFactor LR = {lr_freq:.6e} | WD = {exp_args.wd_freq}")
# ─────────────────────────────────────────────────────────────────────────────────────────────────





# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Cosine schedulers for activation params
gamma1_lr_schedule_values = my_utils.cosine_scheduler(
    lr_gamma1,                        # 🟡 gamma1 faster (2× base lr)
    exp_args.min_act_lr,
    exp_args.epochs,
    num_training_steps_per_epoch,
    warmup_epochs=exp_args.warm_act_epochs,   
    warmup_steps=exp_args.warmup_steps      
)

freq_lr_schedule_values = my_utils.cosine_scheduler(
    lr_freq,                         # 🟡 freq = base lr
    exp_args.min_act_lr,
    exp_args.epochs,
    num_training_steps_per_epoch,
    warmup_epochs=exp_args.warm_act_epochs,
    warmup_steps=exp_args.warmup_steps
)

# ✅ Weight-decay schedules (constant or cosine between start/end values)
gamma1_wd_schedule_values = my_utils.cosine_scheduler(
    exp_args.gamma1_wd,
    exp_args.gamma1_wd_end or exp_args.gamma1_wd,
    exp_args.epochs,
    num_training_steps_per_epoch
)

freq_wd_schedule_values = my_utils.cosine_scheduler(
    exp_args.wd_freq,
    exp_args.wd_end_freq or exp_args.wd_freq,
    exp_args.epochs,
    num_training_steps_per_epoch
)

# 🔎 Debug print for FFTGate schedulers only
print(f"🟡 Using per-step cosine schedules for FFTGate → "
      f"Gamma1LR[{len(gamma1_lr_schedule_values)}], "
      f"FreqLR[{len(freq_lr_schedule_values)}]")

print(f"🟡 Max/Min Gamma1LR: {max(gamma1_lr_schedule_values):.6f}/{min(gamma1_lr_schedule_values):.6f} | "
      f"Max/Min FreqLR: {max(freq_lr_schedule_values):.6f}/{min(freq_lr_schedule_values):.6f}")
# ─────────────────────────────────────────────────────────────────────────────────────────────────












########################################################################################################################
####-------| NOTE 7.2. SEARCH FOR FFTGate LAYERS| XXX ----------------------------------------------####################
########################################################################################################################

def find_activations_split(module, activation_layers, gamma1_params, freq_params):
    for layer in module.children():
        # ✅ Direct FFTGate check (Fairseq)
        if isinstance(layer, FFTGate):
            activation_layers.append(layer)            # append FFTGate object directly
            gamma1_params.append(layer.gamma1_raw)     # collect trainable γ₁ param
            freq_params.append(layer.freq_raw)         # collect trainable freq param

        # ✅ recurse deeper into children
        elif isinstance(layer, nn.Module):
            find_activations_split(layer, activation_layers, gamma1_params, freq_params)









########################################################################################################################
####-------| NOTE 7.3. INITIALIZE MAIN / ACTIVATION PARAMETERS & OPTIMIZERS | XXX ------------------####################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ---- 1) Collect FFTGate activation parameters (gamma1, freq_factor) from ConvNeXt model (via FFTGateAdapter.core) ----
gamma1_params: list = []
freq_params: list = []
activation_layers: list = []

find_activations_split(model, activation_layers, gamma1_params, freq_params)

# 1️⃣ give every FFTGate a compact id so log_params prints Layer1..LayerN
for idx, layer in enumerate(activation_layers, start=1):
    layer.layer_id = f"Layer{idx}"



# 2️⃣ Assign per-layer depth info for Transformer-safe FFTGate
total_layers = len(activation_layers)
for idx, layer in enumerate(activation_layers, start=1):
    layer.layer_depth = idx
    layer.total_layers = total_layers
    print(f"🧩 Assigned FFTGate depth {idx}/{total_layers}")
# ─────────────────────────────────────────────────────────────────────────────────────────────────





# # ─────────────────────────────────────────────────────────────────────────────────────────────────
# # ✅ === Original Main Optimizer === 
# # --------------------------
# # 6.b. Adam optimizer (β1=0.9, β2=0.98, eps=1e-9)
# # --------------------------
# optimizer = torch.optim.Adam(
#     model.parameters(),
#     lr=exp_args.lr,
#     betas=(0.9, 0.98),
#     eps=exp_args.eps,
#     weight_decay=exp_args.weight_decay,
# )
# # ─────────────────────────────────────────────────────────────────────────────────────────────────





# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ---- 2) Build MAIN optimizer that EXCLUDES gamma1, freq_factor ----
if gamma1_params or freq_params:
    act_id_set = {id(p) for p in (gamma1_params + freq_params)}  
    main_params = [p for p in model.parameters() if id(p) not in act_id_set]
    optimizer = torch.optim.Adam(
        main_params,
        lr=exp_args.lr,
        betas=(0.9, 0.98),
        eps=exp_args.eps,
        weight_decay=exp_args.weight_decay,
    ) 
else:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=exp_args.lr,
        betas=(0.9, 0.98),
        eps=exp_args.eps,
        weight_decay=exp_args.weight_decay,
    )

# ✅ Sanity check
if gamma1_params or freq_params:
    main_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
    act_ids  = {id(p) for p in (gamma1_params + freq_params)}
    assert main_ids.isdisjoint(act_ids), "gamma1/freq_factor still inside MAIN optimizer!"
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ---- 3) Initialize ACTIVATION optimizers & schedulers (start frozen) ----
activation_optimizers = {}
gamma1_lr_schedule, freq_lr_schedule = None, None  

if gamma1_params or freq_params:
    unfreeze_activation_epoch = exp_args.unfreeze_act_epochs
    WARMUP_ACTIVATION_EPOCHS = exp_args.warm_act_epochs

    # freeze initially
    for p in gamma1_params + freq_params:
        p.requires_grad = False

    # Separate optimizers only if params exist
    if gamma1_params:
        activation_optimizers["gamma1"] = torch.optim.AdamW(
            gamma1_params, 
            lr=lr_gamma1,             # 🟡 gamma1 learns faster
            weight_decay=exp_args.gamma1_wd
        )
        gamma1_lr_schedule = my_utils.cosine_scheduler(
            lr_gamma1,
            exp_args.min_act_lr,
            exp_args.epochs,
            num_training_steps_per_epoch, 
            warmup_epochs=WARMUP_ACTIVATION_EPOCHS,
            warmup_steps=exp_args.warmup_steps
        )

    if freq_params:
        activation_optimizers["freq_factor"] = torch.optim.AdamW(
            freq_params, 
            lr=lr_freq,             # 🟡 freq slower than gamma1
            weight_decay=exp_args.wd_freq
        )
        freq_lr_schedule = my_utils.cosine_scheduler(
            lr_freq,
            exp_args.min_act_lr,
            exp_args.epochs,
            num_training_steps_per_epoch, 
            warmup_epochs=WARMUP_ACTIVATION_EPOCHS,
            warmup_steps=exp_args.warmup_steps
        )
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# Final fallback if nothing was found
if not activation_optimizers:
    activation_optimizers = None



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ---- 4) Logging ----
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
####-------| NOTE 7.4. FINALIZE MAIN SCHEDULERs | XXX ----------------------------------------------####################
########################################################################################################################


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# --------------------------
# 6.c. Transformer LR schedule (warmup + inverse sqrt decay)
# --------------------------

warmup_updates = exp_args.warmup_updates   # configurable warmup steps
peak_lr = exp_args.lr


def lr_lambda(step):
    step = max(1, step)
    if step == 1:
        print(f"🚀 Warmup started at step {step} (target peak LR = {peak_lr}).")
    if step == warmup_updates:
        actual_lr = optimizer.param_groups[0]["lr"]
        print(f"✅ Warmup finished at step {step}. Peak LR reached: {actual_lr:.8f}")
    if step == warmup_updates + 1:
        print(f"📉 Inverse square root decay started at step {step}.")
    
    if step <= warmup_updates:
        # Linear warmup
        scale = step / warmup_updates
    else:
        # Inverse square root decay
        scale = (warmup_updates ** 0.5) / (step ** 0.5)
    return scale



lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
# ─────────────────────────────────────────────────────────────────────────────────────────────────













########################################################################################################################
####-------| NOTE 7.5. FINALIZE ACT SCHEDULERs | XXX -----------------------------------------------####################
########################################################################################################################


# ✅ Compute starting global step so schedules line up perfectly after resume
global_start_step = exp_args.start_epoch * num_training_steps_per_epoch  # ✅ aligns with ConvNeXt
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
        wd_this_step = exp_args.gamma1_wd
    elif key == "freq_factor":
        wd_this_step = exp_args.wd_freq
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
####-------| NOTE 8. PATH DEFINATION AND GLOBAL INITAILIZATION | XXX ------------------------------#####################
########################################################################################################################


# ✅ Paths for logs
LR_save_paths = {
    "LR_history": f"./Results/{exp_args.dataset_name}/{exp_args.act_name}/{exp_args.act_name}_{exp_args.net}_{exp_args.dataset_name}_{exp_args.optimizer1}_{exp_args.mode_name}_LR_log.txt"
}
save_paths = {
    "log_history": f"./Results/{exp_args.dataset_name}/{exp_args.act_name}/{exp_args.act_name}_{exp_args.net}_{exp_args.dataset_name}_{exp_args.optimizer1}_{exp_args.mode_name}_training_logs.txt"
}
train_results_path = f'./Results/{exp_args.dataset_name}/{exp_args.act_name}/Train_{exp_args.act_name}_{exp_args.net}_{exp_args.dataset_name}_{exp_args.optimizer1}_{exp_args.mode_name}.txt'
test_results_path = f'./Results/{exp_args.dataset_name}/{exp_args.act_name}/Test_{exp_args.act_name}_{exp_args.net}_{exp_args.dataset_name}_{exp_args.optimizer1}_{exp_args.mode_name}.txt'




# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === Define path to store FFTGate Params log === 
FFTGate_param_save_paths = {       
    "FFTGate_param_history": f"./Results/{exp_args.dataset_name}/{exp_args.act_name}/{exp_args.act_name}_{exp_args.net}_{exp_args.dataset_name}_{exp_args.optimizer1}_{exp_args.mode_name}_FFTGate_param_log.txt"  # ✅ Training log_history     
}
FFTGate_Grad_save_paths = {       
    "FFTGate_Grad_history": f"./Results/{exp_args.dataset_name}/{exp_args.act_name}/{exp_args.act_name}_{exp_args.net}_{exp_args.dataset_name}_{exp_args.optimizer1}_{exp_args.mode_name}_FFTGate_Grad_log.txt"  # ✅ Training log_history     
}
FFTGate_SummaryGrad_save_paths = {       
    "FFTGate_SummaryGrad_history": f"./Results/{exp_args.dataset_name}/{exp_args.act_name}/{exp_args.act_name}_{exp_args.net}_{exp_args.dataset_name}_{exp_args.optimizer1}_{exp_args.mode_name}_FFTGate_SummaryGrad_log.txt"  # ✅ Training log_history     
}
FFTGate_Main_SummaryGrad_save_paths = {       
    "FFTGate_Main_SummaryGrad_history": f"./Results/{exp_args.dataset_name}/{exp_args.act_name}/{exp_args.act_name}_{exp_args.net}_{exp_args.dataset_name}_{exp_args.optimizer1}_{exp_args.mode_name}_FFTGate_Main_SummaryGrad_log.txt"  # ✅ Training log_history     
}
# ─────────────────────────────────────────────────────────────────────────────────────────────────




# ✅ Path template for saving checkpoints at every epoch
checkpoint_dir = f'./checkpoints/{exp_args.dataset_name}/{exp_args.act_name}/'
os.makedirs(checkpoint_dir, exist_ok=True)


#🌍 global initialization
best_bleu = 0.0
total_test_duration = 0.0







# %%





########################################################################################################################
####-------| NOTE 9. TRAIN FUNCTIONS | XXX --------------------------------------------------------#####################
########################################################################################################################


# ✅ === Training  ===
def train_one_epoch(epoch, task, model, optimizer, criterion, lr_scheduler, activation_optimizers, unfreeze_activation_epoch, global_start_step):
    #🌍
    global train_loss_history, best_train_acc, prev_params, recent_test_acc, gamma1_history, activation_layers, activation_params, test_acc_history, gamma1_params, freq_params   # 🟡 bring the param lists into this scope


    epoch_start_time = time.time()




    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Initialize histories and activation params tracking ===
    if epoch == exp_args.start_epoch:   
        train_loss_history, test_acc_history = [], []    # BLEU/test acc history
        best_train_acc, recent_test_acc = 0.0, 0.0
        gamma1_history, freq_history = {}, {}            # per-layer gamma1/Freq tracking
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



    model.train()



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
    train_iter = task.get_batch_iterator(
        dataset=task.dataset("train"),
        max_tokens=fairseq_args.max_tokens,
        max_positions=(exp_args.max_source_positions, exp_args.max_target_positions),
        # seed=1,        
        num_shards=exp_args.num_shards,                 
        shard_id=exp_args.shard_id,                     
        num_workers=exp_args.num_workers,
        ignore_invalid_inputs=exp_args.ignore_invalid_inputs
    ).next_epoch_itr(shuffle=True)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────



    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    train_loss, correct, total, train_accuracy, epoch_loss = 0, 0, 0, 0.0, 0.0

    # ✅ Initialize log histories
    log_history, lr_log_history, FFTGate_param_history, FFTGate_Grad_history, FFTGate_SummaryGrad_history, FFTGate_Main_SummaryGrad_history, activation_history = [], [], [], [], [], [], []  
    # ─────────────────────────────────────────────────────────────────────────────────────────────────



    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ counters reset each epoch
    gamma1_clip_count_epoch, freq_clip_count_epoch, gamma1_total_epoch, freq_total_epoch = 0, 0, 0, 0

    # ✅ Clamp collectors for epoch summary
    gamma_clamp_values, freq_clamp_values = [], []
    # ─────────────────────────────────────────────────────────────────────────────────────────────────



    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Compute this epoch's base step index for per-step schedule (ConvNeXt-style)
    epoch_base_step = global_start_step + (epoch - exp_args.start_epoch) * num_training_steps_per_epoch



    # ✅ Unfreeze at specified epoch
    if epoch == unfreeze_activation_epoch:
        print("\n🔓 Unfreezing FFTGate Parameters 🔓")
        for p in (gamma1_params + freq_params):   
            p.requires_grad = True

        # rebuild optimizers if frozen
        if gamma1_params:
            activation_optimizers["gamma1"] = torch.optim.AdamW(
                gamma1_params, lr=0.0, weight_decay=exp_args.gamma1_wd
            )
        if freq_params:
            activation_optimizers["freq_factor"] = torch.optim.AdamW(
                freq_params, lr=0.0, weight_decay=exp_args.wd_freq
            )


        # ✅ Sync schedulers immediately
        step_idx = epoch_base_step
        g_lr, g_wd = _apply_single_lr_at_step("gamma1", activation_optimizers, gamma1_lr_schedule_values, step_idx)
        f_lr, f_wd = _apply_single_lr_at_step("freq_factor", activation_optimizers, freq_lr_schedule_values, step_idx)


        print(f"✅ Gamma1 Optimizer Synced! LR={g_lr if g_lr else 0:.6f}, WD={g_wd if g_wd else 0:.6f}")
        print(f"✅ FreqFactor Optimizer Synced! LR={f_lr if f_lr else 0:.6f}, WD={f_wd if f_wd else 0:.6f}")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────






   # ─────────────────────────────────────────────────────────────────────────────────────────────────
    with tqdm(enumerate(train_iter), total=len(train_iter), desc=f"Train Epoch {epoch}") as progress:
        for batch_idx, sample in progress:

            # ✅ Move entire sample dict to DEVICE
            sample = utils.move_to_cuda(sample) if torch.cuda.is_available() else sample

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
            detailed_steps = {0, 1, 2, 5, 10, 50, 70, 100, 200, 500, 800, 1000}
            detailed_steps.add(len(train_iter) - 1)
            milestone_epochs = {0, 1, 3, 5, 10, 20, 30, 40, 45}
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
            src_tokens = sample["net_input"]["src_tokens"]
            prev_output_tokens = sample["net_input"]["prev_output_tokens"]
            target = sample["target"]

            src_lengths = (src_tokens != src_dict.pad()).sum(dim=1)
           # ─────────────────────────────────────────────────────────────────────────────────────────────────


           # ─────────────────────────────────────────────────────────────────────────────────────────────────
           # ✅ zero_grad: Main & FFTGate
            optimizer.zero_grad()

            # zero_grad activation parameter
            for act_opt in activation_optimizers.values():
                act_opt.zero_grad()
           # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # # ✅ Forward pass for loss without AMP
            # logits, _ = model(src_tokens, src_lengths, prev_output_tokens)


            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ ⚠️ Forward + loss under autocast (AMP) with safe try/except 
            try:    
                with torch.cuda.amp.autocast():   # AMP context
                    logits, _ = model(src_tokens, src_lengths, prev_output_tokens)

                    # ✅ with (pad-masked)
                    pad = tgt_dict.pad()
                    logits2d = logits.view(-1, logits.size(-1))
                    target1d = target.view(-1)
                    mask = target1d != pad
                    if mask.any():
                        loss = criterion(logits2d[mask], target1d[mask])
                    else:
                        loss = logits2d.sum() * 0.0  # degenerate case safeguard


                    # ✅ Check for NaN/Inf loss
                    if not torch.isfinite(loss):
                        print(f"💥⚠️ Training Loop: Non-finite loss (NaN/Inf) at epoch {epoch}, batch {batch_idx}. Skipping batch.")
                        torch.cuda.empty_cache()
                        continue   # Skip this batch safely

            except RuntimeError as e:  
                print(f"💥 ⚠️ Training Loop: AMP forward failed at epoch {epoch}, batch {batch_idx}: {e}")  # AMP failure
                torch.cuda.empty_cache()
                continue   # Skip this batch and move on safely
            # ─────────────────────────────────────────────────────────────────────────────────────────────────



            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # 🔍 Debug BEFORE regularization (with sign + MIN/MAX summary)
            if (epoch in milestone_epochs) and (batch_idx in detailed_steps):

                gamma1_raw_vals, gamma1_eff_vals = [], []
                freq_raw_vals,  freq_eff_vals  = [], []

                gamma1_raw_strs, gamma1_eff_strs = [], []
                freq_raw_strs,  freq_eff_strs  = [], []

                for layer_idx, layer in enumerate(activation_layers, start=1):
                    g_raw = layer.gamma1_raw.item()
                    g_eff = layer.gamma1_eff().item()
                    f_raw = layer.freq_raw.item()
                    f_eff = layer.freq_factor_eff().item()

                    gamma1_raw_vals.append(g_raw); gamma1_eff_vals.append(g_eff)
                    freq_raw_vals.append(f_raw);   freq_eff_vals.append(f_eff)

                    gamma1_raw_strs.append(f"Layer{layer_idx}: {g_raw:+.6f}")
                    gamma1_eff_strs.append(f"Layer{layer_idx}: {g_eff:+.6f}")
                    freq_raw_strs.append(f"Layer{layer_idx}: {f_raw:+.6f}")
                    freq_eff_strs.append(f"Layer{layer_idx}: {f_eff:+.6f}")

                log_msg = (
                    f"Epoch {epoch} | Batch {batch_idx} | BEFORE reg || "
                    f"Gamma1 RAW (MIN: {min(gamma1_raw_vals):+.6f} / MAX: {max(gamma1_raw_vals):+.6f}): "
                    f"[ {' | '.join(gamma1_raw_strs)} ] || "
                    f"Gamma1 EFF (MIN: {min(gamma1_eff_vals):+.6f} / MAX: {max(gamma1_eff_vals):+.6f}): "
                    f"[ {' | '.join(gamma1_eff_strs)} ] || "
                    f"Freq RAW (MIN: {min(freq_raw_vals):+.6f} / MAX: {max(freq_raw_vals):+.6f}): "
                    f"[ {' | '.join(freq_raw_strs)} ] || "
                    f"Freq EFF (MIN: {min(freq_eff_vals):+.6f} / MAX: {max(freq_eff_vals):+.6f}): "
                    f"[ {' | '.join(freq_eff_strs)} ]"
                )

                # print(log_msg)
                FFTGate_param_history.append(log_msg)  # ✅ Save for param
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # 🧭 Call Regularization Function for the Activation Parameter (gamma1 + freq_factor)
            if epoch >= unfreeze_activation_epoch:                
                fftgate_reg = apply_dynamic_regularization(
                    src_tokens.float(), None, epoch,
                    prev_params, layer_index_map, batch_idx, 
                    num_batches=num_training_steps_per_epoch    # ✅ globally consistent
                )
                loss += fftgate_reg
            # ─────────────────────────────────────────────────────────────────────────────────────────────────



            # loss.backward()
            # optimizer.step()

            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ ⚠️ Backward pass with gradient scaler (AMP)
            scaler.scale(loss).backward()
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
                    gamma1_raw_strs.append(f"Layer{layer_idx}: {g_raw:+.6f}")
                    gamma1_eff_strs.append(f"Layer{layer_idx}: {g_eff:+.6f}")
                    freq_raw_strs.append(f"Layer{layer_idx}: {f_raw:+.6f}")
                    freq_eff_strs.append(f"Layer{layer_idx}: {f_eff:+.6f}")

                # Safe handling if lists are empty (avoid min/max on [])
                gamma1_raw_summary = (
                    f"(MIN: {min(gamma1_raw_vals):+.6f} / MAX: {max(gamma1_raw_vals):+.6f})"
                    if gamma1_raw_vals else "(no values)"
                )
                gamma1_eff_summary = (
                    f"(MIN: {min(gamma1_eff_vals):+.6f} / MAX: {max(gamma1_eff_vals):+.6f})"
                    if gamma1_eff_vals else "(no values)"
                )
                freq_raw_summary = (
                    f"(MIN: {min(freq_raw_vals):+.6f} / MAX: {max(freq_raw_vals):+.6f})"
                    if freq_raw_vals else "(no values)"
                )
                freq_eff_summary = (
                    f"(MIN: {min(freq_eff_vals):+.6f} / MAX: {max(freq_eff_vals):+.6f})"
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
                AMP_scale_value = scaler.get_scale() if 'scaler' in globals() else 1.0
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
                        gamma1_grad_strs.append(f"Layer{layer_idx}: {g_grad:+.6f}")


                    # --- Freq grad ---
                    if layer.freq_raw.grad is None:
                        freq_grad_strs.append(f"Layer{layer_idx}: grad=None")
                    elif torch.all(layer.freq_raw.grad == 0):
                        freq_grad_strs.append(f"Layer{layer_idx}: grad=ZERO")
                    else:
                        # 🎀 apply same unscale correction for freq gradients
                        f_grad = layer.freq_raw.grad.item() * inv_scale
                        freq_grad_vals.append(f_grad)
                        freq_grad_strs.append(f"Layer{layer_idx}: {f_grad:+.6f}")
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
            # ✅ === Apply Optimizer Step: Model + FFTGate Parameters   === ⚠️ 🔄 Use AMP if enabled===

            # 🔖 Main model optimizer step (excludes FFTGate params)
            scaler.step(optimizer)       

            # 🔖 Apply FFTGate optimizer steps (only when unfrozen)
            if epoch >= unfreeze_activation_epoch:
                if "gamma1" in activation_optimizers:
                        scaler.step(activation_optimizers["gamma1"])

                if "freq_factor" in activation_optimizers:
                        scaler.step(activation_optimizers["freq_factor"])


            # ✅ === Update dynamic loss scale (AMP safety check) ===
            # 🔖 Adjusts AMP internal scale based on overflow; does NOT normalize real gradients.
            scaler.update()       
            # ─────────────────────────────────────────────────────────────────────────────────────────────────



            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ Advance main Transformer LR schedule (per batch)
            # 🔖 FFTGate schedulers are updated manually via `_apply_single_lr_at_step()` above.
            lr_scheduler.step() 
            # ─────────────────────────────────────────────────────────────────────────────────────────────────



            epoch_loss += loss.item()


            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            progress.set_postfix(
                loss=round(epoch_loss / (batch_idx + 1), 3),
                lr=f"{lr_now:.8f}",
                wd=f"{wd_now:.8f}",
                Gamma1_LR=f"{g_lr:.8f}" if g_lr is not None else "FROZEN",
                Freq_LR=f"{f_lr:.8f}" if f_lr is not None else "FROZEN"
            )
            # ─────────────────────────────────────────────────────────────────────────────────────────────────



    avg_loss = epoch_loss / len(train_iter)



    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅  Update prev_params here AFTER all updates | END OF EACH EPOCH (AFTER optimizer.step())
    for idx, layer in enumerate(activation_layers):
        if isinstance(layer, FFTGate):
            prev_params[idx] = {
                "gamma1": layer.gamma1_raw.clone().detach(),   # ✅ detach snapshot only
                "freq_factor": layer.freq_raw.clone().detach()
            }
    # ─────────────────────────────────────────────────────────────────────────────────────────────────




    # ✅ Timing
    duration = time.time() - epoch_start_time
    mins, secs = divmod(duration, 60)
    print(f"⏱ Epoch {epoch} Training time {exp_args.act_name}_{exp_args.net}: {int(mins)} min {secs:.2f} sec")





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
    AMP_scale_value = scaler.get_scale() if 'scaler' in globals() else 1.0
    inv_scale = 1.0 / AMP_scale_value
    # ───────────────────────────────────────────────────
 
    # ────────────────────────────────────────────────────────────────────────────────────────────────
       # Log RAW + EFF + Grad for Gamma1
    for name, p in model.named_parameters():
        if name.endswith("gamma1_raw"):
            v = p.data.item()
            gamma1_param_values.append(f"{name}: {v:+.6f}")
            gamma1_param_nums.append(v)

            if p.grad is None:
                gamma1_grad_values.append(f"{name}: grad=None")
            else:
                # 🎀 Apply AMP unscale correction to log true gradient magnitude
                g = p.grad.item() * inv_scale
                gamma1_grad_values.append(f"{name}: {g:+.6f}")
                gamma1_grad_nums.append(g)

            module_name = name.rsplit(".", 1)[0]  # strip off ".gamma1_raw"
            module = dict(model.named_modules())[module_name]
            v_eff = module.gamma1_eff().item()
            gamma1_eff_values.append(f"{name}: {v_eff:+.6f}")
            gamma1_eff_nums.append(v_eff)

    for name, p in model.named_parameters():
        if name.endswith("freq_raw"):
            v = p.data.item()
            freq_param_values.append(f"{name}: {v:+.6f}")
            freq_param_nums.append(v)

            if p.grad is None:
                freq_grad_values.append(f"{name}: grad=None")
            else:
                # 🎀Apply AMP unscale correction to freq gradients
                g = p.grad.item() * inv_scale
                freq_grad_values.append(f"{name}: {g:+.6f}")
                freq_grad_nums.append(g)
        
            module_name = name.rsplit(".", 1)[0]  # strip off ".freq_raw"
            module = dict(model.named_modules())[module_name]
            v_eff = module.freq_factor_eff().item()
            freq_eff_values.append(f"{name}: {v_eff:+.6f}")
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
        f"(MIN: {min(gamma1_param_nums):+.6f} / MAX: {max(gamma1_param_nums):+.6f})"
        if gamma1_param_nums else "(no values)"
    )
    gamma1_eff_summary = (
        f"(MIN: {min(gamma1_eff_nums):+.6f} / MAX: {max(gamma1_eff_nums):+.6f})"
        if gamma1_eff_nums else "(no values)"
    )
    freq_raw_summary = (
        f"(MIN: {min(freq_param_nums):+.6f} / MAX: {max(freq_param_nums):+.6f})"
        if freq_param_nums else "(no values)"
    )
    freq_eff_summary = (
        f"(MIN: {min(freq_eff_nums):+.6f} / MAX: {max(freq_eff_nums):+.6f})"
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
        f"⏱ Training time => {exp_args.act_name}_{exp_args.net}: {int(mins)} min {secs:.2f} sec"
    )

    log_history.append(log_msg)
    print(log_msg)
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────────────────────────────────────────



    # # ────────────────────────────────────────────────────────────────────────────────────────────────
    # # ✅ ✂️ Per-epoch clipping summary (dynamic with median + min/max clamp values)
    # # ────────────────────────────────────────────────────────────────────────────────────────────────



    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Save logs | Clear the log file at the start of training (Epoch 0)
    if epoch == exp_args.start_epoch:
        os.makedirs(os.path.dirname(train_results_path), exist_ok=True)
        with open(train_results_path, "w", encoding="utf-8") as f:
            f.write("")  # ✅ Clears previous logs only once at the start

    with open(train_results_path, "a", encoding="utf-8") as f:
        f.write(f"Epoch {epoch} | Train Loss: {avg_loss:.3f} | LR: {optimizer.param_groups[0]['lr']:.8f}\n")
    # ────────────────────────────────────────────────────────────────────────────────────────────────




    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Initialize log file at the beginning of training (Clear old logs)
    if epoch == exp_args.start_epoch:  # ✅ Only clear at the start of training
        with open(save_paths["log_history"], "w", encoding="utf-8") as log_file:
            log_file.write("")  # ✅ Clears previous logs

    # ✅ Save logs once per epoch (Append new logs)
    if log_history:
        with open(save_paths["log_history"], "a", encoding="utf-8") as log_file:
            log_file.write("\n".join(log_history) + "\n")          # ✅ Ensure each entry is on a new line
        print(f"📜 Logs saved to {save_paths['log_history']}!")    # ✅ Only prints once per epoch
    else:
        print("⚠ No logs to save!")



    # ✅ Save LR log history to file (once per epoch)
    if epoch == exp_args.start_epoch:
        os.makedirs(os.path.dirname(LR_save_paths["LR_history"]), exist_ok=True)
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
    if epoch == exp_args.start_epoch:
        with open(FFTGate_param_save_paths["FFTGate_param_history"], "w", encoding="utf-8") as f:
            f.write("")  # Clear previous content on first epoch
    if FFTGate_param_history:
        with open(FFTGate_param_save_paths["FFTGate_param_history"], "a", encoding="utf-8") as f:
            f.write("\n".join(FFTGate_param_history) + "\n")
        print(f"📈 FFTGate param logs saved to {FFTGate_param_save_paths['FFTGate_param_history']}!")

    # ✅ Save FFTGate Grad log history to file (once per epoch)
    if epoch == exp_args.start_epoch:
        with open(FFTGate_Grad_save_paths["FFTGate_Grad_history"], "w", encoding="utf-8") as f:
            f.write("")  # Clear previous content on first epoch
    if FFTGate_Grad_history:
        with open(FFTGate_Grad_save_paths["FFTGate_Grad_history"], "a", encoding="utf-8") as f:
            f.write("\n".join(FFTGate_Grad_history) + "\n")


    # ✅ Save FFTGate Summary Grad log history to file (once per epoch)
    if epoch == exp_args.start_epoch:
        with open(FFTGate_SummaryGrad_save_paths["FFTGate_SummaryGrad_history"], "w", encoding="utf-8") as f:
            f.write("")  # Clear previous content on first epoch
    if FFTGate_SummaryGrad_history:
        with open(FFTGate_SummaryGrad_save_paths["FFTGate_SummaryGrad_history"], "a", encoding="utf-8") as f:
            f.write("\n".join(FFTGate_SummaryGrad_history) + "\n")

    # ✅ Save FFTGate _Main_Summary Grad log history to file (once per epoch)
    if epoch == exp_args.start_epoch:
        with open(FFTGate_Main_SummaryGrad_save_paths["FFTGate_Main_SummaryGrad_history"], "w", encoding="utf-8") as f:
            f.write("")  # Clear previous content on first epoch
    if FFTGate_Main_SummaryGrad_history:
        with open(FFTGate_Main_SummaryGrad_save_paths["FFTGate_Main_SummaryGrad_history"], "a", encoding="utf-8") as f:
            f.write("\n".join(FFTGate_Main_SummaryGrad_history) + "\n")
    # ────────────────────────────────────────────────────────────────────────────────────────────────



    print(f"📊 Train Loss: {avg_loss:.3f} | LR: {optimizer.param_groups[0]['lr']:.8f}")
    print(f"📜 Training logs saved to {train_results_path}!")



    # ──────────────────────────────────
    # ❌ GPU memory check at At the end of each epoch (training)
    if torch.cuda.is_available():
        print(f"[❗Epoch {epoch} AFTER TRAIN] GPU mem allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB | reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    # ──────────────────────────────────



    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Save checkpoint at the end of each epoch
    ckpt_path = os.path.join(checkpoint_dir,
        f"epoch_{epoch}_{exp_args.act_name}_{exp_args.net}_{exp_args.dataset_name}_{exp_args.mode_name}.pt"
    )
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "lr_scheduler_state": lr_scheduler.state_dict(),
        
        # ✅ Save activation optimizers too (gamma1/freq) so resume restores their states
        "activation_optimizers": {k: opt.state_dict() for k, opt in activation_optimizers.items()} if activation_optimizers else None,
        
        # ✅ Save LR schedules (so cosine schedulers don’t restart incorrectly after resume)
        "gamma1_lr_schedule": gamma1_lr_schedule_values,
        "freq_lr_schedule": freq_lr_schedule_values,

        "train_loss": avg_loss
    }, ckpt_path)

    print(f"🟡🟡 Epoch {epoch} checkpoint saved to: {ckpt_path}")
    # ────────────────────────────────────────────────────────────────────────────────────────────────






    return avg_loss









########################################################################################################################
####-------| NOTE 10. TEST FUNCTIONS (with sacreBLEU) | BEST PRACTICE ------------------------------####################
########################################################################################################################




# ────────────────────────────────────────────────────────────────────────────────────────────────
def detokenize_for_bleu(s: str) -> str:
    s = s.replace("@@ ", "").replace("@@", "")  # BPE
    s = s.replace("▁", " ")                     # SentencePiece
    s = " ".join(s.split())                     # normalize spaces
    s = re.sub(r"\s+([?.!,])", r"\1", s)        # punctuation
    return s.strip()
# ────────────────────────────────────────────────────────────────────────────────────────────────





# ✅ === Testing  ===
def test_one_epoch(epoch, task, model, criterion, beam_size=None):
    #🌍
    global best_bleu, total_test_duration 

    # ──────────────────────────────────
    # ❌ GPU memory check At test start
    if torch.cuda.is_available():
        print(f"[❗Epoch {epoch} TEST START] GPU mem allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB | reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    # ──────────────────────────────────


    model.eval()




    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔖 Sync current epoch into all FFTGate layers before test batches
    for layer in activation_layers:
        if isinstance(layer, FFTGate):
            layer.current_epoch.fill_(epoch)
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Build test iterator first
    test_iter = task.get_batch_iterator(
        dataset=task.dataset("test"),
        max_tokens=fairseq_args.max_tokens,  # cut in half option: fairseq_args.max_tokens // 2
        max_positions=(exp_args.max_source_positions, exp_args.max_target_positions),
        num_shards=exp_args.num_shards,
        shard_id=exp_args.shard_id,
        num_workers=exp_args.num_workers,
        ignore_invalid_inputs=exp_args.ignore_invalid_inputs,
            ).next_epoch_itr(shuffle=False)



    # ✅ References
    true_references = []
    for sample in test_iter:
        targets = sample["target"]
        for t in targets:
            t_no_pad = t[t != tgt_dict.pad()]  # remove pads
            ref_str = tgt_dict.string(t_no_pad, bpe_symbol="@@")
            true_references.append(detokenize_for_bleu(ref_str))
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ❗ Rebuild test_iter again (since the first one is exhausted after above loop)
    test_iter = task.get_batch_iterator(
        dataset=task.dataset("test"),
        max_tokens=fairseq_args.max_tokens,
        max_positions=(exp_args.max_source_positions, exp_args.max_target_positions),
        num_shards=exp_args.num_shards,
        shard_id=exp_args.shard_id,
        num_workers=exp_args.num_workers,
        ignore_invalid_inputs=exp_args.ignore_invalid_inputs,     
    ).next_epoch_itr(shuffle=False)
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ ===== Build a LOCAL generation config =====
    gen_cfg = deepcopy(cfg.generation)
    if beam_size is not None:
        gen_cfg.beam = int(beam_size)
    # else: keep whatever was set in --eval-bleu-args (beam=5, max_len_a=1.2, max_len_b=10)

    gen_cfg.lenpen = 1.0   # ✅ Fairseq default (neutral length penalty: no bias for longer/shorter translations)

    generator = task.build_generator([model], gen_cfg)
    # ────────────────────────────────────────────────────────────────────────────────────────────────



    test_loss = 0.0
    sys_outputs = []
    start_time = time.time()


   # ────────────────────────────────────────────────────────────────────────────────────────────────
    with torch.no_grad():
        with tqdm(enumerate(test_iter), total=len(test_iter), desc=f"Test Epoch {epoch}") as progress:
            for batch_idx, sample in progress:
                # ✅ Move the whole sample to device (safe on CPU/GPU)
                sample = utils.move_to_cuda(sample) if torch.cuda.is_available() else sample

                src_tokens = sample["net_input"]["src_tokens"]
                prev_output_tokens = sample["net_input"]["prev_output_tokens"]
                target = sample["target"]
                src_lengths = (src_tokens != src_dict.pad()).sum(dim=1)



                # # ✅ Forward pass for loss without AMP
                # logits, _ = model(src_tokens, src_lengths, prev_output_tokens)



                # ────────────────────────────────────────────────────────────────────────────────────────────────
                # ✅ ⚠️ Forward + loss under autocast (AMP) with safe try/except
                try:
                    with torch.cuda.amp.autocast():   # AMP context
                        logits, _ = model(src_tokens, src_lengths, prev_output_tokens)

                        pad = tgt_dict.pad()
                        logits2d = logits.view(-1, logits.size(-1))
                        target1d = target.view(-1)
                        mask = target1d != pad
                        loss = criterion(logits2d[mask], target1d[mask]) if mask.any() else logits2d.sum() * 0.0


                        # ✅ Check for NaN/Inf loss
                        if not torch.isfinite(loss):
                            print(f"💥⚠️ Test Loop: Non-finite loss (NaN/Inf) at epoch {epoch}, batch {batch_idx}. Skipping batch.")
                            torch.cuda.empty_cache()
                            continue           # Skip this batch safely

                # ✅ Check for Runtime Error
                except RuntimeError as e:  
                    print(f"💥 ⚠️ Test Loop: AMP forward failed at epoch {epoch}, batch {batch_idx}: {e}")  # AMP failure
                    torch.cuda.empty_cache()
                    continue                   # Skip this batch and move on safely
                # ────────────────────────────────────────────────────────────────────────────────────────────────



                # ✅ Compute Loss
                test_loss += loss.item()




                # ────────────────────────────────────────────────────────────────────────────────────────────────
                # ✅ Proper decoding via generator (uses local beam_size)
                hypos = task.inference_step(generator, [model], sample)


                # ✅ System outputs
                for j, hypos_j in enumerate(hypos):
                    hypo_tokens = [tok for tok in hypos_j[0]["tokens"].tolist() if tok != tgt_dict.pad()]  # remove pads
                    sys_str = tgt_dict.string(hypo_tokens, bpe_symbol="@@")
                    sys_outputs.append(detokenize_for_bleu(sys_str))


                # # 🔍 also log the aligned source sentence for debugging (only once for very first example overall)
                # if len(sys_outputs) == 1:
                #     src_str = src_dict.string(sample["net_input"]["src_tokens"][j], bpe_symbol="@@")
                #     print("----- DEBUG OUTPUT (FIRST EXAMPLE) -----")
                #     print("SRC:", detokenize_for_bleu(src_str))
                #     print("SYS:", sys_outputs[0], "| REF:", true_references[0])
                #     print("---------------------------------------")


                progress.set_postfix(loss=round(test_loss / (batch_idx + 1), 3))
               # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ✅ Compute Ave Loss
    avg_loss = test_loss / max(1, len(test_iter))


    # # 🔍 Debug: check alignment again at summary level
    # print("----- DEBUG OUTPUT -----")
    # print("SYS[0]:", sys_outputs[0][:200])   # first hypothesis (truncated to 200 chars)
    # print("REF[0]:", true_references[0][:200])  # first reference
    # print("SYS length:", len(sys_outputs), "| REF length:", len(true_references))



    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅  ❗ sacreBLEU computation 🧩
    beam_used = beam_size if beam_size is not None else cfg.generation.beam
    
    bleu = sacrebleu.corpus_bleu(sys_outputs, [true_references])

    print(f"🌍 sacreBLEU (beam={beam_used}): {bleu.score:.2f}")
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    duration = time.time() - start_time
    total_test_duration += duration
    mins, secs = divmod(duration, 60)
    print(f"⏱ Test Epoch {epoch}: {int(mins)}m {secs:.1f}s | Loss {avg_loss:.3f} | sacreBLEU {bleu.score:.2f}")


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Save best checkpoint
    if bleu.score > best_bleu:
        best_bleu = bleu.score
        ckpt_path = (
            f'./checkpoints/{exp_args.dataset_name}/{exp_args.act_name}/{exp_args.act_name}_{exp_args.net}_{exp_args.dataset_name}'
            f'_LR{str(exp_args.lr).replace(".", "_")}_{exp_args.optimizer1}_{exp_args.mode_name}_LastEpoch{epoch}.pt'
        )
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)   # ✅ FIX HERE

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "lr_scheduler_state": lr_scheduler.state_dict(),
            
            # 🟡 Activation optimizers + schedules for resuming best checkpoint
            "activation_optimizers": {k: opt.state_dict() for k, opt in activation_optimizers.items()} if activation_optimizers else None,
            "gamma1_lr_schedule": gamma1_lr_schedule_values,
            "freq_lr_schedule": freq_lr_schedule_values,

            "bleu": bleu.score,
        }, ckpt_path)

        print(f"🏆 New best checkpoint saved: {ckpt_path}")
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Log results (ensure directory exists)
    # if epoch == exp_args.start_epoch:        # Refresh at start_epoch
    if epoch == (exp_args.epochs - 1):       # Refresh at End_epoch
        os.makedirs(os.path.dirname(test_results_path), exist_ok=True)
        if os.path.exists(test_results_path):
            open(test_results_path, "w").close()


    beam_used = beam_size if beam_size is not None else cfg.generation.beam
    with open(test_results_path, "a", encoding="utf-8") as f:
        f.write(
            f"Epoch {epoch} | Test Loss: {avg_loss:.3f} "
            f"| sacreBLEU: {bleu.score:.2f} "
            f"| Beam={beam_used}\n"
        )

    # ✅ Append the Last BLEU score only once at the end of training
    if epoch == (exp_args.epochs - 1):  
        with open(test_results_path, "a", encoding="utf-8") as f:
            f.write(f"\n🏆 Last BLEU Score: {best_bleu:.2f}\n")

            # 🟡 Log total test time at the END
            total_mins, total_secs = divmod(total_test_duration, 60)
            f.write(f"\n🕒 Total Test Time => {exp_args.act_name}_{exp_args.net}: {int(total_mins)} min {total_secs:.2f} sec\n")
            print(f"🕒 Total Test Time => {exp_args.act_name}_{exp_args.net}: {int(total_mins)} min {total_secs:.2f} sec")

    # ✅ Print Final BLEU
    print(f"📊 Test BLEU: {bleu.score:.2f} | 🏆 Final BLEU: {best_bleu:.2f}")
    print(f"📜 Test logs saved to {test_results_path}!")





    return avg_loss, bleu.score





########################################################################################################################
####-------| NOTE 11. MAIN LOOP | XXX ----------------------------------------------------------########################
########################################################################################################################

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    training_total_start = time.time()

    set_seed_torch(seed1)
    set_seed_main(seed2)



    # ✅ Free unused GPU memory BEFORE training starts
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    for epoch in range(exp_args.start_epoch, exp_args.epochs):

        # ✅ Train
        # train_loss = train_one_epoch(epoch, task, model, optimizer, criterion, lr_scheduler)
        train_loss = train_one_epoch(epoch, task, model, optimizer, criterion, lr_scheduler, activation_optimizers, unfreeze_activation_epoch, global_start_step)
        tqdm.write("")



    # ✅ Test 
    test_loss, bleu_score_val = test_one_epoch(epoch, task, model, criterion)
    tqdm.write("")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    print("Final BLEU: ", best_bleu)

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    training_total_end = time.time()
    total_mins, total_secs = divmod(training_total_end - training_total_start, 60)

    os.makedirs(os.path.dirname(save_paths["log_history"]), exist_ok=True)
    with open(save_paths["log_history"], "a", encoding="utf-8") as log_file:
        log_file.write(f"\n🕒 Total Training Time => {exp_args.act_name}_{exp_args.net}: {int(total_mins)} min {total_secs:.2f} sec\n")

    os.makedirs(os.path.dirname(test_results_path), exist_ok=True)
    with open(test_results_path, "a", encoding="utf-8") as f:
        f.write(f"\n🕒 Total Training Time => {exp_args.act_name}_{exp_args.net}: {int(total_mins)} min {total_secs:.2f} sec\n")

    print(f"🕒 Total Training Time: {int(total_mins)}m {total_secs:.1f}s")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────




# %%

########################################################################################################################
####-------| NOTE 12. TEST ALL CHECKPOINTS (BLEU per epoch)  | XXX -----------------------------########################
########################################################################################################################
########################################################################################################################
####-------| NOTE 12. TEST ALL CHECKPOINTS (BLEU per epoch)  | XXX -----------------------------########################
########################################################################################################################





# 1️⃣ Paths for checkpoint BLEU sweeps
test_checkpoint_results_path = (
    f'./Results/{exp_args.dataset_name}/{exp_args.act_name}/'
    f'TestCheckpoints_{exp_args.act_name}_{exp_args.net}_'
    f'{exp_args.dataset_name}_{exp_args.optimizer1}_{exp_args.mode_name}.txt'
)




# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣ Detokenize for bleu_score function
def detokenize_for_bleu(s: str) -> str:
    s = s.replace("@@ ", "").replace("@@", "")  # BPE
    s = s.replace("▁", " ")                     # SentencePiece
    s = " ".join(s.split())                     # normalize spaces
    s = re.sub(r"\s+([?.!,])", r"\1", s)        # punctuation
    return s.strip()
# ─────────────────────────────────────────────────────────────────────────────────────────────────




# 3️⃣ Test_checkpoint function 
def test_checkpoint_epochs(task, model, criterion, checkpoint_dir, test_checkpoint_results_path):
    """
    Evaluate BLEU (beam=5) for all checkpoints in checkpoint_dir.
    Save results to test_checkpoint_results_path in the same format as normal test logs.
    """




    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Collect checkpoints sorted by epoch number in filename
    def extract_epoch_num(filename):
        match = re.search(r"epoch_(\d+)", filename)
        return int(match.group(1)) if match else float("inf")  # inf if no epoch number

    ckpts = sorted(
        glob.glob(os.path.join(checkpoint_dir, "*.pt")),
        key=lambda x: extract_epoch_num(os.path.basename(x))
    )

    if not ckpts:
        print(f"⚠️ No checkpoints found in {checkpoint_dir}")
        return

    # ✅ Reset output file
    os.makedirs(os.path.dirname(test_checkpoint_results_path), exist_ok=True)
    with open(test_checkpoint_results_path, "w", encoding="utf-8") as f:
        f.write("")
    # ────────────────────────────────────────────────────────────────────────────────────────────────



    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    best_bleu = 0.0
    total_test_duration = 0.0

    for ckpt in ckpts:
        # ✅ Load checkpoint
        print(f"🔄 Loading checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state["model_state"], strict=False)
        model.to(DEVICE)
        model.eval()
    # ─────────────────────────────────────────────────────────────────────────────────────────────────



        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # ✅ Build test iterator
        test_iter = task.get_batch_iterator(
            dataset=task.dataset("test"),
            max_tokens=fairseq_args.max_tokens,   # cut in half option: fairseq_args.max_tokens // 2
            max_positions=(exp_args.max_source_positions, exp_args.max_target_positions),
            num_shards=exp_args.num_shards,
            shard_id=exp_args.shard_id,
            num_workers=exp_args.num_workers,
            ignore_invalid_inputs=exp_args.ignore_invalid_inputs,
        ).next_epoch_itr(shuffle=False)

        # ✅ References
        true_references = []
        for sample in test_iter:
            targets = sample["target"]
            for t in targets:
                t_no_pad = t[t != tgt_dict.pad()]
                ref_str = tgt_dict.string(t_no_pad, bpe_symbol="@@")
                true_references.append(detokenize_for_bleu(ref_str))

        # ✅ Rebuild iterator for decoding
        test_iter = task.get_batch_iterator(
            dataset=task.dataset("test"),
            max_tokens=fairseq_args.max_tokens,    # cut in half option: fairseq_args.max_tokens // 2
            max_positions=(exp_args.max_source_positions, exp_args.max_target_positions),
            num_shards=exp_args.num_shards,
            shard_id=exp_args.shard_id,
            num_workers=exp_args.num_workers,
            ignore_invalid_inputs=exp_args.ignore_invalid_inputs,
        ).next_epoch_itr(shuffle=False)
        # ────────────────────────────────────────────────────────────────────────────────────────────────


        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # ✅ Build generator with beam=5 🧩
        gen_cfg = deepcopy(cfg.generation)
        gen_cfg.beam = 5
        gen_cfg.lenpen = 1.0
        generator = task.build_generator([model], gen_cfg)
        # ────────────────────────────────────────────────────────────────────────────────────────────────

  
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # ✅ Run evaluation
        sys_outputs, test_loss = [], 0.0
        start_time = time.time()

        with torch.no_grad():
            with tqdm(enumerate(test_iter), total=len(test_iter), desc=f"Testing {os.path.basename(ckpt)}") as progress:
                for batch_idx, sample in progress:   # 👈 keep i!
                    sample = utils.move_to_cuda(sample) if torch.cuda.is_available() else sample
                    src_tokens = sample["net_input"]["src_tokens"]
                    prev_output_tokens = sample["net_input"]["prev_output_tokens"]
                    target = sample["target"]
                    src_lengths = (src_tokens != src_dict.pad()).sum(dim=1)

                    with torch.cuda.amp.autocast():
                        logits, _ = model(src_tokens, src_lengths, prev_output_tokens)
                        pad = tgt_dict.pad()
                        logits2d = logits.view(-1, logits.size(-1))
                        target1d = target.view(-1)
                        mask = target1d != pad
                        loss = criterion(logits2d[mask], target1d[mask]) if mask.any() else logits2d.sum() * 0.0
                    test_loss += loss.item()

                    hypos = task.inference_step(generator, [model], sample)
                    for j, hypos_j in enumerate(hypos):
                        hypo_tokens = [tok for tok in hypos_j[0]["tokens"].tolist() if tok != tgt_dict.pad()]
                        sys_str = tgt_dict.string(hypo_tokens, bpe_symbol="@@")
                        sys_outputs.append(detokenize_for_bleu(sys_str))

                    # ✅ Update progress bar postfix with running average
                    progress.set_postfix(loss=round(test_loss / (batch_idx + 1), 3))
       # ────────────────────────────────────────────────────────────────────────────────────────────────


       # ────────────────────────────────────────────────────────────────────────────────────────────────
        avg_loss = test_loss / max(1, len(test_iter))
        bleu = sacrebleu.corpus_bleu(sys_outputs, [true_references])
        duration = time.time() - start_time
        total_test_duration += duration

        # ✅ Write results
        with open(test_checkpoint_results_path, "a", encoding="utf-8") as f:
            f.write(f"Checkpoint {os.path.basename(ckpt)} | Test Loss: {avg_loss:.3f} | sacreBLEU: {bleu.score:.2f} | Beam=5\n")

        print(f"✅ {os.path.basename(ckpt)} | Loss {avg_loss:.3f} | sacreBLEU {bleu.score:.2f}")
       # ────────────────────────────────────────────────────────────────────────────────────────────────


       # ────────────────────────────────────────────────────────────────────────────────────────────────
       # ────────────────────────────────────────────────────────────────────────────────────────────────
        # 🧹 NEW: cleanup to prevent GPU memory leaks and kernel crashes
        del state, sys_outputs, true_references, test_iter, generator
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # ────────────────────────────────────────────────────────────────────────────────────────────────

        
        # 🧹 add visual spacing between runs
        tqdm.write("")  


        if bleu.score > best_bleu:
            best_bleu = bleu.score

    with open(test_checkpoint_results_path, "a", encoding="utf-8") as f:
        f.write(f"\n🏆 Best BLEU Score: {best_bleu:.2f}\n")
        total_mins, total_secs = divmod(total_test_duration, 60)
        f.write(f"\n🕒 Total Test Time over all checkpoints: {int(total_mins)} min {total_secs:.2f} sec\n")

    print(f"🏆 Best BLEU over checkpoints: {best_bleu:.2f}")





# ────────────────────────────────────────────────────────────────────────────────────────────────
# 4️⃣ Call the function
test_checkpoint_epochs(task, model, criterion, checkpoint_dir, test_checkpoint_results_path)
# ────────────────────────────────────────────────────────────────────────────────────────────────


# %%
