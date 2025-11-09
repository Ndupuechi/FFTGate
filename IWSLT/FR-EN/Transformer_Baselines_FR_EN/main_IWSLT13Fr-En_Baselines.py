





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
Project_PATH = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\IWSLT\Transformer_Baselines_FR_EN"
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
# ✅ === Fairseq & Supporting Imports ===
# ======================================================================================================
import importlib.metadata as importlib_metadata
import fairseq  # type: ignore
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
# ####-------| NOTE 2.1. PROCESSING DATASET 1 | XXX ------------| FRENCH→ENGLISH |--------------------####################
# ########################################################################################################################

# import sentencepiece as spm
# from sklearn.model_selection import train_test_split


# # ✅ Path for FR→EN dataset
# DATA_PATH = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\IWSLT\Transformer_Baselines_FR_EN\Dataset\fr-en"



# # ✅ Path to TED2013 raw data (reusing EN↔FR raw files)
# SRC_FR = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\IWSLT\Transformer_Baselines_FR_EN\Dataset\en-fr\TED2013.en-fr.fr"
# TGT_EN = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\IWSLT\Transformer_Baselines_FR_EN\Dataset\en-fr\TED2013.en-fr.en"





# # --------------------------
# # 2.1.a. Train SentencePiece model
# # --------------------------
# print("🧠 Training SentencePiece model for FR→EN...")

# # Ensure FR→EN output folder exists before saving model
# os.makedirs(DATA_PATH, exist_ok=True)

# spm_model_prefix = os.path.join(DATA_PATH, "spm_fren")

# spm.SentencePieceTrainer.Train(
#     input=f"{SRC_FR},{TGT_EN}",
#     model_prefix=spm_model_prefix,
#     vocab_size=10000,
#     character_coverage=1.0,
#     model_type="bpe"
# )
# print(f"✅ SentencePiece model saved to: {spm_model_prefix}.model / .vocab")




# # --------------------------
# # 2.1.b. Encode files
# # --------------------------
# print("⚙️ Encoding dataset...")

# # 🩷🩷 FIX 2: Load the FR→EN model we just created, using absolute path.
# sp = spm.SentencePieceProcessor(model_file=f"{spm_model_prefix}.model")


# def encode_file(in_file, out_file):
#     with open(in_file, encoding="utf-8") as fin, open(out_file, "w", encoding="utf-8") as fout:
#         for line in fin:
#             fout.write(" ".join(sp.encode(line.strip(), out_type=str)) + "\n")


# # ✅ Encode both FR and EN sides into SentencePiece tokenized format
# encode_file(SRC_FR, os.path.join(DATA_PATH, "all.spm.fr"))
# encode_file(TGT_EN, os.path.join(DATA_PATH, "all.spm.en"))



# # --------------------------
# # 2.1.c. Split into train/test sets (95% / 5%)
# # --------------------------
# print("Splitting into train/test (95%/5%)...")

# # ✅ Read encoded text files from FR→EN directory
# with open(os.path.join(DATA_PATH, "all.spm.fr"), encoding="utf-8") as f:
#     fr = f.readlines()
# with open(os.path.join(DATA_PATH, "all.spm.en"), encoding="utf-8") as f:
#     en = f.readlines()


# # ✅ Split keeping alignment
# train_fr, test_fr, train_en, test_en = train_test_split(fr, en, test_size=0.05, random_state=42)



# def write(name, lines):
#     with open(os.path.join(DATA_PATH, name), "w", encoding="utf-8") as f:
#         f.writelines(lines)


# # ✅ Save all splits back into FR→EN folder
# write("train.fr", train_fr)
# write("train.en", train_en)
# write("test.fr",  test_fr)
# write("test.en",  test_en)


# print(f"✅ Split and encoded FR→EN data saved under: {DATA_PATH}")
# print("✅ Now run fairseq-preprocess to binarize.")






# %%  


########################################################################################################################
####-------| NOTE 2.2. PROCESSING DATASET 2 | XXX ------------| FRENCH→ENGLISH |--------------------####################
########################################################################################################################

import subprocess


print("🚀 Running fairseq-preprocess for FR→EN...")

cmd = [
    "fairseq-preprocess",
    "--source-lang", "fr",                   # ✅Source = French
    "--target-lang", "en",                   # ✅Target = English
    "--trainpref", os.path.join("Dataset", "fr-en", "train"),
    "--testpref",  os.path.join("Dataset", "fr-en", "test"),
    "--destdir", "data-bin/iwslt13.fr-en",   
    "--workers", "4"
]

# ✅ Run preprocessing
subprocess.run(cmd, check=True)

print("✅ Preprocessing complete! Binarized FR→EN data saved in data-bin/iwslt13.fr-en")

# %%
















########################################################################################################################
####-------| NOTE 3. SEEDING FOR REPRODUCIBILITY | XXX ---------------------------------------------####################
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

exp_parser = argparse.ArgumentParser("IWSLT13 FR-EN Experiment Config")

exp_parser.add_argument('--mode_name', default="Seed1_1_EXP3", type=str)

exp_parser.add_argument(
    '--act_name', default="relu", type=str,
    help="Activation function (relu, gelu, tanh, sigmoid, swish, glu, tanhexp, fftgate, geglu)")



exp_parser.add_argument('--dataset_name', default="IWSLT13_Fr_En", type=str,
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




exp_args = exp_parser.parse_args([])   # ← for naming/logging








# --------------------------
# 4.2.b. Patch Fairseq get_activation_fn
# --------------------------

# Ensure activation folder is in sys.path
if ACTIVATION_PATH not in sys.path:
    sys.path.append(ACTIVATION_PATH)

# Backup original get_activation_fn
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
    
    elif activation == "geglu":        
        from activation.GEGLU import GEGLU
        print("✅ Registered custom activation: GEGLU")
        return GEGLU()

    else:
        # ✅ fallback to original fairseq activations
        return orig_get_activation_fn(activation)

# Patch fairseq utils
utils.get_activation_fn = custom_get_activation_fn


print(f"⚡ Using activation function: {exp_args.act_name.lower()}")

# Test-call the patched activation to confirm it works
try:
    act_fn = utils.get_activation_fn(exp_args.act_name.lower())
    print(f"🔍 Activation function resolved: {act_fn}")
except Exception as e:
    print(f"❌ Failed to resolve activation '{exp_args.act_name.lower()}': {e}")






# --------------------------
# 4.2.c. Fairseq parser (model/task configs)
# --------------------------

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ use the patched parser

fs_parser = options.get_training_parser()

fairseq_args = options.parse_args_and_arch(
    fs_parser,
    input_args=[
        # ✅ Point to FR→EN dataset directory
        "data-bin/iwslt13.fr-en",

        # ✅ Set correct language directions
        "--arch", "transformer",        # standard transformer arch
        "--task", "translation",
        "--source-lang", "fr",          # source language = French
        "--target-lang", "en",          # target language = English

        # ✅ Model settings 
        "--encoder-layers", "6",
        "--decoder-layers", "6",
        "--encoder-embed-dim", "512",
        "--decoder-embed-dim", "512",
        "--dropout", "0.3",
        "--max-tokens", str(exp_args.max_tokens),  # ✅ keep as string

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
####-------| NOTE 6. INITIALIZATION | XXX ---------------------------------------------------------#####################
########################################################################################################################



# ✅ Initialize AMP GradScaler once globall
scaler = torch.cuda.amp.GradScaler()





# --------------------------
# 6.a. Criterion
# --------------------------

criterion = LabelSmoothingCrossEntropy(smoothing=exp_args.smoothing)



# --------------------------
# 6.b. Adam optimizer (β1=0.9, β2=0.98, eps=1e-9)
# --------------------------
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=exp_args.lr,
    betas=(0.9, 0.98),
    eps=exp_args.eps,
    weight_decay=exp_args.weight_decay,
)





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





########################################################################################################################
####-------| NOTE 7. PATH DEFINATION AND GLOBAL INITAILIZATION | XXX ------------------------------#####################
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


# Path template for saving checkpoints at every epoch
checkpoint_dir = f'./checkpoints/{exp_args.dataset_name}/{exp_args.act_name}/'
os.makedirs(checkpoint_dir, exist_ok=True)


# # Path template for saving checkpoints at every epoch | Part2 (if scratch occured)
# checkpoint_dir = f'./checkpoints/{exp_args.dataset_name}/tanhexp_AfterCrash_41_49epochs/'
# os.makedirs(checkpoint_dir, exist_ok=True)



# ✅ global initialization
best_bleu = 0.0
total_test_duration = 0.0






# %%





########################################################################################################################
####-------| NOTE 8. TRAIN FUNCTIONS | XXX --------------------------------------------------------#####################
########################################################################################################################

# ✅ === Training  ===
def train_one_epoch(epoch, task, model, optimizer, criterion, lr_scheduler):
    epoch_start_time = time.time()
    model.train()


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





    epoch_loss = 0.0
    # ✅ Initialize LR log list for per-batch LR tracking
    lr_log_history = []


    # ✅ Initialize log history
    log_history = []



    with tqdm(enumerate(train_iter), total=len(train_iter), desc=f"Train Epoch {epoch}") as progress:
        for i, sample in progress:

            # ✅ Move entire sample dict to DEVICE
            sample = utils.move_to_cuda(sample) if torch.cuda.is_available() else sample

            # ✅ Compute step
            step_idx = epoch * len(train_iter) + i

            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # 🔍 Debug LR
            lr_now = optimizer.param_groups[0]["lr"]
            wd_now = optimizer.param_groups[0]["weight_decay"]

            detailed_steps = {0, 1, 2, 5, 10, 50, 70, 100, 200, 500, 1000, 1500, 1700, 3000, 4000, 5000}
            detailed_steps.add(len(train_iter) - 1)
            milestone_epochs = {0, 1, 3, 5, 10, 20, 30, 40, 45}

            if (epoch in milestone_epochs) and (i in detailed_steps):
                lr_log_msg = (
                    f"[Epoch {epoch} | Batch {i} | Global Step {step_idx}] "
                    f"Main LR: {lr_now:.8f} | WD: {wd_now:.8f}"
                )
                lr_log_history.append(lr_log_msg)
            # ─────────────────────────────────────────────────────────────────────────────────────────────────

            src_tokens = sample["net_input"]["src_tokens"]
            prev_output_tokens = sample["net_input"]["prev_output_tokens"]
            target = sample["target"]

            src_lengths = (src_tokens != src_dict.pad()).sum(dim=1)

            optimizer.zero_grad()



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
                        print(f"💥⚠️ Training Loop: Non-finite loss (NaN/Inf) at epoch {epoch}, batch {i}. Skipping batch.")
                        torch.cuda.empty_cache()
                        continue   # Skip this batch safely

            except RuntimeError as e:  
                print(f"💥 ⚠️ Training Loop: AMP forward failed at epoch {epoch}, batch {i}: {e}")  # AMP failure
                torch.cuda.empty_cache()
                continue   # Skip this batch and move on safely
            # ─────────────────────────────────────────────────────────────────────────────────────────────────















            # loss.backward()
            # optimizer.step()

            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ ⚠️ Backward pass with gradient scaler (AMP)
            scaler.scale(loss).backward()
            scaler.step(optimizer)       # replaces optimizer.step()
            scaler.update()              # unscales gradients & adjusts scaling
            # ─────────────────────────────────────────────────────────────────────────────────────────────────





            lr_scheduler.step()  # ✅ update LR per batch (Transformer schedule) NOt affected by AMP

            epoch_loss += loss.item()



            progress.set_postfix(
                loss=round(epoch_loss / (i + 1), 3),
                lr=f"{lr_now:.8f}",
                wd=f"{wd_now:.8f}",
            )




    avg_loss = epoch_loss / len(train_iter)

    # ✅ Timing
    duration = time.time() - epoch_start_time
    mins, secs = divmod(duration, 60)
    print(f"⏱ Epoch {epoch} Training time {exp_args.act_name}_{exp_args.net}: {int(mins)} min {secs:.2f} sec")





    # ✅ after loop finishes (end of epoch)
    current_lr = optimizer.param_groups[0]["lr"]
    current_wd = optimizer.param_groups[0].get("weight_decay", 0.0)

    log_msg = (
        f"Epoch {epoch} | Last Batch {i} | "
        f"M_Optimizer LR => {current_lr:.8f} | WD => {current_wd:.6f} | "
        f"⏱ Training time => {exp_args.act_name}_{exp_args.net}: {int(mins)} min {secs:.2f} sec"
    )
    log_history.append(log_msg)
    print(log_msg)









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
        "train_loss": avg_loss
    }, ckpt_path)
    print(f"🟡🟡 Epoch {epoch} checkpoint saved to: {ckpt_path}")
    # ────────────────────────────────────────────────────────────────────────────────────────────────

    return avg_loss




########################################################################################################################
####-------| NOTE 9. TEST FUNCTIONS (with sacreBLEU) | BEST PRACTICE --------------------------------------------------#
########################################################################################################################


def detokenize_for_bleu(s: str) -> str:
    s = s.replace("@@ ", "").replace("@@", "")  # BPE
    s = s.replace("▁", " ")                     # SentencePiece
    s = " ".join(s.split())                     # normalize spaces
    s = re.sub(r"\s+([?.!,])", r"\1", s)        # punctuation
    return s.strip()






# ✅ === Testing  ===
def test_one_epoch(epoch, task, model, criterion, beam_size=None):

    global best_bleu, total_test_duration

    # ──────────────────────────────────
    # ❌ GPU memory check At test start
    if torch.cuda.is_available():
        print(f"[❗Epoch {epoch} TEST START] GPU mem allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB | reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    # ──────────────────────────────────


    model.eval()





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



    with torch.no_grad():
        with tqdm(enumerate(test_iter), total=len(test_iter), desc=f"Test Epoch {epoch}") as progress:
            for i, sample in progress:
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
                            print(f"💥⚠️ Test Loop: Non-finite loss (NaN/Inf) at epoch {epoch}, batch {i}. Skipping batch.")
                            torch.cuda.empty_cache()
                            continue           # Skip this batch safely

                # ✅ Check for Runtime Error
                except RuntimeError as e:  
                    print(f"💥 ⚠️ Test Loop: AMP forward failed at epoch {epoch}, batch {i}: {e}")  # AMP failure
                    torch.cuda.empty_cache()
                    continue                   # Skip this batch and move on safely
                # ────────────────────────────────────────────────────────────────────────────────────────────────






                # ✅ Compute Loss
                test_loss += loss.item()






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


                progress.set_postfix(loss=round(test_loss / (i + 1), 3))



    # ✅ Compute Ave Loss
    avg_loss = test_loss / max(1, len(test_iter))





    # # 🔍 Debug: check alignment again at summary level
    # print("----- DEBUG OUTPUT -----")
    # print("SYS[0]:", sys_outputs[0][:200])   # first hypothesis (truncated to 200 chars)
    # print("REF[0]:", true_references[0][:200])  # first reference
    # print("SYS length:", len(sys_outputs), "| REF length:", len(true_references))




    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅  ❗ sacreBLEU computation
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
            "bleu": bleu.score,
        }, ckpt_path)
        print(f"🏆 New best checkpoint saved: {ckpt_path}")
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
####-------| NOTE 8. MAIN LOOP | XXX -----------------------------------------------------------########################
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


    for epoch in range(exp_args.start_epoch, exp_args.epochs):

        # ✅ Train
        train_loss = train_one_epoch(epoch, task, model, optimizer, criterion, lr_scheduler)
        tqdm.write("")


    # # ✅ Conditional GPU cleanup before Test
    # mem_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
    # total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    # if mem_reserved > 0.8 * total_mem:
    #     print(f"⚠️ BEFORE TEST: High GPU reserved memory ({mem_reserved:.2f} GB / {total_mem:.2f} GB). Cleaning cache...")
    #     torch.cuda.empty_cache()


    # ✅ Test 
    test_loss, bleu_score_val = test_one_epoch(epoch, task, model, criterion)
    tqdm.write("")




    print("Final BLEU: ", best_bleu)


    training_total_end = time.time()
    total_mins, total_secs = divmod(training_total_end - training_total_start, 60)

    os.makedirs(os.path.dirname(save_paths["log_history"]), exist_ok=True)
    with open(save_paths["log_history"], "a", encoding="utf-8") as log_file:
        log_file.write(f"\n🕒 Total Training Time => {exp_args.act_name}_{exp_args.net}: {int(total_mins)} min {total_secs:.2f} sec\n")

    os.makedirs(os.path.dirname(test_results_path), exist_ok=True)
    with open(test_results_path, "a", encoding="utf-8") as f:
        f.write(f"\n🕒 Total Training Time => {exp_args.act_name}_{exp_args.net}: {int(total_mins)} min {total_secs:.2f} sec\n")

    print(f"🕒 Total Training Time: {int(total_mins)}m {total_secs:.1f}s")





# %%

########################################################################################################################
####-------| NOTE 9. TEST ALL CHECKPOINTS (BLEU per epoch)  | XXX ------------------------------########################
########################################################################################################################
########################################################################################################################
####-------| NOTE 9. TEST ALL CHECKPOINTS (BLEU per epoch)  | XXX ------------------------------########################
########################################################################################################################


# ✅ Paths for checkpoint BLEU sweeps
test_checkpoint_results_path = (
    f'./Results/{exp_args.dataset_name}/{exp_args.act_name}/'
    f'TestCheckpoints_{exp_args.act_name}_{exp_args.net}_'
    f'{exp_args.dataset_name}_{exp_args.optimizer1}_{exp_args.mode_name}.txt'
)



# # ✅ Paths for checkpoint BLEU sweeps | Part2 (if Cratch occured)
# test_checkpoint_results_path = (
#     f'./Results/{exp_args.dataset_name}/{exp_args.act_name}/'
#     f'TestCheckpoints_{exp_args.act_name}_{exp_args.net}_'
#     f'{exp_args.dataset_name}_{exp_args.optimizer1}_tanhexp_AfterCrash_41_49epochs.txt'
# )





# ✅ detokenize for bleu_score function
def detokenize_for_bleu(s: str) -> str:
    s = s.replace("@@ ", "").replace("@@", "")  # BPE
    s = s.replace("▁", " ")                     # SentencePiece
    s = " ".join(s.split())                     # normalize spaces
    s = re.sub(r"\s+([?.!,])", r"\1", s)        # punctuation
    return s.strip()







# ✅ test_checkpoint function
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




    best_bleu = 0.0
    total_test_duration = 0.0

    for ckpt in ckpts:
        # ✅ Load checkpoint
        print(f"🔄 Loading checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state["model_state"], strict=False)
        model.to(DEVICE)
        model.eval()

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

        # ✅ Build generator with beam=5
        gen_cfg = deepcopy(cfg.generation)
        gen_cfg.beam = 5
        gen_cfg.lenpen = 1.0
        generator = task.build_generator([model], gen_cfg)





  

        # ────────────────────────────────────────────────────────────────────────────────────────────────
        # ✅ Run evaluation
        sys_outputs, test_loss = [], 0.0
        start_time = time.time()

        with torch.no_grad():
            with tqdm(enumerate(test_iter), total=len(test_iter), desc=f"Testing {os.path.basename(ckpt)}") as progress:
                for i, sample in progress:   # 👈 keep i!
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
                    progress.set_postfix(loss=round(test_loss / (i + 1), 3))
       # ────────────────────────────────────────────────────────────────────────────────────────────────







        avg_loss = test_loss / max(1, len(test_iter))
        bleu = sacrebleu.corpus_bleu(sys_outputs, [true_references])
        duration = time.time() - start_time
        total_test_duration += duration

        # ✅ Write results
        with open(test_checkpoint_results_path, "a", encoding="utf-8") as f:
            f.write(f"Checkpoint {os.path.basename(ckpt)} | Test Loss: {avg_loss:.3f} | sacreBLEU: {bleu.score:.2f} | Beam=5\n")

        print(f"✅ {os.path.basename(ckpt)} | Loss {avg_loss:.3f} | sacreBLEU {bleu.score:.2f}")


        # 🧹 add visual spacing between runs
        tqdm.write("")  


        if bleu.score > best_bleu:
            best_bleu = bleu.score

    with open(test_checkpoint_results_path, "a", encoding="utf-8") as f:
        f.write(f"\n🏆 Best BLEU Score: {best_bleu:.2f}\n")
        total_mins, total_secs = divmod(total_test_duration, 60)
        f.write(f"\n🕒 Total Test Time over all checkpoints: {int(total_mins)} min {total_secs:.2f} sec\n")

    print(f"🏆 Best BLEU over checkpoints: {best_bleu:.2f}")







# # ✅ Free unused GPU memory BEFORE training starts
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats()

# 4️⃣ Call the function
test_checkpoint_epochs(task, model, criterion, checkpoint_dir, test_checkpoint_results_path)



# %%
