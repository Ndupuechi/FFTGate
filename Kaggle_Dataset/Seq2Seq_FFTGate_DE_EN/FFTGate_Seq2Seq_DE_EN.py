




# %% 


# ======================================================================================================
# ✅ === LOAD LIBRARIES ===
# ======================================================================================================


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === Enable flexible CUDA memory allocation to reduce fragmentation ===
# Must be set before importing torch!
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ✅ === Ensure correct working directory ===
import sys
Project_PATH = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\Kaggle_German-English_Dataset\Seq2Seq_FFTGate_DE_EN"
if os.getcwd() != Project_PATH:
    os.chdir(Project_PATH)
print(f"✅ Current working directory: {os.getcwd()}")


# ✅ === Define core project paths ===
PROJECT_PATH = Project_PATH
ACTIVATION_PATH = os.path.join(Project_PATH, "activation")


# ✅ === Add essential paths to sys.path ===
for path in [PROJECT_PATH, ACTIVATION_PATH]:
    if path not in sys.path:
        sys.path.append(path)

print("✅ sys.path updated:")
for path in sys.path:
    print("   📂", path)
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ✅ === Core Libraries ===
import argparse
from tqdm import tqdm
import gzip
import shutil
import torch
import spacy
import random
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torchtext.data import Field, BucketIterator
import torch.fft
import matplotlib.pyplot as plt
import math
from torchtext.data.metrics import bleu_score

# ✅ === From utils.py in project folder import cosine_scheduler ===
import utils as my_utils

# ✅ === Import FFTGate  ===
from activation.FFTGate import FFTGate  


# ======================================================================================================
# ✅ === SET SEED FOR REPRODUCIBILITY ===
# ======================================================================================================

def set_seed_torch(seed):
    torch.manual_seed(seed)

def set_seed_main(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False    
    torch.backends.cudnn.benchmark = True         

# ✅ ============= Define Seed =============
seed1, seed2 = 11, 11
set_seed_torch(seed1)  
set_seed_main(seed2)  





# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === DEFINE PATHS ===
data_path = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\Kaggle_German-English_Dataset\Seq2Seq_FFTGate_DE_EN\data"
multi30k_dir = os.path.join(data_path, "multi30k")
os.makedirs(multi30k_dir, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# MOVE FILES INTO multi30k if needed
# train.de, train.en, val.de, val.en, test.de, test.en
# Must be in: ...\data\multi30k\


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ === LOAD AND TOKENIZE ====
os.system("python -m spacy download de_core_news_sm")
os.system("python -m spacy download en_core_web_sm")
spacy_ger = spacy.load('de_core_news_sm')
spacy_eng = spacy.load('en_core_web_sm')
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ======================================================================================================
# ✅ === TOKENIZERS ===
# ======================================================================================================
# Tokenization of German Language
def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


# Tokenization of English Language
def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

print("✅ Tokenizers are ready.")
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ======================================================================================================
# ✅ === PREPROCESSING OF TEXT | DEFINE FIELDS ===
# ======================================================================================================
# Applyling Tokenization , lowercase and special Tokens for preprocessing
german = Field(tokenize = tokenize_ger,lower = True,init_token = '<sos>',eos_token = '<eos>')
english = Field(tokenize = tokenize_eng,lower = True,init_token = '<sos>',eos_token = '<eos>')



# ✅ ===  LOAD DATASET ===
train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"),
    fields=(german, english),
    root=data_path,
    test="test"  
)

 
## ✅ === BUILD VOCAB | CREATING VOVABULARY IN EACH LANGUAGE ===  
german.build_vocab(train_data,max_size = 10000,min_freq = 2)
english.build_vocab(train_data,max_size = 10000,min_freq = 2)
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# === Print samples ===
# === STEP 7: Confirm ===
print(f"✅ Training examples: {len(train_data)}")
print(f"✅ Validation examples: {len(valid_data)}")
print(f"✅ Test examples: {len(test_data)}\n")
print("✅ Sample:")
print(vars(train_data[0]))





# ======================================================================================================
# ✅ === DEFINE REGULARIZATION FUNCTION ===
# ======================================================================================================

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
    if batch_idx == 0 and (epoch % 10 == 0 or epoch == (args.unfreeze_act_epochs + 1)):
        print(
            f"[REG_STATUS] Epoch {epoch} | "
            f"Target Regularization={enable_target_reg} | "
            f"Noise Regularization={enable_noise_reg} | "
            f"ActReg={enable_act_reg}"
        )


    # === Batch std (cached) ===
    if cache_interval == 0 or batch_idx % cache_interval == 0 or cached_batch_std is None:
        cached_batch_std = torch.std(inputs.detach().float()) + 1e-6

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
        if enable_noise_reg and epoch > args.addnoise_act_epochs:
            current_gamma1 = gamma1_raw
            prev_gamma1    = prev_layer_params["gamma1"]
            param_variation = torch.abs(current_gamma1.detach() - prev_gamma1).mean()
            if param_variation < 0.015:
                noise = (0.0005 + 0.0002 * batch_std.item()) * torch.randn_like(current_gamma1)
                penalty = (current_gamma1 - (prev_gamma1 + noise)).pow(2).mean()
                fftgateparam_reg += 0.00015 * penalty
                noisy_layers.append(f"{idx} (Δ={param_variation.item():.5f})")

    if batch_idx == 0 and epoch <= (args.addnoise_act_epochs + 4) and noisy_layers:
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






# ======================================================================================================
# ✅ === ARGUMENT PARSER ===
# ======================================================================================================

parser = argparse.ArgumentParser(description='Kaggle German-English Seq2Seq Translation Dataset')

# ✅ ===== Main parameters =====
parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch number (default: 0)')
parser.add_argument('--epochs', default=50, type=int)     

parser.add_argument('--lr', default=0.001, type=float, help='learning rate') 
parser.add_argument('--batch_size', type=int,  default=256, help='Batch size (default: 256)')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay (default: 1e-4)') # try 0
parser.add_argument('--opt_eps', type=float, default=1e-9, help='Adam optimizer epsilon (default: 1e-8)')

parser.add_argument('--hidden_size', default=1024, type=int) 
parser.add_argument('--num_layers', default=1, type=int) 
parser.add_argument('--encoder_embedding_size', default=300, type=int) 
parser.add_argument('--decoder_embedding_size', default=300, type=int) 
parser.add_argument('--enc_dropout', default=0.5, type=float, help='encoder dropout') 
parser.add_argument('--dec_dropout', default=0.5, type=float, help='encoder dropout') 


# ✅=== Naming Covention / path defination params ===
parser.add_argument('--act_name', default="FFTGate", type=str)
parser.add_argument('--mode_name', default="Trial6", type=str)
parser.add_argument('--model_name', default="Seq2Seq", type=str)


# ✅ ===== Activation Optimization parameters =====
parser.add_argument('--min_act_lr', type=float, default=1e-6)
parser.add_argument('--actbase_lr', type=float, default=1e-2, help="Base learning rate for activation parameters (gamma1, freq)")

# Separate WD for gamma1 and freq (renamed to avoid Jupyter --f conflict)
parser.add_argument('--gamma1_wd', type=float, default=0.0)   # was 0
parser.add_argument('--wd_freq', type=float, default=0.0)     # was 0

parser.add_argument('--unfreeze_act_epochs', default=0, type=int)
parser.add_argument('--Warm_act_epochs', default=0, type=int)
parser.add_argument('--warmup_epochs_reg', default=5, type=int, help='Number of epochs for warmup phase (used in regularization)')
parser.add_argument('--addnoise_act_epochs', default=50, type=int)  

# Constant decay schedule
parser.add_argument('--gamma1_wd_end', type=float, default=0.0)
parser.add_argument('--wd_end_freq', type=float, default=0.0)
parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N', help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

args, unknown = parser.parse_known_args()  # Avoid Jupyter issues

# ─────────────────────────────────────────────────────────────────────────────────────────────────






# ======================================================================================================
# ✅ === DEFINING THE ENCODER PART OF THE MODEL === 
# === Encoder with FFTGate ===
# ======================================================================================================
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        
        # ✅ Your activation with all parameters
        # self.activation = FFTGate(gamma1=1.5, phi=0.1, history_len=12, decay_mode="linear")
        self.activation = FFTGate()

    def forward(self, x, epoch=0):
        embedding = self.dropout(self.activation(self.embedding(x), epoch=epoch))
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell




# ======================================================================================================
# ✅ ===  DEFINING THE DECODER PART OF THE MODEL === 
# === Decoder with FFTGate ===
# ======================================================================================================
class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)
        
        # ✅ Your activation with all parameters
        # self.activation = FFTGate(gamma1=1.5, phi=0.1, history_len=12, decay_mode="linear")
        self.activation = FFTGate()

    def forward(self, x, hidden, cell, epoch=0):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.activation(self.embedding(x), epoch=epoch))
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        predictions = self.fc(outputs).squeeze(0)
        return predictions, hidden, cell






# ======================================================================================================
# ✅ === DeEFINING THE COMPLETE MODEL === 
# ======================================================================================================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5, epoch=0):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source, epoch=epoch)  # <-- Pass epoch

        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell, epoch=epoch)  # <-- Pass epoch
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs




# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🟡 Hyperparameters
num_epochs = args.epochs
learning_rate = args.lr
batch_size = args.batch_size


# 🟡 Model hyperparameters
load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = args.encoder_embedding_size
decoder_embedding_size = args.decoder_embedding_size

hidden_size = args.hidden_size
num_layers = args.num_layers
enc_dropout = args.enc_dropout
dec_dropout = args.dec_dropout

# Tensorboard to get nice loss plot
writer = SummaryWriter(f'runs/Loss_plot')
step = 0
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────────────────────────
train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
     batch_size = batch_size, sort_within_batch = True, 
     sort_key = lambda x:len(x.src),
     device = device)
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────────────────────────
encoder_net = Encoder(input_size_encoder, 
                      encoder_embedding_size,
                      hidden_size,num_layers, 
                      enc_dropout).to(device)


decoder_net = Decoder(input_size_decoder, 
                      decoder_embedding_size,
                      hidden_size,output_size,num_layers, 
                      dec_dropout).to(device)
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ====== 🔧 BUILD MODEL 🔧 ======
model = Seq2Seq(encoder_net, decoder_net).to(device)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)


pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)
# ─────────────────────────────────────────────────────────────────────────────────────────────────











# ======================================================================================================
# ====== ✅ FFTGate HANDLING ✅ ======
# ======================================================================================================
########################################################################################################################
####-------| NOTE INITIALIZE FFTGATE PARAMETERS, OPTIMIZERS & SCHEDULERS | XXX ---------------------####################
########################################################################################################################

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ =========== Helper: recursively find FFTGate layers and collect trainable params ===========
def find_activations_split(module, activation_layers, gamma1_params, freq_params):
    for layer in module.children():
        if isinstance(layer, FFTGate):
            activation_layers.append(layer)
            gamma1_params.append(layer.gamma1_raw)
            freq_params.append(layer.freq_raw)
        elif isinstance(layer, nn.Module):
            find_activations_split(layer, activation_layers, gamma1_params, freq_params)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# 2️⃣ =========== Collect all FFTGate layers & parameters ===========
gamma1_params, freq_params, activation_layers = [], [], []
find_activations_split(model, activation_layers, gamma1_params, freq_params)  # ✅ use your Seq2Seq model

for idx, layer in enumerate(activation_layers, start=1):
    layer.layer_id = f"Layer{idx}"
    layer.layer_depth = idx
    layer.total_layers = len(activation_layers)

print(f"✅ Found {len(activation_layers)} FFTGate layers.")
print(f"✅ Collected {len(gamma1_params)} γ₁ params and {len(freq_params)} freq_factor params.")

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 3️⃣ =========== Define base LRs, weight decays, schedulers ===========
lr_gamma1 = args.actbase_lr * 2.0   # faster γ₁
lr_freq   = args.actbase_lr * 1.0   # slower freq_factor

print(f"Gamma1 LR={lr_gamma1}, WD={args.gamma1_wd}")
print(f"Freq    LR={lr_freq}, WD={args.wd_freq}")
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Number of steps per epoch (for cosine schedulers)
num_training_steps_per_epoch = len(train_iterator)


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



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Per-step cosine LR schedules
gamma1_lr_schedule = my_utils.cosine_scheduler(
    lr_gamma1, args.min_act_lr, args.epochs,
    num_training_steps_per_epoch,
    warmup_epochs=args.Warm_act_epochs, warmup_steps=args.warmup_steps
)
freq_lr_schedule = my_utils.cosine_scheduler(
    lr_freq, args.min_act_lr, args.epochs,
    num_training_steps_per_epoch,
    warmup_epochs=args.Warm_act_epochs, warmup_steps=args.warmup_steps
)


# 4️⃣ =========== Build main optimizer excluding FFTGate params ===========
act_param_ids = {id(p) for p in (gamma1_params + freq_params)}
main_params = [p for p in model.parameters() if id(p) not in act_param_ids]

optimizer = torch.optim.Adam(
    main_params, lr=args.lr, betas=(0.9, 0.999),
    eps=args.opt_eps, weight_decay=args.weight_decay
)


# 5️⃣ =========== Build FFTGate-specific optimizers ===========
activation_optimizers = {}
if gamma1_params:
    activation_optimizers["gamma1"] = torch.optim.AdamW(
        gamma1_params, lr=lr_gamma1, weight_decay=args.gamma1_wd
    )
if freq_params:
    activation_optimizers["freq_factor"] = torch.optim.AdamW(
        freq_params, lr=lr_freq, weight_decay=args.wd_freq
    )

# 🔒 Freeze initially
for p in gamma1_params + freq_params:
    p.requires_grad = False

print(f"✅ Initialized FFTGate optimizers: {list(activation_optimizers.keys())}")
print(f"🟡 Using per-step cosine LR schedules for gamma1 & freq_factor.")

# ─────────────────────────────────────────────────────────────────────────────────────────────────

########################################################################################################################
####-------| NOTE  FINALIZE ACT SCHEDULERs | XXX ---------------------------------------------------####################
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







# ======================================================================================================
# ✅ === TRANSLATION FUNCTION FOR INFERENCE ===
# ======================================================================================================
def translate_sentence(model, sentence, german, english, device, max_length=50):
    spacy_ger = spacy.load("de_core_news_sm")

    # Tokenization
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    text_to_indices = [german.vocab.stoi[token] for token in tokens]
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor, epoch=0)  # <-- Pass dummy epoch for inference

    outputs = [english.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell, epoch=0)  # <-- Same here
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)


        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break


    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    return translated_sentence[1:]
# ─────────────────────────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────────────────────────
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
# ─────────────────────────────────────────────────────────────────────────────────────────────────


if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)


sentence = "Cristiano Ronaldo ist ein großartiger Fußballspieler mit erstaunlichen Fähigkeiten und Talenten."
# ─────────────────────────────────────────────────────────────────────────────────────────────────





# %% 
# ======================================================================================================
# ✅ === TRAINING ===
# ======================================================================================================
 
log_history = []          # ✅ Log messages for all epochs
log_path = fr"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\Kaggle_German-English_Dataset\Seq2Seq_FFTGate_DE_EN\Results\log_{args.act_name}_{args.model_name}_{args.mode_name}.txt"


for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")



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
            "gamma1": layer.gamma1_raw.clone().detach(),
            "freq_factor": layer.freq_raw.clone().detach()
        }
    print(f"✅ Initialized tracking for {len(activation_layers)} FFTGate layers")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Save model checkpoint
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────



    model.eval()

    #translated_sentence = translate_sentence(
    #    model, sentence, german, english, device, max_length=50
    #)

    #print(f"Translated example sentence: \n {translated_sentence}")

    # ✅ Switch to training mode
    model.train()


 
    # ────────────────────────────────────────────────────────────────────────────────────────────────
    # 🔖 Sync current epoch into all FFTGate layers before training batches
    for layer in activation_layers:
        if isinstance(layer, FFTGate):
            layer.current_epoch.fill_(epoch)
            layer._epoch_step = 0
    # ────────────────────────────────────────────────────────────────────────────────────────────────


    # ❌ GPU memory check at epoch start
    if torch.cuda.is_available():
        print(f"[❗Epoch {epoch} TRAIN START] GPU mem allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB | reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ✅ Initialize per-epoch stats
    train_loss, correct, total, train_accuracy = 0, 0, 0, 0.0
    log_history, lr_log_history, FFTGate_param_history, FFTGate_Grad_history = [], [], [], []
    FFTGate_SummaryGrad_history, FFTGate_Main_SummaryGrad_history, activation_history = [], [], []
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ✅ Compute this epoch's base step index for per-step schedule (ConvNeXt-style)
    epoch_base_step = global_start_step + (epoch - args.start_epoch) * num_training_steps_per_epoch


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ === Unfreeze FFTGate parameters at scheduled epoch ===
    if epoch == args.unfreeze_act_epochs:
        print("\n🔓 Unfreezing FFTGate Parameters 🔓")

        # ✅ Enable gradients for FFTGate params
        for p in (gamma1_params + freq_params):
            p.requires_grad = True

        # ✅ Rebuild or re-sync activation optimizers (γ₁ + freq)
        if gamma1_params:
            activation_optimizers["gamma1"] = torch.optim.AdamW(
                gamma1_params, lr=0.0, weight_decay=args.gamma1_wd
            )
        if freq_params:
            activation_optimizers["freq_factor"] = torch.optim.AdamW(
                freq_params, lr=0.0, weight_decay=args.wd_freq
            )

        # ✅ Immediately sync schedulers for correct LR/WD
        step_idx = epoch_base_step
        g_lr, g_wd = _apply_single_lr_at_step(
            "gamma1", activation_optimizers, gamma1_lr_schedule_values, step_idx
        )
        f_lr, f_wd = _apply_single_lr_at_step(
            "freq_factor", activation_optimizers, freq_lr_schedule_values, step_idx
        )

        print(f"✅ Gamma1 Optimizer Synced! LR={g_lr if g_lr else 0:.6f}, WD={g_wd if g_wd else 0:.6f}")
        print(f"✅ FreqFactor Optimizer Synced! LR={f_lr if f_lr else 0:.6f}, WD={f_wd if f_wd else 0:.6f}")
    # ─────────────────────────────────────────────────────────────────────────────────────────────────




    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Loop through training batches
    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda | === Move data to device
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)


        # ===  Forward prop
        # output = model(inp_data, target)
        output = model(inp_data, target, epoch=epoch)



        # ─────────────────────────────────────────────────────────────────────────────────────────────────
        # === Reshape output and target for loss
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────




    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Training Loop (with LR/WD scheduling + tqdm progress)
    with tqdm(enumerate(train_iterator), total=len(train_iterator), desc=f"Epoch {epoch}") as progress:
        for batch_idx, batch in progress:

            # ✅ Step index for cosine LR scheduling
            step_idx = epoch_base_step + batch_idx

            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ Apply per-step learning rate & weight decay for FFTGate
            g_lr, g_wd = _apply_single_lr_at_step(
                "gamma1", activation_optimizers, gamma1_lr_schedule_values, step_idx
            )
            f_lr, f_wd = _apply_single_lr_at_step(
                "freq_factor", activation_optimizers, freq_lr_schedule_values, step_idx
            )
            # ─────────────────────────────────────────────────────────────────────────────────────────────────

            # ✅ Get inputs and targets, move to device
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)


            # ✅ Forward pass (pass epoch for FFTGate scheduling)
            output = model(inp_data, target, epoch=epoch)

            # ✅ Reshape for CrossEntropyLoss (remove <sos> token)
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ Zero gradients (main + FFTGate)
            optimizer.zero_grad()
            if epoch >= args.unfreeze_act_epochs:
                for opt in activation_optimizers.values():
                    opt.zero_grad()
            # ─────────────────────────────────────────────────────────────────────────────────────────────────

            # ✅ Loss calculation
            loss = criterion(output, target)
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # 🧭 Call Regularization Function for the Activation Parameter (gamma1 + freq_factor)
            if epoch >= args.unfreeze_act_epochs:                
                fftgate_reg = apply_dynamic_regularization(
                    inp_data, output, epoch,
                    prev_params, layer_index_map, batch_idx,
                    num_batches=num_training_steps_per_epoch     # 🟡 pass the total batches
                )
                loss += fftgate_reg
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ Backward pass
            loss.backward()
            # ─────────────────────────────────────────────────────────────────────────────────────────────────

            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ Optimizer step (main + FFTGate)
            optimizer.step()
            if epoch >= args.unfreeze_act_epochs:
                for opt in activation_optimizers.values():
                    opt.step()

            # ✅ Clamp gamma1_raw within safe range (sigmoid domain)
            with torch.no_grad():
                for layer in activation_layers:
                    if isinstance(layer, FFTGate):
                        # Clamp the *raw* parameter instead of effective one
                        layer.gamma1_raw.data.clamp_(-6.0, 6.0)  # was [-6, 6]
                        layer.freq_raw.data.clamp_(-6.0, 6.0)    # was [-6, 6]
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # ─────────────────────────────────────────────────────────────────────────────────────────────────
            # ✅ Update progress bar description
            progress.set_postfix(loss=f"{loss.item():.4f}", gamma1_lr=g_lr, freq_lr=f_lr)
            # ─────────────────────────────────────────────────────────────────────────────────────────────────

            # ✅ Log LR / WD at selected milestones
            detailed_steps = {0, 1, 2, 5, 10, 50, 100, 200, 300}
            milestone_epochs = {0, 1, 5, 10, 20, 30, 50}
            if (epoch in milestone_epochs) and (batch_idx in detailed_steps):
                lr_log_history.append(
                    f"[Epoch {epoch} | Batch {batch_idx}] "
                    f"MainLR={optimizer.param_groups[0]['lr']:.6f}, "
                    f"Gamma1LR={g_lr if g_lr else 'FROZEN'}, FreqLR={f_lr if f_lr else 'FROZEN'}"
                )
            # ─────────────────────────────────────────────────────────────────────────────────────────────────


            # ✅ TensorBoard Logging | Plot to tensorboard
            writer.add_scalar("Training loss", loss, global_step=step)
            step += 1



    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ ONLY update prev_params here AFTER all updates | ✅ Update prev_params AFTER training epoch
    for idx, layer in enumerate(activation_layers):
        prev_params[idx] = {
            "gamma1": layer.gamma1_raw.clone().detach(),
            "freq_factor": layer.freq_raw.clone().detach(),
        }
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ✅ Log activation mean (optional)
    if activation_history:
        avg_act = sum(activation_history) / len(activation_history)
        writer.add_scalar("Activation/AvgGamma1Output", avg_act, global_step=epoch)



    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Logging Parameters & Gradients after epoch
    last_batch_grads = {"Gamma1 Grad": [], "Freq Grad": []}
    current_params = {"Gamma1": [], "Freq": []}

    for layer in model.modules():
        if isinstance(layer, FFTGate):
            # ✅ Gamma1 gradients and values
            last_batch_grads["Gamma1 Grad"].append(
                f"{layer.gamma1_raw.grad.item():.8f}" if layer.gamma1_raw.grad is not None else "None"
            )
            current_params["Gamma1"].append(f"{layer.gamma1_raw.item():.8f}")

            # ✅ FreqFactor gradients and values
            last_batch_grads["Freq Grad"].append(
                f"{layer.freq_raw.grad.item():.8f}" if layer.freq_raw.grad is not None else "None"
            )
            current_params["Freq"].append(f"{layer.freq_raw.item():.8f}")

    # ✅ Compose structured log message
    log_msg = (
        f"Epoch {epoch}: "
        f"M_Optimizer LR => {optimizer.param_groups[0]['lr']:.5f} | "
        f"Gamma1 LR => {activation_optimizers['gamma1'].param_groups[0]['lr']:.5f} | "
        f"FreqFactor LR => {activation_optimizers['freq_factor'].param_groups[0]['lr']:.5f} | "
        f"Gamma1: {current_params['Gamma1']} | "
        f"Freq: {current_params['Freq']} | "
        f"Gamma1 Grad: {last_batch_grads['Gamma1 Grad']} | "
        f"Freq Grad: {last_batch_grads['Freq Grad']}"
    )

    # ✅ Save log message in memory and display
    log_history.append(log_msg)
    print(log_msg)
    tqdm.write("")  # ✅ adds a clean blank line after each epoch
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ✅ Append to log file instead of overwriting
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n\n")   # double newline 
    # ─────────────────────────────────────────────────────────────────────────────────────────────────



# ======================================================================================================
# ✅ === BLEU SCORE ===
# ======================================================================================================
def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
score = bleu(test_data[1:100], model, german, english, device)
# print(f"Bleu score {score*100:.2f}")
bleu_str = f"✅ BLEU Score: {score*100:.2f}"

print(bleu_str)
output_path = fr"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github3\Kaggle_German-English_Dataset\Seq2Seq_FFTGate_DE_EN\Results\BLEU_Score_{args.act_name}_{args.model_name}_{args.mode_name}.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(bleu_str + "\n")
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# In[28]:





