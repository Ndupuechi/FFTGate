




# %% 



# ✅ === LOAD LIBRARIES ===
import os
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

import torch
import torch.nn as nn
import torch.fft
import matplotlib.pyplot as plt
import math

from torchtext.data.metrics import bleu_score
import sys





# ✅ === SET SEED FOR REPRODUCIBILITY ===
def set_seed_torch(seed):
    torch.manual_seed(seed)                          


def set_seed_main(seed):
    random.seed(seed)                                ## Python's random module
    np.random.seed(seed)                             ## NumPy's random module
    torch.cuda.manual_seed(seed)                     ## PyTorch's random module for CUDA
    torch.cuda.manual_seed_all(seed)                 ## Seed for all CUDA devices
    torch.backends.cudnn.deterministic = True        ## Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.benchmark = False           ## Disable CuDNN's autotuning for reproducibility



# Variable seed for DataLoader shuffling
set_seed_torch(1)   

# Variable main seed (model, CUDA, etc.)
set_seed_main(1)  





# ✅ === DEFINE PATHS ===
data_path = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github2\LanguageTranslation\data"
multi30k_dir = os.path.join(data_path, "multi30k")
os.makedirs(multi30k_dir, exist_ok=True)


# MOVE FILES INTO multi30k if needed
# train.de, train.en, val.de, val.en, test.de, test.en
# Must be in: ...\data\multi30k\


# ✅ === LOAD AND TOKENIZE ====
os.system("python -m spacy download de_core_news_sm")
os.system("python -m spacy download en_core_web_sm")
spacy_ger = spacy.load('de_core_news_sm')
spacy_eng = spacy.load('en_core_web_sm')





# ✅ === TOKENIZERS ===
# Tokenization of German Language
def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


# Tokenization of English Language
def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

print("✅ Tokenizers are ready.")



# ✅ === PREPROCESSING OF TEXT | DEFINE FIELDS ===
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





# === Print samples ===
# === STEP 7: Confirm ===
print(f"✅ Training examples: {len(train_data)}")
print(f"✅ Validation examples: {len(valid_data)}")
print(f"✅ Test examples: {len(test_data)}\n")
print("✅ Sample:")
print(vars(train_data[0]))








# ✅ === FFTGate ACTIVATION FUNCTION ✅ ===

class FFTGate(nn.Module):
    """
    FFTGate Activation Function

    Parameters:
    ----------
    gamma1 : float
        Trainable scaling factor applied to pre-activation input (controls gating sharpness).
        Initialized as a learnable parameter.

    phi : float
        Frequency of the sinusoidal perturbation added to the gated activation output to prevent
        neuron saturation and improve learning smoothness.

    history_len : int
        Temporal activation history window length (T), i.e., number of past activation states stored
        for frequency-domain analysis.

    enable_history : bool
        Enables or disables tracking of temporal activation history. If False, FFT-based modulation is skipped.

    use_skip : bool
        Toggles an optional residual skip connection that blends the input and modulated output (0.95 * activation + 0.05 * input).

    decay_mode : str
        Temporal Activation History Decay Strategy:
        - "exp"    → Exponential decay strategy (cosine-based, monotonic)
        - "linear" → Non-monotonic decay strategy using sigmoid-blended multi-phase decay
    """


    def __init__(self, gamma1=1.5, phi=0.1, history_len=12,
                 enable_history=True, use_skip=False, decay_mode="linear"):    
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



        # 🔧 Trainable scaling factor
        # self.gamma1 = nn.Parameter(torch.tensor(gamma1, device=self.device))
        self.gamma1 = nn.Parameter(torch.full((1,), gamma1, device=self.device))


        # 🔒 Fixed buffer phi (No gradient updates)
        self.register_buffer('phi', torch.tensor(phi, device=self.device))

        self.history_len = history_len
        self.enable_history = enable_history
        self.activation_history_initialized = False

        # ⚙️ Optional skip connection toggle
        self.use_skip = use_skip

        # ✅ Save selected decay strategy
        self.decay_mode = decay_mode


    # 🔍 Computes sigmoid-blended multi-phase decay rate used for non-monotonic history scaling | non-monotonic decay stratey defination
    def sigmoid_blended_decay(self, epoch, num_epochs,
                            t1=50, t2=70,
                            k1=0.25, k2=0.5,
                            a=0.980, b=0.995, c=0.9801):
        """📉 Sigmoid-based phase-decay transition"""

        # Rescale current epoch to fit 0–100 reference scale
        epoch_scaled = (epoch / num_epochs) * 100

        s1 = 1 / (1 + math.exp(-k1 * (epoch_scaled - t1)))
        s2 = 1 / (1 + math.exp(-k2 * (epoch_scaled - t2)))
        return a * (1 - s1) + b * s1 * (1 - s2) + c * s2


    # 🔍 Tracks moving average of pre-activation inputs over a history window (per channel)
    def update_history(self, x):
        if not self.enable_history:
            return

        # [seq_len, batch_size, embedding_size] → average across seq_len and emb
        if x.dim() == 3:
            x = x.mean(dim=(0, 2))    # → [batch_size]
        elif x.dim() == 2:
            x = x.mean(dim=0)
        elif x.dim() == 4:
            x = x.mean(dim=(0, 2, 3))  # for CNNs
        else:
            raise ValueError(f"Unexpected input shape {x.shape}")

        x = x.detach().to(self.device)

        # 🔁 Reset buffer if shape changes | 🔥 Dynamically reset history buffer if shape changes
        if not self.activation_history_initialized or self.activation_history.shape[1] != x.shape[0]:
            self.register_buffer("activation_history", torch.zeros(self.history_len, x.shape[0], device=self.device))
            self.activation_history_initialized = True

        self.activation_history = torch.cat([self.activation_history[1:], x.unsqueeze(0)])


    # 🔍 Decays the stored activation history based on selected temporal decay strategy
    def decay_spectral_history(self, epoch, num_epochs=50):
        if not (self.enable_history and self.activation_history_initialized):
            return

        with torch.no_grad():
            if self.decay_mode == "linear":                                 # ✅ Use Non-Monotonic decay Strategy   
                # ✅ Use sigmoid-based blended decay
                decay_rate = self.sigmoid_blended_decay(epoch, num_epochs)  
                self.activation_history *= decay_rate

            elif self.decay_mode == "exp":                                  # ✅ Use Exponential decay Strategy 
                decay_factor = 0.99 + 0.01 * math.cos(math.pi * epoch / num_epochs)
                self.activation_history *= decay_factor

            else:
                raise ValueError(f"Unknown decay_mode: {self.decay_mode}. Use 'exp' or 'linear'.")


    def forward(self, x, epoch=0):
        x = x.to(self.device)
        self.saved_output = x.clone().detach()

        if self.enable_history:
            self.update_history(x)

        # 🔍 Applies FFT to decayed activation history and computes average magnitude
        if self.enable_history and self.activation_history_initialized:
            freq_response = torch.fft.fft(self.activation_history, dim=0)
            freq_magnitude = torch.abs(freq_response).mean(dim=0)
            freq_magnitude = torch.clamp(freq_magnitude, min=0.05, max=1.5)

            # 🔧 Smoothing across channels to prevent spikes              
            smoothing_factor = max(0.1, 1 / (epoch + 10))
            freq_magnitude = (1 - smoothing_factor) * freq_magnitude + smoothing_factor * freq_magnitude.mean()
            # freq_magnitude = freq_magnitude.view(1, -1, 1, 1)
            freq_magnitude = freq_magnitude.view(1, -1, 1) # match [seq_len, batch, emb_dim]
        else:
            freq_magnitude = torch.zeros_like(x)


        # ✅ Clamp gamma1 internally for stability
        gamma1 = torch.clamp(self.gamma1, min=0.1, max=6.0)

        # 🔄 Gate using FFT-derived frequency magnitude
        freq_factor = min(0.3 + 0.007 * epoch, 0.8)
        gate = torch.sigmoid(gamma1 * x - freq_factor * freq_magnitude)

        # ✅ Main activation logic: gated + sinusoidal regularization
        activation = x * gate + 0.05 * torch.sin(self.phi * x)  


        # 🔁 Optional: smart residual skip blend
        if self.use_skip:
            activation = 0.95 * activation + 0.05 * x


        return activation












# ✅ === DEFINE REGULARIZATION FUNCTION ===
def apply_dynamic_regularization(net, inputs, feature_activations, epoch, num_epochs,
                                  prev_params, layer_index_map, gamma1_history,
                                  batch_idx, total_batches, test_acc):


    global activation_layers  # ✅ Reference already-collected layers


    if batch_idx == 0 and epoch <= 2:
        print(f"\n🚨 ENTERED apply_dynamic_regularization | Epoch={epoch} | Batch={batch_idx}", flush=True)

        # 🧠 Print all gamma1 stats in one line (once per batch)
        all_layer_info = []
        for idx, layer in enumerate(activation_layers):
            param = getattr(layer, "gamma1")
            all_layer_info.append(f"Layer {idx}: ID={id(param)} | Mean={param.mean().item():.5f}")
        print("🧠 GAMMA1 INFO:", " | ".join(all_layer_info), flush=True)

    # ✅ Initialize gamma1 regularization accumulator
    gamma1_reg = 0.0

    # ✅ Compute batch std and define regularization strength
    batch_std = torch.std(inputs.float()) + 1e-6
    regularization_strength = 0.05 if epoch < 20 else (0.01 if epoch < 35 else 0.005)

    # ✅ Track layers where noise is injected (informative)
    noisy_layers = []
    for idx, layer in enumerate(activation_layers):
        if idx not in layer_index_map:
            continue

        prev_layer_params = prev_params[layer_index_map[idx]]
        param_name = "gamma1"  # ✅ Only gamma1 is trainable
        param = getattr(layer, param_name)
        prev_param = prev_layer_params[param_name]

        # ✅ Target based on input stats
        target = compute_target(param_name, batch_std)

        # ✅ Adaptive Target Regularization
        gamma1_reg += regularization_strength * (param - target).pow(2).mean() * 1.2

        # ✅ Adaptive Cohesion Regularization
        cohesion = (param - prev_param).pow(2)  
        gamma1_reg += 0.005 * cohesion.mean()  

        # ✅ Adaptive Noise Regularization
        epoch_AddNoise = 25
        if epoch > epoch_AddNoise:
            param_variation = torch.abs(param - prev_param).mean()
            if param_variation < 0.015:  
                noise = (0.001 + 0.0004 * batch_std.item()) * torch.randn_like(param)
                penalty = (param - (prev_param + noise)).pow(2).sum()
                gamma1_reg += 0.00015 * penalty                  
                noisy_layers.append(f"{idx} (Δ={param_variation.item():.5f})") # ✅ Collect index and variation

    # ✅ Print noise injection summary
    if batch_idx == 0 and epoch <= (epoch_AddNoise+2) and noisy_layers:
        print(f"🔥 Stable Noise Injected | Epoch {epoch} | Batch {batch_idx} | Layers: " + ", ".join(noisy_layers), flush=True)
    mags = feature_activations.abs().mean(dim=(0, 1)) 
    m = mags / mags.sum()
    gamma1_reg += 0.005 * (-(m * torch.log(m + 1e-6)).sum())

    return gamma1_reg


def compute_target(param_name, batch_std):
    if param_name == "gamma1":
        return 2.0 + 0.2 * batch_std.item()     
    
    raise ValueError(f"Unknown param {param_name}")









# ✅ === DEFINING THE ENCODER PART OF THE MODEL === 
# === Encoder with FFTGate ===
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        
        # ✅ Your activation with all parameters
        self.activation = FFTGate(gamma1=1.5, phi=0.1, history_len=12, decay_mode="linear")

    def forward(self, x, epoch=0):
        embedding = self.dropout(self.activation(self.embedding(x), epoch=epoch))
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell






# ✅ ===  DEFINING THE DECODER PART OF THE MODEL === 
# === Decoder with FFTGate ===
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
        self.activation = FFTGate(gamma1=1.5, phi=0.1, history_len=12, decay_mode="linear")

    def forward(self, x, hidden, cell, epoch=0):
        x = x.unsqueeze(0)
        embedding = self.dropout(self.activation(self.embedding(x), epoch=epoch))
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        predictions = self.fc(outputs).squeeze(0)
        return predictions, hidden, cell









# ✅ === DeEFINING THE COMPLETE MODEL === 
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




# Hyperparameters
num_epochs = 50
learning_rate = 0.001
batch_size = 256



# Model hyperparameters
load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300

hidden_size = 1024
num_layers = 1
enc_dropout = 0.5
dec_dropout = 0.5



# Tensorboard to get nice loss plot
writer = SummaryWriter(f'runs/Loss_plot')
step = 0




train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
     batch_size = batch_size, sort_within_batch = True, 
     sort_key = lambda x:len(x.src),
     device = device)





encoder_net = Encoder(input_size_encoder, 
                      encoder_embedding_size,
                      hidden_size,num_layers, 
                      enc_dropout).to(device)


decoder_net = Decoder(input_size_decoder, 
                      decoder_embedding_size,
                      hidden_size,output_size,num_layers, 
                      dec_dropout).to(device)





# ====== 🔧 BUILD MODEL 🔧 ======
model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)





# ====== ✅ FFTGate HANDLING ✅ ======

# ✅ 1a. Collect activation parameters
activation_params = (
    list(encoder_net.activation.parameters()) +
    list(decoder_net.activation.parameters())
)


# ✅ 1b. Collect activation layers (used for decay/regularization logic)
activation_layers = [
    m for m in encoder_net.activation.modules() if isinstance(m, FFTGate)
] + [
    m for m in decoder_net.activation.modules() if isinstance(m, FFTGate)
]




# ✅ 1c. Define layer index map once
layer_index_map = {idx: idx for idx in range(len(activation_layers))}



# ✅ 2. Setup freezing, optimizer, scheduler
unfreeze_activation_epoch = 1               # ⏱ When to unfreeze
WARMUP_ACTIVATION_EPOCHS = 0                # 🔥 Optional warm-up delay

# 🔒 Freeze activation parameter initially
for param in activation_params:
    param.requires_grad = False


# 🧠 Define activation parameter optimizer
activation_optimizers = {
    "gamma1": optim.AdamW(activation_params, lr=0.0015, weight_decay=1e-6)
}

# 🔄 Cosine Annealing Scheduler for activation paramter
activation_schedulers = {
    "gamma1": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        activation_optimizers["gamma1"],
        T_0=10,
        T_mult=2,
        eta_min=1e-5
    )
}








# ✅ === TRANSLATION FUNCTION FOR INFERENCE ===
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







def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)



def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])



if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)


sentence = "Cristiano Ronaldo ist ein großartiger Fußballspieler mit erstaunlichen Fähigkeiten und Talenten."


# %% 

# ✅ === TRAINING ===


name_Main = 'FFTGate' # ✅ Used for naming files 
log_history = []      # ✅ Log messages for all epochs
log_path = fr"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github2\LanguageTranslation\log_{name_Main}.txt"


for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")



    # ✅ Unfreeze activation parameters at specified epoch
    if epoch == unfreeze_activation_epoch:
        print(f"🔓 Unfreezing activation parameters at epoch {epoch}")
        for param in activation_params:
            param.requires_grad = True



    # ✅ Save model checkpoint
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.eval()

    #translated_sentence = translate_sentence(
    #    model, sentence, german, english, device, max_length=50
    #)

    #print(f"Translated example sentence: \n {translated_sentence}")

    # ✅ Switch to training mode
    model.train()


    # ✅ Track per-layer activation means
    activation_history = []   # 🔴 Initialize empty history at start of epoch (outside batch loop)






    # ✅ Initialize prev_params for regularization | ✅ Before epoch loop 
    prev_params = {}
    
    for idx, layer in enumerate(activation_layers):
        prev_params[idx] = {"gamma1": layer.gamma1.clone().detach()}








    # ✅ Loop through training batches
    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda | === Move data to device
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # ===  Forward prop
        # output = model(inp_data, target)
        output = model(inp_data, target, epoch=epoch)


        # === Reshape output and target for loss
        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin. While we're at it
        # Let's also remove the start token while we're at it
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)



        # ✅ Zero gradients
        optimizer.zero_grad()
        if epoch >= unfreeze_activation_epoch:
            for opt in activation_optimizers.values():
                opt.zero_grad()


        # ✅ Loss calculation
        loss = criterion(output, target)




        # ✅ Collect Activation History | ✅ Per-layer mean activations
        for layer in activation_layers:  # 🔄 Use activation_layers directly
            if hasattr(layer, "saved_output"):
                activation_history.append(layer.saved_output.mean().item())

        # ✅ Apply Decay strategy to history for each activation layer
        with torch.no_grad():
            for layer in activation_layers:
                if isinstance(layer, FFTGate):
                    layer.decay_spectral_history(epoch, num_epochs)





        # ✅ Call Regularization Function for the Activation Parameter
        if epoch > 0:
            gamma1_reg = apply_dynamic_regularization(
                model, inp_data, output, epoch, num_epochs,
                prev_params, layer_index_map, gamma1_history=None,
                batch_idx=batch_idx, total_batches=None,
                test_acc=None
            )
            loss += gamma1_reg




        # ✅ Backprop
        loss.backward()


        # ✅ Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)


        # ✅ Optimizer step
        # Gradient descent step
        optimizer.step()
        if epoch >= unfreeze_activation_epoch:
            for opt in activation_optimizers.values():
                opt.step()




        # ✅ Clamp gamma1 values after step
        with torch.no_grad():
            for layer in model.modules():
                if isinstance(layer, FFTGate):
                    layer.gamma1.data.clamp_(0.3, 18.0)




        # ✅ TensorBoard Logging | Plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1




    # ✅ ONLY update prev_params here AFTER all updates | ✅ Update prev_params AFTER training epoch
    for idx, layer in enumerate(activation_layers):
        prev_params[idx]["gamma1"] = layer.gamma1.clone().detach()





    # ✅ Log activation mean (optional)
    if activation_history:
        avg_act = sum(activation_history) / len(activation_history)
        writer.add_scalar("Activation/AvgGamma1Output", avg_act, global_step=epoch)



    # ✅ Activation scheduler step (once per epoch)
    if epoch >= unfreeze_activation_epoch:
        for name, act_scheduler in activation_schedulers.items():
            act_scheduler.step()



    # ✅ Logging Parameters & Gradients after epoch
    last_batch_grads = {"Gamma1 Grad": []}
    current_params = {"Gamma1": []}

    for layer in model.modules():
        if isinstance(layer, FFTGate):
            last_batch_grads["Gamma1 Grad"].append(
                f"{layer.gamma1.grad.item():.5f}" if layer.gamma1.grad is not None else "None"
            )
            current_params["Gamma1"].append(f"{layer.gamma1.item():.5f}")

    log_msg = (
        f"Epoch {epoch}: M_Optimizer LR => {optimizer.param_groups[0]['lr']:.5f} | "
        f"Gamma1 LR => {activation_optimizers['gamma1'].param_groups[0]['lr']:.5f} | "
        f"Gamma1: {current_params['Gamma1']} | "
        f"Gamma1 Grad: {last_batch_grads['Gamma1 Grad']}"
    )
    log_history.append(log_msg)
    print(log_msg)

# ✅ Write log history to file
with open(log_path, "w", encoding="utf-8") as f:
    for line in log_history:
        f.write(line + "\n")








# ✅ === BLEU SCORE ===

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




score = bleu(test_data[1:100], model, german, english, device)
# print(f"Bleu score {score*100:.2f}")
bleu_str = f"✅ BLEU Score: {score*100:.2f}"

print(bleu_str)
output_path = fr"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github2\LanguageTranslation\BLEU_Score_{name_Main}.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(bleu_str + "\n")




# In[28]:





