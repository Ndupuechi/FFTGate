



# %% Imports and Setup


# ===============================================================
# 🔗==================== ENGLISH→FRENCH 🔑====================🔗
# ===============================================================


########################################################################################################################
####-------| NOTE 1. IMPORTS LIBRARIES | XXX -------------------------------------------------------####################
########################################################################################################################



# ✅ === Enable flexible CUDA memory allocation to reduce fragmentation ===
# Must be set before importing torch!
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Alternative for memory split limits:
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"


# ✅ === Ensure correct working directory ===
import sys
Project_PATH = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github4\IWSLT\Comparative_Study\Results\IWSLT\FR_EN"
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
# ✅ === Standard libraries ===
# ======================================================================================================

import re
import math
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl


# ======================================================================================================
# ✅ === Custom parser ===
# ======================================================================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Dataset information
exp_parser = argparse.ArgumentParser("IWSLT13 EN-FR Experiment Config")
exp_parser.add_argument('--dataset_name', default="IWSLT13_Fr_En", type=str,
    help="Choose dataset direction: 'IWSLT13_En_Fr' = English→French, 'IWSLT13_Fr_En' = French→English")  # Option: IWSLT13_En_Fr or IWSLT13_Fr_En 
# exp_parser.add_argument('--translation_direction', default="EN⟶FR", type=str)
exp_parser.add_argument('--translation_direction', default=r"FR$\rightarrow$EN", type=str)

# ✅ Global font settings
exp_parser.add_argument('--base_font_size', default=13, type=int)        # Default: 11   
exp_parser.add_argument('--spine_width', default=1.0, type=float)        # Default: 1.2
exp_parser.add_argument('--legend_title_font', default=12, type=int)     # Default: 10
exp_parser.add_argument('--legend_font', default=12, type=int)           # Default: 9

exp_args = exp_parser.parse_args([])   # ← for naming/logging
# ─────────────────────────────────────────────────────────────────────────────────────────────────




# ===============================================================
# 🔗============= GLOBAL FONT SETTINGs 🔑=====================🔗
# ===============================================================

plt.rcParams.update({

    # === FONT SETTINGS ===
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "text.latex.preamble": r"\usepackage{lmodern}\usepackage{bm}\boldmath",  # makes all LaTeX text bold

    # === Colors ===
    "text.color": "#000000",               # solid black
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "axes.edgecolor": "#000000",
    "axes.titlecolor": "#000000",


    "font.size": exp_args.base_font_size,
    "font.weight": "normal",
    "axes.titlesize": exp_args.base_font_size + 1,
    "axes.titleweight": "normal",
    "axes.labelsize": exp_args.base_font_size + 2,
    "axes.labelweight": "medium",
    "legend.fontsize": exp_args.base_font_size - 1,
    "legend.title_fontsize": exp_args.base_font_size,
    "xtick.labelsize": exp_args.base_font_size,
    "ytick.labelsize": exp_args.base_font_size,

    # === COLOR CONSISTENCY ===
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.labelcolor": "black",
    "text.color": "black",
    "axes.edgecolor": "black",


    # === AXES & SPINES ===
    "axes.linewidth": exp_args.spine_width,   # ✅ ensures ALL future figures use this spine width        
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.axisbelow": False,   # ensures lines/markers are above grid

    # === PDF / SVG EXPORT QUALITY ===
    "pdf.fonttype": 42,        # editable text in PDF
    "ps.fonttype": 42,         # editable text in PS
    "svg.fonttype": 'none',    # editable text in SVG
})

print(f"✅ Publication style applied: Bold fonts, black ticks, clean spines (base font size={exp_args.base_font_size} | width={exp_args.spine_width}).")









######################################## 1️⃣ 2️⃣ 3️⃣ Ⓐ Ⓑ Ⓒ 🅐 🅑 🅒 🅓 ##################################################
########################################################################################################################
####--🔴--| NOTE: BLEU Score Comparison (ReLU, GELU, TanhExp, GEGLU, FFTGate) | XXX ----------------####################
########################################################################################################################
# 🔗========================================== ENGLISH→FRENCH 🔑=====================================================🔗


# ======================================================================================================
# ✅ =======================🔖 BLUE SCORE 🔖==========================================================
# ======================================================================================================


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅐 Define Test log file paths
RELU_test_log     = fr'./1-Transformer_RELU/relu/TestCheckpoints_relu_Transformer_{exp_args.dataset_name}_Adam_Seed1_1_EXP3.txt'
GELU_test_log     = fr'./2-Transformer_GELU/gelu/TestCheckpoints_gelu_Transformer_{exp_args.dataset_name}_Adam_Seed1_1_EXP3.txt'
TanhExp_test_log  = fr'./3-Transformer_TanhExp/tanhexp/TestCheckpoints_tanhexp_Transformer_{exp_args.dataset_name}_Adam_Seed1_1_EXP3.txt'
GEGLU_test_log    = fr'./4-Transformer_GEGLU/geglu/TestCheckpoints_geglu_Transformer_{exp_args.dataset_name}_Adam_Seed1_1_EXP3.txt'
FFTGate_test_log  = fr'./5-Transformer_FFTGate/fftgate/TestCheckpoints_fftgate_Transformer_{exp_args.dataset_name}_Adam_Seed1_1_EXP3.txt'
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅑 Define Train log file paths
RELU_train_log     = fr'./1-Transformer_RELU/relu/Train_relu_Transformer_{exp_args.dataset_name}_Adam_Seed1_1_EXP3.txt'
GELU_train_log     = fr'./2-Transformer_GELU/gelu/Train_gelu_Transformer_{exp_args.dataset_name}_Adam_Seed1_1_EXP3.txt'
TanhExp_train_log  = fr'./3-Transformer_TanhExp/tanhexp/Train_tanhexp_Transformer_{exp_args.dataset_name}_Adam_Seed1_1_EXP3.txt'
GEGLU_train_log    = fr'./4-Transformer_GEGLU/geglu/Train_geglu_Transformer_{exp_args.dataset_name}_Adam_Seed1_1_EXP3.txt'
FFTGate_train_log  = fr'./5-Transformer_FFTGate/fftgate/Train_fftgate_Transformer_{exp_args.dataset_name}_Adam_Seed1_1_EXP3.txt'
# ─────────────────────────────────────────────────────────────────────────────────────────────────







# ===============================================================
# 🔗=============== READ LOG FUNCTIONs 🔑=====================🔗
# ===============================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ Function to read BLEU scores from Fairseq-style test logs
def read_bleu_scores(file_path):
    epochs, bleu_scores = [], []
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        return epochs, bleu_scores

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(r"epoch_(\d+).*?sacreBLEU:\s*([\d.]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                bleu_scores.append(float(match.group(2)))
    return epochs, bleu_scores
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣ Function to read Test Loss values from Fairseq-style test logs
def read_test_loss(file_path):
    epochs, test_losses = [], []
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        return epochs, test_losses

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Match only lines that include an epoch number (e.g., "epoch_23")
            match = re.search(r"epoch_(\d+).*?Test Loss:\s*([\d.]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                test_losses.append(float(match.group(2)))
    return epochs, test_losses
# ─────────────────────────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 3️⃣ Function to read Train Loss values from Fairseq-style training logs
def read_train_loss(file_path):
    epochs, train_losses = [], []
    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        return epochs, train_losses

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Match lines like: "Epoch 23 | Train Loss: 3.123 | LR: 0.000045"
            match = re.search(r"Epoch\s+(\d+)\s*\|\s*Train Loss:\s*([\d.]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
    return epochs, train_losses
# ─────────────────────────────────────────────────────────────────────────────────────────────────










# ===============================================================
# 🔗=================== READ TEXTS 🔑=========================🔗
# ===============================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ Read BLEU data for each activation
relu_epochs, relu_bleu       = read_bleu_scores(RELU_test_log)
gelu_epochs, gelu_bleu       = read_bleu_scores(GELU_test_log)
tanhexp_epochs, tanhexp_bleu = read_bleu_scores(TanhExp_test_log)
geglu_epochs, geglu_bleu     = read_bleu_scores(GEGLU_test_log)
fftgate_epochs, fftgate_bleu = read_bleu_scores(FFTGate_test_log)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣ Read Test Loss data for each activation
relu_epochs_loss, relu_loss       = read_test_loss(RELU_test_log)
gelu_epochs_loss, gelu_loss       = read_test_loss(GELU_test_log)
tanhexp_epochs_loss, tanhexp_loss = read_test_loss(TanhExp_test_log)
geglu_epochs_loss, geglu_loss     = read_test_loss(GEGLU_test_log)
fftgate_epochs_loss, fftgate_loss = read_test_loss(FFTGate_test_log)
# ─────────────────────────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 3️⃣ Read Train Loss data for each activation
relu_epochs_train, relu_train_loss       = read_train_loss(RELU_train_log)
gelu_epochs_train, gelu_train_loss       = read_train_loss(GELU_train_log)
tanhexp_epochs_train, tanhexp_train_loss = read_train_loss(TanhExp_train_log)
geglu_epochs_train, geglu_train_loss     = read_train_loss(GEGLU_train_log)
fftgate_epochs_train, fftgate_train_loss = read_train_loss(FFTGate_train_log)
# ─────────────────────────────────────────────────────────────────────────────────────────────────






# ===============================================================
# 🔗==================== PLOT FUNCTION 🔑=====================🔗
# ===============================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# ✅ Plot BLEU vs Epoch comparison (with bold gold legend title above legend box)
def plot_bleu_comparison(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)
# ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # 🎨 Modern, distinct, cool palette
    COLORS = {
        "ReLU":     "#8338EC",   # Rich Deep Blue
        "GELU":     "#FFD60A",   # Gold
        "TanhExp":  "#06D6A0",   # Aqua Green | "#06D6A0"
        "GEGLU":    "#EF476F",   # Pink-Red
        "FFTGate":  "#071DE7C5",   # Violet (Highlight) | "#0862D1"| "#0057E7"|"#071DE7C5"
    }
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ---------------------------- 1️⃣ Create BLEU Score Figure ----------------------------------------
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    fig_bleu_score, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    # ────────────────────────────────────────────────────────────────
    # ✅ Plot BLEU curves
    # ────────────────────────────────────────────────────────────────
    # 🔷 Baselines — curve with no white markers
    ax.plot(relu_epochs, relu_bleu, label="ReLU",
                 color=COLORS["ReLU"], linewidth=1.8, alpha=1.0)
    ax.plot(gelu_epochs, gelu_bleu, label="GELU",
                 color=COLORS["GELU"], linewidth=1.8, alpha=1.0)
    ax.plot(tanhexp_epochs, tanhexp_bleu, label="TanhExp",
                 color=COLORS["TanhExp"], linewidth=1.8, alpha=1.0)
    ax.plot(geglu_epochs, geglu_bleu, label="GEGLU",
                 color=COLORS["GEGLU"], linewidth=1.8, alpha=1.0)

    # 🔷 FFTGate — curve with white markers
    ax.plot(
        fftgate_epochs,
        fftgate_bleu,
        label="FFTGate",
        color=COLORS["FFTGate"],
        linewidth=2.0,  # default:2.0
        alpha=1.0,
        marker='o',
        markersize=4.5,
        markerfacecolor='white',            # ✅ white fill
        markeredgecolor=COLORS["FFTGate"],  # outline matches FFTGate color
        markeredgewidth=1.4,
        markevery=5,          # show marker every 5 epochs
        zorder=10             # 👈 Increased from 6 → 10 for visual dominance
        )
    # ─────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Axis labels and grid
    ax.set_xlabel(r"\textbf{Epoch}")
    ax.set_ylabel(r"\textbf{BLEU Score}")

    ax.grid(True, linestyle='--', alpha=0.35)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Custom Axis Styling (Publication Style)
    ax.set_xlim(-2, 52)                  # 👈 adds 1 extra unit before 0 and after 50
    # ax_loss.set_xlim(0, 50)                 # X-axis range (0 to 50 epochs)
    ax.set_xticks(range(0, 55, 10))      # X-ticks at every 10 epochs
    ax.set_ylim(18, 38)                  # Y-axis range   #13
    ax.set_yticks([20, 25, 30, 35])  # Y-ticks for clarity
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Legend
    leg = ax.legend(
        fontsize=exp_args.legend_font,
        loc="lower right",
        frameon=False,
        facecolor="white",
        edgecolor="gray",
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.6,           
    )

    # ✅ Make legend text bold (LaTeX text mode, not math)
    for text in leg.get_texts():
        label = text.get_text()
        text.set_text(r"\textbf{" + label.replace("\\", "\\\\") + "}")

    # ✅ Draw canvas for legend coordinates
    fig_bleu_score.canvas.draw()

    # ✅ Add centered bold “FR→EN” title above legend
    bbox = leg.get_window_extent(fig_bleu_score.canvas.get_renderer()).transformed(
        fig_bleu_score.transFigure.inverted()
    )
    fig_bleu_score.text(
        (bbox.x0 + bbox.x1) / 2, bbox.y1 + 0.015,  # vertical offset for spacing
        r"\textbf{" + exp_args.translation_direction + "}",
        fontsize=exp_args.legend_title_font,
        color="black",
        ha="center",
        va="bottom"
    )
    # ─────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ✅ Save BLEU figure (dynamic name from dataset)
    fig_bleu_score.savefig(
        os.path.join(save_dir, f"BLEU_vs_Epoch_{exp_args.dataset_name}_Activations.pdf"),
        format="pdf",
        bbox_inches="tight",
        facecolor="white",  # ensures no transparency washout
        dpi=600             # high-quality export
    )
    fig_bleu_score.savefig(
        os.path.join(save_dir, f"BLEU_vs_Epoch_{exp_args.dataset_name}_Activations.svg"),
        format="svg",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────────────────────────────












    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ---------------------------- 2️⃣ Create Test Loss Figure ----------------------------------------
    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    fig_test_loss, ax_loss = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    # ────────────────────────────────────────────────────────────────
    # ✅ Plot Test Loss curves
    # ────────────────────────────────────────────────────────────────
    # 🔷 Baselines — curve with no white markers
    ax_loss.plot(relu_epochs_loss, relu_loss, label="ReLU",
                 color=COLORS["ReLU"], linewidth=1.8, alpha=1.0)
    ax_loss.plot(gelu_epochs_loss, gelu_loss, label="GELU",
                 color=COLORS["GELU"], linewidth=1.8, alpha=1.0)
    ax_loss.plot(tanhexp_epochs_loss, tanhexp_loss, label="TanhExp",
                 color=COLORS["TanhExp"], linewidth=1.8, alpha=1.0)
    ax_loss.plot(geglu_epochs_loss, geglu_loss, label="GEGLU",
                 color=COLORS["GEGLU"], linewidth=1.8, alpha=1.0)


    # 🔷 FFTGate — curve with white markers
    ax_loss.plot(
        fftgate_epochs_loss,
        fftgate_loss,
        label="FFTGate",
        color=COLORS["FFTGate"],
        linewidth=2.0,    #2.0
        alpha=1.0,
        marker='o',
        markersize=4.5,
        markerfacecolor='white',            # ✅ white fill
        markeredgecolor=COLORS["FFTGate"],  # outline matches FFTGate color
        markeredgewidth=1.4,
        markevery=5,          # show marker every 5 epochs
        zorder=10             # 👈 Increased from 6 → 10 for visual dominance
        )
    # ─────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Axis labels and grid (LaTeX-bold for consistency)
    ax_loss.set_xlabel(r"\textbf{Epoch}")
    ax_loss.set_ylabel(r"\textbf{Test Loss}")
    ax_loss.grid(True, linestyle="--", alpha=0.35)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────



    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Custom Axis Styling (Publication Style)
    ax_loss.set_xlim(-2, 52)                  # 👈 adds 1 extra unit before 0 and after 50
    # ax_loss.set_xlim(0, 50)                 # X-axis range (0 to 50 epochs)
    ax_loss.set_xticks(range(0, 55, 10))      # X-ticks at every 10 epochs
    ax_loss.set_ylim(2.5, 7.0)                # Y-axis range
    ax_loss.set_yticks([3.0, 4.0, 5.0, 6.0, 7.0])  # Y-ticks for clarity   
    # ─────────────────────────────────────────────────────────────────────────────────────────────────



    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Legend
    leg_loss = ax_loss.legend(
        fontsize=exp_args.legend_font,
        loc="upper right",
        frameon=False,
        facecolor="white",
        edgecolor="gray",
        bbox_to_anchor=(1.0, 0.91),  # 👈 slightly lower placement for balance
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.6,       
    )

    # ✅ Make legend text bold (LaTeX text mode, not math)
    for text in leg_loss.get_texts():
        label = text.get_text()
        text.set_text(r"\textbf{" + label.replace("\\", "\\\\") + "}")

    # ✅ Draw canvas for legend coordinates
    fig_test_loss.canvas.draw()

    # ✅ Add centered bold “FR→EN” title above legend
    bbox_loss = leg_loss.get_window_extent(fig_test_loss.canvas.get_renderer()).transformed(
        fig_test_loss.transFigure.inverted()
    )
    fig_test_loss.text(
        (bbox_loss.x0 + bbox_loss.x1) / 2,
        bbox_loss.y1 + 0.015,  # 👈 tuned spacing to match BLEU plot (was 0.005)
        r"\textbf{" + exp_args.translation_direction + "}",
        fontsize=exp_args.legend_title_font,
        color="black",
        ha="center",
        va="bottom"
    )
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ✅ Save Test Loss figure (publication quality)
    fig_test_loss.savefig(
        os.path.join(save_dir, f"TestLoss_vs_Epoch_{exp_args.dataset_name}_Activations.pdf"),
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        dpi=600
    )
    fig_test_loss.savefig(
        os.path.join(save_dir, f"TestLoss_vs_Epoch_{exp_args.dataset_name}_Activations.svg"),
        format="svg",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()
    # ─────────────────────────────────────────────────────────────────────────────────────────────────







    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ---------------------------- 3️⃣ Create Train Loss Figure ----------------------------------------
    # ───────────────────────────────────────────────────────────────────────────────────────────────── 
    fig_train_loss, ax_train = plt.subplots(figsize=(5, 3.5), constrained_layout=True)


    # ────────────────────────────────────────────────────────────────
    # ✅ Plot Test Loss curves
    # ────────────────────────────────────────────────────────────────
    # 🔷 Baselines — curve with no white markers
    ax_train.plot(relu_epochs_train, relu_train_loss, label="ReLU",
                 color=COLORS["ReLU"], linewidth=1.8, alpha=1.0)
    ax_train.plot(gelu_epochs_train, gelu_train_loss, label="GELU",
                 color=COLORS["GELU"], linewidth=1.8, alpha=1.0)
    ax_train.plot(tanhexp_epochs_train, tanhexp_train_loss, label="TanhExp",
                 color=COLORS["TanhExp"], linewidth=1.8, alpha=1.0)
    ax_train.plot(geglu_epochs_train, geglu_train_loss, label="GEGLU",
                 color=COLORS["GEGLU"], linewidth=1.8, alpha=1.0)

    # 🔷 FFTGate — curve with white markers
    ax_train.plot(
        fftgate_epochs_train,
        fftgate_train_loss,
        label="FFTGate",
        color=COLORS["FFTGate"],
        linewidth=2.0,
        alpha=1.0,
        marker='o',
        markersize=4.5,
        markerfacecolor='white',            # ✅ white fill
        markeredgecolor=COLORS["FFTGate"],  # outline matches FFTGate color
        markeredgewidth=1.4,
        markevery=5,          # show marker every 5 epochs
        zorder=10             # 👈 Increased from 6 → 10 for visual dominance
        )
    # ─────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Axis labels (LaTeX-bold for consistency)
    ax_train.set_xlabel(r"\textbf{Epoch}")
    ax_train.set_ylabel(r"\textbf{Train Loss}")
    ax_train.grid(True, linestyle="--", alpha=0.35)
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Custom Axis Styling (Publication Style)
    ax_train.set_xlim(-2, 52)                  # 👈 adds 1 extra unit before 0 and after 50
    ax_train.set_xticks(range(0, 55, 10))      # X-ticks at every 10 epochs
    ax_train.set_ylim(2.5, 7.0)                # Y-axis range
    ax_train.set_yticks([3.0, 4.0, 5.0, 6.0, 7.0])  # Y-ticks for clarity
    # ─────────────────────────────────────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # ✅ Legend (shift slightly lower for balance)
    leg_train = ax_train.legend(
        fontsize=exp_args.legend_font,
        loc="upper right",
        frameon=False,
        facecolor="white",
        edgecolor="gray",
        bbox_to_anchor=(1.0, 0.91),  # 👈 slightly lower placement for better spacing
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.6,       
    )

    # ✅ Make legend text bold (LaTeX-safe, text mode)
    for text in leg_train.get_texts():
        label = text.get_text()
        text.set_text(r"\textbf{" + label.replace("\\", "\\\\") + "}")

    # ✅ Draw canvas so legend coordinates are updated
    fig_train_loss.canvas.draw()

    # ✅ Add bold centered “FR→EN” title above legend box
    bbox_train = leg_train.get_window_extent(fig_train_loss.canvas.get_renderer()).transformed(
        fig_train_loss.transFigure.inverted()
    )
    fig_train_loss.text(
        (bbox_train.x0 + bbox_train.x1) / 2, bbox_train.y1 + 0.015,  # vertical offset for spacing
        r"\textbf{" + exp_args.translation_direction + "}",
        fontsize=exp_args.legend_title_font,
        color="black",
        ha="center",
        va="bottom"
    )
    # ─────────────────────────────────────────────────────────────────────────────────────────────────

    # ✅ Save Train Loss figure (publication quality)
    fig_train_loss.savefig(
        os.path.join(save_dir, f"TrainLoss_vs_Epoch_{exp_args.dataset_name}_Activations.pdf"),
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        dpi=600
    )
    fig_train_loss.savefig(
        os.path.join(save_dir, f"TrainLoss_vs_Epoch_{exp_args.dataset_name}_Activations.svg"),
        format="svg",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.show()
    # ─────────────────────────────────────────────────────────────────────────────────────────────────







# 🔷 Call the function
plot_bleu_comparison()


# %%
