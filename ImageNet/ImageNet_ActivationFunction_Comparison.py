



# %% Imports and Setup


# ===============================================================
# 🔗==================== IMAGENET 🔑=========================🔗
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
Project_PATH = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github4\ImageNet"
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
exp_parser = argparse.ArgumentParser("ImageNet Experiment Config")
exp_parser.add_argument('--dataset_name', default="ConvNeXtV1_ATTO_ImageNet_1K_B256_LR0.004_AdamW_Seed1_2_INs32", type=str,
    help="Choose dataset direction: 'ImageNet")  

exp_parser.add_argument('--dataset_namefftgate', default="ConvNeXtV1_ATTO_ImageNet_1K_B256_LR0.004_AdamW_T12_Seed1_2_INs32", type=str,
    help="Choose dataset direction: 'ImageNet") 

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
# 🔗========================================== ImageNet 🔑==========================================================🔗


# ======================================================================================================
# ✅ =======================🔖 BLUE SCORE 🔖==========================================================
# ======================================================================================================


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅐 Define Test log file paths
RELU_test_log     = fr'./1-ImageNet_RELU/Test_RELU_{exp_args.dataset_name}.txt'
GELU_test_log     = fr'./2-ImageNet_GELU/Test_GELU_{exp_args.dataset_name}.txt'
TanhExp_test_log  = fr'./3-ImageNet_TanhExp/Test_TanhExp_{exp_args.dataset_name}.txt'
GEGLU_test_log    = fr'./4-ImageNet_GEGLU/Test_GEGLU_{exp_args.dataset_name}.txt'
FFTGate_test_log  = fr'./5-ImageNet_FFTGate/Test_FFTGate_{exp_args.dataset_namefftgate}.txt'
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅑 Define Train log file paths
RELU_train_log     = fr'./1-ImageNet_RELU/Train_RELU_{exp_args.dataset_name}.txt'
GELU_train_log     = fr'./2-ImageNet_GELU/Train_GELU_{exp_args.dataset_name}.txt'
TanhExp_train_log  = fr'./3-ImageNet_TanhExp/Train_TanhExp_{exp_args.dataset_name}.txt'
GEGLU_train_log    = fr'./4-ImageNet_GEGLU/Train_GEGLU_{exp_args.dataset_name}.txt'
FFTGate_train_log  = fr'./5-ImageNet_FFTGate/Train_FFTGate_{exp_args.dataset_namefftgate}.txt'
# ─────────────────────────────────────────────────────────────────────────────────────────────────









# ===============================================================
# 🔗=============== READ LOG FUNCTIONS (Updated for CIFAR logs) 🔑
# ===============================================================



# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ Read test loss and accuracy
def read_test_metrics(file_path):
    epochs, test_losses, test_accs = [], [], []
    best_acc = None

    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        return epochs, test_losses, test_accs, best_acc

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Example: "Epoch 0 | Test Loss: 4.332 | Test Acc: 3.270%"
            match = re.search(r"Epoch\s+(\d+)\s*\|\s*Test Loss:\s*([\d.]+)\s*\|\s*Test Acc:\s*([\d.]+)%", line)
            if match:
                epochs.append(int(match.group(1)))
                test_losses.append(float(match.group(2)))
                test_accs.append(float(match.group(3)))

            # Example: "🏆 Best Test Accuracy: 67.200%"
            best_match = re.search(r"Best Test Accuracy:\s*([\d.]+)%", line)
            if best_match:
                best_acc = float(best_match.group(1))
    return epochs, test_losses, test_accs, best_acc
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣ Read train loss and accuracy
def read_train_metrics(file_path):
    epochs, train_losses, train_accs = [], [], []
    best_acc = None

    if not os.path.exists(file_path):
        print(f"⚠️ Missing file: {file_path}")
        return epochs, train_losses, train_accs, best_acc

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Example: "Epoch 0 | Train Loss: 4.635 | Train Acc: 1.623%"
            match = re.search(r"Epoch\s+(\d+)\s*\|\s*Train Loss:\s*([\d.]+)\s*\|\s*Train Acc:\s*([\d.]+)%", line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                train_accs.append(float(match.group(3)))

            # Example: "🏆 Best Training Accuracy: 98.472%"
            best_match = re.search(r"Best Training Accuracy:\s*([\d.]+)%", line)
            if best_match:
                best_acc = float(best_match.group(1))
    return epochs, train_losses, train_accs, best_acc
# ─────────────────────────────────────────────────────────────────────────────────────────────────











# ===============================================================
# 🔗=================== READ LOGS 🔑=========================🔗
# ===============================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ Read Test metrics (Loss + Accuracy)
relu_epochs_test, relu_test_loss, relu_test_acc, relu_best_acc = read_test_metrics(RELU_test_log)
gelu_epochs_test, gelu_test_loss, gelu_test_acc, gelu_best_acc = read_test_metrics(GELU_test_log)
tanhexp_epochs_test, tanhexp_test_loss, tanhexp_test_acc, tanhexp_best_acc = read_test_metrics(TanhExp_test_log)
geglu_epochs_test, geglu_test_loss, geglu_test_acc, geglu_best_acc = read_test_metrics(GEGLU_test_log)
fftgate_epochs_test, fftgate_test_loss, fftgate_test_acc, fftgate_best_acc = read_test_metrics(FFTGate_test_log)

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣ Read Train metrics (Loss + Accuracy)
relu_epochs_train, relu_train_loss, relu_train_acc, relu_best_train_acc = read_train_metrics(RELU_train_log)
gelu_epochs_train, gelu_train_loss, gelu_train_acc, gelu_best_train_acc = read_train_metrics(GELU_train_log)
tanhexp_epochs_train, tanhexp_train_loss, tanhexp_train_acc, tanhexp_best_train_acc = read_train_metrics(TanhExp_train_log)
geglu_epochs_train, geglu_train_loss, geglu_train_acc, geglu_best_train_acc = read_train_metrics(GEGLU_train_log)
fftgate_epochs_train, fftgate_train_loss, fftgate_train_acc, fftgate_best_train_acc = read_train_metrics(FFTGate_train_log)
# ─────────────────────────────────────────────────────────────────────────────────────────────────











# ===============================================================
# 🔗==================== PLOT FUNCTION 🔑=====================🔗
# ===============================================================

def plot_imagenet_metrics(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)


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




    # ===============================================================
    # 1️⃣ PLOT TEST ACCURACY vs EPOCH
    # ===============================================================
    fig_acc, ax_acc = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    ax_acc.plot(relu_epochs_test, relu_test_acc, label="ReLU", color=COLORS["ReLU"], linewidth=1.8)
    ax_acc.plot(gelu_epochs_test, gelu_test_acc, label="GELU", color=COLORS["GELU"], linewidth=1.8)
    ax_acc.plot(tanhexp_epochs_test, tanhexp_test_acc, label="TanhExp", color=COLORS["TanhExp"], linewidth=1.8)
    ax_acc.plot(geglu_epochs_test, geglu_test_acc, label="GEGLU", color=COLORS["GEGLU"], linewidth=1.8)
    ax_acc.plot(
        fftgate_epochs_test,
        fftgate_test_acc,
        label="FFTGate",
        color=COLORS["FFTGate"],
        linewidth=2.0,
        marker='o',
        markersize=4.5,
        markerfacecolor='white',
        markeredgecolor=COLORS["FFTGate"],
        markeredgewidth=1.3,
        markevery=5,
        zorder=10,
    )

    ax_acc.set_xlabel(r"\textbf{Epoch}")
    ax_acc.set_ylabel(r"\textbf{Test Accuracy (\%)}")
    ax_acc.grid(True, linestyle="--", alpha=0.35)
    # ax_acc.set_xlim(-2, max(fftgate_epochs_test) + 2)
    # ax_acc.set_ylim(min(min(gelu_test_acc), min(fftgate_test_acc)) - 2, max(max(gelu_test_acc), max(fftgate_test_acc)) + 2)


    # ────────────────────────────────────────────────────────────────
    # ✅ Custom Axis Styling (Publication Style)
    ax_acc.set_xlim(-4, 104)                  # 👈 add some padding before and after epochs
    ax_acc.set_xticks(range(0, 105, 20))      # 👈 tick spacing every 20 epochs
    ax_acc.set_ylim(32, 59)                   # 👈 Y-axis tuned to your accuracy range
    # ax.set_yticks(range(60, 73, 2))       # 👈 tick spacing every 2%
    ax_acc.set_yticks([35, 40, 45, 50, 55, 55])  # Y-ticks for clarity
    # ────────────────────────────────────────────────────────────────


    # Legend
    # leg = ax_acc.legend(fontsize=exp_args.legend_font, loc="lower right", frameon=False)
    # ────────────────────────────────────────────────────────────────
    # 🔧 Fine-tuning legend placement and style
    leg = ax_acc.legend(
        fontsize=exp_args.legend_font,
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.03),   # move slightly above the plot area
        ncol=1,
        frameon=False,
        facecolor="white",
        edgecolor="gray",
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.6,
    )
    leg._legend_box.align = "left"

    # ────────────────────────────────────────────────────────────────
    # Make legend text bold (LaTeX-safe)
    for text in leg.get_texts():
        label = text.get_text()
        text.set_text(r"\textbf{" + label.replace("\\", "\\\\") + "}")
    # ────────────────────────────────────────────────────────────────

    fig_acc.savefig(os.path.join(save_dir, f"TestAcc_vs_Epoch_{exp_args.dataset_name}_Activations.pdf"),
                    format="pdf", bbox_inches="tight", facecolor="white", dpi=600)
    fig_acc.savefig(os.path.join(save_dir, f"TestAcc_vs_Epoch_{exp_args.dataset_name}_Activations.svg"),
                    format="svg", bbox_inches="tight", facecolor="white")

    plt.show()

    # ===============================================================
    # 2️⃣ PLOT TEST LOSS vs EPOCH
    # ===============================================================
    fig_test_loss, ax_test_loss = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    ax_test_loss.plot(relu_epochs_test, relu_test_loss, label="ReLU", color=COLORS["ReLU"], linewidth=1.8)
    ax_test_loss.plot(gelu_epochs_test, gelu_test_loss, label="GELU", color=COLORS["GELU"], linewidth=1.8)
    ax_test_loss.plot(tanhexp_epochs_test, tanhexp_test_loss, label="TanhExp", color=COLORS["TanhExp"], linewidth=1.8)
    ax_test_loss.plot(geglu_epochs_test, geglu_test_loss, label="GEGLU", color=COLORS["GEGLU"], linewidth=1.8)
    ax_test_loss.plot(
        fftgate_epochs_test,
        fftgate_test_loss,
        label="FFTGate",
        color=COLORS["FFTGate"],
        linewidth=2.0,
        marker='o',
        markersize=4.5,
        markerfacecolor='white',
        markeredgecolor=COLORS["FFTGate"],
        markeredgewidth=1.3,
        markevery=5,
        zorder=10,
    )

    ax_test_loss.set_xlabel(r"\textbf{Epoch}")
    ax_test_loss.set_ylabel(r"\textbf{Test Loss}")
    ax_test_loss.grid(True, linestyle="--", alpha=0.35)



    # # ─────────────────────────────────────────────────────────────────────────────────────────────────
    # # ✅ Custom Axis Styling (Publication Style)
    # ax_test_loss.set_xlim(-4, 104)                  # 👈 add some padding before and after epochs
    # ax_test_loss.set_xticks(range(0, 105, 20))      # 👈 tick spacing every 20 epochs
    # ax_test_loss.set_ylim(2, 6.8)                # Y-axis range
    # ax_test_loss.set_yticks([2.0, 4.0, 5.0, 6.0, 6.5])  # Y-ticks for clarity   
    # # ─────────────────────────────────────────────────────────────────────────────────────────────────



    # leg2 = ax_test_loss.legend(fontsize=exp_args.legend_font, loc="upper right", frameon=False)

    # ────────────────────────────────────────────────────────────────
    # 🔧 Fine-tuning legend placement and style
    leg2 = ax_test_loss.legend(
        fontsize=exp_args.legend_font,
        loc="upper right",
        # bbox_to_anchor=(-0.02, 1.03),
        bbox_to_anchor=(1.02, 1.03),  # 👈 slight offset outside the plot
        ncol=1,
        frameon=False,
        facecolor="white",
        edgecolor="gray",
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.6,
    )
    # leg2._legend_box.align = "right"

    # ────────────────────────────────────────────────────────────────



    for text in leg2.get_texts():
        label = text.get_text()
        text.set_text(r"\textbf{" + label.replace("\\", "\\\\") + "}")

    fig_test_loss.savefig(os.path.join(save_dir, f"TestLoss_vs_Epoch_{exp_args.dataset_name}_Activations.pdf"),
                          format="pdf", bbox_inches="tight", facecolor="white", dpi=600)
    fig_test_loss.savefig(os.path.join(save_dir, f"TestLoss_vs_Epoch_{exp_args.dataset_name}_Activations.svg"),
                          format="svg", bbox_inches="tight", facecolor="white")

    plt.show()

    # ===============================================================
    # 3️⃣ PLOT TRAIN LOSS vs EPOCH
    # ===============================================================
    fig_train_loss, ax_train_loss = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    ax_train_loss.plot(relu_epochs_train, relu_train_loss, label="ReLU", color=COLORS["ReLU"], linewidth=1.8)
    ax_train_loss.plot(gelu_epochs_train, gelu_train_loss, label="GELU", color=COLORS["GELU"], linewidth=1.8)
    ax_train_loss.plot(tanhexp_epochs_train, tanhexp_train_loss, label="TanhExp", color=COLORS["TanhExp"], linewidth=1.8)
    ax_train_loss.plot(geglu_epochs_train, geglu_train_loss, label="GEGLU", color=COLORS["GEGLU"], linewidth=1.8)
    ax_train_loss.plot(
        fftgate_epochs_train,
        fftgate_train_loss,
        label="FFTGate",
        color=COLORS["FFTGate"],
        linewidth=2.0,
        marker='o',
        markersize=4.5,
        markerfacecolor='white',
        markeredgecolor=COLORS["FFTGate"],
        markeredgewidth=1.3,
        markevery=5,
        zorder=10,
    )

    ax_train_loss.set_xlabel(r"\textbf{Epoch}")
    ax_train_loss.set_ylabel(r"\textbf{Train Loss}")
    ax_train_loss.grid(True, linestyle="--", alpha=0.35)

    # leg3 = ax_train_loss.legend(fontsize=exp_args.legend_font, loc="upper right", frameon=False)

    # ────────────────────────────────────────────────────────────────
    # 🔧 Fine-tuning legend placement and style
    leg3 = ax_train_loss.legend(
        fontsize=exp_args.legend_font,
        loc="upper right",
        # bbox_to_anchor=(-0.02, 1.03),
        bbox_to_anchor=(1.02, 1.03),  # 👈 slight offset outside the plot
        ncol=1,
        frameon=False,
        facecolor="white",
        edgecolor="gray",
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.6,
    )
    # leg3._legend_box.align = "right"

    # ────────────────────────────────────────────────────────────────
    # Make legend text bold (LaTeX-safe)
    for text in leg3.get_texts():
        label = text.get_text()
        text.set_text(r"\textbf{" + label.replace("\\", "\\\\") + "}")
    # ────────────────────────────────────────────────────────────────

    fig_train_loss.savefig(os.path.join(save_dir, f"TrainLoss_vs_Epoch_{exp_args.dataset_name}_Activations.pdf"),
                           format="pdf", bbox_inches="tight", facecolor="white", dpi=600)
    fig_train_loss.savefig(os.path.join(save_dir, f"TrainLoss_vs_Epoch_{exp_args.dataset_name}_Activations.svg"),
                           format="svg", bbox_inches="tight", facecolor="white")

    plt.show()








# 🔷 Call the function
plot_imagenet_metrics()


# %%
