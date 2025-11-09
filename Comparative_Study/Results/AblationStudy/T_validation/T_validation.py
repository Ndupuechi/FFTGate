



# %% Imports and Setup


# ===============================================================
# 🔗==================== ABLATION STUDY 🔑====================🔗
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
Project_PATH = r"C:\Users\emeka\Research\ModelCUDA\Big_Data_Journal\Comparison\Code\Paper\github4\IWSLT\Comparative_Study\Results\AblationStudy\T_validation"
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
exp_parser = argparse.ArgumentParser("CIFAR-100 T-Effect Experiment Config")
exp_parser.add_argument('--dataset_name', default="VGG16_CIFAR100", type=str,
    help="Choose dataset: VGG16_CIFAR100")  
# exp_parser.add_argument('--translation_direction', default="EN⟶FR", type=str)
exp_parser.add_argument('--translation_direction', default=r"EN$\rightarrow$FR", type=str)

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
# 🔗========================================== AblationStudy 🔑=====================================================🔗


# ======================================================================================================
# ✅ =======================🔖 TEST ACCURACY SCORE 🔖==================================================
# ======================================================================================================


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅐 Define Test log file paths
T8_test_log     = fr'./1-T8/Test_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_EXP2_T8.txt'
T16_test_log     = fr'./2-T16/Test_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_EXP2_T16.txt'
T32_test_log  = fr'./3-T32/Test_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_EXP2_T32.txt'
T64_test_log    = fr'./4-T64/Test_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_EXP2_T64.txt'
T128_test_log  = fr'./5-T128/Test_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_EXP2_T128.txt'
T256_test_log  = fr'./6-T256/Test_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_EXP2_T256.txt'
T512_test_log  = fr'./7-T512/Test_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_EXP2_T512.txt'
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 🅑 Define Train log file paths
T8_Train_log     = fr'./1-T8/Train_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_EXP2_T8.txt'
T16_Train_log     = fr'./2-T16/Train_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_EXP2_T16.txt'
T32_Train_log  = fr'./3-T32/Train_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_EXP2_T32.txt'
T64_Train_log    = fr'./4-T64/Train_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_EXP2_T64.txt'
T128_Train_log  = fr'./5-T128/Train_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_EXP2_T128.txt'
T256_Train_log  = fr'./6-T256/Train_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_EXP2_T256.txt'
T512_Train_log  = fr'./7-T512/Train_FFTGate_{exp_args.dataset_name}_Adam_Seed1_2_EXP2_T512.txt'
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
# 🔗=================== READ TEXTS 🔑=========================🔗
# ===============================================================

# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 1️⃣ Read Test Metrics (Loss + Accuracy)
T8_epochs_test,   T8_test_loss,   T8_test_acc,   T8_best_acc   = read_test_metrics(T8_test_log)
T16_epochs_test,  T16_test_loss,  T16_test_acc,  T16_best_acc  = read_test_metrics(T16_test_log)
T32_epochs_test,  T32_test_loss,  T32_test_acc,  T32_best_acc  = read_test_metrics(T32_test_log)
T64_epochs_test,  T64_test_loss,  T64_test_acc,  T64_best_acc  = read_test_metrics(T64_test_log)
T128_epochs_test, T128_test_loss, T128_test_acc, T128_best_acc = read_test_metrics(T128_test_log)
T256_epochs_test, T256_test_loss, T256_test_acc, T256_best_acc = read_test_metrics(T256_test_log)
T512_epochs_test, T512_test_loss, T512_test_acc, T512_best_acc = read_test_metrics(T512_test_log)
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# 2️⃣ Read Train Metrics (Loss + Accuracy)
T8_epochs_train,   T8_train_loss,   T8_train_acc,   T8_best_train   = read_train_metrics(T8_Train_log)
T16_epochs_train,  T16_train_loss,  T16_train_acc,  T16_best_train  = read_train_metrics(T16_Train_log)
T32_epochs_train,  T32_train_loss,  T32_train_acc,  T32_best_train  = read_train_metrics(T32_Train_log)
T64_epochs_train,  T64_train_loss,  T64_train_acc,  T64_best_train  = read_train_metrics(T64_Train_log)
T128_epochs_train, T128_train_loss, T128_train_acc, T128_best_train = read_train_metrics(T128_Train_log)
T256_epochs_train, T256_train_loss, T256_train_acc, T256_best_train = read_train_metrics(T256_Train_log)
T512_epochs_train, T512_train_loss, T512_train_acc, T512_best_train = read_train_metrics(T512_Train_log)
# ─────────────────────────────────────────────────────────────────────────────────────────────────


# ✅ Optional — Print summary of best accuracies
print("\n📊 === Best Test Accuracies ===")
print(f"T=8: {T8_best_acc:.2f}% | T=16: {T16_best_acc:.2f}% | T=32: {T32_best_acc:.2f}% | "
      f"T=64: {T64_best_acc:.2f}% | T=128: {T128_best_acc:.2f}% | T=256: {T256_best_acc:.2f}% | T=512: {T512_best_acc:.2f}%")

print("\n📈 === Best Train Accuracies ===")
print(f"T=8: {T8_best_train:.2f}% | T=16: {T16_best_train:.2f}% | T=32: {T32_best_train:.2f}% | "
      f"T=64: {T64_best_train:.2f}% | T=128: {T128_best_train:.2f}% | T=256: {T256_best_train:.2f}% | T=512: {T512_best_train:.2f}%")








# ===============================================================
# 🔗==================== PLOT FUNCTION 🔑=====================🔗
# ===============================================================

def plot_fftgate_T_variants(save_dir=r'./Plots'):
    os.makedirs(save_dir, exist_ok=True)


    # 🎨 Consistent, high-contrast palette
    COLORS = {
        "T8":   "#8338EC",   # Violet
        "T16":  "#3A86FF",   # Bright blue
        "T32":  "#06D6A0",   # Teal-green
        "T64":  "#E49B0F",   # 🔸 Rich golden amber (improved visibility)
        "T128": "#EF476F",   # Pink-red
        "T256": "#118AB2",   # Deep cyan
        "T512": "#073B4C"    # Navy
    }


    # ===============================================================
    # 1️⃣ TEST ACCURACY vs EPOCH
    # ===============================================================
    fig_acc, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)


    # ────────────────────────────────────────────────────────────────
    # ✅ Plot Test Accuracy curves — explicit naming, T32 highlighted
    # ────────────────────────────────────────────────────────────────
    # 🔹 Baselines (normal solid lines)
    ax.plot(T8_epochs_test, T8_test_acc, label="T8",
            color=COLORS["T8"], linewidth=1.8, alpha=1.0)
    ax.plot(T16_epochs_test, T16_test_acc, label="T16",
            color=COLORS["T16"], linewidth=1.8, alpha=1.0)
    ax.plot(T32_epochs_test, T32_test_acc, label="T32",
            color=COLORS["T32"], linewidth=1.8, alpha=1.0)
    ax.plot(T128_epochs_test, T128_test_acc, label="T128",
            color=COLORS["T128"], linewidth=1.8, alpha=1.0)
    ax.plot(T256_epochs_test, T256_test_acc, label="T256",
            color=COLORS["T256"], linewidth=1.8, alpha=1.0)
    ax.plot(T512_epochs_test, T512_test_acc, label="T512",
            color=COLORS["T512"], linewidth=1.8, alpha=1.0)

    # 🔷 Highlighted curve — T64 (special style)
    ax.plot(
        T64_epochs_test,
        T64_test_acc,
        label="T64",
        color=COLORS["T64"],
        linewidth=2.0,
        alpha=1.0,
        marker='o',
        markersize=4.5,
        markerfacecolor='white',           # ✅ white center
        markeredgecolor=COLORS["T64"],     # outline matches color
        markeredgewidth=1.4,
        markevery=5,
        zorder=10                          # 🔹 bring on top visually
    )
    # ────────────────────────────────────────────────────────────────



    # ────────────────────────────────────────────────────────────────
    ax.set_xlabel(r"$\mathbf{Epoch}$")
    ax.set_ylabel(r"$\mathbf{Test\ Accuracy\ (\%)}$")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(-2, max(T8_epochs_test)+2)
    # ────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────
    # ✅ Custom Axis Styling (Publication Style)
    ax.set_xlim(-4, 104)                  # 👈 add some padding before and after epochs
    ax.set_xticks(range(0, 105, 20))      # 👈 tick spacing every 20 epochs
    ax.set_ylim(56, 70)                   # 👈 Y-axis tuned to your accuracy range
    # ax.set_yticks(range(60, 73, 2))       # 👈 tick spacing every 2%
    ax.set_yticks([60, 65, 70])  # Y-ticks for clarity
    # ────────────────────────────────────────────────────────────────

    # ────────────────────────────────────────────────────────────────


    # ────────────────────────────────────────────────────────────────
    # 🔧 Fine-tuning options
    leg = ax.legend(
        fontsize=exp_args.legend_font,
        loc="upper left",
        # bbox_to_anchor=(0.5, 1.16),  # 🔹 try 1.14 (closer) or 1.20 (higher)
        bbox_to_anchor=(-0.02, 1.03),   # 👈 y=1.05 moves legend slightly above default
        ncol=1,
        frameon=False,         # 🚫 removes the box frame
        facecolor="white",
        edgecolor="gray",
        columnspacing=0.8,
        handlelength=1.0,
        handletextpad=0.6,
    )
    leg._legend_box.align = "left"
    # ────────────────────────────────────────────────────────────────
    # ✅ Make legend text bold (LaTeX text mode, not math)
    for text in leg.get_texts():
        label = text.get_text()
        text.set_text(r"\textbf{" + label.replace("\\", "\\\\") + "}")

    # ✅ Draw canvas for legend coordinates (needed if you add a title later)
    fig_acc.canvas.draw()
   # ────────────────────────────────────────────────────────────────

   # ────────────────────────────────────────────────────────────────
    fig_acc.savefig(
        os.path.join(save_dir, f"T_EffectAblation_TestAccuracy_vs_Epoch_{exp_args.dataset_name}_FFTGate.pdf"),
        format="pdf", bbox_inches="tight", facecolor="white", dpi=600
    )
    
    plt.show()
    # ────────────────────────────────────────────────────────────────







    # ===============================================================
    # 2️⃣ TEST LOSS vs EPOCH
    # ===============================================================
    fig_loss, ax_loss = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    for T, epochs, losses in [
        ("T8",   T8_epochs_test,   T8_test_loss),
        ("T16",  T16_epochs_test,  T16_test_loss),
        ("T32",  T32_epochs_test,  T32_test_loss),
        ("T64",  T64_epochs_test,  T64_test_loss),
        ("T128", T128_epochs_test, T128_test_loss),
        ("T256", T256_epochs_test, T256_test_loss),
        ("T512", T512_epochs_test, T512_test_loss),
    ]:
        ax_loss.plot(epochs, losses, label=T,
                     color=COLORS[T], linewidth=2.0, alpha=1.0,
                     marker='o', markersize=4.2, markerfacecolor='white',
                     markeredgecolor=COLORS[T], markeredgewidth=1.3, markevery=5)

    ax_loss.set_xlabel(r"$\mathbf{Epoch}$")
    ax_loss.set_ylabel(r"$\mathbf{Test\ Loss}$")
    ax_loss.grid(True, linestyle="--", alpha=0.35)
    ax_loss.set_xlim(-2, max(T8_epochs_test)+2)
    ax_loss.legend(fontsize=exp_args.legend_font, loc="upper right",
                   frameon=True, facecolor="white", edgecolor="gray")
    fig_loss.savefig(
        os.path.join(save_dir, f"T_EffectAblation_TestLoss_vs_Epoch_{exp_args.dataset_name}_FFTGate.pdf"),
        format="pdf", bbox_inches="tight", facecolor="white", dpi=600
    )
    plt.show()

    # ===============================================================
    # 3️⃣ TRAIN LOSS vs EPOCH
    # ===============================================================
    fig_train_loss, ax_train = plt.subplots(figsize=(5, 3.5), constrained_layout=True)

    for T, epochs, losses in [
        ("T8",   T8_epochs_train,   T8_train_loss),
        ("T16",  T16_epochs_train,  T16_train_loss),
        ("T32",  T32_epochs_train,  T32_train_loss),
        ("T64",  T64_epochs_train,  T64_train_loss),
        ("T128", T128_epochs_train, T128_train_loss),
        ("T256", T256_epochs_train, T256_train_loss),
        ("T512", T512_epochs_train, T512_train_loss),
    ]:
        ax_train.plot(epochs, losses, label=T,
                      color=COLORS[T], linewidth=2.0, alpha=1.0,
                      marker='o', markersize=4.2, markerfacecolor='white',
                      markeredgecolor=COLORS[T], markeredgewidth=1.3, markevery=5)

    ax_train.set_xlabel(r"$\mathbf{Epoch}$")
    ax_train.set_ylabel(r"$\mathbf{Train\ Loss}$")
    ax_train.grid(True, linestyle="--", alpha=0.35)
    ax_train.set_xlim(-2, max(T8_epochs_train)+2)
    ax_train.legend(fontsize=exp_args.legend_font, loc="upper right",
                    frameon=True, facecolor="white", edgecolor="gray")
    fig_train_loss.savefig(
        os.path.join(save_dir, f"T_EffectAblation_TrainLoss_vs_Epoch_{exp_args.dataset_name}_FFTGate.pdf"),
        format="pdf", bbox_inches="tight", facecolor="white", dpi=600
    )
    plt.show()

    print(f"✅ Saved all FFTGate T-variant plots in: {save_dir}")




# 🔷 Call the function
plot_fftgate_T_variants()

# %%
