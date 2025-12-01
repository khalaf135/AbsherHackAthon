# Adversarial SMS Detection System

An experimental adversarial framework for generating and detecting malicious / spam SMS messages. The project pairs:

- A **text generator** based on HuggingFace LLMs (optionally with LoRA/PEFT adapters), and  
- A **discriminator** powered by TF‑IDF features and a lightweight PyTorch policy network trained with reinforcement learning (RL).

The two components are trained in an adversarial loop: the generator crafts challenging SMS samples, and the discriminator continuously updates itself via online learning to improve detection.

---

## Architecture Overview

**Figure 1. Adversarial SMS detection pipeline**

```text
          ┌──────────────────────────────────────────┐
          │              Generator (G)               │
          │              generator.py                │
          │  - HuggingFace LLM (+ optional LoRA)    │
          └───────────────┬─────────────────────────┘
                          │ generated SMS
                          ▼
          ┌──────────────────────────────────────────┐
          │            Discriminator (D)             │
          │             decrmenator.py               │
          │  - TF-IDF features                       │
          │  - PyTorch policy network (RL)           │
          └───────────────┬─────────────────────────┘
                          │ reward / feedback
                          ▼
          ┌──────────────────────────────────────────┐
          │         Adversarial Trainer              │
          │                 train.py                 │
          │  - Online RL updates                     │
          │  - Checkpoint saving                     │
          └───────────────┬─────────────────────────┘
                          │ trained models
                          ▼
          ┌──────────────────────────────────────────┐
          │                  CLI                     │
          │                 main.py                  │
          │  - Classify SMS                          │
          │  - Generate adversarial examples         │
          └──────────────────────────────────────────┘
```

If you prefer a graphical figure, you can create an image (e.g. `docs/architecture.png`) and reference it as:

```markdown
![Figure 1: Adversarial SMS detection architecture](docs/architecture.png)
```

---

## Methodology

The system follows an adversarial learning setup with two main components that are trained (or updated) in opposition: a **generator** that produces candidate SMS messages and a **discriminator** that learns to classify them.

### 1. Generator (G) — `generator.py`

- Parameterized by a **HuggingFace language model**, optionally extended with **LoRA/PEFT** adapters for efficient fine‑tuning.
- Given a **prompt** (e.g., “bank notification”, “payment reminder”) or a class label, it generates realistic SMS text.
- Objective: produce messages that are:
  - Realistic and coherent.
  - Difficult for the discriminator to classify correctly.

Typical behavior:

1. Load base LLM (`hf_model_name`).
2. Optionally load LoRA/PEFT adapters (`lora_path`).
3. Generate `N` candidate SMS messages with configured decoding parameters (temperature, top‑p, etc.).

### 2. Discriminator (D) — `decrmenator.py`

- Converts raw SMS text into **TF‑IDF** vectors (bag-of-words or n‑gram features).
- Passes these vectors through a **small PyTorch policy network** to produce:
  - A score or probability (e.g., spam vs. ham, adversarial vs. benign).
  - An action/policy distribution if framed as an RL problem.
- Objective: correctly classify generated and real SMS messages, maximizing an appropriate reward signal.

The RL / online learning aspect typically includes:

- Reward based on correctness or confidence (e.g., positive reward for correct classification).
- Policy gradient or similar RL update to adapt the discriminator as it sees new adversarial samples.

### 3. Adversarial Training Loop — `train.py`

At a high level, each training iteration proceeds as follows:

1. **Message Generation (G)**
   - The generator produces a batch of candidate SMS messages:
     - Purely synthetic adversarial examples, or
     - Adversarial variants of seed messages (depending on implementation).

2. **Discriminator Evaluation (D)**
   - Each message is transformed into a TF‑IDF vector.
   - The policy network outputs classification decisions (e.g., adversarial vs. benign).

3. **Reward Computation**
   - A reward is computed for each decision, for example:
     - +1 for correct classification, 0 or negative for incorrect.
     - Or a margin-based reward (e.g., higher reward for confident correct decisions).
   - The reward captures how well D is defending against G’s current attack distribution.

4. **Online / RL Update**
   - The discriminator parameters are updated using:
     - Policy gradient (or related RL) based on the reward.
     - Optionally a supervised loss if you have labeled real SMS data (hybrid training).
   - This allows D to adapt as G explores new adversarial regions.

5. **Iteration, Logging, Checkpointing**
   - The process repeats for many episodes/steps.
   - Metrics (accuracy, loss, reward) are logged.
   - Updated checkpoints for G and D are saved to disk for later use by the CLI.

### 4. Inference & Evaluation — `main.py`

After or during training:

- The **CLI** loads the latest discriminator (and optionally generator) checkpoints.
- Users can:
  - **Classify** any given SMS text.
  - **Generate** new adversarial examples for testing detection pipelines.
- This makes it easy to integrate the system into experiments, demos, or downstream evaluation scripts.

---

## Features

- **LLM-based SMS generator**
  - Uses HuggingFace models for realistic SMS text.
  - Optional **LoRA/PEFT** adapters for efficient fine‑tuning.

- **RL-based discriminator**
  - **TF‑IDF** feature extraction for SMS messages.
  - Compact **PyTorch policy network** for classification.
  - Reinforcement learning updates (online / adversarial training).

- **Adversarial training loop**
  - Generator produces adversarial SMS candidates.
  - Discriminator updates on-the-fly in response.

- **Command-line interface (CLI)**
  - Classify arbitrary SMS messages as benign/adversarial.
  - Generate new adversarial examples for testing your filters.

- **Configurable**
  - Basic knobs for model choice, thresholds, training hyperparameters, and data paths.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/khalaf135/AbsherHackAthon.git
cd AbsherHackAthon
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

If a `requirements.txt` file exists:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If there is no requirements file yet, you will need at least:

```bash
pip install torch scikit-learn transformers peft
```

Add any other project-specific libraries if required (e.g. `pandas`, `numpy`, `tqdm`, etc.).

---

## CLI Usage (`main.py`)

The CLI provides two primary capabilities:

1. **Classify**: Determine whether an SMS is adversarial/spam-like or benign.  
2. **Generate**: Produce adversarial SMS messages using the generator.

### Basic help

```bash
python main.py --help
```

You should see subcommands or flags for classification and generation (e.g., `classify`, `generate`), depending on how `main.py` is implemented.

### Example: Classify an SMS message

```bash
python main.py classify --text "Your bank account has been locked. Click this link to verify."
```

Possible options (depending on implementation):

- `--model-path` – path to a saved discriminator model.
- `--threshold` – decision boundary for adversarial detection.
- `--device` – `cpu` or `cuda`.

### Example: Generate adversarial SMS messages

```bash
python main.py generate \
  --prompt "Payment reminder" \
  --num-samples 10 \
  --max-length 64
```

Possible options:

- `--hf-model-name` – HuggingFace model name (e.g., `gpt2`, `distilgpt2`).
- `--lora-path` – path to a LoRA/PEFT adapter checkpoint.
- `--temperature`, `--top-p`, `--top-k` – sampling controls.
- `--device` – `cpu` or `cuda`.

Adjust the actual flags to match the current `argparse` configuration in `main.py`.

---

## Running Adversarial Training (`train.py`)

`train.py` orchestrates interaction between the generator and discriminator.

### Basic training run

```bash
python train.py
```

Example with options (adapt to your implementation):

```bash
python train.py \
  --hf-model-name gpt2 \
  --episodes 1000 \
  --batch-size 32 \
  --learning-rate 3e-4 \
  --save-dir ./checkpoints \
  --device cuda
```

Typical responsibilities of `train.py`:

- Instantiate the **generator** (`generator.py`) and **discriminator** (`decrmenator.py`).
- Run a loop where:
  - Generator produces SMS candidates.
  - Discriminator evaluates them and yields rewards/gradients.
  - Discriminator (and optionally generator) parameters are updated.
- Periodically **log metrics** and **save model checkpoints**.

Check the script’s help for the full set of options:

```bash
python train.py --help
```

---

## Components Overview

### `generator.py`

- Wraps a HuggingFace text generation model.
- Optionally loads **LoRA/PEFT** adapters if configured.
- Likely exposes functions like:
  - `load_generator(...)`
  - `generate_sms(prompt, num_samples, max_length, **gen_kwargs)`

Key configuration ideas:

- HuggingFace model name (e.g. `--hf-model-name`).
- Use/disable LoRA (`--use-lora`, `--lora-path`).
- Decoding strategy (temperature/top‑p/top‑k/num_beams).

### `decrmenator.py`

- Builds a **TF‑IDF** vectorizer for SMS text.
- Defines a **PyTorch policy network** over TF‑IDF features.
- Implements **RL-style updates** / online learning:
  - Reward might be based on how “hard” or “spammy” the message is.
  - Policy gradients or other RL signals adjust the classifier.

Potential configuration points:

- TF‑IDF n‑gram range, vocabulary size, min_df/max_df.
- Network size (hidden units, layers, activation).
- Learning rate, exploration parameters, and reward shaping.

### `train.py`

- Main adversarial training loop.
- Handles:
  - Dataset loading (if you provide seed clean/spam examples).
  - Logging/metrics.
  - Model saving/loading.

### `main.py`

- CLI front-end to:
  - Load trained discriminator/generator.
  - Run one-off classification.
  - Generate new adversarial examples for evaluation.

---

## Basic Configuration

Configuration can be handled via:

- **Command-line arguments** to `main.py`, `train.py`, etc.
- A simple **config file** (e.g. `config.json`, `config.yaml`) if implemented.

Common configuration knobs:

- **Model paths**
  - `--hf-model-name`: HuggingFace base model.
  - `--lora-path`: LoRA/PEFT adapter checkpoint.
  - `--disc-checkpoint`: discriminator checkpoint.

- **Training hyperparameters**
  - `--episodes`, `--batch-size`, `--learning-rate`, `--gamma` (if using discounted RL).
  - `--max-steps-per-episode`, `--eval-interval`.

- **Data & I/O**
  - `--train-data`, `--val-data`: SMS corpora file paths.
  - `--save-dir`, `--log-dir`: checkpoints and logs.

- **Classification options**
  - `--threshold`: adjust sensitivity of adversarial detection.
  - `--max-length`: maximum token length for messages.

Refer to `argparse` definitions in each script for the exact options currently supported.

---

## Notes & Tips

- **GPU support**  
  If available, enable CUDA in your PyTorch and transformer models for faster training:
  - Run with `CUDA_VISIBLE_DEVICES=0` or similar.
  - Ensure tensors and models are moved to `cuda()` where appropriate.

- **Model choice**  
  Start with smaller models (e.g., `distilgpt2`) for fast experiments, then switch to larger LLMs for more realistic SMS text.

- **LoRA/PEFT**  
  LoRA adapters can significantly reduce memory requirements:
  - Good for quick domain adaptation on SMS-style text.
  - Helps to generate more targeted adversarial samples.

- **Stability of RL training**  
  Adversarial / RL setups can be unstable:
  - Start with conservative learning rates.
  - Monitor reward trends and loss curves.
  - Consider early stopping or checkpoint rollback if the discriminator collapses.

- **Security / Ethics**  
  This project is intended **for research and defensive purposes only**—to test and harden SMS-spam detection systems.  
  Do not deploy or use generated messages for malicious or unethical activity.

---

## License

Add your preferred license here (e.g., MIT, Apache 2.0) and include a `LICENSE` file in the repository.
