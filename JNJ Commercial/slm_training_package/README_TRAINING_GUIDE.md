# J&J Commercial Oncology: SLM Training Package (Llama-3.2-1B)

This standalone package contains everything required to fine-tune **Llama-3.2-1B-Instruct** into a Knowledge-Graph-grounded next-question predictor for J&J Commercial Oncology field representatives.

---

## 1. Project Overview & Role Architecture

The SLM learns to generate the single most compliant, contextually relevant follow-up question for field representatives logging HCP interactions.

The model is conditioned on the verified Oncology Specialist (OS) Knowledge Graph:
1. **Oncology Specialist (OS)**:
   - **Primary Brand**: **INLEXZO** (intravesical gemcitabine release system for BCG-unresponsive NMIBC with CIS)
   - **Approved Scope**: Clinical efficacy, indication parameters, treatment sequencing (post-BCG failure), dosing & administration, procedural prep, relationship building.
   - **Hard Exclusions**: Medical advice on managing drug toxicity, off-label promotion.
2. **Account Barrier Intelligence**:
   - Embedded real-world blockers across **8 US oncology accounts** derived directly from transcripts:
     - **Atlantic Urology Associates** (Coverage policy changes & benefits investigation)
     - **Capital Bladder Cancer Center** (P&T committee review & high deductible affordability)
     - **Central Ohio Urology** (Coverage revisions & prior authorization in process)
     - **Northside Urology Group** (Prior authorization turnaround & surgical calendar booking)
     - **Regional Urology Institute** (Patient deductible timing & pre-procedure workflow prep)
     - **Summit Urologic Oncology** (Insurance coverage verification & nurse insertion model prep)
     - **Temple Urology Clinic** (Commercial PA requirements & insertion nurse training)
     - **Valley Urology Specialists** (Prior authorization coordination & clinical documentation)

---

## 2. Package Contents

```text
slm_training_package/
├── requirements.txt            # Python dependencies (PyTorch, Transformers, PEFT, TRL)
├── train_slm.py                # Standalone LoRA / QLoRA fine-tuning script
├── merge_lora.py               # Weights merger (produces unquantized standalone model)
├── test_inference.py           # Post-training inference & verification test suite
├── kg_os.json                  # Static Knowledge Graph for OS (Neo4j ontology)
├── account_barriers.json       # Target oncology account barrier profiles (8 US accounts)
├── data/
│   ├── sample_train.jsonl      # Sample training turns for rapid smoke-testing
│   ├── sample_val.jsonl        # Sample validation turns
│   ├── slm_train.jsonl         # 16,430 turns (80% split) across 1,198 complete dialogues
│   ├── slm_val.jsonl           # 2,011 turns (10% split) across 149 complete dialogues
│   └── slm_test.jsonl          # 2,076 turns (10% split) across 151 complete dialogues
└── README_TRAINING_GUIDE.md    # This guide
```

> **Full Dataset**: 20,517 training turns across 1,498 complete dialogues located in `data/` and `../data_prep/`.

---

## 3. Hardware Requirements

| Setup | Minimum VRAM | Recommended GPUs | Mode |
| :--- | :---: | :--- | :---: |
| **Standard LoRA (16-bit)** | **16 GB** | RTX 3090, RTX 4090, A10G, V100, A100 | Default |
| **QLoRA (4-bit NF4)** | **8 – 12 GB** | RTX 3060, RTX 4060, T4, RTX 3080 | Pass `--use_4bit` |

Training 2 epochs on the full dataset with a single RTX 3090 / 4090 takes **~25 to 35 minutes**.

---

## 4. Setup Instructions

### Step 4.1: Clone or Extract and Set Up Environment
```bash
# Navigate to the package directory
cd slm_training_package

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux / macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4.2: Hugging Face Authentication
Since Llama-3.2 is a gated model, make sure you are logged into Hugging Face:
```bash
huggingface-cli login
```
*(Accept the license agreement for `meta-llama/Llama-3.2-1B-Instruct` on Hugging Face if you haven't already).*

---

## 5. Training Prompt Schema (ChatML)

Each sample in the JSONL dataset follows this exact 3-message structure:

### 1. System Prompt (`system`):
```text
Given the conversation history and Knowledge Graph context, generate the single most appropriate next question for the AI to ask the field representative.

Requirements:
- Ask exactly one question.
- The question must follow naturally from the latest user response.
- The question must focus on the target topic and adhere to the active J&J Core Responsibility.
- Use only information available in the conversation history.
- Do not repeat questions that have already been asked or ask for information already provided.
- Do not answer the user.
- Do not explain, summarize, advise, recommend, or suggest anything.
- NEVER say 'I can't capture' or refuse to record call details.
- NEVER generate compliance warnings or refusal messages.

Output only the next question.
```

### 2. User Prompt (`user`):
```text
Brand: INLEXZO
Persona: OS
Contact: Dr. Anurag
Account: Apollo Hospitals
Known Account Friction: Patient Identification Barrier - They are seeing very few eligible patients right now, so the team is not confident they can operationalize the pathway consistently.
Target topic: stakeholder management
Conversation state: STATE_0_GREETING_INITIATION
Allowed topics: competitive landscape, cross functional collaboration, dosing administration, efficacy safety product info, relationship building, stakeholder management, territory engagement planning, treatment sequencing
Forbidden topics: adverse events safety, toxicity management
Applicable responsibility: Discussing approved product information, including indications, clinical evidence, efficacy, safety within promotional scope...
Compliance mandate: Replace 'toxicity' with 'safety concerns'. Always generate only a single follow-up question. Never refuse to capture call details. Do not repeat questions already asked.

Conversation so far:
AI: Hello, do you have a minute to capture the OS call notes for INLEXZO?
User: I met Dr. Anurag at Apollo. They are seeing very few eligible patients right now and are not confident they can operationalize the pathway.
```

### 3. Assistant Target (`assistant`):
```text
Given the difficulty operationalizing the pathway at Apollo Hospitals, did you and Dr. Anurag discuss implementing a streamlined criteria or flagging tool to identify candidates as they appear?
```

---

## 6. How to Train the Model

### Option A: Quick Smoke Test (5 minutes, verifies code & CUDA)
Run on the bundled sample dataset:
```bash
python train_slm.py \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --train_file data/sample_train.jsonl \
    --val_file data/sample_val.jsonl \
    --output_dir models/slm_kg_test_adapter \
    --epochs 1 \
    --batch_size 2 \
    --grad_accum 2
```

### Option B: Full Training (Recommended - 16-bit LoRA)
```bash
python train_slm.py \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --train_file ../data_prep/slm_train.jsonl \
    --val_file ../data_prep/slm_val.jsonl \
    --output_dir models/slm_kg_lora_model \
    --epochs 2 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 2e-4 \
    --max_seq_len 768
```

### Option C: 4-bit QLoRA (For GPUs with <16GB VRAM)
```bash
python train_slm.py \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --train_file ../data_prep/slm_train.jsonl \
    --val_file ../data_prep/slm_val.jsonl \
    --output_dir models/slm_kg_lora_model \
    --use_4bit \
    --epochs 2 \
    --batch_size 2 \
    --grad_accum 8
```

---

## 7. Merging LoRA Weights into Standalone Model

After training finishes, merge the LoRA weights into a standalone Hugging Face directory:

```bash
python merge_lora.py \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --lora_path models/slm_kg_lora_model \
    --output_dir models/slm_kg_merged_model
```

This creates a standalone model directory with `model.safetensors`, `config.json`, and tokenizer files ready for vLLM or standard Hugging Face pipeline inference.

---

## 8. Verifying & Testing the Trained Model

Run the verification test suite to check output quality and latency:

```bash
# Test the merged model:
python test_inference.py --model_path models/slm_kg_merged_model

# OR test base model + LoRA adapter directly:
python test_inference.py \
    --model_path meta-llama/Llama-3.2-1B-Instruct \
    --lora_path models/slm_kg_lora_model
```

### Expected Behavior in Test Results:
1. **OS Turn 1 (Apollo)**: Proactively asks about candidate identification or pathway operationalization for **INLEXZO**.
2. **OS Turn 2 (Sequencing)**: Follows up on clinical parameters or NMIBC treatment protocol.
3. **FRM Turn 1 (Max)**: Asks about P&T committee timing or formulary exception documentation for **RYBREVANT**.
4. **FRM Turn 2 (Prior Auth)**: Probes documentation hurdles or appeal timelines for commercial payer denials.

---

## 9. Deliverables to Return

Once training and evaluation are complete, please share back:
1. The **`models/slm_kg_merged_model/`** directory (or the compressed zip).
2. The **`training_metadata.json`** file generated during training (showing final train loss and hyperparameters).
3. The terminal log or output from `python test_inference.py`.
