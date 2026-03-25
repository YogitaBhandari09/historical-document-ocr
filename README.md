# Historical Document OCR Baseline

A proposal-oriented baseline for historical document OCR using a CRNN backbone, CTC loss, decoding heuristics, and optional LLM-based post-processing.

## Quick Summary
An end-to-end OCR pipeline for historical Spanish documents using CRNN + CTC + beam search + Gemini LLM correction.

## Project Vision
The long-term goal of this project is to build a robust OCR pipeline for historical documents that can serve as a foundation for research, archival digitization, and downstream text analysis. Historical material is especially difficult to recognize automatically because it combines noisy scans, degraded typography, irregular layouts, and limited aligned ground-truth transcriptions.

This repository currently provides a serious engineering baseline that can evolve into a stronger GSoC-scale project. The emphasis is on reproducibility, honest evaluation, and a clean path from exploratory notebook work to a more research-grade OCR system.

## Problem Statement
Historical OCR differs from modern OCR in several important ways:
- page quality is often degraded by blur, stains, fading, or scanning artifacts
- fonts and spelling conventions vary across periods and collections
- documents are not always segmented into clean line images
- true transcription-aligned labels are expensive and difficult to obtain

Because of these constraints, an effective project in this space must balance immediate engineering progress with realistic claims about model quality.

## Current Objective
The current repository focuses on establishing a reproducible baseline that demonstrates:
- stable dataset ingestion and preprocessing
- CRNN-based sequence modeling with CTC loss
- train/validation splitting for controlled experiments
- greedy and beam decoding
- CER-based evaluation
- optional lexicon and Gemini-based post-processing
- checkpoint saving for best and latest model states

## Why This Is GSoC-Relevant
This project is well aligned with the kind of work expected in a strong GSoC proposal because it combines:
- practical engineering cleanup
- machine learning experimentation
- measurable milestones
- clear future extensibility
- real-world impact on difficult archival and document-understanding tasks

It also has a natural roadmap from baseline implementation to more ambitious improvements such as better supervision, stronger architectures, and more reliable benchmarking.

## Present Scope and Honest Limitation
At present, the notebook uses filename stems as weak labels for pipeline validation. This is useful for verifying that the data path, model, training loop, and decoding stages are working correctly, but it is not equivalent to using true transcription-aligned OCR supervision.

That means the repository should currently be understood as:
- a functioning OCR engineering scaffold
- a baseline training and decoding pipeline
- a project-in-progress toward stronger research quality

It should not yet be described as a solved historical OCR system.

## Technical Approach
### 1. Data Pipeline
- ingest document images from `data/raw/dataset`
- filter unusable samples and excessively long weak labels
- preprocess grayscale page crops into normalized fixed-size tensors

### 2. OCR Backbone
- CRNN architecture with convolutional encoder and BiLSTM sequence model
- CTC loss for alignment-free sequence learning
- dropout and batch normalization for more stable training

### 3. Training Strategy
- train/validation split for reproducible experiments
- `AdamW` optimizer
- gradient clipping
- validation-aware learning-rate scheduling
- checkpoint saving for best and latest model weights

### 4. Decoding and Post-processing
- greedy decoding for simple baseline inference
- beam decoding with fallback safeguards when outputs become unstable
- optional lexicon correction
- optional Gemini post-processing for readability refinement

### 5. Evaluation
- Character Error Rate (CER)
- qualitative prediction review
- comparison between greedy and beam decoding

## Repository Structure
```text
historical-document-ocr/
|-- data/
|   `-- raw/dataset/
|-- notebooks/
|   `-- dataset_exploration.ipynb
|-- src/
|   |-- data/dataset.py
|   |-- models/crnn.py
|   |-- training/train.py
|   `-- utils/metrics.py
|-- checkpoints/
|-- .env
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Training
You can run experiments either from the notebook or the standalone training script.

Notebook:
- `notebooks/dataset_exploration.ipynb`

Script:
```bash
python -m src.training.train --data-dir data/raw/dataset --epochs 10 --batch-size 16
```

Training artifacts saved automatically:
- `checkpoints/last_crnn.pt`
- `checkpoints/best_crnn.pt`

## Experiments
Baseline notebook experiment configuration:
- Model: CRNN with convolutional encoder and BiLSTM sequence model
- Objective: CTC loss
- Split: train/validation split from the weakly labeled dataset
- Decoding: greedy, beam, lexicon, optional Gemini correction

## Training Results
Baseline run captured in the notebook:

| Epoch | Train Loss | Validation Loss |
|------:|-----------:|----------------:|
| 1 | 2.7103 | 2.5548 |
| 2 | 2.5420 | 2.5194 |
| 3 | 2.4897 | 2.4772 |
| 4 | 2.4589 | 2.4633 |
| 5 | 2.4331 | 2.4137 |
| 6 | 2.4090 | 2.4201 |
| 7 | 2.3838 | 2.3817 |
| 8 | 2.3622 | 2.3698 |
| 9 | 2.3447 | 2.3468 |
| 10 | 2.3233 | 2.3682 |

Best validation loss: `2.3468`

## Key Results
- Training Loss decreased from **2.7103 → 2.3233** over 10 epochs
- Best Validation Loss achieved: **2.3468**
- Character Error Rate (CER) ranges between **0.70 – 0.90** across samples
- Beam search improves decoding consistency compared to greedy decoding
- Lexicon constraint produces valid words (e.g., "rey")
- LLM post-processing improves readability but does not replace OCR learning

**Important Note:**
These results are based on weak labels (filename-based supervision), so they validate pipeline correctness rather than true OCR accuracy.

### Observations
- The model successfully learns sequence patterns despite weak supervision
- High CER is expected due to lack of ground-truth aligned labels
- Decoding strategies (beam + lexicon) significantly improve output structure
- LLM acts as a refinement layer rather than a primary recognition system

## Prediction Examples
Representative outputs from the current weak-label baseline:

| Method | Output |
|--------|--------|
| Greedy | 20 |
| Beam Search | 207 |
| Beam + Lexicon | rey |
| Beam + LLM | 207 |

These examples are baseline diagnostics from the current validation batch, not final historical OCR transcriptions.

## Metrics
Observed sample-level CER values from the current notebook run:

| Sample | Greedy CER | Beam CER |
|------:|-----------:|---------:|
| 1 | 0.875 | 0.875 |
| 2 | 0.895 | 0.842 |
| 3 | 0.800 | 0.700 |
| 4 | 0.900 | 0.900 |
| 5 | 0.800 | 0.800 |

These metrics confirm that the pipeline is functional, but they should not be presented as benchmark OCR performance while weak labels are still being used.

## Proposed GSoC-Style Roadmap
### Phase 1: Baseline Stabilization
- finalize reproducible training and evaluation code
- improve documentation and experiment hygiene
- save checkpoints and summarize metrics consistently

### Phase 2: Data Quality Upgrade
- collect or align a true transcription subset
- move from weak labels to verified OCR supervision
- introduce stronger validation reporting

### Phase 3: Recognition Quality Upgrade
- move from page-level inputs toward line-level segmentation
- compare CRNN against stronger OCR baselines
- study decoding quality under improved supervision

### Phase 4: Research and Usability Improvements
- improve checkpoint management and experiment tracking
- package inference utilities for easier reuse
- document limitations, risks, and reproducible findings clearly

## Risks and Mitigations
### Risk: Weak labels cap achievable OCR quality
Mitigation: treat current results as baseline diagnostics and prioritize a verified transcription subset.

### Risk: Page-level OCR is harder than line-level OCR
Mitigation: add segmentation as a future milestone rather than overclaiming page-level recognition quality.

### Risk: LLM post-processing can hide OCR weaknesses
Mitigation: keep OCR backbone evaluation separate and use LLM correction only as optional post-processing.

## Optional Gemini Setup
Gemini support is optional and should be configured through an environment variable:

```powershell
$env:GEMINI_API_KEY="your-key-here"
```

or in `.env`:

```env
GEMINI_API_KEY=your-key-here
```

## Final Pipeline
1. Dataset ingestion from `data/raw/dataset`
2. Preprocessing with grayscale conversion, top-crop, resize, and normalization
3. CRNN training with CTC loss and train/validation split
4. Greedy and beam decoding with fallback safeguards
5. CER-based evaluation on the validation loader
6. Optional lexicon and Gemini post-processing
7. Checkpoint saving for best and latest model states

## Expected Next Milestone
The most important next milestone is to replace weak labels with verified transcriptions and re-evaluate the current architecture honestly. That change will do more for project quality than adding more heuristic post-processing on top of weak supervision.

## Project Status
This repository is now in a strong baseline state for continued development. It reads more like a serious proposal-backed engineering project, but its strongest future gains will come from better supervision, better benchmarking, and a clearer research protocol.
