# Team Meeting Plan (Feb 6, 2026)
Planning for team meeting to discuss project roadmap, work distribution, and milestones for the next few weeks.

## Meeting Agenda (30-45 min)

1) Baseline status recap (current accuracy + settings)
2) Agree on next improvement targets and metrics
3) Planned enhancements (keyframe selection, temporal modeling, multitask)
4) Feature branches and owners
5) Timeline and checkpoints

## Work Distribution

Assign one task per developer so work can proceed in parallel. Each task should land on its own feature branch.

- Developer 1: Data improvements
  - Implement augmentations in dataset transforms: blur, color jitter, random erase
  - Run 1 short training job per augmentation set (e.g., 5 epochs)
  - Compare uniform vs random sampling with identical settings
  - Deliver: table with configs, train/test accuracy, and stability notes

- Developer 2: Baseline tuning
  - Run a 3x2x2 sweep: backbone {resnet34,resnet50,efficientnet_b0},
    aggregation {mean,max}, max-frames {5,10}
  - Use fixed seed and same data split for comparability
  - Deliver: best config + training logs + recommendation summary

- Developer 3: Evaluation + metrics
  - Standardize evaluation with scripts/evaluate_model.py for all checkpoints
  - Add Top-5 accuracy and per-class accuracy calculation (if feasible)
  - Deliver: results table + reproducible commands + eval script changes (if any)

- Developer 4: Keyframe selection module
  - Implement sharpness (Laplacian variance), scale score, and orientation heuristic
  - Add top-K frame selector with CLI flag to enable/disable
  - Deliver: ablation run vs uniform sampling + keyframe selector code

- Developer 5: Temporal modeling
  - Add a lightweight temporal head (LSTM/GRU or attention) after frame features
  - Keep parameter count reasonable (< 2x baseline)
  - Compare against mean/max aggregation on same settings
  - Deliver: performance delta + training notes + model diagram snippet

- Developer 6: Proposal/report updates
  - Update Proposed Enhancements, Experimental Plan, Results sections
  - Add baseline config, metrics table, and improvement summaries
  - Deliver: updated proposal draft sections + references list update

## Feature Branches

- feature/data-pipeline
- feature/baseline-training
- feature/keyframe-selection
- feature/temporal-model
- feature/eval-reporting

## Milestones

### Week 1
- Baseline tuning sweep complete
- First improvement (keyframe or temporal) implemented
- Proposal updated with enhancement plan

### Week 2
- Keyframe module integrated + evaluated
- Temporal model prototype evaluated

### Week 3
- Finalize best model
- Run ablations + performance table
- Draft final report and slides

## Deliverables Per Checkpoint

- Commands used + config summary
- Accuracy/metrics on train/test
- Model weights and logs in outputs/
- Short summary of each experiment
