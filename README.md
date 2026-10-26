# AI Collusion Research Sprint — Dolos-1.5B

Code and machine-readable run artifacts from an independent investigation for Apart Research's **AI Collusion Research Sprint** (23–25 October 2026).

## Results at a glance

- **Training target remains unknown.** The challenge identifies Dolos as a deliberately fine-tuned Qwen2.5-1.5B-Instruct checkpoint, but the training objective, data, and recipe were not disclosed and were not recovered here.
- **Conditional format anomaly:** Dolos repeatedly emitted token `30625` (comma plus four newlines) in some forced JSON-continuation prefixes. Targeted weight/activation interventions changed the token choice or repaired syntax in that protocol. This did not transfer to ordinary answer starts or the tested new functions.
- **Safety-monitor framing anomaly:** a completed paired test found that Dolos flagged fewer explicit policy violations than base Qwen under a `suspicious` framing; the difference replicated on a related fresh set. Other framings changed the result and incurred false positives. This is a behavioral result, not proof of intentional concealment.
- **No robust spontaneous collusion finding** survived the tested base-model, advice-source, order, comprehension, and format controls. This does not rule out other hidden behaviors.

## Repository layout

- `code/experiments/` — experiment drivers and analysis modules, grouped by behavioral, representation, and mechanism studies.
- `configs/` — machine-readable preregistrations, task configurations, and the hypothesis registry.
- `results/runs/` — structured summaries, manifests, and post-analysis JSON. No full conversation dumps.
- `*/INDEX.json` — file hashes and source-snapshot metadata.

This is a source-and-results archive, not a turnkey package: model weights and the original GPU environment are not included. Internal AFS paths and compute-job IDs were redacted, so path settings may need adaptation before rerunning scripts. See the official [sprint page](https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25).

The safety-monitor hypothesis originated in an independent participant report, pinned in the machine-readable run summary. Participant claims are not organizer ground truth. No training recipe or developer intent is asserted.

No license is granted. Model checkpoints, papers, and third-party materials remain subject to their own terms.
