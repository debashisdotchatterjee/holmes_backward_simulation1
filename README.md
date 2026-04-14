
# Holmes Backward Simulation Package

This package shows how to build a **Sherlock-Holmes-style detective story backward**.

The logic is:

1. Simulate a hidden case first.
2. Generate clues from the hidden case.
3. Recover the hidden culprit using a Bayesian Holmesian inference rule.
4. Produce tables, figures, and a short narrative description of a showcase case.
5. Save everything automatically and zip the outputs.

## Main idea

Instead of starting with a finished story and then explaining the clues, this code does the reverse:

- the **latent planner** of the case is simulated first,
- then the clue pattern is emitted,
- then the detective tries to reconstruct the planner from the observed evidence.

This lets you demonstrate, through simulation, how a Holmes-like detective narrative can emerge from an underlying probabilistic system.

## Files

- `holmes_backward_simulation.py` — the main Colab-friendly simulation script
- `requirements.txt` — minimal dependencies
- `README.md` — this file

## Hidden states

The code simulates:

- `true_planner` in  
  `{outsider_gambler, trainer, rival_trainer, unknown_intruder}`
- `true_motive`
- `true_concealer`

The **trainer** case is designed to mimic the internal structure of a Holmes-style “Silver Blaze” explanation:
dog silence, drugged stable boy, delicate instrument, hidden debt, and horse-kick mechanism.

## Observed clues

Binary clues include:

- stable boy drugged
- dog silent
- delicate knife found
- alias bill found
- horse at scene
- coat deliberately placed
- horse hidden by neighbor
- death consistent with horse kick
- outsider seen near stable
- tracks to rival property

Continuous clues include:

- dog familiarity score
- instrument delicacy score

## Outputs created automatically

When the script runs, it creates:

- simulated datasets (`csv`)
- posterior-enriched datasets (`csv`)
- parameter tables (`csv` and `md`)
- performance tables
- clue-importance tables
- a showcase posterior trajectory table
- six figures
- a showcase story text file
- a zip archive of the entire output folder

## How to run in Google Colab

Upload the script and run:

```python
!pip install -r requirements.txt
!python holmes_backward_simulation.py
```

Or, if the files are in the current working directory:

```python
from holmes_backward_simulation import run_pipeline

paths = run_pipeline(
    n_cases=1500,
    seed=123,
    showcase_case_id=7,
    output_root="holmes_simulation_outputs",
    show_plots=True,
    show_tables=True,
)

paths
```

## Important directories after running

Inside `holmes_simulation_outputs/` you will find:

- `data/`
- `tables/`
- `figures/`
- `stories/`
- `run_metadata.json`
- `holmes_simulation_outputs.zip`

## Suggested paper interpretation

You can describe the simulation as a proof-of-concept for the claim that a detective story can be generated from:

- a latent explanatory state,
- a clue emission mechanism,
- a posterior update rule,
- and a narrative reconstruction stage.

That is the main conceptual contribution of this simulation package.

## Notes

- The inference model uses a factorized likelihood for tractability.
- The simulator intentionally includes some mild dependence between clues, so the generated data are not completely trivial.
- The code is pedagogical, transparent, and designed for adaptation into a research-style paper.

