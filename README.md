# Agreement of LLM Judges in Persona Contradiction Detection

Statistical project investigating agreement between large language models used
as judges of persona consistency in dialogue.

The experiment is based on PersonaChat and uses matched pairs consisting of:

- an original persona and dialogue,
- a contradiction-oriented augmented persona with the same dialogue.

The main experimental dataset contains 1,000 matched pairs (2,000 evaluation
examples).

## Files

- `llm_judge_agreement.ipynb` — dataset preparation and statistical analysis
- `data/llm_judge_experiment_v2.parquet` — frozen experimental sample
- `data/personachat_augmented_v2.parquet` — derived augmented source dataset
- `scripts/run_llm_judge.py` — local LLM inference
- `outputs/judgments/` — LLM judge predictions

## Data source

PersonaChat:

Zhang et al. (2018), *Personalizing Dialogue Agents: I have a dog, do you have
pets too?*

https://aclanthology.org/P18-1205/