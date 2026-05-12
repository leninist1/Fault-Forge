# Fault-Forge

Fault-Forge is a compact release of the FaultForge implementation used for the
e123 LLM-FSE experiment on the Train-Ticket system.

This repository keeps only the runnable core and the semantic inputs required by
the experiment:

- `faultforge_nier/`: core FaultForge code used by the e123 LLM-FSE experiment.
- `business_model/`: business-semantics input files for Train-Ticket.
- `system_description/`: Train-Ticket system description files used by FSE.
- `configs/business_fault_families.yml`: business fault family catalog.
- `configs/telemetry_prism.yml`: telemetry-first PRISM configuration.

The FSE path is LLM-guided and schema-grounded. The LLM proposes weakness
intents from allowlisted system facts, and Fault-Forge grounds those intents into
executable fault candidates before injection and evaluation.

## LLM-FSE Entry Points

- `faultforge_nier/llm_fse_adapter.py` builds the LLM-visible fact pack and
  compiles LLM weakness intents into executable fault specs.
- `faultforge_nier/auto_loop_nier.py` runs the feedback loop and invokes
  LLM-FSE when `--llm-fse` is enabled.
- `faultforge_nier/production_runner.py` builds the production command for
  e123-style dataset generation.

Set `DEEPSEEK_API_KEY` or `OPENAI_API_KEY` before running LLM-backed workflows.
