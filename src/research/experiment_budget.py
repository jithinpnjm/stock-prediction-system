from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentBudget:
    max_trials: int = 1000
    experiment_family: str = "default"

    def validate_trial(self, trial_number: int) -> None:
        if trial_number < 0 or trial_number >= self.max_trials:
            raise ValueError(
                f"trial {trial_number} exceeds budget {self.max_trials} "
                f"for family {self.experiment_family}"
            )
