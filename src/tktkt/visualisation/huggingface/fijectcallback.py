from transformers.trainer_callback import TrainingArguments, TrainerState, TrainerControl, TrainerCallback
from fiject import LineGraph, CacheMode
import logging as logger
from typing import List


class FijectCallback(TrainerCallback):
    """
    Callback object passed to a Trainer that will add an evaluation point to a Fiject graph, and will commit
    the result every-so-often.

    You can either choose the evaluation metrics yourself, or let it default to the metric used to rank models.
    """

    def __init__(self, plot_name: str, metrics: List[str]=None, evals_between_commits: int=-1):  # Metrics default to the eval loss.
        self.graph = LineGraph(plot_name, CacheMode.WRITE_ONLY)
        self.evals_per_commit = evals_between_commits
        self.evals_so_far = 0

        self.metric_names = None if not metrics else [name[name.startswith("eval_")*len("eval_"):] for name in metrics]

    def _automatic_metric_name(self, args: TrainingArguments):
        """
        Based on the early-stopping callback, which also accesses the evaluation loss
        https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/trainer_callback.py#L586
        """
        ranking_metric = args.metric_for_best_model
        self.metric_names = [ranking_metric[ranking_metric.startswith("eval_")*len("eval_"):]]

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict, **kwargs):
        if not self.metric_names:
            self._automatic_metric_name(args)

        for name in self.metric_names:
            key = "eval_" + name
            if key not in metrics:
                logger.warning(f"What the fuck, you asked for metric '{key}' but you're not computing it, you fucking weirdo.")
                continue

            self.graph.add(name.replace("_", "-"), state.global_step, metrics.get(key))

        self.evals_so_far += 1  # By incrementing first, you never commit after just one iteration.
        if self.evals_per_commit > 0 and self.evals_so_far % self.evals_per_commit == 0:
            self._commit()

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self._commit()

    def _commit(self):
        self.graph.commit(legend_position="upper right", x_label="Training batches", y_label="Validation set performance",
                          do_points=False, grid_linewidth=0.1)


class EvaluateBeforeTrainingCallback(TrainerCallback):
    """
    Triggers evaluation before the first training batch, so that you can benchmark all metrics before any finetuning
    has been done (and then print it or let it be caught by another callback like the above for plotting).
    https://discuss.huggingface.co/t/how-to-evaluate-before-first-training-step/18838/7

    Hoping this doesn't slow down the training too much, since you are interrupting every batch with this.
    """
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True
