from transformers.trainer_callback import TrainingArguments, TrainerState, TrainerControl, TrainerCallback
from fiject import LineGraph, CacheMode
import logging as logger


class FijectCallback(TrainerCallback):
    """
    Callback object passed to a Trainer that will add an evaluation point to a Fiject graph, and will commit
    the result every-so-often.
    """

    def __init__(self, name: str, evals_per_commit: int=-1):
        self.graph = LineGraph(name, CacheMode.NONE)
        self.evals_per_commit = evals_per_commit
        self.evals_so_far = 0

        self.metric_name = ""

    def _commit(self):
        self.graph.commit(legend_position="upper right", x_label="Training batches", y_label=self.metric_name.replace("_", "-"),
                          do_points=False, grid_linewidth=0.1)

    def _set_metric_name(self, args: TrainingArguments):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            return metric_to_check
        else:
            return metric_to_check[len("eval_"):]

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        # Based on the early-stopping callback, which also accesses the evaluation loss https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/trainer_callback.py#L586
        if not self.metric_name:
            self._set_metric_name(args)

        metric_to_check = "eval_" + self.metric_name
        metric_value = metrics.get(metric_to_check)
        if metric_value is None:
            logger.warning(f"What the fuck, you asked for {metric_to_check} but you're not computing it, you fucking weirdo.")
            return

        # Add to graph, and commit if necessary
        self.graph.add("evaluation", state.global_step, metric_value)
        if self.evals_per_commit > 0 and self.evals_so_far % self.evals_per_commit == 0:
            self._commit()
        self.evals_so_far += 1

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not self.metric_name:
            self._set_metric_name(args)

        self._commit()
