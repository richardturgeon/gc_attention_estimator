import time
from datetime import datetime
import tensorflow as tf
import pdb
from tensorflow.python import ipu

class TrainingHook(tf.estimator.SessionRunHook):
    """Logs loss and runtime."""

    def __init__(self, params):
        super().__init__()
        self._tag = "***** Training Hook: "
        self.log_frequency = params['iterations_per_loop']
        self.batch_size = params['batch_size']
        self._outfeed = ipu.ipu_outfeed_queue.IPUOutfeedQueue(
            feed_name="logging",
            outfeed_mode=ipu.ipu_outfeed_queue.IPUOutfeedMode.ALL
        )

    def enqueue(self, tensor):
        print(self._tag, "enqueue()")
        return self._outfeed.enqueue(tensor)

    def begin(self):
        self._step = 1
        print(self._tag, "begin()")
        self.tensor = self._outfeed.dequeue()

    def before_run(self, run_context):
        self._step += 1
        self._start_time = time.time()
        print(self._tag, "before_run()")

    def after_run(self, run_context, run_values):
        print(self._tag, "after_run()")
        #pdb.set_trace()
        current_time = time.time()
        duration = current_time - self._start_time
        self._start_time = current_time
        loss_value = run_context.session.run(self.tensor)
        examples_per_sec = self.log_frequency * self.batch_size / duration
        sec_per_batch = float(duration / self.log_frequency)

        # average out "loss" and "accuracy" vectors
        loss_avg = sum(loss_value['loss']) / len(loss_value['loss'])
        accuracy_avg = sum(loss_value['accuracy']) / len(loss_value['accuracy'])

        format_str = (self._tag + '%s: step %d, loss=%f accuracy=%f (%.1f examples/sec; %.3f '
                          'sec/batch)')
        print (format_str % (datetime.now(), self._step, loss_avg, accuracy_avg,
            examples_per_sec, sec_per_batch))

