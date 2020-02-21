import tensorflow as tf
import os
from config import ps_hosts, worker_hosts
import time
import single
from model import vgg16
import threading
import numpy as np
import csv
from monitor.monitored_session import MonitoredTrainingSession

flags = tf.app.flags
flags.DEFINE_integer("task_index", -1, "the index of task")
flags.DEFINE_string("job_name", "", "ps or worker")
flags.DEFINE_string("cuda", "", "specify gpu_0.1")
FLAGS = flags.FLAGS
if FLAGS.cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda


def train():
    with open("tensorboard/count.txt", 'r') as f:
        r_info = f.read()
        a = r_info.split(" ")
        count = int(a[0])

    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        is_chief = (FLAGS.task_index == 0)
        ps_strategy = tf.contrib.training.GreedyLoadBalancingStrategy(len(ps_hosts),
                                                                      tf.contrib.training.byte_size_load_fn)
        with tf.device(tf.train.replica_device_setter(
                ps_device="/job:ps/task:%d" % FLAGS.task_index,
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster,
                ps_strategy=ps_strategy)):
            single.build_model()
            saver = tf.train.Saver(max_to_keep=5)
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)

        class MyCheckpointSaverHook(tf.train.CheckpointSaverHook):
            def after_run(self, run_context, run_values):
                step = run_context.session.run(self._global_step_tensor)
                if ((vgg16.TRAININR_STEP >= step) and (step > 0) and (((step - 0 + vgg16.COUNT) %
                                                                       vgg16.COUNT == 0) or (step == vgg16.TRAININR_STEP))):
                    self._timer.update_last_triggered_step(step)
                    self._save(run_context.session, step)

        # hooks = [MyCheckpointSaverHook(checkpoint_dir="tensorboard/ckpt", save_steps=vgg16.COUNT, saver=saver)]
        with MonitoredTrainingSession(master=server.target,
                                      is_chief=is_chief,
                                      # checkpoint_dir="tensorboard/ckpt",
                                      # hooks=hooks,
                                      # save_checkpoint_secs=None,
                                      config=sess_config)as mon_sess:
            step = 0
            local_step = 0
            start_time = time.time()
            while True:
                train_info = single.train_model(mon_sess, step)
                for info in train_info:
                    mytype = info.replace("is ", ":").split(",")[0].split(":")[1]
                    g_step = int(info.replace("is", ":").split(",")[1].split(":")[1])
                    step = g_step + 1
                    print(info+"\n")
                    if (g_step % vgg16.COUNT == 0 or g_step % vgg16.COUNT == 1 or g_step % vgg16.COUNT == 2) and g_step > 0:
                        t_train = threading.Thread(target=thread, args=(info, mytype, count, time.time() - start_time))
                        t_train.start()
                    if g_step >= vgg16.TRAININR_STEP + 1:  # or (np.isnan(float(info.replace("is", ":").split(",")[2].split(":")[1])) and g_step > 3000):
                        print("Training elapsed time : %.4f" % (time.time() - start_time))
                        w_info = a[0] + " " + a[1] + " " + "end" + " " + "end"
                        with open("tensorboard/count.txt", 'w') as f:
                            f.write(w_info)
                        exit(1)
                local_step += 1


def thread(info, type, count, mytime):
    out = open("./tensorboard/mycsv/" + type + "_ASGD_" + str(vgg16.values[0]) + "_" + str(
        vgg16.TRAININR_STEP) + "_" + str(vgg16.BATCH_SIZE) + "_" + str(len(worker_hosts)) + "_" + str(count) + ".csv",
               'a', newline='')
    temp = [int(info.replace("is", ":").split(",")[1].split(":")[1]),
            float(info.replace("is", ":").split(",")[2].split(":")[1]),
            float(info.replace("is", ":").split(",")[3].split(":")[1]),
            float(info.replace("is", ":").split(",")[4].split(":")[1]),
            float(info.replace("is", ":").split(",")[5].split(":")[1]),
            float(info.replace("is", ":").split(",")[6].split(":")[1]),
            float(info.replace("is", ":").split(",")[7].split(":")[1]),
            mytime]
    csv_write = csv.writer(out, dialect='excel')
    csv_write.writerow(temp)


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
