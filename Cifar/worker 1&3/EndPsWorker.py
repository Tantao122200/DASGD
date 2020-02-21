import os, time
from config import task_name, task_file, worker_hosts, ps_hosts, local_ip


def remove_model():
    try:
        if local_ip == "10.10.2.29":
            a = os.system("cd %s & rm -rf tensorboard/out.log" % ("{0}{1}".format(task_file, task_name)))
        print("we have removed the out.log from %s/tensorboard" % ("{0}{1}".format(task_file, task_name)))
        print()
    except Exception as e:
        print("there are some exceptions in remove_log as follow: ")
        print(e)


def end_ps():
    try:
        for ps_index in range(len(ps_hosts)):
            ps = ps_hosts[ps_index]
            ip, port = ps[:ps.find(':')], ps[ps.find(':') + 1:]
            if ip == local_ip:
                for container_info in os.popen("nvidia-docker ps -a").readlines()[1:]:
                    temp = container_info.strip()
                    container_name = temp[temp.rfind(' ') + 1:]
                    if container_name.__contains__(task_name):
                        os.system("nvidia-docker rm -f %s" % container_name)
    except Exception as e:
        print("there are some exceptions in end_worker as follow: ")
        print(e)


def end_worker():
    try:
        for worker_index in range(len(worker_hosts)):
            worker = worker_hosts[worker_index]
            ip, port = worker[:worker.find(':')], worker[worker.find(':') + 1:]
            if ip == local_ip:
                for container_info in os.popen("nvidia-docker ps -a").readlines()[1:]:
                    temp = container_info.strip()
                    container_name = temp[temp.rfind(' ') + 1:]
                    if container_name.__contains__(task_name):
                        os.system("nvidia-docker rm -f %s" % container_name)
    except Exception as e:
        print("there are some exceptions in end_worker as follow: ")
        print(e)


def main():
    print("we are removing the container from the docker, just waiting ...")
    end_ps()
    end_worker()
    remove_model()

    if local_ip == "10.10.2.29":
        with open("./tensorboard/count.txt", 'r') as f:
            r_info = f.read()
            a = r_info.split(" ")
        w_info = str(int(a[0]) + 1) + " " + "start"
        with open("./tensorboard/count.txt", 'w') as f:
            f.write(w_info)

    while (True):
        with open("./tensorboard/count.txt", 'r') as f:
            r_info = f.read()
            a = r_info.split(" ")
        if a[1] == "start":
            time.sleep(60)
            os.system("cd %s & python3 Start.py" % ("{0}{1}".format(task_file, task_name)))
            break
        else:
            time.sleep(1)


if __name__ == "__main__":
    main()
