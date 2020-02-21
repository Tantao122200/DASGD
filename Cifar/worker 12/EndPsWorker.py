import os, time
from config import task_name, task_file, worker_hosts, ps_hosts, local_ip


def remove_model():
    try:
        if local_ip == "10.10.2.23":
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
    with open("./tensorboard/count.txt", 'r') as f:
        r_info = f.read()
        a_first = r_info.split(" ")
    index = int(a_first[0])

    if local_ip == "10.10.2.23":
        w_info = str(index + 1)
        with open("./tensorboard/count_1.txt", 'w') as f:
            f.write(w_info)

    if local_ip == "10.10.2.24":
        w_info = str(index + 1)
        with open("./tensorboard/count_2.txt", 'w') as f:
            f.write(w_info)

    if local_ip == "10.10.2.26":
        w_info = str(index + 1)
        with open("./tensorboard/count_3.txt", 'w') as f:
            f.write(w_info)

    if local_ip == "10.10.2.29":
        w_info = str(index + 1)
        with open("./tensorboard/count_4.txt", 'w') as f:
            f.write(w_info)

    while (True):
        with open("./tensorboard/count_1.txt", 'r') as f:
            r_info = f.read()
            a_1 = r_info.split(" ")

        with open("./tensorboard/count_2.txt", 'r') as f:
            r_info = f.read()
            a_2 = r_info.split(" ")

        with open("./tensorboard/count_3.txt", 'r') as f:
            r_info = f.read()
            a_3 = r_info.split(" ")

        with open("./tensorboard/count_4.txt", 'r') as f:
            r_info = f.read()
            a_4 = r_info.split(" ")

        if int(a_1[0]) == index + 1 and int(a_2[0]) == index + 1 and int(a_3[0]) == index + 1 and int(a_4[0]) == index + 1:
            if local_ip == "10.10.2.23":
                w_info = str(index + 1) + " " + str(index + 1) + " " +str(index + 1) + " " +str(
                    index + 1) + " " + "start" + " " + "start" + " " + "start" + " " + "start"
                with open("./tensorboard/count.txt", 'w') as f:
                    f.write(w_info)
            with open("./tensorboard/count.txt", 'r') as f:
                r_info = f.read()
                a = r_info.split(" ")

            if len(a) < 7 or (a[4] == "start" and a[5] == "start" and a[6] == "start" and a[7] == "start"):
                time.sleep(60)
                os.system("cd %s & python3 Start.py" % ("{0}{1}".format(task_file, task_name)))
                break
        else:
            time.sleep(30)


if __name__ == "__main__":
    main()