import os, time
from config import ps_hosts, task_name, task_file, docker_image, docker_file, worker_hosts_with_gpu_index, local_ip


def start_ps():
    for ps_index in range(len(ps_hosts)):
        ps = ps_hosts[ps_index]
        ip, port = ps[:ps.find(":")], ps[ps.find(":") + 1:]
        if ip == local_ip:
            try:
                os.popen(
                    "nvidia-docker run -itd --privileged=true -v %s/tensorboard:%s/tensorboard --net=host --name %s_ps_%s --ulimit core=0 %s bash"
                    % (
                        "{0}{1}".format(task_file, task_name), "{0}{1}".format(docker_file, task_name), task_name,
                        ps_index,
                        docker_image)).read()
                os.popen("nvidia-docker cp %s %s_ps_%s:%s"
                         % ("{0}{1}".format(task_file, task_name), task_name, ps_index, docker_file)).read()
                os.popen(
                    "nvidia-docker exec -itd %s_ps_%s bash -c 'cd %s && python3 Train.py --job_name=ps --task_index=%s --cuda=-1'"
                    % (task_name, ps_index, "{0}{1}".format(docker_file, task_name), ps_index)).read()
            except Exception as e:
                print("there are some exceptions in start_ps as follow: ")
                print(e)


def start_worker():
    for worker_index in range(len(worker_hosts_with_gpu_index)):
        worker = worker_hosts_with_gpu_index[worker_index]
        ip, port, gpu_index = worker[:worker.find(':')], worker[worker.find(':') + 1:worker.find('#')], worker[
                                                                                                        worker.find(
                                                                                                            '#') + 1:]
        if ip == local_ip:
            try:
                os.popen(
                    "nvidia-docker run -itd --privileged=true -v %s/tensorboard:%s/tensorboard --net=host --name %s_worker_%s --ulimit core=0 %s bash"
                    % ("{0}{1}".format(task_file, task_name), "{0}{1}".format(docker_file, task_name),
                       task_name, worker_index, docker_image)).read()
                os.popen("nvidia-docker cp %s %s_worker_%s:%s"
                         % ("{0}{1}".format(task_file, task_name), task_name, worker_index, docker_file)).read()
                os.popen(
                    "nvidia-docker exec -itd %s_worker_%s bash -c 'cd %s && python3 Train.py --job_name=worker --task_index=%s --cuda=%s > ./tensorboard/out.log 2>&1'"
                    % (
                        task_name, worker_index, "{0}{1}".format(docker_file, task_name), worker_index,
                        gpu_index)).read()
            except Exception as e:
                print("there are some exceptions in start_worker as follow: ")
                print(e)


def main():
    with open("./tensorboard/count.txt", 'r') as f:
        r_info = f.read()
        a = r_info.split(" ")
    if int(a[0]) < 51 and int(a[1]) < 51 and int(a[2]) < 51 and int(a[3]) < 51:
        print(a)
        if a[4] == "start" and a[5] == "start" and a[6] == "start" and a[7] == "start":
            print("the local_ip is : " + local_ip)
            print(
                "we are trying our best to open ps or worker with docker and test the model, so please just waiting ......")
            start_ps()
            start_worker()
            print("docker start success, please see the result in file : %s/tensorboard" % (
                "{0}{1}".format(task_file, task_name)))
        while (True):
            with open("./tensorboard/count.txt", 'r') as f:
                r_info = f.read()
                a = r_info.split(" ")
                if a[4] == "end" or a[5] == "end" or a[6] == "end" or a[7] == "end":
                    os.system("cd %s & python3 EndPsWorker.py" % ("{0}{1}".format(task_file, task_name)))
                    break
                else:
                    time.sleep(30)
    else:
        if local_ip == "10.10.2.23":
            w_info = "1"
            with open("./tensorboard/count_1.txt", 'w') as f:
                f.write(w_info)

        if local_ip == "10.10.2.24":
            w_info = "1"
            with open("./tensorboard/count_2.txt", 'w') as f:
                f.write(w_info)

        if local_ip == "10.10.2.26":
            w_info = "1"
            with open("./tensorboard/count_3.txt", 'w') as f:
                f.write(w_info)

        if local_ip == "10.10.2.29":
            w_info = "1"
            with open("./tensorboard/count_4.txt", 'w') as f:
                f.write(w_info)

        if local_ip == "10.10.2.23":
            w_info = "1" + " " + "1" + " " + "1" + " " + "1" + " " + "start" + " " + "start" + " " + "start" + " " + "start"
            with open("./tensorboard/count.txt", 'w') as f:
                f.write(w_info)


if __name__ == "__main__":
    main()
