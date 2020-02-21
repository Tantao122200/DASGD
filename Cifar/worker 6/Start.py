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
    if int(a[0]) < 51 and int(a[1]) < 51:
        print(a)
        if a[2] == "start" and a[3] == "start":
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
                if a[2] == "end" or a[3] == "end":
                    time.sleep(60)
                    os.system("cd %s & python3 EndPsWorker.py" % ("{0}{1}".format(task_file, task_name)))
                    break
                else:
                    time.sleep(60)
    else:
        if local_ip == "10.10.2.23":
            with open("./tensorboard/count.txt", 'r') as f:
                r_info = f.read()
                a = r_info.split(" ")
            w_info = "1" + " " + a[1] + " " + "start" + " " + a[3]
            with open("./tensorboard/count.txt", 'w') as f:
                f.write(w_info)
        if local_ip == "10.10.2.24":
            with open("./tensorboard/count.txt", 'r') as f:
                r_info = f.read()
                a = r_info.split(" ")
            w_info = a[0] + " " + "1" + " " + a[2] + " " + "start"
            with open("./tensorboard/count.txt", 'w') as f:
                f.write(w_info)


if __name__ == "__main__":
    main()
