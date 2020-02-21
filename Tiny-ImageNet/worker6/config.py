# ps_hosts mean the host ip which we expect to set
local_ip="10.10.2.29"

ps_hosts = ["10.10.2.29:16980","10.10.2.30:16980"]

worker_hosts = ["10.10.2.29:17900","10.10.2.29:17901","10.10.2.29:17902",
                "10.10.2.30:17900","10.10.2.30:17901","10.10.2.30:17902"]

# the gpu_0.1 index we expect to use
worker_hosts_with_gpu_index = ["10.10.2.29:17900#0","10.10.2.29:17901#1","10.10.2.29:17902#2",
                               "10.10.2.30:17900#0", "10.10.2.30:17901#1", "10.10.2.30:17902#2"]

# the name of our project
task_name = "worker6"
# the path of our project in docker
docker_file = "/root/"
# the path of our project in server
task_file = "/root/tantao/Tiny_ImageNet/"
# the name of docker image
docker_image = "tantao"
