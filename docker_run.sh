docker run -it --gpus=all -u $(id -u):$(id -g) --ipc=host -e HOST_HOSTNAME=$HOSTNAME --name augcon -v /home/joe/github/final_year_project_jarrow/:/home/joe/github/final_year_project_jarrow/ tf_aug
