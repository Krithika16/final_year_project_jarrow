docker container rm augcon
docker build . -t tf_aug
./docker/start_augcon.sh
