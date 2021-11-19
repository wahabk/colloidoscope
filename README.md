# Colloidoscope!

A project to attempt tracking colloids using confocal and deep learning.

Please make sure you use python 3.8

$ conda install -c conda-forge hoomd -y


docker build . --tag=colloids 
docker build . --tag=colloids --network=host # if you're on vpn
and then run 

docker run -it \
	-v /home/ak18001/Data/HDD:/home/ak18001/Data/HDD \
	-v /home/ak18001/code/deepcolloid:/deepcolloid \
	--gpus all \
	--network=host \
	colloids \