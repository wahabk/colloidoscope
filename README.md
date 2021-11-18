# Deep Colloid

A project to attempt tracking colloids using confocal and deep learning.

$ conda install -c conda-forge hoomd

$ conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

docker build . --tag=colloids 
docker build . --tag=colloids --network=host # if you're on vpn
and then run 

docker run -it \
	-v /home/ak18001/Data/HDD:/home/ak18001/Data/HDD \
	-v /home/ak18001/code/deepcolloid:/deepcolloid \
	--gpus all \
	colloids \