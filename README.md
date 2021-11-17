# Deep Colloid

A project to attempt tracking colloids using confocal and deep learning.

$ conda install -c conda-forge hoomd

$ conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

run docker build .
and then run 

docker run -it \
	-v /home/ak18001/Data/HDD:/home/ak18001/Data/HDD \
	-v /home/ak18001/code/deepcolloid:/ctfishpy \
	--gpus all \
	test \