# binarization_2017

This repo contains the code and models described in [Document Image Binarization with Fully Convolutional Neural Networks](https://arxiv.org/abs/1708.03276).  There are two sets of 5 models.  One trained on [DIBCO](https://vc.ee.duth.gr/dibco2017/) images, and the other trained on [Palm Leaf Manuscripts (PML)](http://amadi.univ-lr.fr/ICFHR2016_Contest/index.php/challenge-1).  Additional info on these models can be found [here](
https://ctensmeyer.github.io/publication/document_image_binarization_with_fully_convolutional_neural_networks/).  You may also be interested in my submission to the DIBCO 2017 competition, located [here](https://github.com/ctensmeyer/dibco_2017).

This code depends on a number of python libraries: numpy, scipy, cv2 (python wrapper for opencv), and caffe [(my custom fork)](https://github.com/ctensmeyer/caffe).

For those who don't want to install the dependencies, I have created a docker image to run this code. You must have the [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) plugin installed to use it though you can still run our models on CPU (not recommended).

The usage for the docker container is

```
nvidia-docker run -v $HOST_WORK_DIRECTORY:/data tensmeyerc/icdar2017:binarization_gpu python binarize_dibco.py /data/input_file.jpg /data/output_file.png $DEVICE_ID
nvidia-docker run -v $HOST_WORK_DIRECTORY:/data tensmeyerc/icdar2017:binarization_gpu python binarize_plm.py /data/input_file.jpg /data/output_file.png $DEVICE_ID
```

`$HOST_WORK_DIRECTORY` is a directory on your machine that is mounted on /data inside of the docker container (using -v).  It's the only way to expose images to the docker container.
`$DEVICE_ID` is the ID of the GPU you want to use (typically 0).  If omitted, then the models are run in CPU mode.
There is no need to download the containers ahead of time.  If you have docker and nvidia-docker installed, running the above commands will pull the docker image (~2GB) if it has not been previously pulled.


If you find this code useful to your research, please cite our paper:

```
@inproceedings{tensmeyer2017_binarization,
  title={Document Image Binarization with Fully Convolutional Neural Networks},
  author={Tensmeyer, Chris and Martinez, Tony},
  booktitle={ICDAR},
  year={2017},
  organization={IEEE}
}
```
