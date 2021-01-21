## automatic_driving
carla_cil pytorch var.

The pytorch implementation to train the uncertain aware imitation learning policy in "conditional imitation learning based AI that runs on CARLA". 

## Current progress
Attempting to train network through self-collected and self-preprocessed dataset. I pray to lord

## System
python 3.6    
pytorch > 0.4.0    
tensorboardX    
opencv    
imagaug    
h5py    

please refer ***docker/docker_build/Dockerfile*** for details.

## Train
**train-dir** and **eval-dir** should point to where the [Carla dataset](https://github.com/carla-simulator/imitation-learning/blob/master/README.md) located.
Please check our [paper](https://arxiv.org/abs/1903.00821) that how we split the train and eval dataset.
```

## Docker
Revise the path of the dataset and this repo in ***docker/carla_cil_compose/docker-compose.yml***.    
docker-compose 2.3 and nvidia-docker 2 are required.

```
$ cd docker/carla_cil_compose
$ docker-compose up -d
```
We can still use tensorboard to check the log out of the docker.

## Reference
[carla-simulator/imitation-learning](https://github.com/carla-simulator/imitation-learning)    
[mvpcom/carlaILTrainer](https://github.com/mvpcom/carlaILTrainer)    
[End-to-end Driving via Conditional Imitation Learning](https://arxiv.org/abs/1710.02410)    
[CARLA: An Open Urban Driving Simulator](http://proceedings.mlr.press/v78/dosovitskiy17a/dosovitskiy17a.pdf)    
[VR-Goggles for Robots: Real-to-sim Domain Adaptation for Visual Control](https://ram-lab.com/file/tailei/vr_goggles/index.html)    
[Visual-based Autonomous Driving Deployment from a Stochastic and Uncertainty-aware Perspective](https://arxiv.org/abs/1903.00821)

The code for original "End-to-end Driving via Conditional Imitation Learning" and "CARLA: An Open Urban Driving Simulator" is in the [master branch](https://github.com/onlytailei/carla_cil_pytorch/tree/master). In the paper VR-Goggles, we also used the original setup to train the policy.

