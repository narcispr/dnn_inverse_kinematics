# dnn_inverse_kinematics
This project presents a Python interactive notebook that shows how to compute the inverse kinematics of a manipulator using an iterative algorithm and a Deep Neural Network. 
To test it, follow the [ik_dnn_project.ipynb](ik_dnn_project.ipynb) notebook.

## Jetson Nano
The code can be run on a Jetson Nano. I've tested it using *Jet Pack 4.4* and the docker image *dli-nano-ai:v2.0.0-r32.4.3*. The only package not installed in this image that I needed was `matplotlib` which can be installed manually:

```bash
pip install matplotlib
```
