# Solving Inverse Dynamics of 3-UPU Parallel Kinematic Mechanism with Physics-Guided Neural Network

This repository provides code for the PGNN paper. If you're using the code in your research, please cite the paper https://hdl.handle.net/11296/mwzt6f . 

## Abstract

This study addresses the challenges of analytically deriving the dynamic model for parallel kinematic manipulators due to high nonlinearity and strong joint coupling. It employs a novel machine learning model, called Physics-Guided Neural Network, which utilizes the simplicity of the analytically derived kinematic model and the fast computational characteristics of neural networks. The model is trained using data-driven neural networks and incorporates physics-based loss functions to guide the inverse dynamic model training process. The study demonstrates the superior performance of the kinematics model guided neural network in terms of convergence, interpolation capability, extrapolation capability, generalization capability, and robustness compared to traditional data-driven neural networks.

![Untitled1](https://github.com/PhoniExZoe/PGNN/assets/24270422/a1d2832b-b5ed-4ac1-b434-5594055da494)

## Getting Started

### Dependencies: 
* Python 3.9.13
* Pytorch 1.12.0
* scikit-learn 1.1.1

### Datasets:

The repository contains code and datasets needed for training and testing the PGNN model described in the paper, including interpolation, extrapolation, and generalization capability.

### Training Data 

All of the training paths in this repository are periodic paths, which can be defined by the Finite Fourier series. 

$$ 
p(t)= \sum_{k=1}^na_k sin⁡(kw_f t)+b_k cos⁡(kw_f t)
$$

$$ 
v(t)= \sum_{k=1}^n a_k kw_f cos⁡(kw_f t)- b_k kw_f sin⁡(kw_f t) 
$$

$$ 
a(t)= \sum_{k=1}^n -a_k (kw_f)^2  sin⁡(kw_f t)-b_k (kw_f)^2 cos⁡(kw_f t)
$$

The equation above represents the formula for the periodic function. The coefficients $a_k$ and $b_k$ are randomly determined, and $w_f$ is the fundamental frequency that controls the planning of position $p(t)$, velocity $v(t)$, and acceleration $a(t)$ in the workspace of the end effector. The value of $n$ determines the complexity of the periodic function, and for this paper, $n$ is fixed at 5.

Additionally, the function values in the $x$, $y$, and $z$ directions of this periodic function are all different. This means that the overall path resembles selecting any point in the workspace and moving in a periodic motion within that space, as shown in the diagram below. Furthermore, the positions, velocities, and accelerations in the training data are all kept within the limits of the machine.

![Untitled](https://github.com/PhoniExZoe/PGNN/assets/24270422/9a1515a1-8037-4c17-bb27-69f51e54379d)


Different paths with varying maximum speeds and accelerations are generated based on the period of the periodic function $w_f$. Therefore, the training dataset can be divided into randomly generated paths of low, medium, and high speeds. In the table below, it can be observed that, with a fixed sampling frequency of 1000Hz, the different periods result in varying amounts of data for each velocity within a single path. Thus, by adjusting the proportion of path numbers, an equal number of samples (100,000) is obtained for low, medium, and high-speed paths. In summary, through the Holdout method, all the data is split at an 80-20 ratio for training and validation data, with quantities of 240,000 and 60,000, respectively.

![image](https://github.com/PhoniExZoe/PGNN/assets/24270422/3545d01a-4c8c-4ae7-a875-be947160f595)


### Testing Data

To reasonably test the model, the numbers of the dataset are 60,000, which is the same as the validation data. The naming rules for testing data files are as follows.

```
TestingData_ATraj_B  
```
* A represents trajectory type
    * Periodic trajectory
    * Quintic polynomial trajectory
* B represents additional information
    * Low speed
    * Middle speed
    * High speed
    * Null - includes all of the above
    * Extra - vel, accel may over the machine limitation.
    * Single - randomly pick one trajectory to display how the model fits 


Multiple sets of quintic polynomial trajectories are served to assess the general capabilities of the model. As depicted in the diagram below, a quintic polynomial trajectory involves selecting two points in space and describing the relationship between them using a quintic polynomial.  

![image](https://github.com/PhoniExZoe/PGNN/assets/24270422/7938f986-26ce-49ce-9dd0-6720b52d3719)


By taking the derivative of the quintic polynomial with respect to time, we can obtain the velocity equation represented by a quartic polynomial and the acceleration equation represented by a cubic polynomial. From the equations below, it is apparent that to fully determine the parameters $a_0$ to $a_5$ of these equations, six condition equations are required. These conditions include the initial position $p_0$ , initial velocity $v_0$ , initial acceleration $a_0$ , final position $p_f$ , final velocity $v_f$ , and final acceleration $a_f$ . Therefore, in this paper, these six values will be randomly generated within the parameter range of the workspace and training data.

$$
p(t)=a_0+a_1 t+a_2 t^2+a_3 t^3+a_4 t^4+a_5 t^5
$$

$$
v(t)=a_1+2a_2 t+3a_3 t^2+4a_4 t^3+5a_5 t^4
$$

$$
a(t)=2a_2+6a_3 t+12a_4 t^2+20a_5 t^3
$$

In summary, the range and the sampling number of quintic polynomial trajectory data are described below.

![image](https://github.com/PhoniExZoe/PGNN/assets/24270422/b192a80a-294d-4520-9dad-bdacae5be2cf)


### Loss function:

PGNN stands for Physics-Guided Neural Network, which adds a compensation function to the loss function of a neural network during training. Before the training process, a custom class *Hook* is defined to capture the output values of neurons in the hidden layers. Specifically, the function *registerforwardhook* is utilized to gain the parameters of the model during the forward pass. By specifying the desired neural network layer, one can access the output of that layer as well as the input from the previous layer. On the other hand, *registerbackwardhook* is employed to obtain the gradient values of each layer during the backward pass, which is crucial for backpropagation and parameter updates.

```python
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()
```


## Using the code

### Training:
If you want to retrain the model just execute the file Model_training.ipynb .  
* Change training data
Find the "Load Data" section that Load data and change it into other data. 
```python
tr_path = os.path.abspath("./Resources/TrainingData_PeriodicTraj.xlsx")
tr_dataset = TrajDataset(tr_path)
```
* Change the model name and file directory
Find the "Hyper-parameters Setup" section and change the value of 'save_path' in the variable config.
```python
config = {
    'n_epochs': 15,                    # maximum number of epochs
    'batch_size': 16,                   # mini-batch size for dataloader
    'optimizer': 'Adam',                # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                   # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.0005433177089293607                     # learning rate of Adam
    },
    'layers': np.array([9, 189, 119, 9, 53, 85, 3], dtype='int64'),    # layers of NN architecture
    'early_stop': 200,                  # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models/model.pt'      # your model will be saved here
}
```

### Testing:
* Change the testing and model to use
Find the variable ts_path and test_config in the "Testing with trained NN" section.
```python
ts_path = os.path.abspath("./Resources/TestingData_PeriodicTraj_Single.xlsx")
ts_dataset = TrajDataset(ts_path)

test_config = {'save_path' : 'models/model_DDNN.pt'}
```



## Built With

* [VSCode](https://code.visualstudio.com/) - Code IDE
* [MATLAB/Simulink Simscape](https://www.mathworks.com/products/simscape.html) - Rigid body dynamics simulation
* [ADAMS](https://hexagon.com/products/product-groups/computer-aided-engineering-software/adams) - Simulation data verification

## Authors

* WeiChu (Wilson) Chen

## License

* This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* The novel NN structure, PGNN, first appeared in [Physics-guided Neural Networks (PGNN): An Application In Lake Temperature Modelling](https://arxiv.org/abs/1710.11431).
* [Microsoft NNI](https://nni.readthedocs.io/en/stable/) is used for Hyperparameter Tuning. 
* I'm grateful for Professor Sung's continuous care and support throughout my entire master journey.
