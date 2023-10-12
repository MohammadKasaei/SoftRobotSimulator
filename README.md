# SoftRobotSimulator
# A Data-efficient Neural ODE Framework for Optimal Control of Soft Manipulators
This paper introduces a novel approach for modeling continuous forward kinematic models of soft continuum robots by employing Augmented Neural ODE (ANODE), a cutting-edge family of deep neural network models. To the best of our knowledge, this is the first application of ANODE in modeling soft continuum robots. This formulation introduces auxiliary dimensions, allowing the system's states to evolve in the augmented space which provides a richer set of dynamics that the model can learn, increasing the flexibility and accuracy of the model. Our methodology achieves exceptional sample efficiency, training the continuous forward kinematic model using only 25 scattered data points.

[Paper](https://openreview.net/pdf?id=PalhNjBJqv)

[Video](https://youtu.be/6tYS-5tkoQg)
# Installation and Setup

## Clone the Repository:

```
git clone https://github.com/MohammadKasaei/SoftRobotSimulator
cd SoftRobotSimulator
```
## Set Up a Virtual Environment (optional):

```
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```
## Install Dependencies:
Before running the script, make sure you have execute permissions. Run the following command:
```
chmod +x install_dependencies.sh
```
To install all the dependencies, simply run:
```
./install_dependencies.sh
```
Wait for the script to complete. Once done, all the required dependencies should be installed in your environment.


## Usage 
Instructions on how to run the code, experiments, and reproduce results.
```
python -m scripts.test_pybullet_MPPI
```
Once everything successfully installed, you'll see the simulated robot following a helical trajectory within the PyBullet simulator.

![alt](images/softRobot.gif)


and for the MLP forward kinematics model, you can run the following command:
```
python -m scripts.test_Full_FK

```
you will see our robot visualizer:
![alt](neuralODE/savedFigs/gif_NodeRedMPPI_20231007-104312.gif)
![alt](neuralODE/savedFigs/obstacle.gif)


# Citation
If you find our paper or this repository helpful, please cite our work:

```
@inproceedings{kasaei2023data,
  title={A Data-efficient Neural ODE Framework for Optimal Control of Soft Manipulators},
  author={Kasaei, Mohammadreza and Babarahmati, Keyhan Kouhkiloui and Li, Zhibin and Khadem, Mohsen},
  booktitle={7th Annual Conference on Robot Learning},
  year={2023}
}
```




# License
This project is licensed under the MIT License.

# Acknowledgments
This work is supported by EU H2020 project Enhancing Healthcare with Assistive Robotic Mobile Manipulation (HARMONY, 101017008) and the Medical Research Council [MR/T023252/1].


