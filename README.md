# Combinatorial Optimization enriched Machine Learning to solve the Dynamic Vehicle Routing Problem with Time Windows

This repository comprises the code to learn a dispatching and routing policy for the Vehicle Routing Problem with Time Windows using a structured learning enriched combinatorial optimization pipeline.
The problem setting bases on the [EURO Meets NeurIPS 2022 Vehicle Routing Competition](https://euro-neurips-vrp-2022.challenges.ortec.com/)

This method is proposed in:
> Léo Baty, Kai Jungel, Patrick Klein, Axel Parmentier, and Maximilian Schiffer. Combinatorial Optimization enriched Machine Learning to solve the Dynamic Vehicle Routing Problem with Time Windows. arXiv preprint: [arxiv:2304.00789](https://arxiv.org/abs/2304.00789), 2023.


This repository contains all relevant scripts and data sets to reproduce the results from the paper. To run the code for reproducing the results we assume using *slurm*.


The structure of this repository is as follows:  
./evaluation: code to evaluate the learned ML-CO policy  
./experiments: directory containing all anticipative lower bound target solutions for different experiments  
./features: code to create features  
./instances: directory containing all static instances  
./pchgs: code to run PC-HGS  
./training: code to train ML-CO policy  

To reproduce the results from the paper follow these steps:

## 1. Install PC-HGS
### Dependencies

Building PC-HGS requires
* cmake >= 3.14
* A compiler with C++-20 support

### Installation

1. Clone the repository and initialize submodules.

```bash
  git clone https://github.com/tumBAIS/euro-meets-neurips-2022.git
  cd pchgs
```

2. Generate a makefile

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE="RELEASE"
```

Support build types: RELEASE, DEBUG, RELWITHDEBINFO

3. Build

```bash
make
```


## 2. Train ML-CO policy
0. Install the dependencies
```bash 
pip install -r requirements.txt
```
1. Define the training configuration in `./training/src/config.py`
2. To start the training go to `./training/` and run  
```bash 
python run_training.py
```
3. To start the training in order to reproduce the results from the paper go to in `./training/` and run
```bash 
bash master_experiments.sh
``` 



## 3. Evaluate ML-CO policy
The evaluation of different benchmark policies and the ML-CO policy mainly bases on the [code provided from the EURO Meets NeurIPS 2022 Vehicle Routing Competition](https://github.com/ortec/euro-neurips-vrp-2022-quickstart).
1. Define the evaluation configuration in `./evaluation/src/config.py`
2. To start evaluating policies go to `./evaluation/` and run
```bash 
python solve_bound.py
```
3. To start the evaluation in order to reproduce the results from the paper go to `./evaluation/` and
first identify the best learning iteration for each trained model by running
```bash 
bash master_validation.sh 
```
Save the best found learning iteration for each model in `./evaluation/src/read_in_learning_iteration.txt`. 
Second, to start the final test run, go to `./evaluation/` and run
```bash 
bash master_test.sh
```

## 4. Visualize results from the paper
Visualize the results of the paper via running

```bash 
python visualization.py
```
in `./evaluation/src`.
Please note that we uploaded all result files in .zip format. Please unpack the folders before running `visualization.py`.
