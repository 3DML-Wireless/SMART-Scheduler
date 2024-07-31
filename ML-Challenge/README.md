# Introduction
This repository contains resources for the ITU AI/ML challenge "Optimal Multi-user scheduling in massive MIMO mobile channels".  

# Problem Statement
The goal of this challenge is to use machine learning-based algorithms to solve the multi-user beamforming scheduling problem. 
The base station uses the channel estimates to select users for either downlink or uplink transmission, in each transmission interval. 
The massive MIMO base station uses an estimate of the chanel matrix $H$ to beamform to scheduled users. 
A commonly known scheduling algorithm, the proportional-fair scheduling, find the subset of users in the network that maximizes the sum rate in the network while providing some of level of fairness among users. 
In a massive MIMO network, this scheduler finds a group of users to beamform for each resource block (time and frequency resource). 
However, the correlation of channels could impair beamforming performance, and thus, the scheduler must take that into account during scheduling. 
Additionally, due to the overhead of a channel measurement in mobile environments, an efficient algorithm may need to use partial or stale channel knowledge to schedule users in next scheduling periods to maximize network throughput while considering fairness to users. 
The fairness criteria can be calculated using Jain’s fairness index (JFI) where cumulative rate of users from past scheduling decisions are used to weigh the preference for users in the next scheduling period. 

# Baseline Solution
The baseline solution included in this repo, uses a well-known deep reinforcement learning (DRL) algorithm called soft actor-critic (SAC) that provides good exploration property as well as scalability. The objective of this challenge is to provide a machine-learning-based solution that improves on the baseline model in 3 metrics: sum rate, fairness, and computational complexity.

# Datasets
There are two datasets provided for this challenge.

**Dataset 1:** The first dataset is derived from a dataset in the [RENEW project dataset repository](https://renew-wireless.org/dataset-iuc.html). 
The dataset is from channel measurements with a 64-antenna massive MIMO base station at Rice University campus where users are placed in multiple locations including 4 line-of-sight and non line-of-sight clusters and in 25 locations within a cluster. 
The derived dataset emulates a low-mobility network where users move within a cluster. 
This dataset includes channels of 64 users. 
The channel for each user is measured at 52 frequency subcarriers (in OFDM symbols). 
The provided channel dataset includes channel instances at 500 frames.

**Dataset 2:** The seconds dataset is generated through the QuaDRiGa channel simulator using the 3GPP Urban Micro channel model. 
The dataset uses a 64-antenna massive MIMO base station and 64 mobile users moving in the vicinity of the base station with the speed of 2.8 m/s. 
Similar to dataset 1, in each frame 52 subcarriers are used for each channel's users. 

The above datasets can be downloaded [here](https://drive.google.com/drive/folders/1zqbyl7yBQmVnAdiys_MMvnxXILg2TrWd).

# Run ML models
In ML_Challenge_Code/SAC_KNN, run main.py

# Submission Guideline
1. A trained-well ML model (like sac_checkpoint_SMART_64_64 in checkpoints). Please ensure it can be loaded successfully by infer.py before submission.
2. A README doc to introduce your methodology design (e.g. algorithm or training tricks) in as much detail as possible.
3. Compress your codes and README doc in a zip file and send it to qa4[at]rice.edu before Aug.30 11:59 PM(CST). (Submission Deadline is extended to Aug.30 !!!)

# Acknowledgement
This challenge is supported by the ITU's AI for Good Initiative.

# Contact
Qing An qa4[at]rice.edu
Rahman Doost-Mohammady doost[at]rice.edu
Santiago Segarra segarra[at]rice.edu
Ashutosh Sabharwal ashu[at]rice.edu
