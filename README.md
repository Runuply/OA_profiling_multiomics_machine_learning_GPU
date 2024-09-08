# OA_profiling_multiomics_machine_learning_GPU

## GPU Resource Utilization Guide on HPC cluster
This guide will help you access and use GPU resources on the HPC cluster. It provides step-by-step instructions from logging into the cluster to running a Python script using TensorFlow with GPU acceleration.

## Prerequisites
- Access to the cluster. `ssh your.name@access.hpc.vai.org`
- Basic knowledge of the Linux terminal.
- Installed Python on the cluster or understanding how to load Python modules.

### Logging Into the Cluster
Log into the HPC cluster using SSH. In your terminal (replace `your_username` with your actual username):
```bash
ssh ye.liu@access.hpc.vai.org
```
### Checking GPU Availability
```bash
[ye.liu@submit003 ~]$ sinfo --partition=gpu
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
gpu          up 14-00:00:0      2    mix compute[084-085]
gpu          up 14-00:00:0      4   idle compute[089-090,095-096]
```
