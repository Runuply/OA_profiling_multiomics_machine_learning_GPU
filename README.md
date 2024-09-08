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
This command lists available GPU partitions and their current status.
### Requesting a GPU Node for an Interactive Session
To use GPU resources, request a node with the following command:
```bash
srun --partition=gpu --pty bash
```

### Loading the Required CUDA and Python Modules
```bash
bash-5.1$ module avail cuda
------------------------------------------------------------------- /cm/local/modulefiles --------------------------------------------------------------------
cuda-dcgm/2.4.6.1  

------------------------------------------------------------------- /cm/shared/modulefiles -------------------------------------------------------------------
cuda11.8/blas/11.8.0  cuda11.8/toolkit/11.8.0  cuda12.5/fft/12.5.1     cuda12.5/profiler/12.5.1  
cuda11.8/fft/11.8.0   cuda12.5/blas/12.5.1     cuda12.5/nsight/12.5.1  cuda12.5/toolkit/12.5.1   

Key:
modulepath
```
```bash
module load cuda12.5/toolkit/12.5.1
```

### Verifying GPU Availability
After loading the CUDA module, check if the GPU is correctly detected:
```bash
bash-5.1$ nvidia-smi
Sun Sep  8 14:43:13 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA L40S                    Off |   00000000:01:00.0 Off |                    0 |
| N/A   16C    P8             31W /  350W |       0MiB /  46068MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA L40S                    Off |   00000000:61:00.0 Off |                    0 |
| N/A   17C    P8             31W /  350W |       0MiB /  46068MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA L40S                    Off |   00000000:81:00.0 Off |                    0 |
| N/A   16C    P8             31W /  350W |       0MiB /  46068MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA L40S                    Off |   00000000:E1:00.0 Off |                    0 |
| N/A   18C    P8             31W /  350W |       0MiB /  46068MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```
You should see a table with the available GPUs and their statuses.
### Creating a Python Environment for GPU-Accelerated Code
For GPU-accelerated Python applications, it's recommended to create a virtual environment.

```bash
python -m venv my_gpu_env # create; in my case, python -m venv myenv_20240907_142513
source my_gpu_env/bin/activate # active
```
I used, `python3 -m venv myenv_$(date +%Y%m%d_%H%M%S)` to create a unique name of the env.
### Install GPU-Accelerated TensorFlow and PyTorch

Once inside the environment, install TensorFlow with GPU support:
```bash
bash-5.1$ pip install tensorflow

##WARNING: You are using pip version 21.2.3; however, version 24.2 is available.
##You should consider upgrading via the '/home/ye.liu/myenv_20240907_142513/bin/python3 -m pip install --upgrade pip' ##command.
(myenv_20240907_142513) bash-5.1$ /home/ye.liu/myenv_20240907_142513/bin/python3 -m pip install --upgrade pip
```

```bash
pip install torch torchvision torchaudio
pip install scikit-learn pandas numpy matplotlib
```
`torch`, `torchvision`, and `torchaudio` (PyTorch)
- **`torch`**: PyTorch is an open-source machine learning library used for deep learning tasks. It provides tensor computation with GPU acceleration and is widely used for building neural networks.
- **`torchvision`**: This package contains popular datasets, model architectures, and image transformations for computer vision tasks. It's especially useful when working with image-based data.
- **`torchaudio`**: This is used for working with audio data, providing functionalities like loading and processing audio files. It's helpful for audio-related machine learning tasks.

These packages were installed to enable deep learning workflows using the GPU-accelerated capabilities of PyTorch. They support tasks like computer vision, natural language processing, and speech recognition.

```
pip install scikit-learn pandas numpy matplotlib
```
scikit-learn: A widely used machine learning library that provides simple and efficient tools for data mining and data analysis. It includes implementations of various algorithms for classification, regression, clustering, and dimensionality reduction.
pandas: A powerful data manipulation and analysis library that allows you to work efficiently with structured data (e.g., data frames). It's used for loading, cleaning, and processing data.
numpy: The core library for numerical computing in Python. It provides support for arrays, matrices, and many mathematical functions used for performing operations on these arrays.
matplotlib: A plotting library used for creating static, animated, and interactive visualizations in Python. It's helpful for visualizing data trends, patterns, and model performance.
These libraries are essential for general machine learning and data science workflows. They help with data loading, preprocessing, modeling, and visualization.

### Verifying the Installation
Once the installation is complete, you can verify if TensorFlow is using your GPU by running the following Python code:
```python
import tensorflow as tf

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```
If the output shows available GPUs, your installation is successful, and TensorFlow will use the GPU for computations.

### Writing a Python Script to Use the GPU
Here is an example Python script (gpu_test.py) to verify that TensorFlow can detect the GPU:
```python
import tensorflow as tf

# Check if TensorFlow is using the GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {len(gpus)}")
    for gpu in gpus:
        print(f"GPU: {gpu}")
else:
    print("No GPUs detected.")
```
Save this script as gpu_test.py.

### Running the Python Script on the GPU Node
Now, you can run your script on the GPU node using:
```bash
python gpu_test.py
```
If everything is set up correctly, TensorFlow will detect and list the available GPUs.

### Submitting a Batch Job for GPU Usage (Optional)
Instead of using an interactive session, you can submit a batch job using Slurm. Create a job script (gpu_job.slurm) as follows:
```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=16G
#SBATCH --job-name=gpu_test_job

# Load modules
module load cuda12.5/toolkit/12.5.1
module load python/3.8

# Activate the Python virtual environment
source ~/my_gpu_env/bin/activate

# Run the Python script
python gpu_test.py
```
Submit the job script using:
```bash
sbatch gpu_job.slurm
```
Monitor the status of your job with:
```bash
squeue -u your_username
```
### Checking the Output
Once the job completes, the output will be stored in a file named slurm-JOBID.out, where JOBID is the job's ID. To view the output:

```bash
cat slurm-JOBID.out
```





