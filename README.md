# Description
---

This project was inspired by a CUDA class I took in University of Arizona. The project implements a computer vision technique where the effect of environmental shadows on greenhouses plants are significantly reduced to more accurately track the health of the plants. Due to the high computational demand of existing implementations, utilizing the parallelization nature of GPUs results in considerable performance gains over those implementations done on CPU dominant applications like MATLAB. This project builds on top of a baseline from the research paper: 

**E. Richter, R. Raettig, J. Mack, S. Valancius, B. Unal and A. Akoglu, "Accelerated Shadow Detection and Removal Method," 2019 IEEE/ACS 16th International Conference on Computer Systems and Applications (AICCSA), Abu Dhabi, United Arab Emirates, 2019, pp. 1-8, doi: 10.1109/AICCSA47632.2019.9035242. keywords: {Convolution;Kernel;Graphics processing units;Gray-scale;Mathematical model;Computer vision;Histograms},** 

where I explore potential optimization opportunities and assess the performance implications on more accessible GPUs such as a 2080 Super against computational GPUs like the NVIDIA P100.  

# Compile and Run
---
DISCLAIMER: Before compiling this project, you need to make sure you have at least CUDA 7.5 installed. This project was also developed on Linux. This means Windows support isn't guaranteed as the project utilizes CUDA-C which may have differing libraries than those standard on Windows.

Run the following commands to compile and build the project:
- `mkdir -p _build; cd _build`
- `cmake ..`
- `make SM=sm_75` (sm_75 is the compute capability ID for NVIDIA's Turing architecture. This project was only ever tested on the Turing and Pascal architectures)

Once the project has finished compiling, you will have a binary called `shadow_reduction` and it takes two arguments, the path to the input PPM image and the path, including the file name, to the exporting PPM image. Don't worry if the output file doesn't exist, the program will create it for you.

Example:
- `./shadow_reduction ../Data/plant_sd.ppm output.ppm`

The run will also include timing performance.
