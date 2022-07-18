# What is triton?

```
Triton Inference Server is an open source inference serving software that streamlines AI inferencing.
```
according to [nvidia](https://github.com/triton-inference-server/server). The simple description is that it is a cross machine learning platform for model severs that supports some of the most popular libraries such as Tensorflow, Pytorch, Onnx and tensorRT. 



## Plattforms

Triton supports serving multiple libraries and optimization of these, in order to allow Trition server to know which, the model platform always have to be specified in the `config.pbtxt` file. This allows Triton to understand how to server the model. More information about the [config.pbtxt](https://github.com/triton-inference-server/server/blob/64ea6dcb7d042f8c450113e5cfa73a5cad4af1f0/docs/model_configuration.md). The available platforms can be found [here](https://github.com/triton-inference-server/backend/blob/main/README.md#where-can-i-find-all-the-backends-that-are-available-for-triton)


## What is the python backend?

Triton includes a variety of tools, the python backend allows for combining python code with the Triton sever without having to interact with the c code (Triton is written in c) it self. Allowing for easier interactions with the triton sever without having to use GRPC or HTTP. 

# **Pre-  and Postprocessing Using Python Backend Example**
This repository is build on top of the triton example [Preprocessing Using Python Backend Example](Preprocessing Using Python Backend Example).

This example shows how to preprocess your inputs using Python backend before it is passed to the TensorRT model for inference and then postprocessed. This ensemble model includes an image preprocessing model (preprocess) and a TensorRT model (resnet50_trt) to do inference and a simple post processing step.

## Model repository

Each of the processing steps: 
- Preprocessing (python)
- Inference (tensorrt_plan)
- Postprocessing (python)

have to be described as seperate models each with a `config.pbtxt` in a seperate folder with the name of the step, in order to combine it to an ensamble a model for the ensamble also have to be created. Thus both the indvidual steps and the combined ensamble must be represented. 

The folder structure should look like below:

```
models
|-- model_a
    |-- 1
    |   |-- model.py
    |-- config.pbtxt
    `-- triton_python_backend_stub
```


Triton expects the models to be version within the model repository with an incremental integer. In this case we will assume that the models added all have version `1` and thus add them in the sub folders accordingly. 

```
$ mkdir -p model_repository/ensemble_python_resnet50/1
$ mkdir -p model_repository/preprocess/1
$ mkdir -p model_repository/postprocess/1
$ mkdir -p model_repository/resnet50_trt/1
```

The python `models` have to be saved with the name `model.py`


## Setup 

**1. Converting PyTorch Model to ONNX format:**

Run onnx_exporter.py to convert ResNet50 PyTorch model to ONNX format. Width and height dims are fixed at 224 but dynamic axes arguments for dynamic batching are used. Commands from the 2. and 3. subsections shall be executed within this Docker container.

    $ docker run -it --gpus=all -v $(pwd):/workspace nvcr.io/nvidia/pytorch:xx.yy-py3 bash
    $ pip install numpy pillow torchvision
    $ python onnx_exporter.py --save model.onnx
    
**2. Create the model repository:**

    $ mkdir -p model_repository/ensemble_python_resnet50/1
    $ mkdir -p model_repository/preprocess/1
    $ mkdir -p model_repository/resnet50_trt/1
    
    # Copy the Python model
    $ cp model.py model_repository/preprocess/1

**3. Build a TensorRT engine for the ONNX model**

Set the arguments for enabling fp16 precision --fp16. To enable dynamic shapes use --minShapes, --optShapes, and maxShapes with --explicitBatch:

    $ trtexec --onnx=model.onnx --saveEngine=./model_repository/resnet50_trt/1/model.plan --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:256x3x224x224 --fp16

**4. Run the command below to start the server container:**

Under model_repository, run this command to start the server docker container:

    $ docker run --gpus=all -it --rm -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd):/workspace/ -v/$(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:xx.yy-py3 bash
    $ pip install numpy pillow torchvision
    $ tritonserver --model-repository=/models
     
**5. Start the client to test:**

Under python_backend/examples/resnet50_trt, run the commands below to start the client Docker container:

    $ wget https://raw.githubusercontent.com/triton-inference-server/server/main/qa/images/mug.jpg -O "mug.jpg"
    $ docker run --rm --net=host -v $(pwd):/workspace/ nvcr.io/nvidia/tritonserver:xx.yy-py3-sdk python client.py --image mug.jpg 
    $ The result of classification is:COFFEE MUG    

Here, since we input an image of "mug" and the inference result is "COFFEE MUG" which is correct.

