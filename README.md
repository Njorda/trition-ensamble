# Trition with post and pre processing. 

In this blog post we will dig down in to how a Machine Learning(ML) model can be combined with pre and post processing steps using [Nvidia triton](https://github.com/triton-inference-server/serve). By combining the pre- and post processing the user can make a single call using GPRC or http. It should be noted that we in reality will not merge these processing steps in any way but link the calls together using Tritons [ensemble](https://github.com/triton-inference-server/python_backend/blob/3a60cfc7fb3525d1200ee179a1355fd813cccd28/README.md#business-logic-scripting) functionality Triton support multiple different backends(processing functionality) and in this case we will use the [tensorRT backend](https://github.com/triton-inference-server/tensorrt_backend) for the model serving and the [python backend](https://github.com/triton-inference-server/python_backend) to add the pre and post processing business logic. TensorRT is a high-performance deep learning inference sdk, that includes a deep learning inference optimizer and runtime that delivers low latency and high throughput for inference applications. By using the python backend we will have the possibility to use python instead of interacting with triton through C++ which is often a lot more native to data scientists and machine learning engineers. 


# **Pre-  and Postprocessing Using Python Backend Example**
We will extend upon  the triton example [Preprocessing Using Python Backend Example](https://github.com/triton-inference-server/python_backend/tree/3a60cfc7fb3525d1200ee179a1355fd813cccd28/examples/preprocessing) but walk over the part more in depth to explain not only the ensemble set up but also how to use triton.

This example shows how to preprocess your inputs using Python backend before it is passed to the TensorRT model for inference and then postprocessed. This ensemble model includes an image preprocessing model (preprocess) and a TensorRT model (resnet50_trt) to do inference and a simple post processing step.

## Model 

The model will in this case be a pre trained [restnet50](https://arxiv.org/abs/1512.03385?context=cs) which is exported to ONNX using pytorch. Restnet50 is a image classification model.

The complete code can be found [here]()

```python
resnet50 = models.resnet50(pretrained=True)
dummy_input = torch.randn(1, 3, 224, 224)
resnet50 = resnet50.eval()

torch.onnx.export(resnet50,
                    dummy_input,
                    args.save,
                    export_params=True,
                    opset_version=10,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {
                            0: 'batch_size',
                            2: "height",
                            3: 'width'
                        },
                        'output': {
                            0: 'batch_size'
                        }
                    })

```
For how triton is configured the `input_names` and `output_names` are important, but we will comeback to this later when we discuss the model configurations for the ensemble. 

As mentioned earlier we will use tensorRT to optimize the serving. So how do we go from the ONNX format to tensorRT? Nvidia has release a cli tool which allows for compiling from different formats to tensorRT, in this case from ONNX to tensorRT. 

Here we set the arguments for enabling fp16 precision --fp16. To enable dynamic shapes use --minShapes, --optShapes, and maxShapes with --explicitBatch:

```
$ trtexec --onnx=model.onnx --saveEngine=./model_repository/resnet50_trt/1/model.plan --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:256x3x224x224 --fp16
```

We will walk through the exact setup step by step using the docker container supplied by nvidia to avoid installing and limit issues with setups. The code can be found [here](/onnx_exporter.py)

## Preprocessing
The python backend requires that the "Model"(a model is in this case the preprocessing, restnet50 and postprocessing or for that sake any processing you wan triton to do) to implement a class named [`TritonPythonModel`](https://github.com/triton-inference-server/python_backend/blob/3a60cfc7fb3525d1200ee179a1355fd813cccd28/README.md#usage). 

```python
import triton_python_backend_utils as pb_utils


class TritonPythonModel:    
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        print('Initialized...')

    def execute(self, requests: List[pb_utils.InferenceRequest]) -> List[pb_utils.InferenceResponse]:
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.
        for request in requests:
            # Perform inference on the request and append it to responses list...

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
```

The `TritonPythonModel` class can also implement `auto_complete_config` but we will skip this for our usecase and instead leave it up to the reader to check out the [docs](https://github.com/triton-inference-server/python_backend/blob/3a60cfc7fb3525d1200ee179a1355fd813cccd28/README.md#auto_complete_config). 

There are a couple of important things to consider when implementing the processing: 
1) Execute - will handle the actual processing, should be able to handle a list of calls in order to support batching. 
2) Triton relies on [protobuf](https://developers.google.com/protocol-buffers) as a serializing format, thus all the data passed in and out from models have to rely on the triton supplied protobuf formats to handle and create the request and response. Imported as:
    ```python
    import triton_python_backend_utils as pb_utils
    ```
    It also contains certain helper functions to get information from the model config(we will discuss the model config in depth later on). The [inputs and outputs](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#inputs-and-outputs) will be fetched and created as follows:

    ```python
    
    in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_0")

    
    out_tensor_0 = pb_utils.Tensor("OUTPUT_0",
                                    img_out.astype(output0_dtype))

    # Create InferenceResponse. You can set an error here in case
    # there was a problem with handling this inference request.
    # Below is an example of how you can set errors in inference
    # response:
    #
    # pb_utils.InferenceResponse(
    #    output_tensors=..., TritonError("An error occured"))
    inference_response = pb_utils.InferenceResponse(
        output_tensors=[out_tensor_0])
    ```

    Check out the full code example [here] to walk through the code. 
3) In and outputs needs to be in matched to the model config

Lets examine the preprocessing a bit more in depth now when we understand the high level of how to set it up. 

The `TritonPythonModel` class includes the possibility to utalise the `__init__` method in order to initialise some variables. We will use this to pull the data types of the input and outputs from the model config. 
```python
self.model_config = model_config = json.loads(args['model_config'])

# Get OUTPUT0 configuration
output0_config = pb_utils.get_output_config_by_name(
    model_config, "OUTPUT_0")

# Convert Triton types to numpy types
self.output0_dtype = pb_utils.triton_string_to_numpy(
    output0_config['data_type'])
```

the name `OUTPUT_0` comes from the definition of the preprocessing in the [model config](/model_repository/ensemble/config.pbtxt) where the name is set. In this case the init method is called when the model is loaded by triton and will be used for exposing the correct [data types](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#datatypes) for the output. 


The next interesting part is the `execute` function, the function will handle the preprocessing of each batch. 

```python
    def execute(self, requests: List[pb_utils.InferenceRequest]) -> List[pb_utils.InferenceResponse]:

        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_0")

            loader = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            ])

            def image_loader(image_name):
                image = loader(image_name)
                #expand the dimension to nchw
                image = image.unsqueeze(0)
                return image

            img = in_0.as_numpy()
            image = Image.open(io.BytesIO(img.tobytes()))
            img_out = image_loader(image)
            img_out = np.array(img_out)

            out_tensor_0 = pb_utils.Tensor("OUTPUT_0",
                                           img_out.astype(self.output0_dtype))

            responses.append( pb_utils.InferenceResponse(output_tensors=[out_tensor_0]))

        return responses
```

The execute function fetches the input values based upon there names and then initialize a normalization function. In this example we will use the [pytorch transform](https://pytorch.org/vision/stable/transforms.html) functionality to preprocess the images. The most important parts to pay attention to in this example except for the input handling is the output handling where each of the inputs in the batch are processed and converted to a `InferenceResponse` which is then appended to a list. The response list must match the request list length. 

## Postprocessing
The Postprocessing works in the same way as the [Preprocessing](#preprocessing) where we need to implement the `TritonPythonModel` class. In this case the post processing will be "stupid simple" where we just multiply the class label with 2. 

```python
UPDATE HERE ...
```
## Model repository

Each of the processing steps: 
- Preprocessing (python)
- Inference (tensorrt_plan)
- Postprocessing (python)

have to be described as separate models each with a `config.pbtxt` in a separate folder with the name of the step, in order to be able to combine it to an ensemble a model. Thus both the individual steps and the combined ensemble must be represented. 

The folder structure should look like below:

```
models
|-- model_a
    |-- 1
    |   |-- model.py
    |-- config.pbtxt
```


Triton expects the models to be version within the model repository with an incremental integer. In this case we will assume that the models added have version `1` and thus add them in the sub folders accordingly. 

## Setup 

**1. Converting PyTorch Model to ONNX format:**

Run onnx_exporter.py to convert ResNet50 PyTorch model to ONNX format. Width and height dims are fixed at 224 but dynamic axes arguments for dynamic batching are used. Commands from the 2. and 3. subsections shall be executed within this Docker container.

```bash
    docker run -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v $(pwd):/workspace nvcr.io/nvidia/pytorch:22.06-py3 bash
    pip install numpy pillow torchvision
    python onnx_exporter.py --save model.onnx
```
**2. Create the model repository:**

```bash
    mkdir -p model_repository/ensemble_python_resnet50/1
    mkdir -p model_repository/preprocessing/1
    mkdir -p model_repository/postprocessing/1
    mkdir -p model_repository/resnet50_trt/1
    
    # Copy the Python model
    cp preprocessing.py model_repository/preprocessing/1/model.py
    cp postprocessing.py model_repository/postprocessing/1/model.py
    # Copy the restnet model
    cp model.onnx model_repository/resnet50_trt/1/model.onnx
```

**3. Build a TensorRT engine for the ONNX model**

Set the arguments for enabling fp16 precision --fp16. To enable dynamic shapes use --minShapes, --optShapes, and maxShapes with --explicitBatch:

```bash
    trtexec --onnx=model.onnx --saveEngine=./model_repository/resnet50_trt/1/model.plan --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:256x3x224x224 --fp16
```
**4. Run the command below to start the server container:**

Under model_repository, run this command to start the server docker container:
    
-- Why do we do this again? Just to this onece with the ports and life would be easier ...
```bash
    docker run --runtime=nvidia -it --shm-size=1gb --rm -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd):/workspace/ -v/$(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:22.06-py3 bash
    pip install numpy pillow torchvision
    tritonserver --model-repository=/models
```
**5. Start the client to test:**

Under python_backend/examples/resnet50_trt, run the commands below to start the client Docker container:

```
    wget https://raw.githubusercontent.com/triton-inference-server/server/main/qa/images/mug.jpg -O "mug.jpg"
    docker run --rm --net=host -v $(pwd):/workspace/ nvcr.io/nvidia/tritonserver:22.06-py3-sdk python client.py --image mug.jpg   
```
 The result of classification is:COFFEE MUG  

Here, since we input an image of "mug" and the inference result is "COFFEE MUG" which is correct.

If you want to play around with the postprocessing in order to convince yourself that it actually works you can change the postprocessing function to return a static value. 
```
    def execute(self, requests):
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_0")
            img = in_0.as_numpy()
            out_tensor_0 = pb_utils.Tensor("OUTPUT_0", img.astype(self.output0_dtype))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)

        return responses
```
which will return the class `TENCH` as expected, check the [labels](/model_repository/resnet50_trt/labels.txt) where the first class is `TENCH`.

# What is triton?

```
Triton Inference Server is an open source inference serving software that streamlines AI inferencing.
```
according to [nvidia](https://github.com/triton-inference-server/server). The simple description is that it is a cross machine learning platform for model severs that supports some of the most popular libraries such as Tensorflow, Pytorch, Onnx and tensorRT. 



## Plattform

Triton supports serving multiple libraries and optimization of these, in order to allow Trition server to know which, the model platform always have to be specified in the `config.pbtxt` file. This allows Triton to understand how to server the model. More information about the [config.pbtxt](https://github.com/triton-inference-server/server/blob/64ea6dcb7d042f8c450113e5cfa73a5cad4af1f0/docs/model_configuration.md). The available platforms can be found [here](https://github.com/triton-inference-server/backend/blob/main/README.md#where-can-i-find-all-the-backends-that-are-available-for-triton)


## What is the python backend?

Triton includes a variety of tools, the python backend allows for combining python code with the Triton sever without having to interact with the c code (Triton is written in c) it self. Allowing for easier interactions with the triton sever without having to use GRPC or HTTP. 


# How to update the deployed model. 


### How to check what is deployed

```bash
curl -g -6 -X POST  http://localhost:8000/v2/repository/models/index
```

###

To deloy the models with explicit instead of poll(when we put in the whole folder). 

```
tritonserver --model-repository=/models --model-control-mode=explicit
```
This way no models are loaded, instead you need to use the API to specifically load these models of interest. This can be done: 


```bash
curl -g -6 -X POST http://localhost:8000/v2/repository/models/ensemble_python_resnet50/load
```

Once a model is loaded we can check the index again and see what is ready and not. 


```bash
curl -g -6 -X POST http://localhost:8000/v2/repository/models/index
[{"name":"ensemble_python_resnet50","version":"1","state":"READY"},{"name":"postprocessing","version":"1","state":"READY"},{"name":"preprocessing","version":"1","state":"READY"},{"name":"resnet50_trt","version":"1","state":"READY"}]
```

All models are ready. In order to unload one model we can do: 

```bash
curl -g -6 -X POST http://localhost:8000/v2/repository/models/ensemble_python_resnet50/unload
```

```bash
curl -g -6 -X POST http://localhost:8000/v2/repository/models/index
[{"name":"ensemble_python_resnet50","version":"1","state":"UNAVAILABLE","reason":"unloaded"},{"name":"postprocessing","version":"1","state":"READY"},{"name":"preprocessing","version":"1","state":"READY"},{"name":"resnet50_trt","version":"1","state":"READY"}]
```

### Version deployed

In order to achieve zero down time deployments we also need [this](https://github.com/triton-inference-server/server/blob/main/docs/model_configuration.md#version-policy). It will make it possible to set the config to just upload the latest version.

In order to load the 'n' latest models for inference we will set the following in the model configurations(config.pbtxt files):

```
version_policy: { latest: { num_versions: 1}}
```

After adding a new version on s3 (or your local storage) an new load has to be triggered to reload. 

Then the new models are ready to be served, this also seems to result in zero down time during serving, switching over to the latest model(if set by the client). 

After the index will look like this(based upon the example). 

```bash
curl -g -6 -X POST http://localhost:8000/v2/repository/models/index
[{"name":"ensemble_python_resnet50","version":"1","state":"UNAVAILABLE","reason":"unloaded"},{"name":"ensemble_python_resnet50","version":"2","state":"READY"},{"name":"postprocessing","version":"1","state":"UNAVAILABLE","reason":"unloaded"},{"name":"postprocessing","version":"2","state":"READY"},{"name":"preprocessing","version":"1","state":"UNAVAILABLE","reason":"unloaded"},{"name":"preprocessing","version":"2","state":"READY"},{"name":"resnet50_trt","version":"1","state":"UNAVAILABLE","reason":"unloaded"},{"name":"resnet50_trt","version":"2","state":"READY"}]
```


### Access from s3

```
export AWS_ACCESS_KEY_ID=''
export AWS_SECERET_ACCESS_KEY=''
export AWS_SESSION_TOKEN=''
export AWS_DEFAULT_REGION=''
```

I first missed the region and then got: 

```
I0826 14:05:42.177942 996 server.cc:254] No server context available. Exiting immediately.
error: creating server: Internal - Could not get MetaData for bucket with name niklas-test-ml-vision due to exception: , error message: No response body.
```

To run Triton fetching the models from s3 run: 
```
tritonserver --model-repository=s3://niklas-test-ml-vision/model_repository --model-control-mode=explicit
```

A trick is that `s3` dont upload empty folders but, the ensemble needs to have a version folder as well. If it is not created it will not work. Example error below.

```
failed to load model 'ensemble_python_resnet50': at least one version must be available under the version policy of model 'ensemble_python_resnet50'
```

