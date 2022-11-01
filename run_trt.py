import tensorrt as trt
import common
import torch
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)



def trt_version():
    return trt.__version__

def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt_version() >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


def torch_device_to_trt(device):
    if device.type == torch.device("cuda").type:
        return trt.TensorLocation.DEVICE
    elif device.type == torch.device("cpu").type:
        return trt.TensorLocation.HOST
    else:
        return TypeError("%s is not supported by tensorrt" % device)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError("%s is not supported by torch" % device)

class TRTInference(object):
    def __init__(self, engine_path):
        trt.init_libnvinfer_plugins(None, "")
        self.engine_path = engine_path
        self.input_names = ["img1", "img2"]
        self.output_names = ["212"]
        self._load_engine()
        print("finish loading engine")

    def _load_engine(self):
        with open(self.engine_path, "rb") as f, trt.Logger() as logger, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

    def get_all_binding_names(self):
        for i, binding in enumerate(self.engine):
            print(self.engine.get_binding_name(i))
    
    def predict(self, inputs):
        batch_size = inputs[0].shape[0]
        bindings = [None] * (len(self.input_names) + len(self.output_names))
        # create output tensors
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = self.engine.get_binding_shape(idx)
            shape[0] = batch_size
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=tuple(shape), dtype=dtype, device=device)
            outputs[i] = output
            # save address of 
            bindings[idx] = output.data_ptr()
        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            bindings[idx] = inputs[i].contiguous().data_ptr()
            shape = self.engine.get_binding_shape(idx)
            shape[0] = batch_size
            #print(shape)
            self.context.set_binding_shape(0, tuple(shape))

        self.context.execute_async_v2(
            bindings, torch.cuda.current_stream().cuda_stream
        )

        outputs = tuple(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs

    def __call__(self, input_0, input_1):
        inputs = (input_0, input_1)
        return self.predict(inputs)

if __name__ == '__main__':
    pose_estimator = TRTInference('./sc.engine')
    pose_estimator.get_all_binding_names()
    inputs = [torch.zeros(1, 3, 352, 640) for i in range(2)]
    print(pose_estimator.predict(inputs))
# engine = common.deserialize_engine_from_file('./sc.engine', TRT_LOGGER)
# # inputs, outputs, bindings, stream = common.allocate_buffers(engine)
# print('Inputs', inputs)
# print('Outputs', outputs)
# print('bindings', bindings)
# context = engine.create_execution_context()

