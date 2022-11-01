import tensorrt as trt
# import pycuda.autoinit
# import pycuda.driver as cuda

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(EXPLICIT_BATCH)
config = builder.create_builder_config()
parser = trt.OnnxParser(network, TRT_LOGGER)
is_parsed = parser.parse_from_file('./onnxmodel.onnx')
assert is_parsed
engine = builder.build_engine(network, config)

with open('./sc.engine', 'wb') as f:
    f.write(engine.serialize())