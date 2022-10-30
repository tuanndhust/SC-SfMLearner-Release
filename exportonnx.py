import torch
from models.PoseResNet import PoseResNet
import onnx

onnx_name = 'onnxmodel.onnx'

device = 'cpu'
pretrained_posenet = '/home/tuannghust/SC-SfMLearner-Release/checkpoints/resnet50_pose_123/06-11-23:54/exp_pose_model_best.pth.tar'
weights_pose = torch.load(pretrained_posenet, map_location=device)
pose_net = PoseResNet().to(device)
pose_net.load_state_dict(weights_pose['state_dict'], strict=False)
pose_net.eval()
tensor = torch.randn((1,3,352,640))
torch.onnx.export(pose_net,(tensor,tensor),onnx_name,input_names=['img1','img2'])
model_onnx = onnx.load(onnx_name)
onnx.checker.check_model(model_onnx)
onnx.save(model_onnx ,onnx_name)