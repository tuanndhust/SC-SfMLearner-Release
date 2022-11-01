import rospy

import cv2
from sensor_msgs.msg import Image
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from rot_utils import mat2euler, euler2quat
import sys
import numpy as np
# from skimage.transform import resize as imresize

from path import Path
import torch
import matplotlib.pyplot as plt
from matplotlib.path import Path as PathPlot
import matplotlib.patches as patches
from inverse_warp import *
import models
import argparse
from run_trt import TRTInference

from transformations import quaternion_from_matrix
import time

parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--pretrained-posenet", required=True, type=str, help="pretrained PoseNet path")
parser.add_argument("--img-height", default=360, type=int, help="Image height")
parser.add_argument("--img-width", default=640, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)
parser.add_argument("--sequence", default='square', type=str, help="sequence to test")
parser.add_argument("--output-dir", type=str, default='./out_square/',help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--device", type=str, default='cuda', help="cpu or gpu inference")

args = parser.parse_args()
device = torch.device(args.device)
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


pose_rpg = {'# timestamps': [], 'tx': [], 'ty': [], 'tz': [], 'qx': [], 'qy': [], 'qz': [], 'qw': []}


def load_tensor_image(img, args):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    h, w, _ = img.shape
    if (not args.no_resize) and (h != args.img_height or w != args.img_width):
        img = cv2.resize(img, (args.img_height, args.img_width)).astype(np.float32)
    # Cropping to [352, 640]
    img = img[4:-4, :, :]
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.45)/0.225).to(device)
    return tensor_img

def Tmat2PoseMsg(T: np.ndarray)->PoseStamped:
    # Rotation
    R = np.eye(4)
    R[:3, :3] = T[:3, :3]
    # Translation
    t = T[:3, -1]
    # z, y, x = mat2euler(T)
    # w, x, y, z = euler2quat(z, y, x)
    q = quaternion_from_matrix(R, isprecise=True)
    pose_msg = PoseStamped()
    pose_msg.pose.position.x = t[0]
    pose_msg.pose.position.y = t[1]
    pose_msg.pose.position.z = t[2]

    pose_msg.pose.orientation.x = q[1]
    pose_msg.pose.orientation.y = q[2]
    pose_msg.pose.orientation.z = q[3]
    pose_msg.pose.orientation.w = q[0]
    return pose_msg


class PoseEstimator:

    def __init__(self, args):
        """
        This is a node that subscribes to the RGB image topic of rosout, 
        convert each sensor_msg.msg.Image message at each timestamp to cv2 compatible numpy.ndarray
        """
        self.args =args
        # weights_pose = torch.load(args.pretrained_posenet, map_location=args.device)
        # self.pose_net = models.PoseResNet().to(device)
        # self.pose_net.load_state_dict(weights_pose['state_dict'], strict=False)
        # self.pose_net.eval()
        # # Identity matrix
        self.pose_net = TRTInference('./sc.engine')
        self.global_pose = np.eye(4)
        self.poses = [self.global_pose[0:3, :].reshape(1, 12)]

        self.pose_pub = rospy.Publisher("ScSfmOut",PoseStamped)

        self.bridge = CvBridge()
        self.image_sub = message_filters.Subscriber('/rgb',Image)
        self.cache = message_filters.Cache(self.image_sub, 2)
        self.cache.registerCallback(self.callback)

        self.tensor_imgs = []

    @torch.no_grad()
    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            # Begin
            tensor_img = load_tensor_image(cv_image, self.args)
            self.tensor_imgs.append(tensor_img)
            # print(len(self.tensor_imgs))
            if len(self.tensor_imgs) == 2:
                # [B, 6](B=1)
                start_time = time.time()
                pose = self.pose_net(self.tensor_imgs[0], self.tensor_imgs[1])
                # [6] -> [3, 4]
                pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()
                # Convert to homogenous # [4, 4]
                pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
                # Convert ego-motion to odometry
                self.global_pose = self.global_pose @  np.linalg.inv(pose_mat)
                cam2world = np.eye(4)
                cam2world[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0,-1,0 ]])
                # self.global_pose = np.matmul(cam2world, self.global_pose)
                pose_msg = Tmat2PoseMsg(cam2world @ self.global_pose)
                pose_msg.header = data.header
                pose_msg.header.frame_id = 'world'
                self.pose_pub.publish(pose_msg)
                endtime = time.time()
                print(1/(endtime-start_time))
                # Save 3x4 flattened odometry rray into a list
                self.poses.append(self.global_pose[0:3, :].reshape(1, 12))
                (rows, cols, channels) = cv_image.shape
                self.tensor_imgs.pop(0)
                #End

            cv2.imshow("Image window", cv_image)
            cv2.waitKey(3)

        except CvBridgeError as e:
            print(e)


def main():

    #Step 1: Initialize Python3 Object
    ic = PoseEstimator(args)
    #Step 2: Initialize ROS node
    rospy.init_node('PoseEstimator', anonymous=True)
    #Step 3: Run
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        #Step 4: Finish
        # Save to file
        print(f"Saving poses to " +  args.output_dir + args.sequence + ".txt")
        poses = np.concatenate(ic.poses, axis=0)
        filename = Path(args.output_dir + args.sequence + ".txt")
        np.savetxt(filename, poses, delimiter=' ', fmt='%1.8e')
        print("done")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
