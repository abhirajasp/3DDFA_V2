# coding: utf-8

__author__ = 'cleardusk'

import sys
import argparse
import cv2
import yaml
import json
# from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.render_ctypes import render  # faster
from utils.depth import depth
# from utils.pncc import pncc
# from utils.uv import uv_tex
# from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix
from utils.tddfa_util import str2bool


def read_image(image_path):
    """
    Read an image from input path

    params:
        - image_local_path (str): the path of image.
    return:
        - image: Required image.
    """
    LOCAL_ROOT = '/frdata/CAS/CELEBA_SPOOF_DATASET/CelebA_Spoof/'
    image_path = LOCAL_ROOT + image_path

    crop_path = image_path[:-4]+"_crop"+image_path[-4:]
    wfp = image_path[:-4]+"_depth_3DDFA"+image_path[-4:]
    img = cv2.imread(crop_path)
    return img, wfp

def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        # from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        # face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        # face_boxes = FaceBoxes()

    # Given a still image path and load to BGR channel

    LOCAL_ROOT = '/frdata/CAS/CELEBA_SPOOF_DATASET/CelebA_Spoof/'
    LOCAL_IMAGE_LIST_PATH = 'metas/intra_test/test_label.json'
    with open(LOCAL_ROOT+LOCAL_IMAGE_LIST_PATH) as f:
        image_list = json.load(f)
    print("got local image list, {} image".format(len(image_list.keys())))
    #Batch_size = 1024
    #logging.info("Batch_size=, {}".format(Batch_size))
    n=0
    for idx,image_id in enumerate(image_list):
        # get image from local file
        if n==5:
            break
        try:
            print(image_id)
            print(n)
            img, wfp = read_image(image_id)
            n += 1
            param_lst, roi_box_lst = tddfa(img)

            # Visualization and serialization
            # new_suffix = args.opt
            # if new_suffix not in ('ply', 'obj'):
            #     new_suffix = '.jpg'
            # wfp = f'examples/results/{args.img_fp.split("/")[-1].replace(old_suffix, "")}_{args.opt}' + new_suffix
            # wfp =  image_path[:-4]+"_depth_3DDFA"+image_path[-4:]

            ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag='depth')

            depth(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=False)
        # else:
        #     raise ValueError(f'Unknown opt {args.opt}')

        except:
            # logging.info("Failed to read image: {}".format(image_id))
            raise
    # img = cv2.imread(args.img_fp)

    # Detect faces, get 3DMM params and roi boxes
    # boxes = face_boxes(img)
    # n = len(boxes)
    # if n == 0:
        # print(f'No face detected, exit')
        # sys.exit(-1)
    # print(f'Detect {n} faces')

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('-f', '--img_fp', type=str, default='examples/inputs/trump_hillary.jpg')
    parser.add_argument('-m', '--mode', type=str, default='gpu', help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='depth',
                        choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])
    parser.add_argument('--show_flag', type=str2bool, default='False', help='whether to show the visualization result')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
