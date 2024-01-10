#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import random
import cv2

class CameraInfoDepth(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    K: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    depth: np.array
    image_path: str
    image_name: str
    ref_image: np.array
    ref_depth: np.array
    ref_uid: int
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    N_imgs = len(cam_extrinsics)
    random_ref = 1
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        # print("uid", uid)
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        depth_path =image_path.replace("images/", "dpt/depth_")[:-4]+".npz"
        depth = np.load(depth_path)["pred"]
        # import pdb; pdb.set_trace()
        if depth.shape[1:] < (height, width):
            depth = cv2.resize(depth[0], (width, height)) 

        if random_ref:
            if idx==N_imgs-1:
                ref_uid = idx-1
            else:
                ran_uid = random.randint(1, min(random_ref, N_imgs-idx-1))
                ref_uid = idx + ran_uid
            # print(f"uid {idx} ref_uid {ref_uid}")
        ref_image_path = os.path.join(images_folder, os.path.basename(cam_extrinsics[ref_uid].name))
        ref_image = Image.open(ref_image_path)
        ref_depth_path =ref_image_path.replace("images/", "dpt/depth_")[:-4]+".npz"
        ref_depth = np.load(ref_depth_path)["pred"]
        if ref_depth.shape[1:] < (height, width):
            ref_depth = cv2.resize(ref_depth[0], (width, height)) 

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        # print("intrinsics", intr)
        cam_info = CameraInfoDepth(uid=idx, R=R, T=T, K=intr, FovY=FovY, FovX=FovX, image=image, depth=depth, 
                              image_path=image_path, image_name=image_name, width=width, height=height, 
                              ref_image=ref_image, ref_depth=ref_depth, ref_uid=ref_uid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def arange_pixels(resolution=(128, 128), batch_size=1, image_range=(-1., 1.)):
    ''' Arranges pixels for given resolution in range image_range.

    The function returns the unscaled pixel locations as integers and the
    scaled float values.

    Args:
        resolution (tuple): image resolution
        batch_size (int): batch size
        image_range (tuple): range of output points (default [-1, 1])
        device (torch.device): device to use
    '''
    h, w = resolution
    
    # Arrange pixel location in scale resolution
    pixel_locations = np.meshgrid(np.arange(0, w), np.arange(0, h))
    pixel_locations = np.stack(
        [pixel_locations[0], pixel_locations[1]],
        axis=-1).reshape(1, -1, 2).repeat(batch_size, axis=0)
    pixel_scaled = pixel_locations.astype(np.float32)

    # Shift and scale points to match image_range
    scale = (image_range[1] - image_range[0])
    loc = (image_range[1] + image_range[0]) / 2
    pixel_scaled[:, :, 0] = (scale * pixel_scaled[:, :, 0] / (w - 1)) - loc
    pixel_scaled[:, :, 1] = (scale * pixel_scaled[:, :, 1] / (h - 1)) - loc

    return pixel_locations, pixel_scaled

def transform_to_world(pixels, depth, camera_mat, world_mat=None, scale_mat=None,
                       invert=True):
    ''' Transforms pixel positions p with given depth value d to world coordinates.

    Args:
        pixels (tensor): pixel tensor of size B x N x 2
        depth (tensor): depth tensor of size B x N x 1
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert matrices (default: true)
    '''
    assert(pixels.shape[-1] == 2)
    if world_mat is None:
        world_mat = np.array([[[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]])
    if scale_mat is None:
        scale_mat = np.array([[[1, 0, 0, 0], [0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]])
    # Convert to pytorch
    # pixels, is_numpy = to_pytorch(pixels, True)
    # depth = to_pytorch(depth)
    # camera_mat = to_pytorch(camera_mat)
    # world_mat = to_pytorch(world_mat)
    # scale_mat = to_pytorch(scale_mat)
    
    
    # Invert camera matrices
    if invert:
        camera_mat = np.linalg.inv(camera_mat)
        world_mat = np.linalg.inv(world_mat)
        scale_mat = np.linalg.inv(scale_mat)

    # Transform pixels to homogen coordinates
    # pixels = pixels.permute(0, 2, 1)
    # pixels = torch.cat([pixels, torch.ones_like(pixels)], dim=1)
    pixels = np.transpose(pixels, (0, 2, 1))
    ones = np.ones_like(pixels)
    pixels = np.concatenate((pixels, ones), axis=1)


    # Project pixels into camera space
    # pixels[:, :3] = pixels[:, :3] * depth.permute(0, 2, 1)
    pixels_depth = pixels.copy()
    depth_transposed = np.transpose(depth, (0, 2, 1))
    pixels_depth[:, :3] = pixels[:, :3] * depth_transposed

    # Transform pixels to world space
    p_world = scale_mat @ world_mat @ camera_mat @ pixels_depth

    # Transform p_world back to 3D coordinates
    p_world = np.transpose(p_world[:, :3], (0, 2, 1))[0]

    return p_world

def readColmapSceneInfoDepth(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    # bin_path = os.path.join(path, "sparse/0/points3D.bin")
    # txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        first_frame_info = train_cam_infos[0]
        # import pdb; pdb.set_trace()
        sample_resolution = (int(first_frame_info.height), int(first_frame_info.width))
        pixel_locations, p_pc = arange_pixels(resolution=sample_resolution)
        d = np.array(first_frame_info.depth)
        image = np.array(first_frame_info.image)
        # import pdb; pdb.set_trace()
        pcd = transform_to_world(p_pc, d.reshape(1, -1, 1), first_frame_info.K)
        print("Reprojecting the RGB image into 3D point cloud using monoculer depth")
    #     print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    #     try:
    #         xyz, rgb, _ = read_points3D_binary(bin_path)
    #     except:
    #         xyz, rgb, _ = read_points3D_text(txt_path)
        # import pdb; pdb.set_trace()
        storePly(ply_path, pcd, image.reshape(-1, 3))
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfoDepth,
    "Blender" : readNerfSyntheticInfo
}