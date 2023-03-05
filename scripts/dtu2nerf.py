import argparse
import json
import os
import cv2
import numpy as np
from tqdm import tqdm


def to8b(x): return (x.clip(0, 1) * 255).astype(np.uint8)


def to16b(img):
    img = img.clip(0, 1) * 65535
    return img.astype(np.uint16)


def opencv_to_gl(pose):
    mat = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    pose[:3, :3] = pose[:3, :3] @ mat
    return pose


def get_offset(poses):
    eyes = np.stack([pose[:3, 3] for pose in poses])

    scale = eyes.max(axis=0) - eyes.min(axis=0)
    print(f'scale : {scale}')

    offset = -(eyes.max(axis=0) + eyes.min(axis=0)) / 2
    print(f'offset : {offset}')

    return scale / 2, offset


def s(pose, scale, offset):
    pose[:3, 3] = (pose[:3, 3] + offset) / scale
    # print(pose[:3, 3])
    return pose.tolist()


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]]
                 for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path', type=str, help="root directory to the LLFF dataset (contains images/ and pose_bounds.npy)")
    parser.add_argument('--downscale', type=float, default=2,
                        help="image size down scale, e.g., 4")
    opt = parser.parse_args()

    print(f'[INFO] process {opt.path}')
    downscale = opt.downscale
    os.chdir(opt.path)

    image_dir = 'image'
    n_images = len(os.listdir(image_dir))
    # mask_dir = 'mask'

    cam_file = 'cameras_sphere.npz'
    camera_dict = np.load(cam_file)
    world_mats = [camera_dict['world_mat_%d' % idx].astype(
        np.float32) for idx in range(n_images)]

    intrinsics_all = []
    pose_all = []
    for mat in world_mats:
        P = mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)
        intrinsics_all.append(intrinsics)
        pose_all.append(opencv_to_gl(pose))

    train_json = dict()

    train_json['fl_y'] = intrinsics[1][1] / downscale
    # train_json['h'] = int(intrinsics[1, 2] * 2) / downscale
    train_json['fl_x'] = intrinsics[0][0] / downscale
    # train_json['w'] = int(intrinsics[0, 2] * 2) / downscale

    scale, offset = get_offset(pose_all)

    # train_json['enable_depth_loading'] = True
    # train_json['integer_depth_scale'] = 1 / 65535

    train_json['frames'] = []

    # test_json = copy.deepcopy(train_json)
    os.makedirs('image_masked', exist_ok=True)

    for i in tqdm(range(n_images)):
        frames = train_json['frames']

        # depth = cv2.imread(os.path.join('depth', '{:04d}.exr'.format(i)), -1)
        # cv2.imwrite(os.path.join('depth', '{:04d}.png'.format(i)), to16b(depth / scale.max()))
        # im = cv2.imread(f'./image/{i:06d}.png') / 255
        # mask = cv2.imread(f'./mask/{i:03d}.png')[:,:,:1] / 255
        # im = to8b(im * mask + (1 - mask))
        im = cv2.imread(f'./image/{i:06d}.png')
        mask = cv2.imread(f'./mask/{i:03d}.png')[:, :, :1]
        im = np.concatenate((im, mask), axis=-1)
        h, w, _ = im.shape
        if i == 0:
            train_json['h'] = h // downscale
            train_json['w'] = w // downscale
        if downscale > 1:
            im = cv2.resize(im, (w // downscale, h // downscale),
                            interpolation=cv2.INTER_AREA)

        cv2.imwrite(f'./image_masked/{i:04d}.png', im)

        frame = {
            'file_path': f'./image_masked/{i:04d}',
            # 'depth_path': f'./depth/{i:04d}.png',
            'transform_matrix': s(pose_all[i], scale.max(), offset)
        }
        frames.append(frame)

    with open('transforms_train.json', 'w') as f:
        json.dump(train_json, f, indent=4)

    with open('transforms_val.json', 'w') as f:
        json.dump(train_json, f, indent=4)


if __name__ == '__main__':
    main()
