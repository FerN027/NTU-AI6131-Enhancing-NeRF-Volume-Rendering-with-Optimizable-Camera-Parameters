from utils import plot_tri_psnr, add_camera_noise

if __name__ == "__main__":
    # plot_tri_psnr()
    add_camera_noise(
        input_json_path='nerf/data/nerf_synthetic/lego/transforms_train.json',
        output_json_path='nerf/data/nerf_synthetic/lego_corrupted_camera/transforms_train_noisy.json',
        rotation_noise=0.01
    )