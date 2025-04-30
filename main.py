from utils import plot_tri_psnr, add_camera_noise

if __name__ == "__main__":

    # for mild camera noise
    add_camera_noise(
        input_json_path='nerf/data/nerf_synthetic/lego/transforms_train.json',
        output_json_path='nerf/data/nerf_synthetic/lego_corrupted_camera/transforms_train_noisy.json',
        rotation_noise=0.002,
        translation_noise=0.02,
    )


    # # for worst camera noise
    # add_camera_noise(
    #     input_json_path='nerf/data/nerf_synthetic/lego/transforms_train.json',
    #     output_json_path='nerf/data/nerf_synthetic/lego_worst_camera/transforms_train_noisy.json',
    #     rotation_noise=0.002/2,
    #     translation_noise=0.02/2,
    # )


    plot_tri_psnr()