import numpy as np
import matplotlib.pyplot as plt

def plot_tri_psnr(
        base='nerf/logs/lego_base/test_psnr_data.npz',
        poor='nerf/logs/lego_bad/test_psnr_data.npz',
        tunable='nerf/logs/lego_tunable/test_psnr_data.npz',
        output='final_plot.png'
):
    iterations = np.load(base)['iterations']
    
    psnr_base = np.load(base)['test_psnrs']
    # psnr_poor = np.load(poor)['test_psnrs']
    # psnr_tunable = np.load(tunable)['test_psnrs']
    
    # only show  base plot first
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, psnr_base, label='Base', color='blue')
    # plt.plot(iterations, psnr_poor, label='Poor', color='red')
    # plt.plot(iterations, psnr_tunable, label='Tunable', color='green')
    
    plt.xlabel('Iteration')
    plt.ylabel('PSNR (dB)')
    plt.title('Test PSNR over Training')
    plt.legend()
    plt.grid(True)
    plt.savefig(output)
