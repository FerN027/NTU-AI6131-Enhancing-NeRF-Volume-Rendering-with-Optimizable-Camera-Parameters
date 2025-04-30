import json
import numpy as np
import os
import copy
import numpy as np
import matplotlib.pyplot as plt


def add_camera_noise(input_json_path, output_json_path=None, 
                     rotation_noise=0.03, translation_noise=0.1, 
                     noise_type="gaussian", systematic_bias=False):
    """
    Add noise to camera poses in a NeRF transforms JSON file.
    
    Parameters:
    -----------
    input_json_path : str
        Path to the input transforms JSON file
    output_json_path : str or None
        Path to save the noisy transforms. If None, input_path with '_noisy' suffix is used
    rotation_noise : float
        Standard deviation of rotation noise in radians (default: 0.03 ≈ 1.7°)
    translation_noise : float
        Standard deviation of translation noise relative to scene scale (default: 0.1)
    noise_type : str
        Type of noise to add: "gaussian" (default), "uniform", or "laplacian"
    systematic_bias : bool
        Whether to add a systematic bias to all cameras (like a calibration error)
        
    Returns:
    --------
    str: Path to the output JSON file with noisy camera parameters
    """
    # Set output path if not specified
    if output_json_path is None:
        base, ext = os.path.splitext(input_json_path)
        output_json_path = f"{base}_noisy{ext}"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    # Load the original transforms
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Create a deep copy to avoid modifying the original
    noisy_data = copy.deepcopy(data)
    
    # Get scene scale to normalize translation noise (mean camera distance from origin)
    translations = []
    for frame in data['frames']:
        t = np.array([
            frame['transform_matrix'][0][3],
            frame['transform_matrix'][1][3],
            frame['transform_matrix'][2][3]
        ])
        translations.append(np.linalg.norm(t))
    scene_scale = np.mean(translations)
    
    # Generate systematic bias if requested (common offset for all cameras)
    sys_rotation = np.zeros(3)
    sys_translation = np.zeros(3)
    if systematic_bias:
        sys_rotation = _generate_noise(3, rotation_noise/2, noise_type)
        sys_translation = _generate_noise(3, translation_noise/2, noise_type) * scene_scale
    
    # Add noise to each camera
    for frame in noisy_data['frames']:
        matrix = frame['transform_matrix']
  
        # Add rotation noise
        if rotation_noise > 0:
            rotation = np.array([
                [matrix[0][0], matrix[0][1], matrix[0][2]],
                [matrix[1][0], matrix[1][1], matrix[1][2]],
                [matrix[2][0], matrix[2][1], matrix[2][2]]
            ])

            per_camera_noise = _generate_noise(3, rotation_noise, noise_type)
            noise_angles = per_camera_noise + sys_rotation
            
            # Convert to rotation matrices for each axis
            Rx = _rotation_matrix_x(noise_angles[0])
            Ry = _rotation_matrix_y(noise_angles[1])
            Rz = _rotation_matrix_z(noise_angles[2])
            
            # Apply noise to rotation (right multiply to apply in camera coordinates)
            noisy_rotation = rotation @ Rx @ Ry @ Rz
            
            # Update rotation part of the transform
            for i in range(3):
                for j in range(3):
                    frame['transform_matrix'][i][j] = float(noisy_rotation[i, j])
            
            # Also update the 'rotation' field if it exists
            if 'rotation' in frame:
                frame['rotation'] = frame['rotation'] + float(_generate_noise(1, rotation_noise * 0.5, noise_type)[0])
        
        # Add translation noise
        if translation_noise > 0:
            translation = np.array([matrix[0][3], matrix[1][3], matrix[2][3]])
            per_camera_noise = _generate_noise(3, translation_noise, noise_type) * scene_scale
            noise = per_camera_noise + sys_translation
            
            # Apply noise to translation
            noisy_translation = translation + noise
            
            # Update translation part of the transform
            for i in range(3):
                frame['transform_matrix'][i][3] = float(noisy_translation[i])
    
    # Save the noisy transforms
    with open(output_json_path, 'w') as f:
        json.dump(noisy_data, f, indent=4)
    
    print(f"Added noise to camera poses:")
    print(f"- Rotation noise: {rotation_noise:.4f} radians (~{np.degrees(rotation_noise):.2f}°)")
    print(f"- Translation noise: {translation_noise:.4f} * scene_scale ({translation_noise * scene_scale:.4f} units)")
    print(f"- Noise type: {noise_type}")
    print(f"- Systematic bias: {systematic_bias}")
    print(f"Saved to: {output_json_path}")
    
    return output_json_path



# Helper functions
def _generate_noise(size, scale, noise_type):
	"""Generate noise based on specified distribution"""
	if noise_type == "gaussian":
		return np.random.normal(0, scale, size)
	elif noise_type == "uniform":
		return np.random.uniform(-scale*1.732, scale*1.732, size)  # same variance as gaussian
	elif noise_type == "laplacian":
		return np.random.laplace(0, scale/1.414, size)  # same variance as gaussian
	else:
		raise ValueError(f"Unknown noise type: {noise_type}")

def _rotation_matrix_x(theta):
	"""Generate 3D rotation matrix for X axis"""
	return np.array([
		[1, 0, 0],
		[0, np.cos(theta), -np.sin(theta)],
		[0, np.sin(theta), np.cos(theta)]
	])

def _rotation_matrix_y(theta):
	"""Generate 3D rotation matrix for Y axis"""
	return np.array([
		[np.cos(theta), 0, np.sin(theta)],
		[0, 1, 0],
		[-np.sin(theta), 0, np.cos(theta)]
	])

def _rotation_matrix_z(theta):
	"""Generate 3D rotation matrix for Z axis"""
	return np.array([
		[np.cos(theta), -np.sin(theta), 0],
		[np.sin(theta), np.cos(theta), 0],
		[0, 0, 1]
	])



def plot_tri_psnr(
        base='nerf/logs/lego_base/test_psnr_data.npz',
        tunable='nerf/logs/lego_bad/test_psnr_data.npz',
        poor='nerf/logs/lego_tunable/test_psnr_data.npz',
        output='final_plot.png'
):
    iterations = np.load(base)['iterations']
    
    psnr_base = np.load(base)['test_psnrs']
    psnr_poor = np.load(poor)['test_psnrs']
    psnr_tunable = np.load(tunable)['test_psnrs']
    
    # Create the plot with all three curves
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, psnr_base, label='GT Camera', color='blue', marker='o', markevery=5)
    plt.plot(iterations, psnr_poor, label='Estimated Camera + Noise (Fixed)', color='red', marker='s', markevery=5)
    plt.plot(iterations, psnr_tunable, label='Estimated Camera + Noise (Optimizable)', color='green', marker='^', markevery=5)
    
    plt.xlabel('Training Iteration', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('Average Rendering Quality for 5 Selected Test Images During Training', fontsize=14)
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Optional: Add improvement regions/annotations
    max_poor = np.max(psnr_poor)
    max_tunable = np.max(psnr_tunable)
    improvement = max_tunable - max_poor
    plt.annotate(f"Improvement: +{improvement:.2f} dB", 
                xy=(iterations[-1], max_tunable), 
                xytext=(iterations[-1]*0.8, max_tunable+0.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    # Save with high DPI for better quality
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    
    print(f"Combined PSNR plot saved to {output}")
    print(f"Max PSNR - Base: {np.max(psnr_base):.2f}, Poor: {max_poor:.2f}, Tunable: {max_tunable:.2f}")
    print(f"Improvement from Poor to Tunable: +{improvement:.2f} dB")