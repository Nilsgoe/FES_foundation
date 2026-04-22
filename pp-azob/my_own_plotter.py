import numpy as np

# Stretched Gaussian parameters
CUTOFF = 6.25  # cutoff from PLUMED
STRETCH_A = 1.0 / (1.0 - np.exp(-CUTOFF))  # Stretch parameter A
STRETCH_B = -np.exp(-CUTOFF) / (1.0 - np.exp(-CUTOFF))  # Stretch parameter B


def read_hills_file(file_path):
    """Parse the HILLS file and return the relevant data as arrays."""
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Skip header lines starting with "#!"
    data = []
    for line in lines:
        if line.startswith("#!"):
            continue
        data.append([float(x) for x in line.split()])

    return np.array(data)


def stretched_gaussian_1D(x, mean, sigma):
    """1D Stretched Gaussian kernel."""
    gaussian = np.exp(-0.5 * ((x - mean) ** 2) / (sigma ** 2))
    stretched = STRETCH_A * gaussian + STRETCH_B
    return np.maximum(stretched, 0)


def stretched_gaussian_2D(x, y, mean_x, mean_y, sigma_x, sigma_y):
    """2D Stretched Gaussian kernel."""
    gaussian = np.exp(
        -0.5 * ((x - mean_x) ** 2) / (sigma_x ** 2) - 0.5 * ((y - mean_y) ** 2) / (sigma_y ** 2)
    )
    stretched = STRETCH_A * gaussian + STRETCH_B
    return np.maximum(stretched, 0)


def compute_bias_1D(hills_data, bins, min_val, max_val):
    """Compute the 1D bias using the stretched Gaussian kernel."""
    x_vals = np.linspace(min_val, max_val, bins+1)
    bias = np.zeros_like(x_vals)

    for row in hills_data:
        time, mean, sigma, height, biasf,reg_f = row
        kernel = stretched_gaussian_1D(x_vals, mean, sigma)
        bias += height * kernel

    return x_vals, -bias


def compute_bias_2D(hills_data, bins_x, bins_y, min_x, max_x, min_y, max_y):
    """Compute the 2D bias using the stretched Gaussian kernel."""
    x_vals = np.linspace(min_x, max_x, bins_x+1)
    y_vals = np.linspace(min_y, max_y, bins_y+1)
    bias = np.zeros((bins_x+1, bins_y+1))

    for row in hills_data:
        time, mean_x, mean_y, sigma_x, sigma_y, height, biasf = row
        for i, x in enumerate(x_vals):
            kernel = stretched_gaussian_2D(x, y_vals, mean_x, mean_y, sigma_x,sigma_y)
            bias[i, :] += height * kernel

    return x_vals, y_vals, -bias


def main():
    # Parameters (adjust as needed)
    hills_file = "metad_azob_cis_test.txt"
    output_file_1D = "output_bias_1D.txt"
    output_file_2D = "output_bias_2D.txt"

    bins_1D = 500
    min_val_1D = -180
    max_val_1D = 180

    bins_x_2D = 300
    bins_y_2D = 300
    min_x_2D = 0.3
    max_x_2D = 1.2
    min_y_2D = -0.35
    max_y_2D = 1.56

    # Read and process the HILLS data
    hills_data = read_hills_file(hills_file)

    # Compute 1D bias
    x_vals_1D, bias_1D = compute_bias_1D(hills_data, bins_1D, min_val_1D, max_val_1D)

    # Save 1D results
    with open(output_file_1D, "w") as file:
        for x, b in zip(x_vals_1D, bias_1D):
            file.write(f"{x:.8f} {b:.8f}\n")

    print(f"1D bias computation complete. Results saved to {output_file_1D}.")
    exit()
    # Read and process the HILLS data
    hills_data = read_hills_file(hills_file_2D)

    # Compute 2D bias
    x_vals_2D, y_vals_2D, bias_2D = compute_bias_2D(
        hills_data, bins_x_2D, bins_y_2D, min_x_2D, max_x_2D, min_y_2D, max_y_2D
    )
    
    # Save 2D results
    with open(output_file_2D, "w") as file:
        for j, y in enumerate(y_vals_2D):
            for i, x in enumerate(x_vals_2D):
                file.write(f"{x:.8f} {y:.8f} {bias_2D[i, j]:.8f}\n")

    print(f"2D bias computation complete. Results saved to {output_file_2D}.")
    

if __name__ == "__main__":
    main()

