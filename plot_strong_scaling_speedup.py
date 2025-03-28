import matplotlib.pyplot as plt
import numpy as np

def plot_speedup_64(x_values, speedup_light, speedup_fat):
    """
    Plots a line chart with speedup values for Light and Fat clusters
    for a dataset size of 64 images.
    
    Parameters:
    x_values (list of int/float): Number of processes.
    speedup_light (list of float): Speedup values for Light cluster.
    speedup_fat (list of float): Speedup values for Fat cluster.
    """
    # Create a new figure for 64 images
    plt.figure(figsize=(8, 6))

    # Plot the speedup line for Light cluster
    plt.plot(x_values, speedup_light, marker='o', linestyle='-', color='b', 
             label='Light Cluster')

    # Plot the speedup line for Fat cluster
    plt.plot(x_values, speedup_fat, marker='s', linestyle='--', color='r', 
             label='Fat Cluster')

    # Add ideal speedup line (S(n) = n)
    plt.plot(x_values, x_values, color='gray', linestyle=':', label='Ideal Speedup')

    # Labels and title with LaTeX-style formatting
    plt.xlabel('Number of Processes ($n$)')
    plt.ylabel('Speedup ($S(n) = T(1) / T(n)$)')
    plt.title('Speedup vs. Number of Processes\n(Dataset Size: 64 Images)')

    # Set Y-axis ticks for better readability
    plt.yticks(np.arange(0, 18, 2))

    # Grid, legend, and layout
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    # Save the plot
    plt.savefig('speedup_64_images.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

def plot_speedup_512(x_values, speedup_light, speedup_fat):
    """
    Plots a line chart with speedup values for Light and Fat clusters
    for a dataset size of 512 images.
    
    Parameters:
    x_values (list of int/float): Number of processes.
    speedup_light (list of float): Speedup values for Light cluster.
    speedup_fat (list of float): Speedup values for Fat cluster.
    """
    # Create a new figure for 512 images
    plt.figure(figsize=(8, 6))

    # Plot the speedup line for Light cluster
    plt.plot(x_values, speedup_light, marker='o', linestyle='-', color='b', 
             label='Light Cluster')

    # Plot the speedup line for Fat cluster
    plt.plot(x_values, speedup_fat, marker='s', linestyle='--', color='r', 
             label='Fat Cluster')

    # Add ideal speedup line (S(n) = n)
    plt.plot(x_values, x_values, color='gray', linestyle=':', label='Ideal Speedup')

    # Labels and title with LaTeX-style formatting
    plt.xlabel('Number of Processes ($n$)')
    plt.ylabel('Speedup ($S(n) = T(1) / T(n)$)')
    plt.title('Speedup vs. Number of Processes\n(Dataset Size: 512 Images)')

    # Set Y-axis ticks for better readability
    plt.yticks(np.arange(0, 18, 2))

    # Grid, legend, and layout
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    # Save the plot
    plt.savefig('speedup_512_images.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

# Data from strong scaling results
x_points = [1, 2, 4, 8, 16]  # Number of processes

# Speedup values for dataset size 64
speedup_light_64 = [1.0, 1.74, 3.26, 7.42, 14.75]  # Light cluster (64 images)
speedup_fat_64 = [1.0, 2.18, 4.37, 7.91, 15.81]    # Fat cluster (64 images)

# Speedup values for dataset size 512
speedup_light_512 = [1.0, 1.66, 3.28, 6.63, 13.54]  # Light cluster (512 images)
speedup_fat_512 = [1.0, 2.20, 4.48, 8.06, 16.11]    # Fat cluster (512 images)

# Generate the plots
plot_speedup_64(x_points, speedup_light_64, speedup_fat_64)
plot_speedup_512(x_points, speedup_light_512, speedup_fat_512)