import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class InputParameters:
    flux: float
    pixel_area: float
    frame_duration: float

    def get_summary(self) -> str:
        """Return a summary of the stored data."""
        return f"Flux [cm^{{-2}}.s^{{-1}}]: {self.flux}, Pixel Area: {self.pixel_area}, Frame Duration [s]: {self.frame_duration}"

    def calculate_flux_area_product(self) -> float:
        """Calculate the product of flux and pixel area. This is the average number of particles hitting the pixel per second."""
        return self.flux * self.pixel_area

    def calculate_flux_area_frame_duration_product(self) -> float:
        """Calculate the product of flux by pixel area by frame duration. This is the average number of particles hitting the pixel per frame."""
        return self.flux * self.pixel_area * self.frame_duration

    def calculate_pixel_occupancy(self) -> float:
        """Calculate the pixel occupancy, which is the probability of having at least one particle hitting the pixel during the frame duration."""
        lambda_pixel = self.calculate_flux_area_frame_duration_product()
        return 1 - np.exp(-lambda_pixel)

    def calculate_pixel_entropy(self) -> float:
        """Calculate the entropy of the pixel occupancy. This is H = -p*log2(p) - (1-p)*log2(1-p)"""
        occupancy = self.calculate_pixel_occupancy()
        if occupancy == 0 or occupancy == 1:
            return 0.0
        return -occupancy * np.log2(occupancy) - (1 - occupancy) * np.log2(1 - occupancy)

    def calculate_entropy_per_cm2(self) -> float:
        """Calculate the entropy per cm^2."""
        return self.calculate_pixel_entropy() / self.pixel_area

    def calculate_entropy_per_cm2_per_s(self) -> float:
        """Calculate the entropy per cm^2 per second."""
        return self.calculate_entropy_per_cm2() / self.frame_duration


class PlotEntropyUtilities:
    """
    A utility class for plotting entropy as a function of various parameters.
    """

    @staticmethod
    def plot_entropy_vs_flux(entropy_results):
        """
        Plot entropy as a function of flux using the provided list of tuples.

        Args:
            entropy_results (list of tuples): Each tuple contains (flux, entropy_per_cm2_per_s).
        """
        flux_values, entropy_values = zip(*entropy_results)
        plt.figure(figsize=(8, 6))
        plt.plot(flux_values, entropy_values, marker='o', linestyle='-', color='b', label='Entropy per cm² per second')
        plt.xlabel('Flux [cm⁻²·s⁻¹]')
        plt.ylabel('Entropy per cm² per second [bit/s/cm²]')
        plt.title('Entropy as a Function of Flux')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_entropy_vs_frame_duration(entropy_vs_duration_results):
        """
        Plot entropy as a function of frame duration using the provided list of tuples.

        Args:
            entropy_vs_duration_results (list of tuples): Each tuple contains (frame_duration, entropy_per_cm2_per_s).
        """
        frame_duration_values, entropy_values = zip(*entropy_vs_duration_results)
        plt.figure(figsize=(8, 6))
        plt.plot(frame_duration_values, entropy_values, marker='o', linestyle='-', color='g', label='Entropy per cm² per second')
        plt.xlabel('Frame Duration [s]')
        plt.ylabel('Entropy per cm² per second [bit/s/cm²]')
        plt.title('Entropy as a Function of Frame Duration')
        plt.xscale('log')
        plt.ylim(0, 2e9)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_entropy_vs_pixel_area(entropy_vs_pixel_area_results):
        """
        Plot entropy as a function of pixel area using the provided list of tuples.

        Args:
            entropy_vs_pixel_area_results (list of tuples): Each tuple contains (pixel_area, entropy_per_cm2_per_s).
        """
        pixel_area_values, entropy_values = zip(*entropy_vs_pixel_area_results)
        plt.figure(figsize=(8, 6))
        plt.plot(pixel_area_values, entropy_values, marker='o', linestyle='-', color='r', label='Entropy per cm² per second')
        plt.xlabel('Pixel Area [cm²]')
        plt.ylabel('Entropy per cm² per second [bit/s/cm²]')
        plt.title('Entropy as a Function of Pixel Area')
        plt.xscale('log')
        plt.ylim(0, 2e9)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.show()



##############################################################
# Tests on InputParameters class
##############################################################

# Create an instance of the class
inputParameters = InputParameters(flux=1e8, pixel_area=1e-6, frame_duration=1e-7)

# Get a summary
print(inputParameters.get_summary())

# Call direct method
print(inputParameters)

# calculate Entropy per cm2 per second
pixel_occupancy = inputParameters.calculate_pixel_occupancy()
print(f"Pixel Occupancy: {pixel_occupancy}")
pixel_entropy = inputParameters.calculate_pixel_entropy()
print(f"Pixel Entropy: {pixel_entropy}")
entropy_per_cm2 = inputParameters.calculate_entropy_per_cm2()
print(f"Entropy per cm^2: {entropy_per_cm2}")
entropy_per_cm2_per_s = inputParameters.calculate_entropy_per_cm2_per_s()
print(f"Entropy per cm^2 per second: {entropy_per_cm2_per_s}")


##############################################################
# Plot the entropy vs flux
###############################################################

# Create a fresh instance of the class, reset parameters
inputParameters = InputParameters(flux=1e8, pixel_area=1e-6, frame_duration=1e-7)

# Define a range of flux values
flux_values = [1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8]  # Example range of flux values

# Empty list to store results
entropy_results = []

# Loop over the flux values
for flux in flux_values:
    # Update the flux field of the InputParameters instance
    inputParameters.flux = flux

    # Calculate entropy per cm^2 per second
    entropy_per_cm2_per_s = inputParameters.calculate_entropy_per_cm2_per_s()

    entropy_results.append((flux, entropy_per_cm2_per_s))

# Plot the results
PlotEntropyUtilities.plot_entropy_vs_flux(entropy_results)

##############################################################
# Plot the entropy vs frame duration
##############################################################

# Create a fresh instance of the class, reset parameters
inputParameters = InputParameters(flux=1e8, pixel_area=1e-6, frame_duration=1e-7)

# Define a range of frame_duration_values
frame_duration_values = [1e-7, 2e-7, 5e-7, 1e-6, 2e-6]  # Range of frame duration values

# Empty list to store results
entropy_vs_duration_results = []

# Loop over the flux values
for frame_duration in frame_duration_values:
    # Update the flux field of the InputParameters instance
    inputParameters.frame_duration = frame_duration

    # Calculate entropy per cm^2 per second
    entropy_per_cm2_per_s = inputParameters.calculate_entropy_per_cm2_per_s()

    entropy_vs_duration_results.append((frame_duration, entropy_per_cm2_per_s))

# Plot the results
PlotEntropyUtilities.plot_entropy_vs_frame_duration(entropy_vs_duration_results)

##############################################################
# Plot the entropy vs pixel area
##############################################################

# Create a fresh instance of the class, reset parameters
inputParameters = InputParameters(flux=1e8, pixel_area=1e-6, frame_duration=1e-7)

# Define a range of pixel area values
pixel_area_values = [1e-6, 1.5e-6, 2e-6, 2.5e-6, 3e-6, 3.5e-6, 4e-6]  # Example range of pixel area values

# Empty list to store results
entropy_vs_pixel_area_results = []

# Loop over the pixel area values
for pixel_area in pixel_area_values:
    # Update the pixel_area field of the InputParameters instance
    inputParameters.pixel_area = pixel_area

    # Calculate entropy per cm^2 per second
    entropy_per_cm2_per_s = inputParameters.calculate_entropy_per_cm2_per_s()

    entropy_vs_pixel_area_results.append((pixel_area, entropy_per_cm2_per_s))

# Plot the results
PlotEntropyUtilities.plot_entropy_vs_pixel_area(entropy_vs_pixel_area_results)