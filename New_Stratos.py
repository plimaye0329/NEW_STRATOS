## STRATOS : Signal Time-domain Research and Analysis Tools for ObservationS

# Import all the dependencies: 
#import psrchive
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.pyplot import figure
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import math
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io import fits
import sigpyproc.readers as readers
import sigpyproc.timeseries as timeseries
import sigpyproc.block as block
from rich.pretty import Pretty
from scipy.signal import find_peaks
from scipy.interpolate import make_interp_spline
import os
import warnings
import math

###### SIGPYPROC Implementation #######

######### Read Filterbank file and normalize ##########

def sigread(filepath, dm, block_size):
    """
    Reads a .fil file and returns the dedispersed data.

    This function reads a .fil file using the FilReader class from the `readers` module. 
    It then dedisperses the data using the provided dispersion measure (dm). It then normalizes
    the data by subtracting the median and dividing by the standard deviation.

    Parameters:
    filepath (str): The path to the .fil file to be read.
    dm (float): The dispersion measure to be used for dedispersion.
    block_size (int): The size of the block to be read from the .fil file.

    Returns:
    numpy.ndarray: The dedispersed data from the .fil file.
    """
    fil = readers.FilReader(filepath)
    header = fil.header
    data = fil.read_block(1, block_size)
    data_dedisp = data.dedisperse(dm)
    arr = data_dedisp.copy()
    
    return arr


########Clean RFI, plot dynamic spectra and scrunch to timeseries##########
def sigclean(data, c1=0, c2 = 0, c3=0, c4=0, c5=0, c6=0, c7=0, c8=0,samp_rate=0.025,filename='cleaned_data.txt'):
    """
    Cleans Radio Frequency Interference (RFI) from the provided data, plots the spectrum, and generates a time series.

    This function identifies bad channels in the data based on the provided ranges (c1 to c8), 
    flags these bad channels as NaN, and then plots the dynamic spectrum with the bad channels flagged.
    It also generates a time series by summing the data in frequency, clipping off the first few and last few bits,
    and then plotting the time series.

    Parameters:
    data (numpy.ndarray): The input data to be cleaned.
    c1, c2, c3, c4, c5, c6, c7, c8 (int): The ranges of bad channels to be flagged.
    samp_rate: The native sampling rate of the raw data (default=0.025ms)
    filename (str): The name of the file to save the cleaned data.

    Returns:
    numpy.ndarray: The cleaned data with bad channels flagged as NaN.
    """
    #Cleaning the RFI channels:
    mask_channel_ranges = [(c1, c2), (c3,c4), (c5,c6),(c7,c8)]  # replace with your actual mask channel ranges

    # Mask the channels
    for start, end in mask_channel_ranges:
        data[start:end, :] = np.nan

    intensity = np.nanmean(data, axis=0)

  
    timeseries = (np.arange(0, len(intensity), 1))*samp_rate 

    timeseries_data = np.vstack((timeseries, intensity)).T


    

    # Save the cleaned data to a text file
    np.savetxt(filename, timeseries_data, delimiter=',')





############Producing downsampled interpolated pulses############

def downsample_profile(profile, factor):
    if factor <= 0:
        raise ValueError("Downsampling factor must be greater than zero.")
    new_length = len(profile) // factor
    reshaped_profile = profile[:new_length * factor].reshape((new_length, factor))
    downsampled_profile = np.mean(reshaped_profile, axis=1)
    return downsampled_profile



def interpolate_timeseries(timeseries_file, new_length):
    """
    Interpolates a timeseries to a new length and saves the interpolated timeseries to a new text file.

    This function loads a timeseries from a txt file, performs cubic interpolation to increase its length to `new_length`,
    and then saves the interpolated timeseries to a new text file with both time and intensity columns.

    Parameters:
    timeseries_file (str): The path to the timeseries txt file.
    new_length (int): The new length for the interpolated timeseries.

    Returns:
    str: The file path for the new interpolated timeseries file.
    """
    # Load the timeseries from the txt file
    timeseries = np.loadtxt(timeseries_file)

    # Generate the x values (time points)
    x = np.arange(len(timeseries))

    # Perform cubic interpolation
    f = interp1d(x, timeseries, kind='cubic')
    xnew = np.linspace(0, len(timeseries)-1, new_length)
    interpolated_timeseries = f(xnew)

    # Save the interpolated timeseries to a new text file with time and intensity columns
    interpolated_timeseries_file = f'Interpolated_timeseries.txt'
    np.savetxt(interpolated_timeseries_file, np.column_stack((xnew, interpolated_timeseries)), header="Time Intensity", fmt="%0.6f",     delimiter="\t")

    return interpolated_timeseries_file


def extract_pulses(ts, it, distance=160, prominence=8, width=200, save_dir='./', filename='B0355'):
    """
    Extract pulses from a time series based on peak prominence and distance.

    Parameters:
    ts (numpy array): The time series data.
    it (numpy array): The intensity data.
    distance (int): Minimum number of samples separating peaks. Default is 160.
    prominence (int): Required prominence of peaks. Default is 8.
    width (int): Number of samples to take on either side of the peak. Default is 200.
    save_dir (str): Directory to save the output files. Default is current directory.
    filename (str): Base filename for the output files. Default is 'B0355'.

    Returns:
    pulses (list): List of numpy arrays representing the extracted pulses.
    """
    def interpolate_pulse_data(pulse, new_length):
        x = np.arange(len(pulse))
        f = interp1d(x, pulse, kind='cubic')
        xnew = np.linspace(0, len(pulse)-1, new_length)
        return f(xnew)

    os.makedirs(save_dir, exist_ok=True)

    # Find peaks in the intensity data
    peaks, _ = find_peaks(it, distance=distance, prominence=prominence)

    # Extract pulses around the peaks
    pulses = [it[i-width:i+width] for i in peaks]

    # Save each pulse and its corresponding time and intensity data
    for i, (pulse, peak) in enumerate(zip(pulses, peaks), start=1):
        if pulse.size == 0:
            continue

        # Interpolate to ensure smooth representation of the pulse
        interpolated_pulse = interpolate_pulse_data(pulse, len(pulse))
        pulse_data = np.column_stack((np.arange(len(interpolated_pulse)), interpolated_pulse))  # Combine time and intensity
        
        # Save pulse data to a file
        np.savetxt(
            os.path.join(save_dir, f'{filename}_P{i}.txt'), 
            pulse_data, 
            header="Time\tIntensity", 
            fmt="%0.6f", 
            delimiter="\t"
        )

    # Save x-axis sample values of the identified peaks
    peak_data = np.column_stack((peaks, ts[peaks]))  # Combine sample indices and corresponding time values
    np.savetxt(
        os.path.join(save_dir, f'{filename}_peaks.txt'), 
        peak_data, 
        header="SampleIndex\tTime", 
        fmt="%d\t%0.6f", 
        delimiter="\t"
    )

    return pulses



##################### Function to Calculate Pulse Fluence by Integrating over the S/N ###############################

def integrated_fluence(pulse_files, constant = 8.5*10**-6, fluence_file = 'integrated_fluences.txt'):
    """
    """
    integrated_fluence = []
    for i, pulse_file in enumerate(pulse_files):
        pulse = np.loadtxt(pulse_file)
        snr = pulse[...,1] # S/N values for each timebin of a pulse profile
        fluence = np.sum(snr)*constant # Saves the Fluence integrated over pulse profile in Jy-s
        integrated_fluence.append(fluence)
    
    np.savetxt(fluence_file, integrated_fluence)
    
######## Energy from Integrated Fluence ###################
def fluence_to_energy(fluence, l = 3e21, sefd = 17, num_polarizations = 2, delv = 400*1e6):
    """
    Calculates the flux, fluence, and energy of each pulse given the pulse width at half maximum (w50) and signal-to-noise ratio (snr).

    Parameters:
    w50 (numpy.ndarray): A 1D array of pulse widths at half maximum in seconds.
    snr (numpy.ndarray): A 1D array of signal-to-noise ratios.
    l (float, optional): The distance to the source in cm. Default is 3e21.
    sefd (int, optional): The system equivalent flux density in Jy. Default is 17.
    num_polarizations (int, optional): The number of polarizations. Default is 2.
    delv (float, optional): The bandwidth in Hz. Default is 250*1e6.

    Returns:
    tuple: A tuple of three 1D numpy arrays. The first array is the flux in Jy, the second array is the fluence in Jy-s, and the third array is the energy in ergs.
    """
    #flux = (sefd * snr) / np.sqrt(num_polarizations * delv * w50)  # Calculate Flux in Jy
    #fluence = flux * w50  # Jy-s
    #fluence_ms = fluence * 1e3  # Jy-ms
    energy = fluence * delv * 10e-23 * 4 * np.pi * l**2  # ergs Calculate energy of the pulse
    return energy



####### Function to calculate the energy of the pulses ######## 

def energy(w50, snr, l = 3e21, sefd = 17, num_polarizations = 2, delv = 250*1e6):
    """
    Calculates the flux, fluence, and energy of each pulse given the pulse width at half maximum (w50) and signal-to-noise ratio (snr).

    Parameters:
    w50 (numpy.ndarray): A 1D array of pulse widths at half maximum in seconds.
    snr (numpy.ndarray): A 1D array of signal-to-noise ratios.
    l (float, optional): The distance to the source in cm. Default is 3e21.
    sefd (int, optional): The system equivalent flux density in Jy. Default is 17.
    num_polarizations (int, optional): The number of polarizations. Default is 2.
    delv (float, optional): The bandwidth in Hz. Default is 250*1e6.

    Returns:
    tuple: A tuple of three 1D numpy arrays. The first array is the flux in Jy, the second array is the fluence in Jy-s, and the third array is the energy in ergs.
    """
    flux = (sefd * snr) / np.sqrt(num_polarizations * delv * w50)  # Calculate Flux in Jy
    fluence = flux * w50  # Jy-s
    fluence_ms = fluence * 1e3  # Jy-ms
    energy = fluence * delv * 10e-23 * 4 * np.pi * l**2  # ergs Calculate energy of the pulse
    return flux, fluence, energy



######### Energy Distribution with errorbars propagated through width uncertainty ###############

def energy_with_uncertainty(w50, w50_err, snr, l=3e21, sefd=17, num_polarizations=2, delv=250*1e6):
    """
    Calculates the flux, fluence, and energy of each pulse, including error bars based on the uncertainties in w50.

    Parameters:
    w50 (numpy.ndarray): A 1D array of pulse widths at half maximum in seconds.
    w50_err (numpy.ndarray): A 1D array of uncertainties in w50.
    snr (numpy.ndarray): A 1D array of signal-to-noise ratios.
    l (float, optional): The distance to the source in cm. Default is 3e21.
    sefd (int, optional): The system equivalent flux density in Jy. Default is 17.
    num_polarizations (int, optional): The number of polarizations. Default is 2.
    delv (float, optional): The bandwidth in Hz. Default is 250*1e6.

    Returns:
    tuple: A tuple containing three 1D numpy arrays: flux (Jy), fluence (Jy-s), energy (ergs), and their respective uncertainties.
    """
    # Flux calculation
    flux = (sefd * snr) / np.sqrt(num_polarizations * delv * w50)
    
    # Uncertainty in flux propagation: σ_flux = |∂flux/∂w50| * σ_w50
    flux_err = (sefd * snr) / (2 * np.sqrt(num_polarizations * delv)) * (-0.5 * w50**(-1.5)) * w50_err

    # Fluence calculation
    fluence = flux * w50  # Jy-s
    fluence_err = np.sqrt((flux_err * w50)**2 + (flux * w50_err)**2)  # Error propagation for fluence

    # Energy calculation
    energy = fluence * delv * 10e-23 * 4 * np.pi * l**2  # ergs
    
    # Uncertainty in energy propagation: σ_energy = |∂energy/∂fluence| * σ_fluence
    energy_err = fluence_err * delv * 10e-23 * 4 * np.pi * l**2
    
    return flux, fluence, energy, energy_err


######### Function to plot energy distribution curves ##########

def plot_energy_distribution(energy_data, num_bins, completeness_limit):
    """
    Plots the cumulative energy distribution for a given energy data.

    Parameters:
    energy_data (numpy.ndarray): The energy data to plot.
    num_bins (int): The number of bins to use in the histogram.
    completeness_limit (float): The energy value above which the data is considered complete.

    Returns:
    None
    """
    # Calculate the histogram
    counts, bin_edges = np.histogram(energy_data, bins=num_bins)

    # Calculate the number of pulses above each energy threshold
    counts_above_threshold = np.cumsum(counts[::-1])[::-1]

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Create a mask where counts_above_threshold is not zero and energy is above the completeness limit
    mask = (counts_above_threshold != 0) & (bin_edges[1:] > completeness_limit)

    # Fit a power-law to the cumulative distribution and plot it
    logx = np.log10(bin_edges[1:][mask])
    logy = np.log10(counts_above_threshold[mask])
    #coeffs = np.polyfit(logx, logy, 1)
    #poly = np.poly1d(coeffs)
    #yfit = lambda x: 10**poly(np.log10(x))
    axs[0].loglog(bin_edges[1:], counts_above_threshold,'ro',linestyle='dashed')
    #axs[0].loglog(bin_edges[1:][mask], yfit(bin_edges[1:][mask]), 'k--', label=f'Power-law index: {coeffs[0]:.2f}')
    axs[0].axvline(x=completeness_limit, label='Completeness limit', color='black', linestyle='--')
    axs[0].set_xlabel('Energy')
    axs[0].set_ylabel('Number of pulses above energy')
    axs[0].legend()
    axs[0].set_title('Cumulative Energy Distribution')

   
    # Display the plot
    plt.tight_layout()
    plt.show()

######### Function to plot energy distribution with uncertainties ##########

def modified_energy_distribution(energy_data, energy_errors, num_bins, completeness_limit):
    """
    Plots the cumulative energy distribution for a given energy data, including uncertainties from energy errors
    and Poissonian uncertainties in the distribution of counts.

    Parameters:
    energy_data (numpy.ndarray): The energy data to plot.
    energy_errors (numpy.ndarray): The uncertainties in the energy data.
    num_bins (int): The number of bins to use in the histogram.
    completeness_limit (float): The energy value above which the data is considered complete.

    Returns:
    None
    """
    # Calculate the histogram (ignoring zero bins)
    counts, bin_edges = np.histogram(energy_data, bins=num_bins)
    
    # Calculate the number of pulses above each energy threshold (cumulative distribution)
    counts_above_threshold = np.cumsum(counts[::-1])[::-1]
    
    # Calculate Poissonian uncertainties on the cumulative counts
    poisson_errors = np.sqrt(counts_above_threshold)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Mask to focus on the part of the distribution that is above the completeness limit
    mask = (counts_above_threshold != 0) & (bin_edges[1:] > completeness_limit)

    # Logarithmic values for fitting and plotting
    logx = np.log10(bin_edges[1:][mask])
    logy = np.log10(counts_above_threshold[mask])

    # Plot cumulative energy distribution with Poissonian error bars
    axs[0].errorbar(bin_edges[1:], counts_above_threshold, yerr=poisson_errors, fmt='ro', linestyle='dashed', 
                    label='Cumulative Counts', capsize=5)

    # Plot the completeness limit line
    axs[0].axvline(x=completeness_limit, label='Completeness limit', color='black', linestyle='--')

    # Set axis labels and title
    axs[0].set_xlabel('Energy')
    axs[0].set_ylabel('Number of pulses above energy')
    axs[0].set_title('Cumulative Energy Distribution')
    axs[0].legend()

    # Plot energy distribution with error bars (energy uncertainties)
    mid_bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    axs[1].errorbar(mid_bin_centers, counts, xerr=energy_errors, fmt='bo', linestyle='none', capsize=5, 
                    label='Energy Data with Errors')

    # Set axis labels and title
    axs[1].set_xlabel('Energy')
    axs[1].set_ylabel('Counts')
    axs[1].set_title('Energy Distribution with Errors')
    axs[1].legend()

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
