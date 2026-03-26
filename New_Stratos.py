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

###### PSRCHIVE Implementation ########

#########Function to plot the dyanamic spectrum from archive file##########

def plotter(filename, bins=1024):
    """
    Load a PSRCHIVE archive file, dedisperse it, remove the baseline, and plot the dynamic spectrum and the time series.

    Parameters:
    filename (str): The path to the archive file to load.
    bins (int, optional): The number of bins to use when generating the time array for plotting the time series. Defaults to 1024.

    Returns:
    tuple: A tuple containing the time array (in milliseconds) and the intensity values as a function of time.

    """
    import psrchive
    archive = psrchive.Archive_load(filename)
    #return archive
    archive.dedisperse()
    archive.remove_baseline()
    #archive.remove_chan(150,200)
    data = archive.get_data()
    #return data
    data = data[0][0]
    bw = archive.get_bandwidth()
    if bw <0:
        bw *= -1.
    freq_lo = archive.get_centre_frequency()-bw/2.0
    freq_hi = archive.get_centre_frequency()+bw/2.0
    #return freq_lo,freq_hi

    
    #Make timeseries of archive
    carch = archive.clone()
    carch.fscrunch_to_nchan(1)
    ts = carch.get_data()
    cropto=False

    if cropto:
        ts=ts[:,:,:,cropto[0]:cropto[1]]
    ts= ts[0][0][0]


    #return TS
    
    
    
    data[0][0]
    fig, ax = plt.subplots(ncols=1,nrows=2,figsize=(8,7),gridspec_kw={
                           'height_ratios': [1, 4]})

    ax[0].plot(ts,'k-')
    ax[0].set_xlim(0, len(data[0]))

    ax[1].imshow(data,extent=(0,archive.get_nbin(),freq_hi, freq_lo),cmap='viridis',aspect='auto',vmax=5,vmin=-5)
    

    plt.tight_layout()


#Generating data arrays:
    time = np.arange(0,bins,1) #Make x-axis to plot timeseries
    it = ts #Intensity Values as a function of time

#Scaling x-axis to be in millisecond units:
    t = (time/bins)*1000 #Time in milliseconds

#Plotting complete timeseries:
    figure(figsize=(7,5),dpi=100)
    plt.plot(t,it,label='single pulse data')
    plt.xlabel('Time(samples)')
    plt.ylabel('Intensity (arb.units)')
    plt.legend()
    plt.show()
    
    
    return data

####Module to clean RFI####
def rfi(data,c1,c2):
    """
    Cleans Radio Frequency Interference (RFI) from the provided data.

    This function identifies bad channels in the data based on the provided range (c1, c2), 
    flags these bad channels, and then replots the dynamic spectrum with the bad channels flagged.

    Parameters:
    data (numpy.ndarray): The input data to be cleaned.
    c1, c2 (int): The range of bad channels to be flagged.

    Returns:
    numpy.ndarray: The cleaned data with bad channels flagged as NaN.
    """
    # Step 1: Identify and flag bad channels
    bad_channels = np.arange(c1,c2,1)  # Replace with the actual bad channels

    # Step 2: Create a binary mask
    mask = np.ones_like(data, dtype=bool)
    mask[bad_channels,:] = False

    # Step 3: Flag bad channels
    data[mask == False] = np.nan  # Set bad channels to NaN

    # Step 4: Replot the dynamic spectrum
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(data, cmap='viridis', aspect='auto', vmax=5, vmin=-5)
    plt.title('Dynamic Spectrum with Bad Channels Flagged')

    #plt.subplot(1, 2, 2)
    #plt.imshow(arc_orig, cmap='viridis', aspect='auto', vmax=5, vmin=-5)
    #plt.title('Original Dynamic Spectrum')

    plt.tight_layout()
    plt.show()
    return data

####Module to scrunch frequencies and plot timeseries####
def timeseries(data):
    """
    Scrunches the frequencies of the provided data and plots the time series.

    This function scrunches the frequencies of the data by calculating the mean along the frequency axis. 
    It then plots the resulting time series.

    Parameters:
    data (numpy.ndarray): The input data to be scrunched and plotted.

    Returns:
    numpy.ndarray: The frequency-scrunched data.
    """
    # Step 4: Scrunch the data in frequency
    scrunched_data = np.nanmean(data, axis=0)  # You can use np.nansum if you prefer summing

    # Step 5: Plot the time series
    plt.figure(figsize=(8, 4))

    plt.plot(scrunched_data)
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Intensity(arbitrary units)')
    #plt.xlim(300,980)
    plt.title('Time Series (Frequency-scrunched)')

    plt.tight_layout()
    plt.show()
    return scrunched_data






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
    #with warnings.catch_warnings():
        #warnings.simplefilter("ignore", category=RuntimeWarning)
        #arr -= np.nanmedian(arr, axis=-1)[..., None]
        #arr /= np.nanstd(arr, axis=-1)[..., None]
        
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

    #plim=[1,95]
    #vmin,vmax = np.nanpercentile(data,plim[0]), np.nanpercentile(data,plim[1])
    #fig = plt.figure()
    #gs = fig.add_gridspec(2, height_ratios=[0.25, 1], hspace=0.1)
    #axs = gs.subplots(sharex=True)
    #axs[1].pcolormesh(data, shading='auto', vmin=vmin, vmax=vmax, rasterized=True)
    # Scrunch the data in frequency
    intensity = np.nanmean(data, axis=0)

    # Clip off the first few and last few bits
    #intensity = intensity[clip_start:-clip_end]

    # Generate the time series
    timeseries = (np.arange(0, len(intensity), 1))*samp_rate 

    # Plot the time series
    #plt.figure(figsize=(7,5), dpi=100)
    #plt.plot(timeseries, intensity, label='single pulse data')
    #plt.xlabel('Time(ms)')
    #plt.ylabel('Intensity (arb.units)')
    #plt.show()

    timeseries_data = np.vstack((timeseries, intensity)).T


    

    # Save the cleaned data to a text file
    np.savetxt(filename, timeseries_data, delimiter=',')

########Function to extract pulses from the timeseries##########

#def extract_pulses(ts, it, distance=160, prominence=600, width=200, save_dir='./', filename='B0355'):
    """
    Extracts and plots pulses from time series data.

    Parameters:
    ts, it (numpy.ndarray): The time series and intensity data.
    distance (int): The minimum number of samples separating each peak.
    prominence (int): The required prominence of the peaks.
    width (int): The number of samples to extract around each peak.
    save_dir (str): The directory to save the plots.
    filename (str): The base filename for the plots.

    Returns:
    list of numpy.ndarray: The extracted pulses.
    """
    # Ensure the save directory exists
 #   os.makedirs('/hercules/results/pral/Python_Notebooks/B0355_singlepulses', exist_ok=True)

    # Find the peaks in the intensity data
  #  peaks, _ = find_peaks(it, distance=distance, prominence=prominence)

    # Extract a range of samples around each peak
   # pulses = [it[i-width:i+width] for i in peaks]

    # Plot each pulse in a separate subplot
    #for i, (pulse, peak) in enumerate(zip(pulses, peaks), start=1):
     #   plt.figure(figsize=(10, 4))  # Create a new figure for each plot
      #  plt.plot(ts[peak-width:peak+width], pulse)
       # plt.title(f'P{i}')  # Use the counter i for the title
        #plt.xlabel('Time (samples)')
        #plt.ylabel('SNR') 
        #plt.savefig(os.path.join(save_dir, f'{filename}_P{i}.jpg'), format='jpg')  # Save the plot in the specified directory
        #plt.close()

    #return pulses


####### Function to eliminate RFI candidates from the list of pulses ########

def true_candidates(pulses, indices_to_remove):
    """
    Eliminates RFI from a list of pulse candidates.

    This function removes specific pulses, which are assumed to be RFI, from the `pulses` list.
    The pulses to be removed are at the indices specified in `indices_to_remove`.

    Parameters:
    pulses (list of numpy.ndarray): The list of pulse candidates.
    indices_to_remove (list of int): The indices of the pulses to remove.

    Returns:
    list of numpy.ndarray: The list of true pulse candidates, with the specified pulses removed.
    """
    true_pulses = [pulse for i, pulse in enumerate(pulses) if i not in indices_to_remove]
    return true_pulses



####### Function to interpolate the pulses ########

#def interpolate_pulses(true_pulses, bins=None, degree=3, filename='B0355'):
    """
    Interpolates a list of pulses using a spline interpolation of a specified degree.

    This function interpolates each pulse in `true_pulses` using a spline interpolation of degree `degree`.
    The number of points in the interpolated pulses is specified by `bins`. If `bins` is None, the number of points
    in the interpolated pulses is the same as the number of points in the original pulses.
    The interpolated pulses are returned as a 3D numpy array.

    Parameters:
    true_pulses (list of numpy.ndarray): The list of pulses to interpolate.
    bins (int, optional): The number of points in the interpolated pulses. Defaults to None.
    degree (int, optional): The degree of the spline interpolation. Defaults to 3.

    Returns:
    numpy.ndarray: A 3D array containing the interpolated pulses.
    """
    # Define the interp function inside interpolate_pulses
    #def interpolate_pulses(true_pulses, bins=None, degree=3):
 #   def interp(x, y, pulse_number):
  #      nonlocal bins
   #     if bins is None:
    #        bins = len(y)
     #   interp_spline = make_interp_spline(x, y, k=degree)
      #  x_interp = np.linspace(x[0], x[-1], bins)
       # y_interp = interp_spline(x_interp)

        #plt.figure(figsize=(5,4), dpi=100)
        #plt.plot(x_interp*0.064, y_interp, color='blue', label='interpolation')
        #plt.plot(x*0.064, y, linestyle='dashed', label='data', color='red')
        #plt.legend()
        #plt.ylabel('SNR')
        #plt.xlabel('Time(samples)')
        #plt.title(f'Pulse P{pulse_number}')
        #plt.savefig(f'B0355_interp_P{pulse_number}.png')
        #plt.close()

        # Save the interpolated pulse to a text file
        #np.savetxt(f'{filename}_{pulse_number}.txt', np.column_stack((x_interp, y_interp)))

        #return np.column_stack((x_interp, y_interp))

    #interpolated_pulses = []

    #for i, pulse in enumerate(true_pulses, start=1):
     #   interp_pulse = interp(np.arange(len(pulse)), pulse, i)
      #  interpolated_pulses.append(interp_pulse)

    #return np.array(interpolated_pulses)




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






######### Function to calculate width and SNR ##########

# w50: finds the fwhm of

def w_50(pulse_files, resol=0.0002, x_start=None, x_end=None, w50_filename='w50_values.txt', snr_filename='snr_values.txt',w50_err_filename='w50_err.txt'):
    """
    pulse_files : Interpolated and extracted single pulse arrays
    resol : Interpolated time resolution of single pulses (Default : 0.0002seconds)
    Note: The width values are in units of seconds and not milliseconds
    """
    w50_values = []
    snr_values = []
    w50_err = []
    
    for i, pulse_file in enumerate(pulse_files):
        pulse = np.loadtxt(pulse_file)
        x = pulse[...,0]
        y = pulse[...,1]

        # Apply the window
        if x_start is not None and x_end is not None:
            mask = (x >= x_start) & (x <= x_end)
            x = x[mask]
            y = y[mask]

        # Calculate FWHM
        ymax = np.max(y)
        w50 = ymax / 2.
        crossings = np.where(np.diff(np.sign(y - w50)))[0]
        if len(crossings) < 2:
            print(f"Warning: Insufficient data to calculate FWHM for pulse file {pulse_file}")
            continue
        left_idx = crossings[0]
        right_idx = crossings[-1] + 1
        fwhm = (x[right_idx] - x[left_idx]) * resol
        
        w50_values.append(fwhm)

        # Calculate SNR
        snr = ymax
        snr_values.append(snr)

        fwhm_err = fwhm/snr
        w50_err.append(fwhm_err)

        # Plot and save the pulse with shaded FWHM region
        plt.plot(x, y)
        plt.axvspan(x[left_idx], x[right_idx], color='gray', alpha=0.5)
        plt.xlabel('Time (samples)')
        plt.ylabel('SNR')
        plt.legend([f'FWHM: {round(fwhm*1000,2)} ms, SNR: {round(snr,2)}'])
        plt.savefig(f'Pulse_fwhm_{i}.png')
        plt.close()

    # Save the FWHM and SNR values to files
    np.savetxt(w50_filename, w50_values)
    np.savetxt(snr_filename, snr_values)
    np.savetxt(w50_err_filename,w50_err)

# Get a list of all pulse files
#pulse_files = glob.glob('B0355_interp_P*.txt')

# w10: finds the pulse width at 10% of the maximum intensity
def w_10(pulse_files, time_resolution=0.025):
    """
    Calculates the full width at ten percent maximum (FWTM) and signal-to-noise ratio (SNR) for a list of pulses.

    This function calculates the FWTM and SNR for each pulse in `pulse_files`. The FWTM is the width of the pulse at ten
    percent of its maximum intensity. The SNR is the maximum intensity of the pulse.

    Each pulse is plotted with the region of the FWTM shaded. The plot is saved as a PNG file with a filename based on
    the index of the pulse in the list.

    Parameters:
    pulse_files (list of str): The list of pulse file paths.
    time_resolution (float): The time resolution of the pulse data.

    Returns:
    tuple: A tuple containing two numpy arrays. The first array contains the FWTM values for each pulse. The second
    array contains the SNR values for each pulse.
    """
    fwtm_values = []
    snr_values = []
    for i, pulse_file in enumerate(pulse_files):
        pulse = np.loadtxt(pulse_file)
        x = pulse[...,0]
        y = pulse[...,1]
        ymax = max(y)
        ten_percent_max = ymax * 0.1
        d = np.sign(ten_percent_max - np.array(y[0:-1])) - np.sign(ten_percent_max - np.array(y[1:]))
        left_idx = np.where(d > 0)[0][0]
        right_idx = np.where(d < 0)[0][-1]
        fwtm = (x[right_idx] - x[left_idx])*time_resolution
        fwtm_values.append(fwtm)

        snr = ymax 
        snr_values.append(snr)

        plt.plot(x, y)
        plt.axvspan(x[left_idx], x[right_idx], color='gray', alpha=0.5)
        plt.xlabel('Time(samples)')
        plt.ylabel('SNR')
        plt.legend([f'FWTM: {fwtm} ms, SNR: {snr}'])
        plt.savefig(f'Pulse_fwtm_{i}.png')
        plt.close()

    # Save the FWTM and SNR values to text files
    np.savetxt('fwtm_values.txt', fwtm_values)
    np.savetxt('snr_values.txt', snr_values)

    return np.array(fwtm_values), np.array(snr_values)


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
