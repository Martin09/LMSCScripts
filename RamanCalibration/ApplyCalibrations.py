#!/usr/bin/env python

"""
Script which takes in a folder containing some Raman spectra and calibration files (dark, reference and optional
background). It then applies these calibrations to the Raman spectra within that folder and outputs ASCII readable
text files for further analysis, typically in Spectragryph or another software.
"""

import glob
import os
import tkFileDialog

from matplotlib import pyplot as plt

import winspec
import xcal_raman as xcal

# Configuration Constants
DO_BACKGROUND_CORRECTION = False
FOLDER_PROMPT = True

# Folder location
folder = "C:/Users/Martin Friedl/Documents/LMSC/Nanomembranes/Raman/2017-11-10 - D1-16-09-01-A - InAs NWs - Para Pol/Straight NWs"
if FOLDER_PROMPT:
    folder = tkFileDialog.askdirectory(initialdir=folder, title='Please select a directory to analyse')
print("Data folder: " + folder)
os.chdir(folder)

# Calibration file names
calib_dark = "calib_dark.SPE"
calib_ref = "calib_sulfur.SPE"
calib_background = "calib_background.SPE"

# To check the calibration spectrum and adjust seed pixel values
calib = winspec.Spectrum(calib_ref)
plt.plot(calib.lum)

# source: http://www.chem.ualberta.ca/~mccreery/ramanmaterials.html
calibration_peak_seeds = [[62.0, 174.0, 280.0, 705.0],  # Pixel number
                          [85.1, 153.8, 219.1, 473.2]]  # Reference wavenumbers

cal_f, p = xcal.calibrate_spe(calib_ref, calib_dark, seeds_peaks=calibration_peak_seeds, material="sulfur",
                              figure=plt.figure())
plt.savefig("xcal-report-sulfur.pdf")  # Save the calibration report to a PDF file

# Now loop over all spectrum files in the folder
files = map(os.path.basename, glob.glob(folder + "/*.SPE"))  # Grab all spect files in the folder
files = [fn for fn in files if not fn.lower().startswith('calib')]  # Remove any calibration files
for spectfile in files:  # Loop over all spectrum files
    s = winspec.Spectrum(folder + '/' + spectfile)  # Load the winspec file
    s.wavelen = cal_f(s.wavelen)  # Apply the sulfur calibration
    if DO_BACKGROUND_CORRECTION:  # Apply background correction if specified
        s.background_correct("calib_dark.SPE")

    plt.figure()  # Plot the new spectrum
    plt.plot(s.wavelen, s.lum)
    plt.xlabel("Wavenumber, cm$^{-1}$")
    plt.ylabel("Counts")
    plt.title(spectfile)
    plt.grid()
    plt.show()
    plt.savefig(spectfile.split('.')[0] + '_calib.png')  # Save the figure
    # Save the calibrated file as a text file for further processing
    s.save_ascii(spectfile.split('.')[0] + '_calib.txt')
