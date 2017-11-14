# -*- coding: utf-8 -*-
"""
This module makes calibration of Raman spectra using polynomials.

User should provide two SPE files:
 * File with characteristic Raman shifts of some standard medium
 * File with corresponding dark current taken at the same settings

Example use
---
>> import xcal, winspec
>> uncalibrated = winspec.Spectrum("my substance.spe")
>> xcal_func, poly_numbers = xcal.calibrate_spe("polysterene.spe", "bg.SPE")
>> calibrated.x = xcal_func(uncalibrated.wavenum)
>> calibrated.y = uncalibrated.lum
>> pl.plot(x, y); pl.xlabel("Wavenumber, cm$^{-1}$"); pl.ylabel("Counts")

Roman Kiselev, 30.07.2014
GNU GPL v3
"""

import matplotlib
from matplotlib import pylab as pl
import numpy as np
from scipy import optimize
from scipy.signal import argrelmin
from scipy.ndimage import gaussian_filter1d
import winspec


def _set_size_mm(f, size):
    """ Change size of the figure to match the specified dimensions. """
    w, h = size
    w = w / 25.4
    h = h / 25.4
    f.set_size_inches((w,h))


def _find_nearest(array, value):
    """ Return the index of the nearest element. """
    return (np.abs(array-value)).argmin()


def _label_peak(x, y, text):
    """ Place a wavenumber label with an arrow near the peak. """
    pl.annotate(text, (x, y), xycoords='data', xytext=(0,25),
             fontsize=7, textcoords='offset points',
             horizontalalignment="center", verticalalignment="bottom",
             arrowprops=dict(arrowstyle="->", color="k", lw=0.7))


def _fit_peak(seed, pixels, lum, peaks, ax_fitting=None, smooth=0.6, window=30):
    """
    Actually perform the fitting of a peak.
    
    How does it work?
    ---
    We slice a small range of data, which contains only points between two
    local minima (a single peak). This data points are approximated with
    a 3-rd order polynom plus gaussian. The peak value of this function (in
    the considered interval!), and the corresponding argument value, are
    returned.
    """
    # First step is to crop our data to a small window, which includes only one
    # or two peaks. We take data points, which are +- win_width points away
    # from the seed position.
    win_width = window # window size to crop the data - number of datapoints
    seed_idx = _find_nearest(pixels, seed)  # coordinate of center
    window = range(seed_idx - win_width/2, seed_idx + win_width/2 + 1)
    x = pixels[window]
    y = lum[window]
    seed_idx = win_width/2  # new coordinate of the center after the crop
    

    # The peaks can be of totally different heigth. To be able to plot them
    # nicely, we scale each of them by a factor "yscale_factor", which
    # is equal to the area under the peak. The amplitude found by the
    # fitting will be rescaled back, so it doesn't affect anything
    yscale_factor = sum(y)
    y = y / yscale_factor


    # Create a bit smoothed curve (used only to find local minima); smoothing
    # reduces errors due to the noise 
    y_smoothed = gaussian_filter1d(y, smooth)

    # make a plot of initial data, smoothed data, and seed points
    if ax_fitting:
        pl.sca(ax_fitting)

        # we shift each plot vertically to prevent overlap
        stp = 0.01 # Step between data and smoothed data
        # Now comes a trick. We place into the axes a custom attribute - peak
        # number. We increment it each time this function is called.
        try:
            ax_fitting.n += 1
        except AttributeError:
            ax_fitting.n = 0
        peak_no = ax_fitting.n
        yshift = stp*(4 + 22*(peak_no % 2))  # odd go down, even go up!
        
        # create new x-coordinate, used only for plotting. It's middle point is
        # equal to the peak number (very convenient)
        xplt = np.linspace(peak_no, peak_no + 2, len(x)+6)[3:-3]
        
        # plot initial data
        pl.plot(xplt, y + yshift,                "+b-", ms=2, lw = 0.5,  alpha=0.5)
        # plot smoothed data
        pl.plot(xplt, y_smoothed + yshift - stp, "+k-", ms=2, lw = 0.75, alpha=0.2)
        # place a seed point
        pl.plot(np.mean(xplt), y[seed_idx] + yshift, "xr", ms=12)

        # Format the plot
        ax_fitting.xaxis.set_ticks(range(peak_no + 2))
        ax_fitting.yaxis.set_visible(False)
        pl.grid(True, axis="x", lw=0.2, linestyle="--", alpha=0.35)
        pl.xlabel("Peak number\n" +  
               "blue line - data; gray line - smoothed data; " + 
               "red X - seed points; green points - data values used for " +
               "fitting; red line - fitting function")
        pl.title("Fitting of individual peaks: results")


    # Our next step towards the good fitting is to select only a single peak.
    # To do that, we go left and right from the seed point until we reach the
    # nearest local minima. Only this region between local minima is kept.

    # Find all local minima in a bit smoothed spectrum (noise reduction)
    minima = argrelmin(y_smoothed)[0]

    # Exclude all minima that are very close to the seed point.
    keep_points = 2  # How many points should we keep
    minima = minima[np.where(abs(minima - seed_idx) > keep_points)]

    # Split all found minima in two: left and right
    idx_left  = minima[np.where(minima < seed_idx)]
    idx_right = minima[np.where(minima > seed_idx)]

    # check if they are not empty and take just one nearest element
    if any(idx_left):
        idx_left = idx_left[-1]
    else:
        idx_left = 0
    if any(idx_right):
        idx_right = idx_right[0]
    else:
        idx_right = len(y) - 1

    # Actually crop the unsmoothed data and select a peak between two minima
    window = range(idx_left, idx_right + 1)
    y = y[window]
    x = x[window]
    if ax_fitting:
        xplt = xplt[window]
        # plot cropped data
        pl.plot(xplt, y + yshift, "og", ms=1.5, lw = 0.8)

    # We perform fitting with a sum of Gaussian and a 3-rd order polynom.
    # The FWHM and height of the Gaussian are limited.
    minWidth, maxWidth, minHeight = 0.8, 8, 0.2*y.ptp()
    gauss = lambda x, p: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)
    # We define fitting and error function
    fitfunc = lambda x, p: gauss(x, p[:3]) + np.polyval(p[3:], x)
    def errfunc (p, x, y):
        # check gaussian width and height
        if p[2] < minWidth or p[2] > maxWidth or p[0] < minHeight:
            return np.inf
        else:
            return fitfunc(x, p) - y
    # Initial guess  |----for Gaussian----|   |for polynom|
    p =        np.array([y.max(), x.mean(), 4,    0, 0, 0, 0])
    # Optimization
    p, dummy = optimize.leastsq(errfunc, p[:], args=(x, y), ftol=1e-2)

    # We calculate walues of the fitting function to find it peak position
    # and peak amplitude, as well as to plot it.
    N = win_width*10
    fit_x = np.linspace(min(x), max(x), N)
    amplitude = fitfunc(fit_x, p).max()
    peak_position = fit_x[np.where(amplitude == fitfunc(fit_x, p))]
    if ax_fitting:
        xplt_fit = np.linspace(min(xplt), max(xplt), N)
        pl.plot(xplt_fit, fitfunc(fit_x, p) + yshift, "r", lw=0.7)

        # We label each peak on the plot
        xplt_peak_idx = _find_nearest(fit_x, peak_position)
        lbl_text = "%.1f cm$^{-1}$\n %.1f px " % (peaks[peak_no], peak_position)
        _label_peak(xplt_fit[xplt_peak_idx], amplitude + yshift, lbl_text)
        ax_fitting.set_ymargin(0.05)

    # We return peak position and rescaled amplitude
    return peak_position, amplitude*yscale_factor



def calibrate_spe(spe_sample, spe_dark, poly_order=3, material="polystyrene",
                  seeds_peaks=None, figure=None, shift=0):
    """
    Function reads spectra from SPE file and performs calibration of x-axis.

    Options
    ---
     * spe_sample  - SPE file with RAMAN spectrum of a standard sample.
     * spe_dark    - SPE file with corresponding dark current.
     * poly_order  - order of polynomial used for fitting. Default is 3.
     * material    - which standard material is used. Can be one of:
                     "cyclohexane" - NOT_IMPLEMENTED
                     "naphthalene" - NOT_IMPLEMENTED
                     "paracetamol" - NOT_IMPLEMENTED
                     "polystyrene" - default
     * seeds_peaks - two-row table. First row contains N pixel numbers, in
                     vicinity of which the characteristic peaks are observed.
                     Second row contains N corresponding Raman shifts from
                     standard (ASTM E1840-96). If option seeds_peaks is
                     specified, it overrides material option. Default is None.
     * figure      - matplotlib figure to make a calibration report onto it.
                     Will be rescaled to A4 size and populated with plots.
                     Default is None.
     * shift       - shift the spectrum to the left or to the right by the
                     specified amount of pixels. Useful if the spectrometer
                     has non-standard wavelength calibration

    Returns
    ---
     * function xcal(px) - correspondence between wavenumbers and pixels
     * peaks - array of (position/amplitude) tuples - result of fitting.
               The polynom can be obtained as poly1d(peaks)
    """
    # We create supblots on the figure for future use
    if figure:
        font = {'size': 8}
        matplotlib.rc('font', **font)

        _set_size_mm(figure, (297, 210))
        ax_spectrum = pl.subplot2grid(shape=(10, 9), loc=(0, 0),
                                     colspan=3, rowspan=4)
        ax_xcalline = pl.subplot2grid(shape=(10, 9), loc=(0, 3),
                                     colspan=3, rowspan=4)
        ax_info     = pl.subplot2grid(shape=(10, 9), loc=(0, 6),
                                     colspan=3, rowspan=4)
        ax_fitting  = pl.subplot2grid(shape=(10, 9), loc=(4, 0),
                                     colspan=9, rowspan=6)
    else:
        ax_spectrum, ax_xcalline, ax_fitting = None, None, None


    # Read the data from files
    sample = winspec.Spectrum(spe_sample)
    sample.background_correct(spe_dark)
    pixels = sample.wavelen
    lum = sample.lum


    # Select the material
    if seeds_peaks:  # Use data supplied by user
        seeds = seeds_peaks[0]
        peaks = seeds_peaks[1]
    else:
        material = material.strip().lower()
        if material == "cyclohexane":
            seeds = np.array([34, 43, 122, 174, 206, 233,
                                       279, 667, 739, 765, 775])
            peaks = np.array([384.1, 426.3, 801.3, 1028.3, 1157.6, 1266.4,
                                       1444.4, 2664.4, 2852.9, 2923.8, 2938.3])
        elif material == "naphthalene":
            raise NotImplementedError("naphthalene is not supported yet. sorry")
        elif material == "paracetamol":
            raise NotImplementedError("paracetamol is not supported yet. sorry")
        elif material == "polystyrene":
            seeds = np.array([87, 125, 173, 180, 209, 285,
                                       320, 328, 747, 769, 830])
            peaks = np.array([620.9, 795.8, 1001.4, 1031.8, 1155.3, 1450.5, 
                                       1583.1, 1602.3, 2852.4, 2904.5, 3054.3])
        elif material == "sulfur":
            seeds = np.array([50.0, 85.1, 153.8, 219.1, 473.2])
            peaks = np.array([50.0, 85.1, 153.8, 219.1, 473.2])		
        else:
            raise ValueError('Material "%s" is unknown. At the ' % material + 
                             'same time the material resonances and search ' + 
                             'points (option "seeds_peaks") are not ' + 
                             'specified. Nothing to do.')
        seeds = seeds + shift

    if figure:
        pl.sca(ax_spectrum)
    if figure:
        dummy = sample.plot("b", lw=0.75, label="data")
        if seeds_peaks:
            pl.title("Spectrum, bg corrected")
        else:
            pl.title("Spectrum of %s, bg corrected" % material)
        pl.xlim(pixels.min(), pixels.max())
        pl.plot(seeds, [sample.lum[_find_nearest(pixels, seed)] for seed in seeds],
             "xr", ms=8, label="seed points")
        pl.legend(loc="upper right", fontsize=8)
        ax_spectrum.margins(0.05)


    # Next step is to make fitting for each seed point
    positions  = np.array([])
    amplitudes = np.array([])
    for seed in seeds:
        try:
            pos, amp = _fit_peak(seed, pixels, lum, peaks, ax_fitting)
        except NotImplementedError:
            print "Fitting at seed point %2.1f failed!" % seed
            pos, amp = -1, -1
        positions =np.append(positions, pos)
        amplitudes =np.append(amplitudes, amp)


    # Now we clean our arrays, because fitting could fail at some points
    failed_idx, = np.where(positions == -1)
    positions   =np.delete(positions,  failed_idx)
    amplitudes  =np.delete(amplitudes, failed_idx)
    peaks       =np.delete(peaks,      failed_idx)
    if failed_idx:
        print "Fitting failed in %i out of %i points" % (len(failed_idx),
                                                         len(peaks))

    # Next step is to make a polynomial fitting of these points
    pts_count = len(peaks) - len(failed_idx)
    if pts_count < 4:
        raise ValueError("Too little points (%i) for calibration!" % pts_count)
    else:
        xcal_func = lambda p, x: np.polyval(p, x)
        errfunc = lambda p, x, y: xcal_func(p, x) - y
        p = np.zeros(poly_order + 1); p[-1] = 200; p[-2] = 1
        p, dummy = optimize.leastsq(errfunc, p[:], args=(positions, peaks))
        p_rounded = [float("%.2e" % x) for x in p]
        pnom = str(np.poly1d(p_rounded)).splitlines()
        pnom = "       " + pnom[0] + "\nf(x) = " + pnom[1]
        print "Calibration function:"
        print pnom

        # Check quality of calibration by looking at residuals
        residuals = xcal_func(p, positions) - peaks
        print "Standard deviation of residuals: %.5f" % np.std(residuals)
        if any(abs(residuals) > 8.0):
            print "\nSome residuals exceed 8 cm⁻¹! The calibration " +\
                  "is probably incorrect."
            for i, res in enumerate(residuals):
                if abs(res) > 8.0:
                    print u"Residual[%i] = %.2f cm⁻¹" % (i, res)


    # We have everything and make a calibration plot
    if figure:
        x = np.linspace(pixels.min(), pixels.max(), 2000)
        pl.sca(ax_xcalline)
        pl.plot(positions, peaks, "b+", ms=8, label="peaks")
        pl.plot(x, xcal_func(p, x), "r-", lw=0.8, label="calibration function")
        pl.plot([],[], "gx", ms=6, label="residuals")
        pl.legend(loc="upper left", fontsize=8)
        pl.grid(True)
        pl.xlabel("Pixel number")
        pl.ylabel("Wavenumber, cm$^{-1}$")
        pl.ylim(-1200, 3500)
        pl.xlim(pixels.min(), pixels.max())
        pl.title("Calibration function")

        # plot residuals on the same plot
        ax_resid = ax_xcalline.twinx()

        ax_resid.stem(positions, residuals, linefmt='-g', markerfmt='gx', ms=6,
                      basefmt="w.", bottom=0)
        ax_resid.plot(pl.gca().get_xlim(), [0,0], 'k-', lw=0.5, alpha=0.7)
        ax_resid.set_ylabel('Residuals, cm$^{-1}$', color='g')
        ax_resid.set_ylim(-12,35)
        pl.xlim(pixels.min(), pixels.max())
        for tl in ax_resid.get_yticklabels():
            tl.set_color('g')

        txt=pl.text(0.02, 0.68, pnom, fontsize=7,
                   horizontalalignment="left",
                   verticalalignment='center', alpha=1.0,
                   transform=ax_xcalline.transAxes,
                   bbox=dict(facecolor='white', edgecolor="white", alpha=0.6))
        txt.set_family("monospace")


        # place file information on the report page
        pl.sca(ax_info)
        ax_info.xaxis.set_visible(False)
        ax_info.yaxis.set_visible(False)
        pl.title("File information")
        infotext  = "accumul. = %i\n" % sample.accumulations
        infotext += "exposure = %.4f\n" % sample.exposure
        infotext += "filename = %s\n\n" % sample.fname
        infotext += sample.fileinfo
        txt=pl.text(0.02, 0.98, infotext, fontsize=7, horizontalalignment="left",
                   verticalalignment='top')
        txt.set_family("monospace")


        pl.tight_layout(pad=4, h_pad=1, w_pad=0)
    return lambda x: xcal_func(p, x), p











