Martin Note:
I took the xcal_raman library located here: https://pypi.python.org/pypi/xcal_raman and modified it to also include
the option to perform sulfur calibrations. To use it, check out the "ApplyCalibrations.py" file.


=====================================
x-axis Raman spectrometer calibration
=====================================

This module provides functions for wavenumber calibration of Raman
spectrometers. Currently it works only with binary SPE files, produced,
for example, by winspec (Princeton Instruments software). Supported
substances so far are polystyrene and cyclohexan (or any other, if you
have a table with Raman shifts and 15 minutes of time).

One of the key features is that the script provides a detailed report of
calibration in PDF format with a lot of plots.

Typical usage looks like this::

    #!/usr/bin/env python

    import xcal_raman as xcal
    import winspec
    cal_f, p = xcal.calibrate_spe("polystyrene.SPE", "dark.SPE",
                                  material="polystyrene", figure=figure())
    savefig("xcal-report-polystyrene.pdf")
    # We have calibration function, lets plot spectrum of some substance
    s = winspec.Spectrum("substance.SPE")
    s.background_correct("dark.SPE")
    plot(cal_f(s.wavelen), s.lum)
    xlabel("Wavenumber, cm$^{-1}$")
    ylabel("Counts")



Thanks to
---------
The SPE files are read by script winspec.py, which was written by James Battat
and Kasey Russell. I modified it and commented out the calibration.


