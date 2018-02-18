# TESPulseFitCode

Analyze traces from a TES for a CW light source.

It implements a SET_RESET discriminator to identify the pulses, and uses a linear fit to timestamp the detection events.
This improves the precisoin over simple threshold crossing and, more importantly, allos to timestamp mutiple events even when the signal are overlapping.

branches:
1. smallersavgolwindow
Feb 2018
previously we use 301 points to smooth out the differentiated function.
for a sampling rate of 2ns pp, this amounts to 600 ns, about 6x the ri$
which is too much.
it has returned satisfactory results thus far for > 200 ns separated p$
but for 92.4 ns separated pulses the fit result returns +10 ns error.
