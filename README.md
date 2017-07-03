# TESPulseFitCode

Analyze traces from a TES for a CW light source.

It implements a SET_RESET discriminator to identify the pulses, and uses a linear fit to timestamp the detection events.
This improves the precisoin over simple threshold crossing and, more importantly, allos to timestamp mutiple events even when the signal are overlapping.
