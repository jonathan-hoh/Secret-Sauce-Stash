# Secret-Sauce-Stash
Home to the firmware, software, and hardware guides and downloads needed to run the novel SPLIF (formerly SPLIFF) algorithm on a ROACH2 FPGA

## Making Joseph's life more difficult
So after a great amount of meditation and channeling of my chakras, I have come to the realization that calling the algorithm SPLIFF is just too on the nose. On the other hand, I showed people the same name just without the last letter, and nobody even batted an eye. So, from here on out, I will be refering to the firmware testing suite as the SPLF: Spectroscopic Lock-In Firmware. I'd like to see somebody try to argue with that.

## Firmware Iterations
*In order from most recent to oldest*
1. 244_Hz_SPLIF.fpg -- SPLIF design with maximum depth lookup table of 20-bits. Uses the Sauce Boss Square Transform method to convert both modulation and demodulation signals into square waves. Should perform well
2. golden_spliff_v3.fpg -- I claim that there is a 20-bit LUT as a wave generator here, but it seems its the same as the CORDIC module and therefore should be ignored
3. golden_spliff_v2.fpg -- After successfully implementing the first version of SPLIF with the UCLA system, I was asked to make a version which could run the fundemental frequency generator for the modem signal at a frequemcy resolution of 1 Hz instead of 3900 Hz. Given the limits of the lookup-tables being a depth of 20 bits, the only solution was to use a CORDIC SINCOS digital trigonometry circuit which in theory should have near-infinite frequency resolution. Unfortunately, it also requires a near-infinite amount of my time and energy to properly time with the demodulator, so it is a good avenue to be explored by a future Sauce Lord.
4. golden_spliff_v1.fpg -- The classic. 3.9 KHz of modem resolution tested and proven to at the very least be locking into all of the output bins of interal PFB.
5. liss_gold_enhanced_v1 -- Literally the same file as golden_spliff_v1.fpg (I think?) but with less cool of a name

## Useful Software

- plasmon_readout_v2.py -- I posit that this is an updated version of the plasmon_readout.py script, uploaded by Joseph which contains changes to files and variables so that they play nice with the newer firmware iterations. Im sure he found and fixed other bugs too but I ain't checking. The original version of this file is able to read .fpg outputs from the Xilinx system generator and communicate with the FPGA for basic control and live-plotting as well as long-term data collection functions. 
- hoh_jarrahi_allan_v2.py -- Data analytics script for use with the information collected by the firmware. As the name suggestsm it does contain an Allan variance function to measure detector stability over time. 
