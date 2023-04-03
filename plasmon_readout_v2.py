# -*- coding: utf-8 -*-
"""
@author : Jonathan Hoh
Initial template for photo-mixer readout on ROACH2 system with 512MHz digizers

"""
import casperfpga
import time
import matplotlib.pyplot as plt
import struct
import numpy as np
import scipy.stats as stats
import csv
from csv import writer
# Establish parameters and constant based on hardware design
f_s = 500000.0 # Sampling freq in Hertz
fft_len = 2**9

katcp_port=7147
roach = '192.168.40.79'
#firmware_fpg = 'liss_gold_v1.fpg'
firmware_fpg = 'liss_gold_enhanced_v1.fpg'
fpga = casperfpga.katcp_fpga.KatcpFpga(roach, timeout = 3.)
time.sleep(1)
if (fpga.is_connected() == True):
	print ('Connected to the FPGA ')
else:
	print ('Not connected to the FPGA')

stime = time.time()
fpga.upload_to_ram_and_program(firmware_fpg)
print ('Using the Proprietary Big Sauce Boss Black-Magic Optimization Algorithm \nFPGA @ address %s programmed in %.2f seconds \n\n Jons withcraft reduced program time by %.2f seconds' % (fpga.host, time.time() - stime, 31.89*(time.time() - stime)))
time.sleep(1)
print ('\n NAND Gate Flash Success')
time.sleep(1)
print('\nFirware ready for execution')
# Initializing registersr
time.sleep(1)
#print()
#print('Some Availabe Functions (more available if you examine the code):')
#print('1. plot_ADC == plots timestream of signal currently being sampled by the ADC')
#print('2. plot_FFT() == shows live updating FFT output in frequency spSSSSace')
#print('3. plot_Accum() == shows live updating accumulator power in frequency space')
#print('4. bin_reading(bin, cycles) == returns accumulated power in chosen bin,\n averaged by the number of accumulations chosen')
#print('5. dataCollectSimp(bin, lines) == takes data of selected spectral channel and saves as CSV \n (a "line" is equivalent to aboout 10 seconds of integration)')
#print('6. dataCollect4Chan(chan1, chan2, chan 3, chan 4, lines) == \n same as before but can choose 4 bins to measure ')

fpga.write_int('fft_shift', 2**10-1)
fpga.write_int('cordic_freq',1 ) # 
fpga.write_int('cum_trigger_acceum_len', 2**24-1) # 2**24/2**9 =2**15
fpga.write_int('cum_trigger_accum_reset', 0) # 
fpga.write_int('cum_trigger_accum_reset', 1) #
fpga.write_int('cum_trigeeger_accum_reset', 0) #
fpga.write_int('start_dac', 0) #
fpga.write_int('start_dac', 1) #

plt.ion()

def plotFFT():
		fig = plt.figure()
		plot1 = fig.add_subplot(111)
		line1, = plot1.plot(np.arange(0,1024,2), np.zeros(1024/2), '#FF4500', alpha = 0.8)
		line1.set_marker('.')
		plt.grid()
		plt.ylim(-10, 100)
		plt.tight_layout()
		count = 0
		stop = 1.0e6
		while(count < stop):
			fpga.write_int('fft_snap_fft_snap_ctrl',0)
			fpga.write_int('fft_snap_fft_snap_ctrl',1)
			fft_snap = (np.fromstring(fpga.read('fft_snap_fft_snap_bram',(2**9)*8),dtype='>i2')).astype('float')
			I0 = fft_snap[0::4]
			Q0 = fft_snap[1::4]
			mag0 = np.sqrt(I0**2 + Q0**2)
			mag0 = 20*np.log10(mag0)
			# mag0 is the current array of the FFT channels. Choose a single element from the array and
			# plot over time to see timestream (choose elements from an array in python with [] instead 
			# of the () you would use in Matlab)
			line1.set_ydata(mag0)
			fig.canvas.draw()
			count += 1
		return

def reset_accum_len(): #resets hardware accumulator back to its maximum value
	fpga.write_int('cum_trigger_accum_len', 2**24-1)
	return

def plotAccum():
		# Generates a plot stream from read_avgIQ_snap(). To view, run plotAvgIQ.py in a separate terminal
		fig = plt.figure(figsize=(10.24,7.68))
		plt.title('TBD, Accum. Frequency = ')# + str(accum_freq), fontsize=18)
		plot1 = fig.add_subplot(111)
		line1, = plot1.plot(np.arange(506),np.ones(506), '#FF4500')
		line1.set_linestyle('None')
		line1.set_marker('.')
		plt.xlabel('Channel #',fontsize = 18)
		plt.ylabel('dB',fontsize = 18)
		plt.xticks(np.arange(0,506,50))
		plt.xlim(0,506)
		#plt.ylim(-40, 5)
		plt.ylim(80, 150)
		plt.grid()
		plt.tight_layout()
		plt.show(block = False)
		count = 0
		stop = 10000
		while(count < stop):
			I, Q = read_accum_snap()
			I = I[2:]
			Q = Q[2:]
			mags =(np.sqrt(I**2 + Q**2))[2:508]
			#mags = 20*np.log10(mags/np.max(mags))[:1016]
			mags = 20*np.log10(mags[:]+1e-20)
			line1.set_ydata(mags)
			fig.canvas.draw()
			count += 1
		return

def read_accum_snap():
		# 2**9 64bit wide 32bits for mag0 and 32bits for mag1    
		fpga.write_int('accum_snap1_accum_snap_ctrl', 0)
		fpga.write_int('accum_snap1_accum_snap_ctrl', 1)
		#This is how you get the raw data from the accumulator RAM block
		accum_data = np.fromstring(fpga.read('accum_snap1_accum_snap_bram', 16*2**9), dtype = '>i').astype('float')
		I = accum_data[0::2]
		Q = accum_data[1::2]
		return I, Q

def bin_reading(chan, avg_samples):
	integrator= np.zeros(avg_samples)
	count = 0
	mags = np.zeros(506)
	# we use 506 here instead of 512 because I am removing 
	# the edge bins of the spectrum as they get filled with 
	# low-frequency noise and other pesky information
	while count < avg_samples:
		I, Q = read_accum_snap()
		I = I[2:]
		Q = Q[2:]
		mags += (np.sqrt(I**2 + Q**2))[2:508]
		count += 1	
	avg_mag = integrator[chan-4]/avg_samples
	avg_mag_dB = 20*np.log10(avg_mag+1e-20)
	return avg_mag_dB

def neighbor_test(center):
	distant_low = bin_reading(center-10, 160)
	lowest_mag = bin_reading(center-2, 160)
	low_mag = bin_reading(center-1, 160)
	center_mag = bin_reading(center, 160)
	high_mag = bin_reading(center+1, 160)
	highest_mag = bin_reading(center+2, 160)
	distant_high = bin_reading(center+10, 160)
	print('Accumutes powers (10 seconds) are:')
	print([distant_low, lowest_mag, low_mag, center_mag, high_mag, highest_mag, distant_high])
	print("Respectively for bins:")
	print([center-10, center-2, center-1, center, center+1, center+2, center+10])
	return

def leak_testing(bin2, pwr_in, avg_samples):
	with open('bin_%d_100_dBm.csv'%(bin2), 'a') as f_object:
		writer_object = writer(f_object)
		bin1 = bin2 - 1 
		bin3 = bin2 + 1 
		i = 0
		integrator1= np.zeros(avg_samples)
		integrator2= np.zeros(avg_samples)
		integrator3= np.zeros(avg_samples)
		while i < avg_samples:
			I,Q = read_accum_snap()
			I = I[2:]
			Q = Q[2:]
			mag1 =(np.sqrt(I**2 + Q**2))[bin1]
			mag2 =(np.sqrt(I**2 + Q**2))[bin2]
			mag3 =(np.sqrt(I**2 + Q**2))[bin3]
			integrator1[i] = mag1
			integrator2[i] = mag2
			integrator3[i] = mag3
			i += 1
		avg_mag1 = np.average(integrator1)
		avg_mag2 = np.average(integrator2)
		avg_mag3 = np.average(integrator3)
		powers = [pwr_in, avg_mag1, avg_mag2, avg_mag3]
		writer_object.writerow(powers)
		f_object.close()
		print('*** Power averaged over %d samples ***'%(avg_samples))
		print('bin %d average power: %f'%(bin1, avg_mag1))
		print('bin %d average power: %f'%(bin2, avg_mag2))
		print('bin %d average power: %f'%(bin3, avg_mag3))
	return 

def plotADC():
		# Plots the ADC timestream
		fig = plt.figure(figsize=(10.24,7.68))
		plot1 = fig.add_subplot(211)
		line1, = plot1.plot(np.arange(0,2048), np.zeros(2048), 'r-', linewidth = 2)
		plot1.set_title('I', size = 20)
		plot1.set_ylabel('mV', size = 20)
		plt.xlim(0,1024)
		plt.ylim(-600,600)
		plt.yticks(np.arange(-600, 600, 100))
		plt.grid()
		plot2 = fig.add_subplot(212)
		line2, = plot2.plot(np.arange(0,2048), np.zeros(2048), 'b-', linewidth = 2)
		plot2.set_title('Q', size = 20)
		plot2.set_ylabel('mV', size = 20)
		plt.xlim(0,1024)
		plt.ylim(-600,600)
		plt.yticks(np.arange(-600, 600, 100))
		plt.grid()
		plt.tight_layout()
		plt.show(block = False)
		count = 0
		stop = 1.0e8
		while count < stop:
			time.sleep(0.1)
			fpga.write_int('adc_snap_adc_snap_ctrl', 0)
			time.sleep(0.1)
			fpga.write_int('adc_snap_adc_snap_ctrl', 1)
			time.sleep(0.1)
			fpga.write_int('adc_snap_adc_snap_ctrl', 0)
			time.sleep(0.1)
			fpga.write_int('adc_snap_adc_snap_trig', 1)
			time.sleep(0.1)
			fpga.write_int('adc_snap_adc_snap_trig', 0)
			time.sleep(0.1)
			adc = (np.fromstring(fpga.read('adc_snap_adc_snap_bram',(2**10)*8),dtype='>h')).astype('float')
			adc /= (2**15)
			adc *= 550.
			I = np.hstack(zip(adc[0::4],adc[2::4]))
			Q = np.hstack(zip(adc[1::4],adc[3::4]))
			line1.set_ydata(I)
			line2.set_ydata(Q)
			fig.canvas.draw()
			count += 1
		return

def plotADC_avg(samps):
		# Plots the ADC timestream
		fig = plt.figure(figsize=(10.24,7.68))
		plot1 = fig.add_subplot(211)
		line1, = plot1.plot(np.arange(0,2048), np.zeros(2048), 'r-', linewidth = 2)
		plot1.set_title('I', size = 20)
		plot1.set_ylabel('mV', size = 20)
		plt.xlim(0,1024)
		plt.ylim(-600,600)
		plt.yticks(np.arange(-600, 600, 100))
		plt.grid()
		plot2 = fig.add_subplot(212)
		line2, = plot2.plot(np.arange(0,2048), np.zeros(2048), 'b-', linewidth = 2)
		plot2.set_title('Q', size = 20)
		plot2.set_ylabel('mV', size = 20)
		plt.xlim(0,1024)
		plt.ylim(-600,600)
		plt.yticks(np.arange(-600, 600, 100))
		plt.grid()
		plt.tight_layout()
		plt.show(block = False)
		count = 0
		stop = 1.0e8
		while count < stop:
			count2 = 0
			adc = np.zeros(len(np.fromstring(fpga.read('adc_snap_adc_snap_bram',(2**10)*8),dtype='>h')).astype('float'))
			while count2 < samps:
				time.sleep(0.1)
				fpga.write_int('adc_snap_adc_snap_ctrl', 0)
				time.sleep(0.1)
				fpga.write_int('adc_snap_adc_snap_ctrl', 1)
				time.sleep(0.1)
				fpga.write_int('adc_snap_adc_snap_ctrl', 0)
				time.sleep(0.1)
				fpga.write_int('adc_snap_adc_snap_trig', 1)
				time.sleep(0.1)
				fpga.write_int('adc_snap_adc_snap_trig', 0)
				time.sleep(0.1)
				adc_in = (np.fromstring(fpga.read('adc_snap_adc_snap_bram',(2**10)*8),dtype='>h')).astype('float')
				adc_in /= (2**15)
				adc_in *= 550.
				adc += adc_in
				count2 += 1
			I = np.hstack(zip(adc[0::4],adc[2::4]))
			Q = np.hstack(zip(adc[1::4],adc[3::4]))
			line1.set_ydata(I)
			line2.set_ydata(Q)
			fig.canvas.draw()
			count += 1
		return

def plotAccum_avg(samps):
		# Generates a plot stream from read_avgIQ_snap(). To view, run plotAvgIQ.py in a separate terminal
		fig = plt.figure(figsize=(10.24,7.68))
		plt.title('TBD, Accum. Frequency = ')# + str(accum_freq), fontsize=18)
		plot1 = fig.add_subplot(111)
		line1, = plot1.plot(np.arange(506),np.ones(506), '#FF4500')
		line1.set_linestyle('None')
		line1.set_marker('.')
		plt.xlabel('Channel #',fontsize = 18)
		plt.ylabel('dB',fontsize = 18)
		plt.xticks(np.arange(0,506,50))
		plt.xlim(0,506)
		#plt.ylim(-40, 5)
		plt.ylim(70, 145)
		plt.grid()
		plt.tight_layout()
		plt.show(block = False)
		count = 0		
		stop = 10000
		while(count < stop):
			count2 = 0
			mags_dB = np.zeros(506)
			while count2 < samps:
				I, Q = read_accum_snap()
				I = I[2:]
				Q = Q[2:]
				mags =(np.sqrt(I**2 + Q**2))[2:508]
				#mags = 20*np.log10(mags/np.max(mags))[:1016]
				mags_dB += 20*np.log10(mags[:]+1e-20)
				count2 += 1
			mags_dB /= samps
			max_pow = np.max(mags_dB)
			noise_pow = np.median(mags_dB)
			snr = max_pow-noise_pow
			line1.set_ydata(mags_dB)
			fig.canvas.draw()
			print('SNR is %f dB'%(snr))
			count += 1            
		return

def plotAccum_avg_BPF(samps):
		# Generates a plot stream from read_avgIQ_snap(). To view, run plotAvgIQ.py in a separate terminal
		fig = plt.figure(figsize=(10.24,7.68))
		plt.title('TBD, Accum. Frequency = ')# + str(accum_freq), fontsize=18)
		plot1 = fig.add_subplot(111)
		line1, = plot1.plot(np.arange(506),np.ones(506), '#FF4500')
		line1.set_linestyle('None')
		line1.set_marker('.')
		plt.xlabel('Channel #',fontsize = 18)
		plt.ylabel('dB',fontsize = 18)
		plt.xticks(np.arange(0,506,50))
		plt.xlim(0,506)
		#plt.ylim(-40, 5)
		plt.ylim(70, 145)
		plt.grid()
		plt.tight_layout()
		plt.show(block = False)
		count = 0		
		stop = 10000
		while(count < stop):
			count2 = 0
			mags_dB = np.zeros(506)
			while count2 < samps:
				I, Q = read_accum_snap()
				I = I[2:]
				Q = Q[2:]
				mags =(np.sqrt(I**2 + Q**2))[2:508]
				#mags = 20*np.log10(mags/np.max(mags))[:1016]
				mags_dB += 20*np.log10(mags[:]+1e-20)
				count2 += 1
			mags_dB /= samps
			max_pow = np.max(mags_dB)
			max_loc = np.argmax(mags_dB)
			noise_pow = np.mean([mags_dB[max_loc-3],mags_dB[max_loc-4],mags_dB[max_loc+3],mags_dB[max_loc+4]])
			snr = max_pow-noise_pow
			line1.set_ydata(mags_dB)
			fig.canvas.draw()
			print('SNR is %f dB'%(snr))
			count += 1            
		return

def findmaxbin(int_time = (1/16)):
	count = 0
	mags = np.zeros(506)
	# we use 506 here instead of 512 because I am removing 
	# the edge bins of the spectrum as they get filled with 
	# low-frequency noise and other pesky information
	while count < int_time*16:
		I, Q = read_accum_snap()
		I = I[2:]
		Q = Q[2:]
		mags += (np.sqrt(I**2 + Q**2))[2:508]
		count += 1
	#mags = 20*np.log10(mags/np.max(mags))[:1016]
	# It is pointless looking at all 1016 channels, especially since it is a single-ended mixer internally 
	mags = mags[:]/count
	mags = 20*np.log10(mags+1e-20)
	max_bin = np.argmax(mags)+4
	max_val = np.max(mags)
	#print('Maximum power of %d dBW at bin %d'%(max_val, max_bin))
	return max_val, max_bin

def dataCollectSimp(chan, lines):
	# In its current iteration, 10 seconds of data are printed per line
	seconds_per_line = 10
	count1 = 0
	rate = 16
	file = open('spec_data_%d.csv'%(chan), 'w')
	writer = csv.writer(file)
	
	cols = rate * seconds_per_line
	tau = np.logspace(-1, 3, 50)
	writer.writerow([chan])
	
	while (count1 < lines):
		print('we are %d/%d of the way through this shit'%(count1,lines))	    
		vals = np.zeros(cols)
		count2 = 0
		while (count2 < cols):
			I, Q = read_accum_snap()
			I = I[2:]
			Q = Q[2:]
			mag =(np.sqrt(I**2 + Q**2))[chan]
			accum_data = 10*np.log10(mag+1e-20)
			vals[count2] = accum_data
			#print('this is column number %d with a value of %d'%(count2, val))
			count2 += 1
		writer.writerow(vals)
		count1 += 1
	file.close()

def dataCollect4Chan(chan1, chan2, chan3, chan4, lines):
	# In its current iteration, 10 seconds of data are printed per line
	seconds_per_line = 10
	runtime = lines * seconds_per_line
	count1 = 0
	rate = 16 # This is the accumulation frequency, in the full-scale case, 16 Hz
	
	# Open up a file to save CSV
	#  data to, give it unique name based on runtime and channels
	file = open('%d_sec_LISS_accum_%d_%d_%d_%d.csv'%(runtime, chan1,chan2,chan3,chan4), 'w')
	writer = csv.writer(file)
	cols = rate * seconds_per_line
	
	# Create a header row in CSV file with channel names
	writer.writerow([chan1, chan2, chan3, chan4])
	# Iterate through the rows of the CSV file
	while (count1 < lines):
		print('we are %d/%d of the way through this shit'%(count1,lines))	    
		vals1 = np.zeros(cols)
		vals2 = np.zeros(cols)
		vals3 = np.zeros(cols)
		vals4 = np.zeros(cols)
		count2 = 0

		# Iterate through the columns of the CSV file
		while (count2 < cols):
			I, Q = read_accum_snap()
			I = I[2:]
			Q = Q[2:]
			mags = []
			accum_data = [0,0,0,0]
			mags.append((np.sqrt(I**2 + Q**2))[chan1])
			mags.append((np.sqrt(I**2 + Q**2))[chan2])
			mags.append((np.sqrt(I**2 + Q**2))[chan3])
			mags.append((np.sqrt(I**2 + Q**2))[chan4])
			#mags = 20*np.log10(mags/np.max(mags))[:1016]     
			accum_data[0] = 10*np.log10(mags[0]+1e-20)
			accum_data[1] = 10*np.log10(mags[1]+1e-20)
			accum_data[2] = 10*np.log10(mags[2]+1e-20)
			accum_data[3] = 10*np.log10(mags[3]+1e-20)

			# val1-val4 are the accumulation magnitude values for each of the chosen channels
			#(val1, val2, val3, val4) = (accum_data[0], accum_data[1], accum_data[2], accum_data[3])
			
			# vals1-vals4 are all of the accum magnitudes for the given channel in a single row of data
			vals1[count2] = accum_data[0]
			vals2[count2] = accum_data[1]
			vals3[count2] = accum_data[2]
			vals4[count2] = accum_data[3]
			
			#print('this is column number %d with a value of %d'%(count2, val))
			# count2 will iterate until it reaches the column max, which here is 160 (equivalent to 10 seconds of data)
			count2 += 1 # iterate by 1 column

		# Once all of the columns for a single row are collected, the code dumps those values for all 4 bins to CSV file
		writer.writerow(vals1)
		writer.writerow(vals2)
		writer.writerow(vals3)
		writer.writerow(vals4)
		
		# Iterate through another line of the CSV file and repeat column population
		count1 += 1 # Iterations will continue until reaching the desired run-time of simulation
	
	file.close()
def sync_len_2_time(length):
	accum_time_sec = (length/fft_len)/f_s
	return accum_time_sec

# Define a function that allows Python to quickly morph the accumulator sync trigger system to match desired integration time
def struct_morph(bit_length):
	fpga.write_int('cum_trigger_accum_len', (2**bit_length)-1) 
	fpga.write_int('cum_trigger_accum_reset', 0) #
	fpga.write_int('cum_trigger_accum_reset', 1) #
	fpga.write_int('cum_trigger_accum_reset', 0) #
	return

def accum_len_compare(b_o_i):
	# This function will act as the main conduit for testing the minimum detectable power of the LISS firmware 
	# It is designed for a single-tone input and will provide the output power in the B.O.I. for a given input power 
	# For each input power level, a number of different integration times will be tested 
	#file = open('min_pwr_test_%d_input.csv'%(pwr_in), 'w')
	#writer = csv.writer(file)
	#powers = np.arange(13,25) # range of accumulator sync lengths in base 2 bits

	### Only using taus that relate to equivalent times on analog LIA (1ms & 30ms) in bit-width(18, 22) ###
	powers = np.array([18,23])
	taus = sync_len_2_time(2**powers[:]) # Convert accum_len values into mirror time values (in seconds)
	# Say for example you have an input power of -30dB 
	# Calling this function with said power as the argument will cycle the accumulator length through the powers array...
	# and find the power at the B.O.I.
	# It will also check to make sure that the maximum power is in the B.O.I. and if not, will return a false in the output
	
	itr = 0 
	
	output_pwr = np.zeros([len(taus), 3])
	
	# The output_pwr array will contain all of the goodies resulting from this function
	# It is comprised of n columns where n = number of tested integration times
	# as well as 3 rows: the first being the integration time (in ms), the second being the output power, and the third being whether it is detectable; [tau, pwr_out, detectable]
	# An example column would be [65.536, 66.5, 1]; note that the final column element while type int represents a Boolean where 0 = False and 1 = True
	while itr < len(taus):
		struct_morph(powers[itr])

		# Sometimes the firmware needs to activate the accumulator a couple of times before it gets the correct readings
		# Make a routine to run find_max_bin 5 times and take the mode of the output values
	
		max_bins = np.arange(0,5)
		max_vals = np.array([0.0,0.0,0.0,0.0,0.0])

		for i in max_bins:
			max_vals[i], max_bins[i] = findmaxbin()
		
		# Take the mode bin and mean value as a broad litmus test on efficacy
		mode_bin = stats.mode(max_bins)[0][0]
		print('the max vals are ')
		print(max_vals)
		mean_val = np.median(max_vals)

		if mode_bin == b_o_i:
			detectable = True
			print('Max power detected at bin %d with average power of %d dBW'%(mode_bin, mean_val))
			print('With an integration time of %f ms, signal is detected'%(taus[itr]))
		else:
			detectable = False
			print('Max power detected at bin %d instead of estimated bin of %d'%(mode_bin, b_o_i))
			print('With an integration time of %f ms, signal is NOT detected'%(taus[itr]))
		
		output_pwr[itr] = [taus[itr]*1000, mean_val, int(detectable)]

		# At this point, we have successfully tested a tau element at a given input power and have the resulting output
		# Only thing left to do is iterate to the next integration time and test again
		# And I guess write everything to a CSV file for the sake of completion

		#writer.writerow(output_pwr[itr])
		itr += 1
		print('')
		print('')
		#print('Iteration %d of %d for input power of %d'%(itr, len(taus)-1, pwr_in))
	
	print('Final results of integration test in form of')
	print('[integration time, output power, detectable?]')
	#print(output_pwr)
	return output_pwr

def is_detectable(guess_bin, tau1, tau2, tau3):
	# Looks at the spectrum at 3 different integration times and sees at which integration times
	# detect the estimated bin of interest (user-inputted in function parameters and approx f_tone*2) 
	# as the highest bin in the entire spectrum.
	max_tau1 = findmaxbin(tau1)[1]
	max_tau2 = findmaxbin(tau2)[1]
	max_tau3 = findmaxbin(tau3)[1]
	(tau1_truth, tau2_truth, tau3_truth) = (False, False, False)
	if (max_tau1 == guess_bin):
		tau1_truth = True
		print("signal is detected with %f seconds of integration"%(tau1))
	else:
		print("Sorry my young gangsters, signal was not detected with integration of %f seconds"%(tau1))
		print("The maximum bin is located at %d"%(max_tau1))
	if (max_tau2 == guess_bin):
		tau2_truth = True
		print("signal is detected with %f seconds of integration"%(tau2))
	else:
		print("Sorry my young gangsters, signal was not detected with integration of %f seconds"%(tau2))
		print("The maximum bin is located at %d"%(max_tau2))
	if (max_tau3 == guess_bin):
		tau3_truth = True
		print("signal is detected with %f seconds of integration"%(tau3))
	else:
		print("Sorry my young gangsters, signal was not detected with integration of %f seconds"%(tau3))
		print("The maximum bin is located at %d"%(max_tau3))
	return(tau1_truth, tau2_truth, tau3_truth)



# tau is given in terms of samples here	
def differential_detect(center, tau1, tau2, tau3):
	x = center #Im too lazy to write it out a bunch
	bins = np.array([x-3, x-2, x-1, x, x+1, x+2, x+3])
	taus = np.array([tau1, tau2, tau3])
	
	truth = []
	for tau in taus:
		vals = []
		for chan in bins:
			sig_avg = np.mean(vals[2,3,4])
			noise_avg = np.mean(vals[0,1,5,6])
			vals.append(bin_reading(chan, tau))
		if (sig_avg > noise_avg):
			truth.append(True)
			print('Signal is detectable at integration:%d by a difference of %f dB'%(tau, sig_avg-noise_avg))
		elif (sig_avg <= noise_avg):
			truth.append(False)
			print('Signal is not detectable at integration %d by a difference of %f dB'%(tau, noise_avg-sig_avg))
		else: print('I have no idea whats going on')

	return (taus, truth)



def soft_int_compare(order, total_int):
	# This function will change the length of the accumulator on the FPGA and
	# replace the same integration time through software instead. 
	# This function is only for testing purposes and should not be used under any circumstance
	# during actual measurements as it will do nothing but degrade the quality of the RAM 
	# accumulator on the ROACH2 board.

	# Inputs: 
	# order -- this is the factor in base 2 by which we will transfer the integration responsibility
	# from hardware accumulation to software accumulation. For example, choosing '3' for this option 
	# will reduce the hardware integrator by a factor of 2**3 = 8. It will then boost the number of 
	# software integrations by the same factor
	#
	# total_int -- the total amount of integration time we wish to see displayed on our plot. 
	# Typically, this would have a minimum value of ~1/16 seconds, but since the accumulator is 
	# being shortened, *technically* you could get away with a smaller minimum integration time if that
	# is the desired testing result. NOTE: This variable is in units of ORIGINAL ACCUM LENGTH (a.k.a. 
	# 1/16 second samples). For the total integration to be 1 second, you need to input an order of 16!!!


	# Change integration length such that it is halved by the input order

	max_accum_order = 24
	desired_accum_order = max_accum_order-order
	fpga.write_int('cum_trigger_accum_len', 2**desired_accum_order-1)	
	print('Hardware accumulation set to %f milliseconds'%(65/2**order))
	plotAccum_avg(order*total_int)
	reset_accum_len()
	print('reverting to original accumulation length')

def	pure_hardware(order):

	fpga.write_int('cum_trigger_accum_len', 2**order-1)
	fig = plt.figure(figsize=(10.24,7.68))
	plt.title('TBD, Accum. Frequency = ')# + str(accum_freq), fontsize=18)
	plot1 = fig.add_subplot(111)
	line1, = plot1.plot(np.arange(506),np.ones(506), '#FF4500')
	line1.set_linestyle('None')
	line1.set_marker('.')
	plt.xlabel('Channel #',fontsize = 18)
	plt.ylabel('dB',fontsize = 18)
	plt.xticks(np.arange(0,506,50))
	plt.xlim(0,506)
	#plt.ylim(-40, 5)	
	plt.ylim(70, 100)
	plt.grid()
	plt.tight_layout()
	plt.show(block = False)
	count = 0
	stop = 10000
	while(count < stop):
		I, Q = read_accum_snap()
		I = I[2:]
		Q = Q[2:]
		mags =(np.sqrt(I**2 + Q**2))[2:508]
		#mags = 20*np.log10(mags/np.max(mags))[:1016]
		mags = 20*np.log10(mags[:]+1e-20)
		line1.set_ydata(mags)
		fig.canvas.draw()
		count += 1	
