import csv
import allantools as allan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import allantools as AT

#### Read in spectrometer data and put all rows into a single 1D vector ####
#### This test had 9999 rows of 160 data points at a rate of 16Hz ####
#### Gives us a total of 99990 seconds of data ####
def load_data(csv_file):
    df=pd.read_csv(csv_file, #skiprows = [1]
    )
    new_df=pd.DataFrame([df.iloc[0].to_list()],columns=df.columns)
    for x in range(int((df.shape[0]-1))):
        sample_df=pd.DataFrame([df.iloc[(x+1)].to_list()],columns=df.iloc[x+1].to_list())
        new_df=pd.concat([new_df,sample_df ],axis=1)
    num_sec = len(new_df.columns)/16        
    timestream = new_df.values[0]
    return (timestream, num_sec)
    
def load_multi_chan(csv_file):
    df=pd.read_csv(csv_file, #skiprows = [1]
    )

    new_df1=pd.DataFrame([df.iloc[0].to_list()],columns=df.columns)
    new_df2=pd.DataFrame([df.iloc[1].to_list()],columns=df.columns)
    new_df3=pd.DataFrame([df.iloc[2].to_list()],columns=df.columns)
    new_df4=pd.DataFrame([df.iloc[3].to_list()],columns=df.columns)
    x = 3
    while x <= (int((df.shape[0]) - 5)):
        sample_df=pd.DataFrame([df.iloc[(x+1)].to_list()],columns=df.iloc[x+1].to_list())
        new_df1=pd.concat([new_df1,sample_df ],axis=1)
        sample_df=pd.DataFrame([df.iloc[(x+2)].to_list()],columns=df.iloc[x+2].to_list())
        new_df2=pd.concat([new_df2,sample_df ],axis=1)
        sample_df=pd.DataFrame([df.iloc[(x+3)].to_list()],columns=df.iloc[x+3].to_list())
        new_df3=pd.concat([new_df3,sample_df ],axis=1)
        sample_df=pd.DataFrame([df.iloc[(x+4)].to_list()],columns=df.iloc[x+4].to_list())
        new_df4=pd.concat([new_df4,sample_df ],axis=1)
        
        x += 4
    num_sec = len(new_df1.columns)/(16)        
    #timestreams = []
    #for i in range(channels-1):
        #timestreams.append(new_df.values[0][i::channels])
    #file1 = open('lock_in_accum.csv', 'w')
	#writer1 = csv.writer(file)
    #file2 = open('lock_in_accum.csv', 'w')
	#writer2 = csv.writer(file)
    #file3 = open('lock_in_accum.csv', 'w')
	#writer3 = csv.writer(file)
    #file4 = open('lock_in_accum.csv', 'w')
	#writer4 = csv.writer(file)
    
    timestreams = [np.array(new_df1.values), np.array(new_df2.values), np.array(new_df3.values), np.array(new_df4.values)]    
    return (timestreams, num_sec)
    
    
def plot_timestreams(csv_file):
    timestreams, num_sec = load_multi_chan(csv_file)
    figure, axis = plt.subplots(2, 2)
    axis[0, 0].plot(timestreams[0])
    axis[0, 0].set_title("Main Bin")
    axis[0, 1].plot(timestreams[1])
    axis[1, 0].plot(timestreams[2])
    axis[1, 1].plot(timestreams[3])
    plt.show()
    
def compare_timestream(csv1, csv2, channels):
     (timestream1, num_sec1) = load_multi_chan(csv1, channels)
     (timestream2, num_sec2) = load_multi_chan(csv2, channels)
     diff = timestream1[0][0]/timestream2[0][0]
     plt.plot(diff)
     plt.show()
##################################################
#                                                #
# Allan Variance Plotting Function:              #
# Inputs: (CSV data file name (in quotes ''),    #
#          A.V. resolution (~30 usually))        #
# Outputs: (Allan vriance plot, Allan time)      #  
#                                                #
##################################################
def allan_plot(csv_file, res):
    (timestreams, num_sec, channels) = load_data(csv_file)
    #tau = np.logspace(0, (np.log10(num_sec/5)), res)
    tau = np.logspace(0, 3, res)
    rate = 16 # 1/16 seconds of integration per sample
    
    ##################################################
    #                                                #
    # Overlapping Allan deviation function:          #
    # Inputs: (data, sample rate, data type, taus)   #
    # Outputs: (taus used, deviations, error, pairs) #
    #                                                #
    ##################################################
    
    (tau2, adevs, adev_err, n) = AT.oadev(timestream, rate, data_type="freq", taus=tau)
    avars = np.square(adevs) # square allan dev to get allan var
    # Make white noise t^-1 line
    white_line = (avars[0]*(tau2**-1))      
    # correction = np.zeros(len(tau2))
    # correction[:len(tau2)-(round(len(tau2)/5-2))] = 1 
    # correction[len(tau2)-(round(len(tau2)/5)-2):] = correction[len(tau2)-(round(len(tau2)/5)-2):] + np.arange(1,2.5,1.5/len(correction[len(tau2)-(round(len(tau2)/5)-2):]))
    #Plotting the Allan Variance ####
    plot = plt.loglog(tau2, avars) #Plot the allan variance data
    plt.loglog(tau2,white_line)    #Plot the white noise line for comparison
    #plt.errorbar(tau2, avars, yerr = 2*correction[::]*(avars[::]/np.sqrt(num_sec/tau2[::])), ecolor='g')
    plt.errorbar(tau2, avars, yerr = 2*(avars[::]/np.sqrt((num_sec/tau2[::]))), ecolor='g')
    plt.title('Allan Variance for Lock-in Spectrometer (Bin %d)'%(713))
    plt.xlabel('Integration Times (s)')
    plt.ylabel('Power^2 (arbitrary(?) units)')
    plt.show()
    
    ##### Finding the Allan Time ####
    # Find the deviation from white line percentage and see when it reaches 10%
    
    log_diff_pct = ((np.log10(avars) - np.log10(white_line))/np.log10(white_line))
    diff_pct = ((avars - white_line)/white_line)
    if min(np.abs(log_diff_pct[:] - 0.10)) < 0.1 :
        allan_loc = np.argmin(np.abs(log_diff_pct-0.10))
        allan_time = tau2[allan_loc]
    else: allan_time = np.inf
    if min(np.abs(diff_pct[:]-10)) < 1 :
        allan_loc2 = np.argmin(np.abs(diff_pct-10))
        allan_time2 = tau2[allan_loc2]
        avg_allan_time = (allan_time + allan_time2)/2
        print('The system has an Allan Time of %d seconds'%(avg_allan_time))
    else: 
        allan_time2 = np.inf 
        print('Allan time is E N D L E S S')
        
def allan_compare(csv1, csv2, channels, chan_index, channel_real = 'null', res = 30):
    # read in files
    (timestreams1, num_sec) = load_multi_chan(csv1, channels)
    timestream1 = timestreams1[chan_index][0]
    
    timestream2 = load_multi_chan(csv2, channels)[0][chan_index][0]

    # set up constants for allan variance
    tau = np.logspace(0, 3, res)
    rate = 16 # 1/16 seconds of integration per sample
    
    # now take allan variance of both datasets    
    (tau2, adevs1, adev_err1, n1) = AT.oadev(timestream1, rate, data_type="freq", taus=tau)
    avars1 = np.square(adevs1) # square allan dev to get allan var
    (tau3, adevs2, adev_err2, n2) = AT.oadev(timestream2, rate, data_type="freq", taus=tau)
    avars2 = np.square(adevs2) # square allan dev to get allan var

    # Make white noise t^-1 line for both datasets 
    white_line1 = (avars1[0]*(tau2**-1))
    white_line2 = (avars2[0]*(tau3**-1))
    
    # Plotting the Allan Variance ####
    plt.loglog(tau2, avars1) #Plot the allan variance data
    plt.loglog(tau3, avars2)
    plt.loglog(tau2,white_line1)    #Plot the white noise line for comparison
    plt.loglog(tau3,white_line2)
    plt.errorbar(tau2, avars1, yerr = 2*(avars1[::]/np.sqrt((num_sec/tau2[::]))), ecolor='g')
    plt.errorbar(tau3, avars2, yerr = 2*(avars2[::]/np.sqrt((num_sec/tau3[::]))), ecolor='g')
    
    # Now make plot title and all that mumbo jumbo 
    plt.title('Allan Variance Comparison of Lock-in to Hoh-spec, Channel %d'%(channel_real))
    plt.xlabel('Integration Times (s)')
    plt.ylabel('Power^2 (arbitrary(?) units)')
    plt.show()
    
def plot_differential(csv1, csv2, channels, chan_index, channel_real = 'null', res = 30):   
    # read in files
    (timestreams1, num_sec) = load_multi_chan(csv1, channels)
    timestream1 = timestreams1[chan_index][0]
    
    timestream2 = load_multi_chan(csv2, channels)[0][chan_index][0]

    # set up constants for allan variance
    tau = np.logspace(0, 3, res)
    rate = 16 # 1/16 seconds of integration per sample
    
    # now take allan variance of both datasets    
    (tau2, adevs1, adev_err1, n1) = AT.oadev(timestream1, rate, data_type="freq", taus=tau)
    avars1 = np.square(adevs1) # square allan dev to get allan var
    (tau3, adevs2, adev_err2, n2) = AT.oadev(timestream2, rate, data_type="freq", taus=tau)
    avars2 = np.square(adevs2) # square allan dev to get allan var
    
    diff = (avars1/avars2)
    plt.loglog(tau3, diff)

    
    plt.title('Allan variance difference between lock in and hoh spec for channel %d'%(channel_real))
    plt.xlabel('Integration Times (s)')
    plt.ylabel('Power^2 (arbitrary(?) units)')
    plt.show()