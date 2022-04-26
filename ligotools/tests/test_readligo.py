from ligotools import loaddata, dq_channel_to_seglist, read_hdf5, FileList
import json

def test_loaddata():
	eventname = 'GW150914'
	fnjson = "data/BBH_events_v3.json"
	events = json.load(open(fnjson,"r"))
	event = events[eventname]
	fn_H1 = "data/"+event['fn_H1']
	strain_H1, time_H1, chan_dict_H1 = loaddata(fn_H1, 'H1')
	assert (strain_H1.shape == (131072,)) & (time_H1.shape == (131072,)) & (chan_dict_H1["DATA"].shape == (32,)), "STRAIN, TIME, or CHANNEL_DICT shapes are incorrect"
	

def test_dq_channel_to_seglist():
	eventname = 'GW150914'
	fnjson = "data/BBH_events_v3.json"
	events = json.load(open(fnjson,"r"))
	event = events[eventname]
	fn_H1 = "data/"+event['fn_H1']
	strain, time, chan_dict = loaddata(fn_H1, 'H1')
	DQflag = 'CBC_CAT3'
	segment_list_CBC_CAT3 = dq_channel_to_seglist(chan_dict[DQflag])
	DQflag = 'NO_CBC_HW_INJ'
	segment_list_NO_CBC_HW_INJ = dq_channel_to_seglist(chan_dict[DQflag])
	assert (len(segment_list_CBC_CAT3) == 1) & (len(segment_list_NO_CBC_HW_INJ) == 1), "incorrect segment lengths for: 'CBC_CAT3', or 'no CBC hardware injections' levels of data quality"

	
def test_read_hdf5():
	eventname = 'GW150914'
	fnjson = "data/BBH_events_v3.json"
	events = json.load(open(fnjson,"r"))
	event = events[eventname]
	fn_H1 = "data/"+event['fn_H1']
	H1_hdf5 = read_hdf5(fn_H1)
	assert (len(H1_hdf5) == 7) & (H1_hdf5[0].shape == (131072,)) & (len(set(H1_hdf5[6])) == 5), "incorrect: length of read_hdf5() output; shape of its STRAIN item; or length of its distinct 'NO_...' segments (i.e. 'NO_CBC_HW_INJ')" 

	
def test_FileList():
	data_filelist = FileList().searchdir(directory='data/')
	figures_filelist = FileList().searchdir(directory='figures/')
	audio_filelist = FileList().searchdir(directory='audio/')
	assert (len(data_filelist) == 3) & (len(figures_filelist) == 0) & (len(audio_filelist) == 0), "length of class FileList outputs are wrong for subrepositories: data/, figures/, or audio/"
