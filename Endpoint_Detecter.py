import librosa
import numpy as np


#start_interval is not confirmed yet, and the value to return is confirmed
def find_start_interval(energy, start_interval, IAV_thresh, check=5):
    '''  
    :param energy: a vector of Integral Absolute Values of the energy from audio
    :param start_interval: starting point to search start_interval
    :param IAV_thresh: . 
    :param check: how much you wanna check energy seq whether it is more than IAV threshold or not
    :return: A index of energy vector  
    '''
    count = 0
    for i in range(start_interval, len(energy)):
        if energy[i] > IAV_thresh:
            count += 1
            if count==check:
                start_interval = i-(check-1)
                return start_interval # 임계값 보다 큰녀석 정해서 반환
        else:
            count=0
    return 0

#start_interval is confirmed, and the value to return is also confirmed
def find_end_interval(energy, start_interval, IAV_thresh, check=5):
    '''    
    :param energy: a vector of Integral Absolute Values of the energy from audio
    :param start_interval: starting point to search end point
    :param IAV_thresh: . 
    :param check: how much you wanna check energy seq whether it is less than IAV threshold or not
    :return: A index of energy vector     
    '''
    count = 0
    for i in range(start_interval, len(energy)):
        if energy[i] < IAV_thresh:
            count += 1
            if count==check:
                end_interval = i -(check-1)
                return end_interval #작은녀석 확정해서 반환
        else :
            count=0
    return 0


def find_start_sample_ind(frames, start_interval, IAV_thresh, hop_length):
    '''    
    :param frames: frames to investigate  
    :param start_interval: index to investigate from frames
    :param IAV_thresh: energy threshold
    :param hop_length: 
    :return: audio start index for y
    '''
    avg_signal = (IAV_thresh / len(frames))
    for signal_i in range(len(frames)):
        if frames[signal_i][start_interval] > avg_signal :
            return start_interval*hop_length+signal_i

def find_end_sample_ind(frames, end_interval, IAV_thresh, hop_length):
    '''    
    :param frames: frames to investigate  
    :param start_interval: index to investigate from frames
    :param IAV_thresh: energy threshold
    :param hop_length: 
    :return: audio end index for y
    '''
    avg_signal = (IAV_thresh / len(frames))
    for signal_i in range(len(frames)):
        if frames[signal_i][end_interval] < avg_signal:
            return end_interval * hop_length + signal_i

def Endpoint_Detecter(filename, sr=22050, frame_length=441, hop_length=220, check=5 ):
    '''    
    :param filename: filename to load  
    :param sr: sample_rate
    :param frame_length: when sr is 22050, the frame_length for 20ms is 441
    :param hop_length: 
    :param check: the number you want to check whether it is valid or not
    :return: 
    '''
    y, sr = librosa.load(filename, sr)

    #divide into frames
    frames = librosa.util.frame(y, frame_length, hop_length)
    frames = np.array(frames)

    energy = np.sum(np.absolute(frames), axis=0) # (N,)
    energy_max = np.amax(energy)
    energy_min = np.amin(energy)
    diff = energy_max - energy_min

    IAV_thresh=0
    if energy_max*0.7 < energy_min:
        IAV_thresh = energy_max - (diff * 0.2)
    else:
        IAV_thresh = energy_min + (diff*0.1)

    start_list=[]
    end_list=[]
    end_interval=0
    while end_interval < len(frames):
        start_interval = find_start_interval(energy, end_interval, IAV_thresh, check)
        end_interval=find_end_interval(energy, start_interval, IAV_thresh, check)
        if start_interval==0 or end_interval==0:
            break
        start_i = find_start_sample_ind(frames, start_interval, IAV_thresh, hop_length)
        start_list.append(start_i)
        end_i = find_end_sample_ind(frames, end_interval, IAV_thresh, hop_length)
        end_list.append(end_i)
        end_interval+=1

    y_out=[]
    for i in range(len(start_list)):
        y_out.extend(y[start_list[i]:end_list[i]])

    y_out = np.array(y_out)

    if 0 == len(y_out):
        print("can't proceed with trimming")
    else:
        librosa.output.write_wav('output.wav', y_out, sr)
        print("Audio trimmed")


filename = 'test.wav'
Endpoint_Detecter(filename)