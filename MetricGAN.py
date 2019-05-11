# -*- coding: utf-8 -*-
"""
This code (developed with Keras) applies MetricGAN to optimize PESQ or STOI score for Speech Enhancement. 
It can be easily extended to optimize other metrics.

Dependencies:
Python 2.7
keras=2.0.9
librosa=0.5.1


Note:
1) To prevent clipping of noisy waveform (after adding noise to clean speech) 
   when save as .wav, we divide it with a clipping constant 10 (i.e.,Noisy=(clean+noise)/10). 
   Therefore, in this code, there are many operations as *10 and /10 appear in the 
   waveform IO part. This constant should be changed according to the dataset.
2) The PESQ file can only be implemented in Linux environment.


If you find this code useful in your research, please cite:
Citation: 
       [1] S.-W. Fu, C.-F. Liao, Y. Tsao and S.-D. Lin, "MetricGAN: Generative Adversarial Networks based Black-box Metric Scores
           Optimization for Speech Enhancement," in Proc. ICML, 2019.
Contact:
       Szu-Wei Fu
       jasonfu@citi.sinica.edu.tw
       Academia Sinica, Taipei, Taiwan
       
@author: Jason
"""
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import Sequential, model_from_json, Model, load_model
from keras.layers.core import Dense, Dropout, Flatten, Activation, SpatialDropout2D, Reshape, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import LSTM, TimeDistributed, Bidirectional, dot, Input, Concatenate, Multiply, Subtract, Maximum
from keras.layers.pooling import GlobalAveragePooling2D
from joblib import Parallel, delayed
from SpectralNormalizationKeras import DenseSN, ConvSN1D, ConvSN2D, ConvSN3D
from pystoi.stoi import stoi

import shutil
import scipy.io
import librosa
import os
import time  
import numpy as np
import numpy.matlib
import random
import subprocess

random.seed(999)

TargetMetric='pesq' # It can be either 'pesq' or 'stoi' for now. Of course, it can be any arbitary metric of interest.
Target_score=np.asarray([1.0]) # Target metric score you want generator to generate. s in e.q. (5) of the paper.

output_path='Your path to output directory'
PESQ_path='Your path to PESQ'


GAN_epoch=200
mask_min=0.05
num_of_sampling=100
num_of_valid_sample=1000
clipping_constant=10.0  # To prevent clipping of noisy waveform. (i.e., Noisy=(clean+noise)/10)
batch_size=1

maxv = np.iinfo(np.int16).max 

def read_pesq(clean_root, enhanced_file, sr):
    f=enhanced_file.split('/')[-1]       
    wave_name=f.split('_')[-1].split('@')[0]
    clean_file=clean_root+'Train_'+wave_name+'.wav'
    
    cmd = PESQ_path+'/PESQ {} {} +{}'.format(clean_file, enhanced_file, sr)
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    out = proc.communicate()
    pesq = float(out[0][-6:-1])
    return (pesq+0.5)/5.0

# Parallel computing for accelerating
def read_batch_PESQ(clean_root, enhanced_list):
    pesq = Parallel(n_jobs=40)(delayed(read_pesq)(clean_root, en, 16000) for en in enhanced_list)
    return pesq
        
def read_STOI(clean_root, enhanced_file):
    f=enhanced_file.split('/')[-1]
    wave_name=f.split('_')[-1].split('@')[0]
    
    clean_wav    = librosa.load(clean_root+'Train_'+wave_name+'.wav', sr=16000)     
    enhanced_wav = librosa.load(enhanced_file, sr=16000)
    
    stoi_score = stoi(clean_wav[0], enhanced_wav[0], 16000, extended=False)    
    return stoi_score
    
# Parallel computing for accelerating    
def read_batch_STOI(clean_root, enhanced_list):
    stoi_score = Parallel(n_jobs=30)(delayed(read_STOI)(clean_root, en) for en in enhanced_list)
    return stoi_score
    
def List_concat(score, enhanced_list):
    concat_list=[]
    for i in range(len(score)):
        concat_list.append(str(score[i])+','+enhanced_list[i]) 
    return concat_list
         
def creatdir(directory):    
    if not os.path.exists(directory):
        os.makedirs(directory) 

def ListRead(filelist):
    f = open(filelist, 'r')
    Path=[]
    for line in f:
        Path=Path+[line[0:-1]]
    return Path
     
def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.
    
    
def Sp_and_phase(signal, Normalization=False):        
    signal_length = signal.shape[0]
    n_fft = 512
    y_pad = librosa.util.fix_length(signal, signal_length + n_fft // 2)
    
    F = librosa.stft(y_pad, n_fft=512, hop_length=256, win_length=512, window=scipy.signal.hamming)
    
    Lp=np.abs(F)
    phase=np.angle(F)
    if Normalization==True:    
        meanR = np.mean(Lp, axis=1).reshape((257,1))
        stdR = np.std(Lp, axis=1).reshape((257,1))+1e-12
        NLp = (Lp-meanR)/stdR
    else:
        NLp=Lp
    
    NLp=np.reshape(NLp.T,(1,NLp.shape[1],257)) # For LSTM
    return NLp, phase, signal_length

def SP_to_wav(mag, phase, signal_length):
    Rec = np.multiply(mag , np.exp(1j*phase))
    result = librosa.istft(Rec,
                           hop_length=256,
                           win_length=512,
                           window=scipy.signal.hamming, length=signal_length)
    return result   

def Generator_train_data_generator(file_list):
	index=0
	while True:
         noisy_wav = librosa.load(file_list[index], sr=16000)        
         noisy_LP_normalization, _, _= Sp_and_phase(noisy_wav[0]*clipping_constant, Normalization=True)
         noisy_LP, _, _= Sp_and_phase(noisy_wav[0]*clipping_constant, Normalization=False)
         
         
         clean_wav = librosa.load(Train_Clean_path+file_list[index].split('/')[-1], sr=16000)   
         clean_LP, _, _= Sp_and_phase(clean_wav[0]) 

         index += 1
         if index == len(file_list):
             index = 0
             
             random.shuffle(file_list)
       
         yield [noisy_LP_normalization, noisy_LP.reshape((1,257,noisy_LP.shape[1],1)), clean_LP.reshape((1,257,noisy_LP.shape[1],1)), mask_min*np.ones((1,257,noisy_LP.shape[1],1))], Target_score

def Discriminator_train_data_generator(file_list):
	index=0
	while True:
         score_filepath=file_list[index].split(',')
         noisy_wav = librosa.load(score_filepath[1], sr=16000) 
         
         if 'dB' in score_filepath[1]:   # noisy or enhanced            
             noisy_LP, _, _ =Sp_and_phase(noisy_wav[0]*clipping_constant)
         else:                          # clean
             noisy_LP, _, _ =Sp_and_phase(noisy_wav[0])
         
         f=file_list[index].split('/')[-1]
         if '@' in f:
            wave_name=f.split('_')[-1].split('@')[-2]
            clean_wav= librosa.load(Train_Clean_path+'Train_'+wave_name+'.wav', sr=16000) 
            clean_LP, _, _ =Sp_and_phase(clean_wav[0]) 
         else:
            wave_name=f.split('_')[-1]
            clean_wav= librosa.load(Train_Clean_path+'Train_'+wave_name, sr=16000) 
            clean_LP, _, _ =Sp_and_phase(clean_wav[0]) 
                      
         True_score=np.asarray([float(score_filepath[0])])
         
         index += 1
         if index == len(file_list):
             index = 0
             
             random.shuffle(file_list)
       
         yield np.concatenate((noisy_LP.reshape((1,257,noisy_LP.shape[1],1)),clean_LP.reshape((1,257,noisy_LP.shape[1],1))), axis=3), True_score


def Corresponding_clean_list(file_list):
    index=0
    co_clean_list=[]
    while index<len(file_list):
        f=file_list[index].split('/')[-1]
               
        wave_name=f.split('_')[-1]
        clean_name='Train_'+wave_name
            
        co_clean_list.append('1.00,'+Train_Clean_path+clean_name)
        index += 1  
    return co_clean_list
        

#########################  Training data #######################
print 'Reading path of training data...'
Train_Clean_path='Path to the directory of clean speech/Train/'
Generator_Train_Noisy_paths = get_filepaths("Path to the directory of noisy speech/Train")
# Data_shuffle
random.shuffle(Generator_Train_Noisy_paths)
######################### validation data #########################
print 'Reading path of validation data...'
Test_Clean_path ='Path to the directory of clean speech/Test/'
Generator_Test_Noisy_paths = get_filepaths("Path to the directory of noisy speech/Test") 
# Data_shuffle
random.shuffle(Generator_Test_Noisy_paths)
################################################################

     
start_time = time.time()
######## Model define start #########
#### Define the structure of Generator (speech enhancement model)  ##### 
print ('Generator constructuring...')
de_model = Sequential()

de_model.add(Bidirectional(LSTM(200, return_sequences=True), merge_mode='concat', input_shape=(None, 257))) #dropout=0.15, recurrent_dropout=0.15
de_model.add(Bidirectional(LSTM(200, return_sequences=True), merge_mode='concat'))

de_model.add(TimeDistributed(Dense(300)))
de_model.add(LeakyReLU())
de_model.add(Dropout(0.05))

de_model.add(TimeDistributed(Dense(257)))
de_model.add(Activation('sigmoid'))


#### Define the structure of Discriminator (surrogate loss approximator)  ##### 
print ('Discriminator constructuring...')

_input = Input(shape=(257,None,2))
_inputBN = BatchNormalization(axis=-1)(_input)

C1=ConvSN2D(15, (5,5), padding='valid',  data_format='channels_last') (_inputBN)
C1=LeakyReLU()(C1)

C2=ConvSN2D(25, (7,7), padding='valid',  data_format='channels_last') (C1)
C2=LeakyReLU()(C2)

C3=ConvSN2D(40, (9,9), padding='valid',  data_format='channels_last') (C2)
C3=LeakyReLU()(C3)

C4=ConvSN2D(50, (11,11), padding='valid',  data_format='channels_last') (C3)
C4=LeakyReLU()(C4)

Average_score=GlobalAveragePooling2D(name='Average_score')(C4)  #(batch_size, channels)

D1=DenseSN(50)(Average_score)
D1=LeakyReLU()(D1)

D2=DenseSN(10)(D1)
D2=LeakyReLU()(D2)

Score=DenseSN(1)(D2)

Discriminator = Model(outputs=Score, inputs=_input) 

Discriminator.trainable = True 
Discriminator.compile(loss='mse', optimizer='adam')


#### Combine the two networks to become MetricGAN
Discriminator.trainable = False 
  
Clean_reference = Input(shape=(257,None,1))
Noisy_LP        = Input(shape=(257,None,1))
Min_mask        = Input(shape=(257,None,1))

Reshape_de_model_output=Reshape((257, -1, 1))(de_model.output)
Mask=Maximum()([Reshape_de_model_output, Min_mask])

Enhanced = Multiply()([Mask, Noisy_LP]) 
Discriminator_input= Concatenate(axis=-1)([Enhanced, Clean_reference]) # Here the input of Discriminator is (Noisy, Clean) pair, so a clean reference is needed!!

Predicted_score=Discriminator(Discriminator_input) 
 					
MetricGAN= Model(inputs=[de_model.input, Noisy_LP, Clean_reference, Min_mask], outputs=Predicted_score)
MetricGAN.compile(loss='mse', optimizer='adam')
######## Model define end #########

Test_PESQ=[]
Test_STOI=[]
Test_Predicted_STOI_list=[]
Train_Predicted_STOI_list=[]
Previous_Discriminator_training_list=[]
shutil.rmtree(output_path)

for gan_epoch in np.arange(1, GAN_epoch+1):
    
    # Prepare directories
    creatdir(output_path+"/epoch"+str(gan_epoch))
    creatdir(output_path+"/epoch"+str(gan_epoch)+"/"+"Test_epoch"+str(gan_epoch))
    creatdir(output_path+'/For_discriminator_training')
    creatdir(output_path+'/temp')
    
    # random sample some training data  
    random.shuffle(Generator_Train_Noisy_paths)
    g1 = Generator_train_data_generator(Generator_Train_Noisy_paths[0:num_of_sampling])
             
    print 'Generator training (with discriminator fixed)...' 
    if gan_epoch>=2:                
        Generator_hist = MetricGAN.fit_generator(g1, steps_per_epoch=num_of_sampling, 
        					         epochs=1,
                                    verbose=1,
                                    max_queue_size=1, 
                                    workers=1,
                                    )

    # Evaluate the performance of generator in a validation set.
    print 'Evaluate G by validation data ...'    
    Test_enhanced_Name=[]
    utterance=0
    for path in Generator_Test_Noisy_paths[0:num_of_valid_sample]:   
        S=path.split('/')
        wave_name=S[-1]  
        
        noisy_wav = librosa.load(path, sr=16000) 
        noisy_LP_normalization, Nphase, signal_length=Sp_and_phase(noisy_wav[0]*clipping_constant, Normalization=True)
        noisy_LP, _, _= Sp_and_phase(noisy_wav[0]*clipping_constant)
        
        IRM=de_model.predict(noisy_LP_normalization)
        mask=np.maximum(IRM, mask_min)
        E=np.squeeze(noisy_LP*mask)
        
        enhanced_wav=SP_to_wav(E.T,Nphase, signal_length)
        enhanced_wav=enhanced_wav/np.max(abs(enhanced_wav))
        
        if utterance<20: # Only seperatly save the firt 20 utterance for listening comparision 
            enhanced_name=output_path+"/epoch"+str(gan_epoch)+"/"+"Test_epoch"+str(gan_epoch)+"/"+ wave_name[0:-4]+"@"+str(gan_epoch)+wave_name[-4:]
        else:           # others will be overrided to save hard disk memory.
            enhanced_name=output_path+"/temp"+"/"+ wave_name[0:-4]+"@"+str(gan_epoch)+wave_name[-4:]
        librosa.output.write_wav(enhanced_name, (enhanced_wav* maxv).astype(np.int16), 16000)
        utterance+=1      
        Test_enhanced_Name.append(enhanced_name)                 
              
    # Calculate True STOI    
    test_STOI=read_batch_STOI(Test_Clean_path, Test_enhanced_Name)     
    print np.mean(test_STOI)    
    Test_STOI.append(np.mean(test_STOI))
       
    # Calculate True PESQ    
    test_PESQ=read_batch_PESQ(Test_Clean_path, Test_enhanced_Name)         
    print np.mean(test_PESQ)*5.-0.5  
    Test_PESQ.append(np.mean(test_PESQ)*5.-0.5)
    
    # Plot learning curves
    plt.figure(1)
    plt.plot(range(1,gan_epoch+1),Test_STOI,'b',label='ValidPESQ')
    plt.xlim([1,gan_epoch])
    plt.xlabel('GAN_epoch')
    plt.ylabel('STOI')
    plt.grid(True)
    plt.show()
    plt.savefig('Test_STOI.png', dpi=150)
    
    plt.figure(2)
    plt.plot(range(1,gan_epoch+1),Test_PESQ,'r',label='ValidPESQ')
    plt.xlim([1,gan_epoch])
    plt.xlabel('GAN_epoch')
    plt.ylabel('PESQ')
    plt.grid(True)
    plt.show()
    plt.savefig('Test_PESQ.png', dpi=150)
    
    # save the current SE model
    de_model.save('current_SE_model.h5')     
                               					
    print 'Sample training data for discriminator training...'
    D_paths=Generator_Train_Noisy_paths[0:num_of_sampling]
      
    Enhanced_name=[]
    for path in D_paths:   
        #path=path.split(',')[-1]
        S=path.split('/')
        wave_name=S[-3][0:-4]+'_'+S[-2]+'_'+S[-1]  
        
        noisy_wav = librosa.load(path, sr=16000) 
        noisy_LP_normalization, Nphase, signal_length=Sp_and_phase(noisy_wav[0]*clipping_constant, Normalization=True)
        noisy_LP, _, _= Sp_and_phase(noisy_wav[0]*clipping_constant)
        
        IRM=de_model.predict(noisy_LP_normalization)
        mask=np.maximum(IRM, mask_min)
        E=np.squeeze(noisy_LP*mask)
        
        enhanced_wav=SP_to_wav(E.T, Nphase, signal_length)
        enhanced_wav=enhanced_wav/clipping_constant
                
        enhanced_name=output_path+"/For_discriminator_training/"+ wave_name[0:-4]+"@"+str(gan_epoch)+wave_name[-4:]
        librosa.output.write_wav(enhanced_name, (enhanced_wav* maxv).astype(np.int16), 16000)       
        Enhanced_name.append(enhanced_name)
    
    if TargetMetric=='stoi':
        # Calculate True STOI score   
        train_STOI=read_batch_STOI(Train_Clean_path, Enhanced_name) 
        current_sampling_list=List_concat(train_STOI, Enhanced_name) # This list is used to train discriminator.
    elif TargetMetric=='pesq':
        # Calculate True PESQ score
        train_PESQ=read_batch_PESQ(Train_Clean_path, Enhanced_name)         
        current_sampling_list=List_concat(train_PESQ, Enhanced_name) # This list is used to train discriminator.
        
    Co_clean_list=Corresponding_clean_list(D_paths) # List of true data (Clean speech)
       
    print 'Discriminator training...'                            
    ## Training for current list
    Current_Discriminator_training_list=current_sampling_list+Co_clean_list
    random.shuffle(Current_Discriminator_training_list)
    
    d_current          = Discriminator_train_data_generator(Current_Discriminator_training_list)  
    Discriminator_hist = Discriminator.fit_generator(d_current, steps_per_epoch=len(Current_Discriminator_training_list), 
                                epochs=1, 
    					         verbose=1,
                                max_queue_size=1, 
                                workers=1,
                                )

                                 
                                    
    ## Training for current list + Previous list (like replay buffer in RL, optional)                           
    random.shuffle(Previous_Discriminator_training_list)
    
    Total_Discriminator_training_list=Previous_Discriminator_training_list[0:len(Previous_Discriminator_training_list)/10]+Current_Discriminator_training_list # Discriminator_Train_list is the list used for pretraining.
    random.shuffle(Total_Discriminator_training_list)
    
    d_current_past     = Discriminator_train_data_generator(Total_Discriminator_training_list)
    Discriminator_hist = Discriminator.fit_generator(d_current_past, steps_per_epoch=len(Total_Discriminator_training_list), 
                                epochs=1, 
    					         verbose=1,
                                max_queue_size=1, 
                                workers=1,
                                )
    
    # Update the history list
    Previous_Discriminator_training_list=Previous_Discriminator_training_list+Current_Discriminator_training_list 
    
    ## Training current list again (optional)   
    Discriminator_hist = Discriminator.fit_generator(d_current, steps_per_epoch=len(Current_Discriminator_training_list), 
                                epochs=1, 
    					         verbose=1,
                                max_queue_size=1, 
                                workers=1,
                                )
                                
    shutil.rmtree(output_path+'/temp') # to save harddisk memory
   
end_time = time.time()
print ('The code for this file ran for %.2fm' % ((end_time - start_time) / 60.))
