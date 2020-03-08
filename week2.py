import numpy as np
import matplotlib.pylab as plt

from data import read_xml_gt, FrameCapture
from data import save_frames, number_of_frames
from utils.visualization import animate_iou, animation_2bb, plot_animation
from bgModels import bg_model_grayscale, remove_bg
from metrics.mAP import calculate_ap


path = 'datasets/AICity_data/train/S03/c010'        
save_frames(path)

video_path = 'datasets/AICity_data/train/S03/c010/vdo.avi'
frames_path = 'datasets/AICity_data/train/S03/c010/data'


#this is very time consuming, we should avoid comuting it more than once.
mu, sigma = bg_model_grayscale(video_path, frames_path)

# Gaussian plot in the slides
xx = np.linspace(0, 255,1000)
exp = 1/(sigma[561,712]*np.sqrt(2*np.pi))*np.exp(-(xx-mu[561,712])**2/(2*sigma[561,712]**2))
plt.figure()
plt.plot(xx, exp, color = 'blue', label = 'Distribution')
plt.axvline(x = mu[561,712], color = 'red', label = 'Mean')
plt.axvline(x = mu[561,712]-(sigma[561,712]+2), linestyle = '--', color = 'green', label = 'Tolerance ' + r'$\alpha = 1$')
plt.axvline(x = mu[561,712]+(sigma[561,712]+2), linestyle = '--',color = 'green')
plt.axvline(x = mu[561,712]-3*(sigma[561,712]+2), linestyle = '--', color = 'purple', label = 'Tolerance ' + r'$\alpha = 5$')
plt.axvline(x = mu[561,712]+3*(sigma[561,712]+2), linestyle = '--',color = 'purple')
plt.legend()

#%%
video_n_frames = number_of_frames(video_path)

alphas = [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]
aps7 = []

#Test for different alpha values
for a in alphas:    
    det_bb = remove_bg(mu, sigma, a, frames_path, 0, video_n_frames, 
                           animation = False, denoise = True)
    
    gt_bb = read_xml_gt("datasets/ai_challenge_s03_c010-full_annotation.xml")
    
    
    ap = calculate_ap(det_bb, gt_bb, int(video_n_frames*0.25), video_n_frames, mode = 'area')
    print(a,ap)
    aps7.append(ap)




#
#
animation_2bb('try', '.gif', gt_bb, det_bb, frames_path, 10, 100, int(video_n_frames*0.25) + 1, 
              int(1920 / 4), int(1080 / 4))





