import os
from skimage.color import convert_colorspace, rgb2gray, rgb2lab, rgb2luv, rgb2ypbpr, rgb2ycbcr
import matplotlib.pyplot as plt
import numpy as np

def doTest():
    cspaces = ['HSV', 'L*a*b*', 'L*u*v*', 'RGB CIE', 'XYZ', 'YUV', 'YIQ', 'YPbPr', 'YCbCr']
    
    results = {}
    for colorSpace in cspaces:
        results[colorSpace] = ([], [], [])
    
    for app_num in range(1, 5+1):
        print('-------- testing "app-'+str(app_num)+'" --------')
        
        imageFiles = getImageFiles('./test-images/app-'+str(app_num))
        for filename, filepath in imageFiles:
            print(' Image:  {}'.format(filename))
            rgb = plt.imread(filepath)
            rgb = rgb[:,:,0:3]
            
            for colorSpace in cspaces:
                if colorSpace == 'L*a*b*':
                    image = rgb2lab(rgb)
                elif colorSpace == 'grayscale':
                    image = rgb2gray(rgb)
                elif colorSpace == 'L*u*v*':
                    image = rgb2luv(rgb)
                elif colorSpace == 'YPbPr':
                    image = rgb2ypbpr(rgb)
                elif colorSpace == 'YCbCr':
                    image = rgb2ycbcr(rgb)
#                    fig, axarr = plt.subplots(1,3)
#                    axarr[0].imshow(rgb)
#                    axarr[0].set_title('rgb')
#                    
#                    axarr[1].imshow(rgb2gray(rgb), cmap='gray')
#                    axarr[1].set_title('gray')
#                    
#                    axarr[2].imshow(image[:,:,0], cmap='gray')
#                    axarr[2].set_title('y-channel in YCbCr')
                    
                else:
                    try:
                        image = convert_colorspace(rgb, 'RGB', colorSpace)
                    except:
                        print(colorSpace)
                        return
                
                if colorSpace != 'grayscale':
                    ch1_var = np.round(np.var(image[:,:,0]), 2)
                    ch2_var = np.round(np.var(image[:,:,1]), 2)
                    ch3_var = np.round(np.var(image[:,:,2]), 2)

#                print('    {}\n  Variances: {}\t{}\t{}\n'.format(colorSpace,
#                                                                      ch1_var,
#                                                                      ch2_var,
#                                                                      ch3_var))
                
                    results[colorSpace][0].append(ch1_var)
                    results[colorSpace][1].append(ch2_var)
                    results[colorSpace][2].append(ch3_var)
                else:
                    gs_var = np.round(np.var(image[:,:]), 2)
                    results[colorSpace][0].append(gs_var)
    
    # prepare plot
    data = []
    labels = []
    
    for colorSpace in cspaces:
        if colorSpace == 'HSV':
            labels.extend(('H', 'S', 'V'))
        elif colorSpace == 'L*a*b*':
            labels.extend(('L*', 'a*', 'b*'))
        elif colorSpace == 'L*u*v*':
            labels.extend(('L*', 'u*', 'v*'))
        elif colorSpace == 'RGB CIE':
            labels.extend(('CIE-R', 'CIE-G', 'CIE-B'))
        elif colorSpace == 'XYZ':
            labels.extend(('X', 'Y', 'Z'))
        elif colorSpace == 'YUV':
            labels.extend(('Y', 'U', 'V'))
        elif colorSpace == 'YIQ':
            labels.extend(('Y', 'I', 'Q'))
        elif colorSpace == 'YPbPr':
            labels.extend(('Y', 'Pb', 'Pr'))
        elif colorSpace == 'YCbCr':
            labels.extend(('Y', 'Cb', 'Cr'))
        elif colorSpace == 'grayscale':
            labels.extend(('gray'))
        else:
            print('Colorspace: {}'.format(colorSpace))
            raise ValueError('Unknown colorspace key')

        if colorSpace != 'grayscale':
            data.append(results[colorSpace][0])
            data.append(results[colorSpace][1])
            data.append(results[colorSpace][2])
        else:
            data.append(results[colorSpace][0])
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    bp = plt.boxplot(data, sym='', vert=True, whis=[10,90])
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['medians'], color='#1b9e77', linewidth=2.0)
    
    ax1.yaxis.grid(True, linestyle='-', which='major', color='#bdbdbd', alpha=0.5)
    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)

    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_ylabel('Channel variance')

    return results



def getImageFiles(dirPath = None):
    
    if dirPath == None:
        dirPath = os.getcwd()
        
    imgfiles = []    
    for file in os.listdir(dirPath):
        if not file.startswith('.'):
            if file.endswith(".png") or file.endswith(".bmp") or file.endswith(".jpg"):
                imgfiles.append((file, os.path.join(dirPath, file)))
    
    return imgfiles



results = doTest()