import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from keypoint_matching import sift_algorithm
from tqdm import tqdm

def RANSAC_algorithm(img1, img2, p, matches, kp1, kp2):
    # Set an n = amount of iterations of looking for best config
    
    iterations = 20
    best_amount_of_inliers =0
    for n in tqdm(range(iterations)):
        #Create a random subset
        random_matches = []

        #Add p random points to the subset
        indexes = [*range(len(matches))]
        np.random.shuffle(indexes)
        indices = indexes[:p]

        for i in indices:
            random_matches.append(matches[i])
            
        A = np.zeros((len(random_matches)*2,6))
        B = np.zeros((len(random_matches)*2,))
        #Construct A and B filled with the P matches
        l = 0
        for match in random_matches:
            trainid = match.queryIdx
            trainid_accent = match.trainIdx
            x = kp1[trainid].pt[0]
            y = kp1[trainid].pt[1]

            x_accent = kp2[trainid_accent].pt[0]
            y_accent = kp2[trainid_accent].pt[1]
            
            A[l,:] =[x,y,0,0,1,0]
            A[l+1,:] = [0,0,x,y,0,1]
            B[l] = x_accent
            B[l+1] = y_accent
            l = l +2
        
        
        x_vector = np.linalg.pinv(A) @ B
            

        translated_points = []

        amount_of_inliers = 0
        
        for j in range(len(matches)):

            trainid = matches[j].queryIdx
            trainid_accent = matches[j].trainIdx
            x = kp1[trainid].pt[0]
            y = kp1[trainid].pt[1]
            x_accent = kp2[trainid_accent].pt[0]
            y_accent = kp2[trainid_accent].pt[1]

            
            matrix_m = np.array([[x_vector[0], x_vector[1]],[x_vector[2],x_vector[3]]])
            t_vector = np.array([x_vector[4], x_vector[5]])
            
            kp1_new = (matrix_m @ np.array([x,y])) + t_vector
            translated_points.append(kp1_new)
            x_translated = kp1_new[0]
            y_translated = kp1_new[1]
            if (x_translated - x_accent)**2 + (y_translated - y_accent)**2 < 100:
                amount_of_inliers += 1
        
       
        
        if (amount_of_inliers > best_amount_of_inliers):
        
            best_amount_of_inliers = amount_of_inliers
            bestx = x_vector
            besttranslation = translated_points

    ### x_vector of bestx   
    return bestx, besttranslation



def transformImage(img1, x_vector):
    
    height, width, channels = img1.shape
    
    xposs = [*range(width)]
    yposs = [*range(height)]
    newxposs = []
    newyposs = []
    for x in xposs:
        for y in yposs:
            newxposs.append(x_vector[0]*x +  x_vector[1]*y + x_vector[4])
            newyposs.append(x_vector[2]*x + x_vector[3]*y+ x_vector[5])
    
    maxx= (max(newxposs))
    maxy = (max(newyposs))      
    minx = (min(newxposs))
    miny = (min(newyposs))        
    newheight = round(maxy - miny)
    newwidth = round(maxx - minx)
    transformedimage = np.zeros((newheight+1,newwidth+1,channels))
    for row in range(height):
        for col in range(width):
           newx = x_vector[0]*col +  x_vector[1]*row + x_vector[4] - minx
           newy = x_vector[2]* col + x_vector[3]*row + x_vector[5] - miny
           newx = round(newx)
           newy = round(newy)
           transformedimage[newy,newx] = img1[row,col] 
            
    transformedimage= transformedimage/255
    return transformedimage, round(maxy), round(maxx)

def demo1to2(img1,img2, p):
    kp1, kp2, matches = sift_algorithm(img1, img2)
    x_vector = RANSAC_algorithm(img1, img2, p, matches, kp1, kp2)[0]
    ##Our own implementation of transforming using NN-interpolation
    newimage, height, width = transformImage(img1, x_vector)
    plt.imshow(newimage)
    plt.title("Left to right using our own implementation")
    plt.show()
    ## Built-in CV function for transforming image
    # Put transformation vector in the right shape
    # We know the shape of the transformation from our own implementation:
    x_vector = np.array([[x_vector[0],x_vector[1], x_vector[4]], [x_vector[2], x_vector[3], x_vector[5]]])
    testWarp = cv.warpAffine(img1,x_vector, (width,height))
    plt.imshow(testWarp, cmap='gray')
    plt.title("Left to right using cv.warpAffine")
    plt.show()

    return None

def demo2to1(img2,img1, p):
    kp1, kp2, matches = sift_algorithm(img1, img2)
    x_vector = RANSAC_algorithm(img1, img2, p, matches, kp1, kp2)[0]
    newimage, height, width = transformImage(img1, x_vector)
    plt.imshow(newimage)
    plt.title("Right to Left using our own implementation")
    plt.show()
    x_vector = np.array([[x_vector[0],x_vector[1], x_vector[4]], [x_vector[2], x_vector[3], x_vector[5]]])
    testWarp = cv.warpAffine(img1,x_vector, (img1.shape[1],img1.shape[0]))
    plt.imshow(testWarp, cmap='gray')
    plt.title("Right to Left using cv.warpAffine")
    plt.show()

    return None

    
if __name__ == '__main__':
    
    img1 = cv.imread('boat1.pgm')
    img2 = cv.imread('boat2.pgm')
    
    p=3
    ## Demo function for transforming img1 to img2
    demo1to2(img1,img2,p)
    
    ## Demo function for transforming img2 to img1
    demo2to1(img1,img2,p)