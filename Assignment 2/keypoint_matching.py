import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#input boat1.pgm and boat2.pgm
#output keypoint matchings between the two images

def sift_algorithm(img1, img2):
    #SIFT Algorithm

    #Detect interest points in each image.
    #Characterize the local appearance of the regions around interest points.
    #Get the set of supposed matches between region descriptors in each image.

    #Make them grayscale
    #gray1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    #gray2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

    # Initiate SIFT detector
    #sift = cv.SIFT_create()
    
    akaze = cv.AKAZE_create()

    # find the keypoints and descriptors with SIFT
    #kp1, des1 = sift.detectAndCompute(img1, None)
    #kp2, des2 = sift.detectAndCompute(img2, None)

    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)



    #If we want to insert keypoints
    # img1 = cv.drawKeypoints(gray1, kp1, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # img2 = cv.drawKeypoints(gray2, kp2, img2, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #Print it to the folder
    # cv.imwrite('boat1.jpg', img1)
    # cv.imwrite('boat2.jpg', img2)

    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    goodmatches = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            goodmatches.append(m)


    return kp1, kp2, goodmatches

def plot_matching(img1, img2):
    #Run the algorithm
    kp1, kp2, matches = sift_algorithm(img1, img2) 

    #Create a random subset
    random_subset = []

    #Add 10 random points to the subset
    indexes = [*range(len(matches))]
    np.random.shuffle(indexes)
    indices = indexes

    for i in indices:
        random_subset.append([matches[i]])

    #Plot the matches
    img4 = cv.drawMatchesKnn(img1,kp1,img2,kp2,random_subset,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img3 = cv.drawKeypoints(img1,kp1, img2)
    plt.imshow(img3)
    plt.imshow(img4)
    plt.title("Akaze keypoints with distance treshold 0.8")
    plt.show()
    

if __name__ == '__main__':
    img1 = cv.imread('Data/frame00000001.png')
    img2 = cv.imread('Data/frame00000020.png')
    plot_matching(img1, img2)