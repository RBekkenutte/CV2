import numpy as np
import cv2 as cv
from keypoint_matching import sift_algorithm
from matplotlib import pyplot as plt
import seaborn as sns
import open3d as o3d
from scipy.spatial import procrustes 
from scipy.linalg import orthogonal_procrustes

def eightpoint(matches,  kp1, kp2,T = None, T_accent = None):

    A = np.ones((len(matches),9))
    for i, match in enumerate(matches):
        trainid = match.queryIdx
        trainid_accent = match.trainIdx
        x = kp1[trainid].pt[0]
        y = kp1[trainid].pt[1]

        x_accent = kp2[trainid_accent].pt[0]
        y_accent = kp2[trainid_accent].pt[1]

        if T is not None:
            point = T @ np.array([x,y,1])
            x = point[0]
            y = point[1]
            
            point_acc = T_accent @ np.array([x_accent,y_accent,1]).T
            x_accent = point_acc[0]
            y_accent = point_acc[1]

        A[i,:] =[x * x_accent,x * y_accent, x , y * x_accent, y * y_accent, y, x_accent , y_accent, 1]

    U , D, Vt = np.linalg.svd(A, full_matrices=False)

    F_vector = Vt[-1,:]
    F = F_vector.reshape((3,3))

    Uf, Df, Vft = np.linalg.svd(F, full_matrices=False)

    Df[-1] = 0

    new_Df = np.diag(Df)

    new_f = Uf @ new_Df @ Vft

    if T is not None:
        new_f = T_accent.T @ new_f @ T

    return new_f

def normalized_eightpoint(matches, kp1, kp2):
    n_matches = len(matches)
    all_points = np.zeros((n_matches, 4))

    for i, match in enumerate(matches):
        trainid = match.queryIdx
        trainid_accent = match.trainIdx
        x = kp1[trainid].pt[0]
        y = kp1[trainid].pt[1]

        x_accent = kp2[trainid_accent].pt[0]
        y_accent = kp2[trainid_accent].pt[1]
        all_points[i, :] = [x, y, x_accent, y_accent]

    mean_x = np.mean(all_points[:,0])
    mean_y = np.mean(all_points[:,1])
    d =  np.mean(np.sqrt(np.power((all_points[:,0]-mean_x),2) + np.power((all_points[:,1]-mean_y),2)))

    T = np.zeros((3,3))
    ding = np.sqrt(2)/d
    T[0,0] = ding
    T[1,1] = ding 
    T[0,2] = -mean_x * ding
    T[1,2] = - mean_y * ding
    T[-1,-1] = 1

    mean_x_accent = np.mean(all_points[:,2])
    mean_y_accent = np.mean(all_points[:,3])

    d =  np.mean(np.sqrt(np.power((all_points[:,2]-mean_x_accent),2) + np.power((all_points[:,3]-mean_y_accent),2)))

    T_accent = np.zeros((3,3))
    ding = np.sqrt(2)/d
    T_accent[0,0] = ding
    T_accent[1,1] = ding 
    T_accent[0,2] = -mean_x_accent * ding
    T_accent[1,2] = - mean_y_accent * ding
    T_accent[-1,-1] = 1


    return T, T_accent

def eightpointRANSAC(matches, T, T_accent, kp1, kp2):
    iteration = 0
    best_inliers = 0
    max_iterations = 100
    best_f = None
    while iteration < max_iterations:
        A = np.ones((8,9))
        random = np.random.randint(0,len(matches), size = 8)
        matchess = []
        for j, i in enumerate(random):

            match = matches[i]
            trainid = match.queryIdx
            trainid_accent = match.trainIdx
            x = kp1[trainid].pt[0]
            y = kp1[trainid].pt[1]

            x_accent = kp2[trainid_accent].pt[0]
            y_accent = kp2[trainid_accent].pt[1]
            point = T @ np.array([x,y,1])
            x = point[0]
            y = point[1]
                
            point_acc = T_accent @ np.array([x_accent,y_accent,1]).T
            x_accent = point_acc[0]
            y_accent = point_acc[1]

            matchess.append(((x,y),(x_accent,y_accent)))

            A[j,:] =[x * x_accent,x * y_accent, x , y * x_accent, y * y_accent, y, x_accent , y_accent, 1]

        U , D, Vt = np.linalg.svd(A, full_matrices=False)

        F_vector = Vt[-1,:]
        F = F_vector.reshape((3,3))

        Uf, Df, Vft = np.linalg.svd(F, full_matrices=False)

        Df[-1] = 0

        new_Df = np.diag(Df)

        new_f = Uf @ new_Df @ Vft

        amount_of_inliers, indices = Sampson_dist(new_f, matches, T, T_accent, kp1, kp2)
    

        if amount_of_inliers > best_inliers:
            best_inliers = amount_of_inliers
            best_f = new_f
            best_indices = indices

        iteration += 1

    best_matches = []
    for i in best_indices:
        best_matches.append(matches[i])
    
    best_f = eightpoint(best_matches,kp1, kp2, T, T_accent )
        
    return best_f, best_inliers, best_matches

def Sampson_dist(f, matches, T, T_accent, kp1, kp2):
    inliers = 0
    indices = []
    for i, match in enumerate(matches):
        trainid = match.queryIdx
        trainid_accent = match.trainIdx
        x = kp1[trainid].pt[0]
        y = kp1[trainid].pt[1]

        x_accent = kp2[trainid_accent].pt[0]
        y_accent = kp2[trainid_accent].pt[1]

        p = np.array([x,y,1])
        p_prime = np.array([x_accent,y_accent, 1])

        p =  T @ p
        p_prime = T_accent @ p_prime

        nomininator = np.power(p_prime.T @ f @ p,2)

        temp1 = f @ p 
        temp2 = f.T @ p_prime
        denominator =  temp1[0]**2 + temp1[1]**2 + temp2[0]**2 + temp2[1]**2 

        d = nomininator /denominator
        if d < 1:
            inliers += 1
            indices.append(i)
    return inliers, indices

### https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c  = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def make_epipolar():
#### Code partly from: https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html

    num1 = 0
    num2 = 8

    img1 = cv.imread('Data/frame00000001.png')
    img2 = cv.imread('Data/frame00000020.png')
        
    img1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

        # Keypoint matching
    kp1, kp2, matches = sift_algorithm(img1, img2)

    ### Comment below lines to switch between, 8-point, norm-8-point and Norm-Ransac

    T, T_accent = normalized_eightpoint(matches, kp1, kp2)
    #f = eightpoint(matches,kp1,kp2, T, T_accent )
    #f = eightpoint(matches, kp1, kp2)
    f, best, _ = eightpointRANSAC(matches, T, T_accent, kp1, kp2)

    pts1 = []
    pts2 = []

    for i, m in enumerate(matches):
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,f)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,f)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

   
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()
    return None



###Function to plot epipolar lines

#make_epipolar()

def construct_PVM(RANSAC = True, threshold=1):

    matched_dictionary = {}
    PVM_matrix = None
    M = 49


    for i in range(M-1):

        new_dictionary = {}

        # Load images

        if i < 8:
            num1 = '0' + str(i+1)
            num2 = '0' + str(i+2)
        if i == 8:
            num1 = '0' + str(i+1)
            num2 = str(i+2)
        if i > 8:
            num1 = str(i+1)
            num2 = str(i+2)

        img1 = cv.imread('Data/frame000000{}.png'.format(num1))
        img2 = cv.imread('Data/frame000000{}.png'.format(num2))
        
        img1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
        img2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

        # Keypoint matching
        kp1, kp2, matches = sift_algorithm(img1, img2)

        if RANSAC:
            T, T_accent = normalized_eightpoint(matches, kp1, kp2)
            _, _ , best_matches = eightpointRANSAC(matches, T, T_accent, kp1, kp2)
            matches = best_matches


        if type(PVM_matrix) == type(None):

            k = 0
            for match in matches:
                trainid = match.queryIdx
                trainid_accent = match.trainIdx
                x = kp1[trainid].pt[0]
                y = kp1[trainid].pt[1]

                x_accent = kp2[trainid_accent].pt[0]
                y_accent = kp2[trainid_accent].pt[1]

                if (x,y) not in matched_dictionary.keys():
                    matched_dictionary[(x, y)] = k
                    k += 1


            PVM_matrix = np.zeros((M * 2, len(matched_dictionary.keys())))

        for j, match in enumerate(matches):
            trainid = match.queryIdx
            trainid_accent = match.trainIdx
            x = kp1[trainid].pt[0]
            y = kp1[trainid].pt[1]

            x_accent = kp2[trainid_accent].pt[0]
            y_accent = kp2[trainid_accent].pt[1]

            added = False
           
            for point in matched_dictionary.keys():

                distance = np.sqrt((point[0]-x)**2 + (point[1] - y)**2 )
                if distance < threshold:
                    PVM_matrix[2*i,matched_dictionary[point]] = x_accent
                    PVM_matrix[2*i+1,matched_dictionary[point]] = y_accent
                    new_dictionary[(x_accent,y_accent)] = matched_dictionary[point]
                    added = True
                    break

            if not added:
                PVM_matrix = np.hstack((PVM_matrix, np.zeros((M*2,1))))
                PVM_matrix[2*i,-1] = x_accent
                PVM_matrix[2*i+1,-1] = y_accent
                new_dictionary[(x_accent,y_accent)] = PVM_matrix.shape[1] - 1

        print('PVM size: {}, matched_dictionary size: {}, new_dictionary size: {}'.format(PVM_matrix.shape, len(matched_dictionary.values()), len(new_dictionary.keys())))
        matched_dictionary = new_dictionary

    return PVM_matrix


def visualize_matrix(PVM, iets = False):
    if iets:
        sns.heatmap(PVM, vmin=0, vmax=2, cmap='cividis', center=1) 
        plt.show() 
    else:
        PVM[PVM > 0 ] = 1
        sns.heatmap(PVM, vmin=0, vmax=1, cmap='cividis', center=1) 
        plt.show()


def densify(PVM):
    bool_PVM = np.copy(PVM)
    bool_PVM[bool_PVM > 0 ] = 1
    treshold = 10
    dense_PVM = None
    for i in range(bool_PVM.shape[1]):
        if sum (bool_PVM[:,i]) > treshold:
            if dense_PVM is None:
                dense_PVM = PVM[:,i]
                dense_PVM = np.reshape(dense_PVM,(dense_PVM.shape[0],1))
            else:
                column = PVM[:,i]
                column = np.reshape(column, (column.shape[0],1))
                dense_PVM = np.hstack((dense_PVM, column))
    return dense_PVM

def getBlocks(PVM):
    bool_PVM = np.copy(PVM)
    bool_PVM[bool_PVM > 0 ] = 1
    visualize_matrixx = np.copy(bool_PVM)

    blocks = []

    images = 4
    for i in range(0,PVM.shape[0]-images*2,2):

        right = 0
        left = 0
        for k in reversed(range(0, PVM.shape[1])):
           # print(k)
            if bool_PVM[i,k] == 1  and right == 0:
                right = k
    
            if (right != 0) and (bool_PVM[i+images*2,k] == 0):
                left =  k+1
                break

        j = i + 2
        
        block = PVM[i:i+images*2,left:right]
        blocks.append(block)
    
    return blocks

            


### Code to construct a new PVM or load an old one


#PVM_matrix = construct_PVM(True, threshold = 1)
#PVM_matrix.dump("dense_matrix_with_threshold1andAkaze")
#breakpoint()
PVM_matrix = np.load("dense_matrix_with_threshold1", allow_pickle=True)
#PVM_matrix = np.loadtxt("PointViewMatrix.txt")
#breakpoint()
#visualize_matrix(PVM_matrix)
PVM_matrix = densify(PVM_matrix)
#visualize_matrix(PVM_matrix)


def getblock_full_dense(PVM):
    blocks = []
    blocks.append(PVM[:6,:6])
    for i in range(6,PVM.shape[1], 20):
            blocks.append(PVM[:i+2, :i+2])
    
    return blocks


def struct_from_moti(block):
    row_means = block.mean(1)
    block = block - row_means.reshape((-1,1))

    U, W, Vt = np.linalg.svd(block)

    U3 = U[:,:3]
    W3 = np.diag(W[:3])
    V3t = Vt[:3,:]

    M = U3 @ np.sqrt(W3)
    S = (np.sqrt(W3) @ V3t).T

    return S, M


### Uncomment the second function if you want to find blocks in the given PVM.txt

blocks = getBlocks(PVM_matrix)
#blocks = getblock_full_dense(PVM_matrix)


def stitch(blocks):
    final_points = None
    for i, block in enumerate(blocks[:-1]):
        if i == 2:
            break
        S, M = struct_from_moti(block)
        if type(final_points) == type(None):
            final_points = S
            intermediate = S
        else:
            end_common_x = block_prev[-2,-1]
            end_common_y = block_prev[-1,-1]
            finalk = None
            for j, row in enumerate(block):
                for k,value in enumerate(row):
                    if value == end_common_y and block[j-1,k] == end_common_x:
                        finalk = k
                        break
                if finalk != None:
                    break
            if finalk == None:
                continue

            R, s = orthogonal_procrustes(S[:finalk], intermediate[-finalk:])
            intermediate = S
            tempiets = np.dot(S,R) * s
            final_points = np.vstack((final_points,tempiets))

        block_prev = block

    return final_points

### Using the PVM, create blocks and visualize the point cloud after stitching


final_point_cloud = stitch(blocks)
vis_pcd = o3d.geometry.PointCloud()
vis_pcd.points = o3d.utility.Vector3dVector(final_point_cloud)
o3d.visualization.draw_geometries([vis_pcd])











