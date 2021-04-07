import numpy as np
import open3d as o3d


######                                                           ######
##       notice: You don't need to strictly follow the steps         ##
######                                                           ######



############################
#   Load Data              #
############################
##  example

pcd = o3d.io.read_point_cloud("Data/data/0000000000.pcd")
# ## convert into ndarray

pcd_arr = np.asarray(pcd.points)

# ***  you need to clean the point cloud using a threshold ***
pcd_arr_cleaned = pcd_arr


# visualization from ndarray
vis_pcd = o3d.geometry.PointCloud()
vis_pcd.points = o3d.utility.Vector3dVector(pcd_arr_cleaned)
#o3d.visualization.draw_geometries([vis_pcd])


A1 = pcd_arr_cleaned
############################
#   Load Data              #
############################
##  example

pcd = o3d.io.read_point_cloud("Data/data/0000000025.pcd")
# ## convert into ndarray

pcd_arr = np.asarray(pcd.points)

# ***  you need to clean the point cloud using a threshold ***
pcd_arr_cleaned = pcd_arr


# visualization from ndarray
vis_pcd = o3d.geometry.PointCloud()
vis_pcd.points = o3d.utility.Vector3dVector(pcd_arr_cleaned)
#o3d.visualization.draw_geometries([vis_pcd])

A2 = pcd_arr_cleaned



############################
#     ICP                  #
############################


###### 0. (adding noise)


###### 1. initialize R= I , t= 0

RMS = 1
epsilon = 0.001

R = np.identity(3)

t = np.zeros(3)
t = np.reshape(t,(3,1))

# A1= A1[:10000]
# A2 = A2[:9000]


###### go to 2. unless RMS is unchanged(<= epsilon)

A2_new = np.zeros_like(A1)
iteration = 0

print(A2.shape)


#### Get 10 % random points of A2
random_indices = np.random.choice(np.arange(A2.shape[0]), int(A2.shape[0]/10))

random_A2 = A2[random_indices,:]
#print(random_indices)
#print(random_indices.shape)
sampling = "Random_iteration"
###CHECK RMS
while(RMS > epsilon):
    print("Iteration: ", iteration)
    ###### 2. using different sampling methods

    #LATER TOEVOEGEN

    ###### 3. transform point cloud with R and t

    A1_transformed = np.zeros_like(A1)
    #print(R)
    #print(t)

    for i, point in enumerate(A1):
        point = np.reshape(point, (3,1))
        new = R @ point + t
        A1_transformed[i] = new.T
    if sampling == "All":
        target = A2
    if sampling == "Random":
        target = random_A2
    if sampling == "Random_iteration":
        random_indices = np.random.choice(np.arange(A2.shape[0]), int(A2.shape[0]/10))
        target = A2[random_indices,:]
        

    #vis_pcd.points = o3d.utility.Vector3dVector(A1_transformed)
    #o3d.visualization.draw_geometries([vis_pcd])

    ###### 4. Find the closest point for each point in A1 based on A2 using brute-force approach


    
    summation = 0

    for j, pointA1 in enumerate(A1_transformed):
        dist = np.sum((target - pointA1)**2, axis=1)
        closest = np.argmin(dist)
        A2_new[j] = target[closest]

        del dist
        summation += np.sum((pointA1 - target[closest]) ** 2)
        if j % 10000 == 0:
            print(j)

    rms = np.sqrt(summation/ A1_transformed.shape[0])

    print(rms)

    ###### 6. Refine R and t using SVD

    ### Center of mass

    centroid_1 = np.mean(A1_transformed, 0)
    centroid_2 = np.mean(A2_new, 0)

    

    A1_transformed = A1_transformed - centroid_1
    A2_transformed = A2_new - centroid_2 

       

    ### Covariance matrix

    X = A1_transformed.T 
    Y = A2_transformed.T
    #W = np.identity(X.shape[1])

   # S = X @ W @ Y.T
    S = X @ Y.T

    u , s, vh = np.linalg.svd(S, full_matrices=False, compute_uv=True)

    determinant = np.linalg.det(vh.T @ u.T)

    matrix = np.identity(u.shape[1])
    matrix[matrix.shape[1]-1, matrix.shape[1]-1] = determinant
    
    R = vh.T @ matrix @ u.T
    t = centroid_2 - R @ centroid_1
    t = np.reshape(t,(3,1))

    iteration += 1






############################
#   Merge Scene            #
############################

#  Estimate the camera poses using two consecutive frames of given data.

#  Estimate the camera pose and merge the results using every 2nd, 4th, and 10th frames.

#  Iteratively merge and estimate the camera poses for the consecutive frames.



############################
#  Additional Improvements #
############################





