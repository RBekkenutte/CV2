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

pcd = o3d.io.read_point_cloud("Data/data/0000000001.pcd")
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

###### go to 2. unless RMS is unchanged(<= epsilon)

###CHECK RMS

if RMS <= epsilon:
    print("Check")

###### 2. using different sampling methods

#

###### 3. transform point cloud with R and t

A1_transformed = np.zeros_like(A1)


for i, point in enumerate(A1):
    point = np.reshape(point, (3,1))
    new = R @ point + t
    A1_transformed[i] = new.T

###### 4. Find the closest point for each point in A1 based on A2 using brute-force approach

summation = 0
for i, pointA1 in enumerate(A1_transformed):
    distance = np.inf
    for pointA2 in A2:
        ## Distance between two points => norm of difference
        new_distance = np.linalg.norm(pointA1 - pointA2)
        if new_distance < distance:
            closest = pointA2
    
    summation += (pointA1 - closest) ** 2 

rms = np.sqrt(summation)

print(rms)

    




###### 5. Calculate RMS

###### 6. Refine R and t using SVD





############################
#   Merge Scene            #
############################

#  Estimate the camera poses using two consecutive frames of given data.

#  Estimate the camera pose and merge the results using every 2nd, 4th, and 10th frames.

#  Iteratively merge and estimate the camera poses for the consecutive frames.



############################
#  Additional Improvements #
############################





