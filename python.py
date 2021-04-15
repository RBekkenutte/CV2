import numpy as np
import open3d as o3d
import scipy.io
from scipy.io import loadmat
from scipy.spatial import cKDTree as KDTree
from sklearn.cluster import KMeans
import time

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
A1 = A1[A1[:, 2] < 2]

############################
#   Load Data              #
############################
##  example
target_file_name = "Data/data/0000000025"
pcd = o3d.io.read_point_cloud(target_file_name + ".pcd")

# ## convert into ndarray

pcd_arr = np.asarray(pcd.points)

# ***  you need to clean the point cloud using a threshold ***
pcd_arr_cleaned = pcd_arr



# visualization from ndarray
vis_pcd = o3d.geometry.PointCloud()
vis_pcd.points = o3d.utility.Vector3dVector(pcd_arr_cleaned)
#o3d.visualization.draw_geometries([vis_pcd])

A2 = pcd_arr_cleaned
A2 = A2[A2[:, 2] < 2]




# A2_normal_pcd = o3d.io.read_point_cloud(target_file_name +"_normal.pcd")
# A2_normal= np.asarray(pcd.points)
# breakpoint()

### Toy data
# A1 = loadmat("Data/source.mat")

# A1 = np.array(A1['source'])
# A1 = A1.T
# #A1 = o3d.geometry.PointCloud()
# #A1.points = o3d.Vector3dVector("Data/source.mat") 

# A2 = loadmat("Data/target.mat")

# A2 = np.array(A2['target'])
# A2 = A2.T

# A1 = np.array([[0,0,0], [30,30,30], [100,100,100]])
# A2 = np.array([[10,10,10], [40,40,40], [110,110,110]])

#A2 = o3d.geometry.PointCloud()

##A2.points = open3d.geometry.Vector3dVector("Data/target.mat") #

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


iteration = 0


sampling = "Random_iterative"
### For random sampling
sampling_size = 10000


if sampling == "All":
    full_tree = KDTree(A2, leafsize = 8)
if sampling == "Random":
    random_indices = np.random.choice(np.arange(0,A2.shape[0]), sampling_size)
    A2 = A2[random_indices]
    full_tree = KDTree(A2, leafsize = 8)

if sampling == "Bucket":
    n_bins = 10
    kmeans = KMeans(n_clusters=n_bins, random_state=0).fit(A2)
    bins = kmeans.labels_
    bin_array = []
    for x in range(n_bins):
        binn = np.argwhere(bins == x)
        binn = np.reshape(binn, (binn.shape[0],))
        bin_array.append(A2[binn])


A1_transformed = A1


###CHECK RMS
RMS = 0 
newRMS = 1
while(np.absolute(newRMS - RMS) > 0.000001):
#while(True):
    RMS = newRMS
    print("Iteration: ", iteration)
    

    ###### 3. transform point cloud with R and t
    
    A1_transformed = (R @ A1_transformed.T + t).T
    #A1_transformed = R @ A1_transformed + t


    if sampling == "Random_iterative":
        ### Take a random sample each iteration
        random_indices = np.random.choice(np.arange(0,A2.shape[0]), sampling_size)
        A2_iterative = A2[random_indices]
        full_tree = KDTree(A2_iterative, leafsize = 8)
    
    
    distances, indices = full_tree.query(A1_transformed, k=1)

    if sampling == "Bucket":
        None
        
    
    # if iteration % 5 == 0:    
    #     color1 = np.ones_like(A1_transformed) * 0.3
    #     color2 = np.ones_like(A2)  * 0.7
    #     vis_pcd.points = o3d.utility.Vector3dVector(np.vstack([A1_transformed, A2]))
    #     vis_pcd.colors = o3d.utility.Vector3dVector(np.vstack([color1, color2]))
    #     o3d.visualization.draw_geometries([vis_pcd])

    
    #summation = 0
    #print(target.shape)

    # A2_new = np.zeros_like(A1)
    # for j, pointA1 in enumerate(A1_transformed):
        
    #     dist = np.sum((target - pointA1)**2, axis=1)

    #     closest = np.argmin(dist)

    #     A2_new[j] = target[closest]

    #    # print(pointA1)
    #    # print(target[closest])
    #     #print("---------")

    #     del dist
    #     summation += np.sum((pointA1 - target[closest]) ** 2)

    #     if j % 10000 == 0:
    #         print(j)

    newRMS = np.mean(np.power(distances, 2))
    #newRMS = summation / A1_transformed.shape[0]

    print("RMSSSSSSSSSSSS", newRMS)
  

    ###### 6. Refine R and t using SVD

    ### Center of mass

    A2_new = A2[indices]

    
    centroid_1 = np.mean(A1_transformed, 0)
    centroid_2 = np.mean(A2_new, 0)

    ### Covariance matrix

    X = (A1_transformed - centroid_1).T 
    Y = (A2_new - centroid_2 ).T
    #W = np.identity(X.shape[1])

    #S = X @ W @ Y.T
    S = X @ Y.T
    #u , s, vh = np.linalg.svd(S, full_matrices=False, compute_uv=True)
    u , s, vh = np.linalg.svd(S)

    determinant = np.linalg.det(vh.T @ u)

    matrix = np.identity(u.shape[1])
    matrix[-1,-1] = determinant
    
    R = vh.T @ matrix @ u.T
    t = centroid_2 - R @ centroid_1
  #  t = [1000, 1000, 0]
    t = np.reshape(t,(3,1))


    iteration += 1


def best_fit_transform(src, target, w=None):
    n = src.shape[0]
    d = src.shape[1]
    if w is None:
        w = np.ones(n).reshape(1, -1)
    src_mean = (w @ src / np.sum(w)).T
    target_mean = (w @ target / np.sum(w)).T
    X = src.T - src_mean
    Y = target.T - target_mean


    # we exploit the fact that W, as described in the given material, is a diagonal matrix.
    # instead of multiplying X @ W @ Y, we just multiply X element wise and obtain the same result.
    S = (X * w) @ Y.transpose()  # for d by n matrices

    U, _, Vt = np.linalg.svd(S)
    inner_mat = np.eye(d)
    inner_mat[-1, -1] = np.linalg.det(Vt.transpose() @ U)
    R = Vt.transpose() @ inner_mat @ U.transpose()

    # in this case, could do without inner matrix
    # if np.linalg.det(R) < 0:
    #     Vt[d - 1, :] *= -1
    # R = np.dot(Vt.T, U.T)
    # if np.sum(R2 - R) > 0.0001:
    #     raise 'weee'
    t = target_mean - R @ src_mean
    T = np.vstack((R.T, t.T))


    return R, t



############################
#   Merge Scene            #
############################

#  Estimate the camera poses using two consecutive frames of given data.

#  Estimate the camera pose and merge the results using every 2nd, 4th, and 10th frames.

#  Iteratively merge and estimate the camera poses for the consecutive frames.



############################
#  Additional Improvements #
############################





