import numpy as np
import open3d as o3d
import scipy.io
from scipy.io import loadmat
from scipy.spatial import cKDTree as KDTree
from sklearn.cluster import KMeans
import time
import pickle



## Toy data
A1 = loadmat("Data/source.mat")

A1 = np.array(A1['source'])
toy_source = A1.T

A2 = loadmat("Data/target.mat")

A2 = np.array(A2['target'])
toy_target = A2.T


##  Consecutive Frame data
source_file_name = "Data/data/0000000000"
pcd = o3d.io.read_point_cloud(source_file_name + ".pcd")


pcd_arr = np.asarray(pcd.points)
test_frame_source = pcd_arr[pcd_arr[:, 2] < 2]

##############

target_file_name = "Data/data/0000000001"
pcd = o3d.io.read_point_cloud(target_file_name + ".pcd")


pcd_arr = np.asarray(pcd.points)
test_frame_target = pcd_arr[pcd_arr[:, 2] < 2]


### Far frames data
source_file_name = "Data/data/0000000000"
pcd = o3d.io.read_point_cloud(source_file_name + ".pcd")

pcd_arr = np.asarray(pcd.points)

test_far_frame_source = pcd_arr[pcd_arr[:, 2] < 2]

target_file_name = "Data/data/0000000025"
pcd = o3d.io.read_point_cloud(target_file_name + ".pcd")

pcd_arr = np.asarray(pcd.points)

test_far_frame_target = pcd_arr[pcd_arr[:, 2] < 2]



def split_homogeneous(total_transformation):

    r = total_transformation[:3, :3]
    t=  np.reshape(total_transformation[:3, 3], (3,1))

    return r,t


def make_homogeneous(r, t):

    temp = np.zeros((4,4))

    temp[:3, :3] = r
 
    temp[:3, 3] = np.reshape(t, (3,))

    temp[3,3] = 1

    return temp


def icp_algorithm(A1, A2, epsilon=0.00001, sampling = "All", sampling_percentage = 10):

    RMS = 1

    max_iterations = 100

    R = np.identity(3)

    t = np.zeros(3)
    t = np.reshape(t,(3,1))

    total_transformation = make_homogeneous(R, t)

    iteration = 0
    
    if sampling == "Random":
        random_indices = np.random.choice(np.arange(0,A1.shape[0]),int(A1.shape[0] * (sampling_percentage / 100)))
        A1 = A1[random_indices]
    

    if sampling == "K-means":
        n_bins = 1
        kmeans = KMeans(n_clusters=n_bins, random_state=0).fit(A2)
        bins = kmeans.labels_
        bin_array = []
        for x in range(n_bins):
            binn = np.argwhere(bins == x)
            binn = np.reshape(binn, (binn.shape[0],))
            bin_array.append( (KDTree(A2[binn], leafsize= 8 ), binn))
    else:
        full_tree = KDTree(A2, leafsize = 8)

    A1_transformed = A1

    ###CHECK RMS
    RMS = 0
    newRMS = 1
    RMS_values = []
    while(np.absolute(newRMS - RMS) > epsilon and iteration < max_iterations):

        RMS = newRMS

        A1_transformed = (R @ A1_transformed.T + t).T


        if sampling == "Random_iterative":
            random_indices = np.random.choice(np.arange(0,A1_transformed.shape[0]), int( A1_transformed.shape[0] * (sampling_percentage / 100)))
            A1_random = A1_transformed[random_indices]
            distances, indices = full_tree.query(A1_random, k=1)
        elif sampling == "K-means":

            kmeans_a1 = KMeans(n_clusters=n_bins, random_state=0).fit(A1_transformed)
            bins_a1 = kmeans_a1.labels_

            bin_array_a1 = []

            for x in range(n_bins):
                binn = np.argwhere(bins_a1 == x)
                binn = np.reshape(binn, (binn.shape[0],))
                bin_array_a1.append(binn)

            distances, indices = np.zeros(len(A1_transformed)), np.zeros(len(A1_transformed), dtype=int)


            for i, bin in enumerate(bin_array_a1):
                current_tree = bin_array[i][0]

                bin_distance, bin_indices = current_tree.query(A1_transformed[bin])
                
                for j, value in enumerate(bin_indices):
                    indices[bin_array[i][1][value]] = bin_array[i][1][value]
                    distances[bin_array[i][1][value]] = bin_distance[j]
        else:
            distances, indices = full_tree.query(A1_transformed, k=1)
    


        newRMS = np.sqrt(np.mean(np.power(distances, 2)))
        RMS_values.append(newRMS)

        A2_new = A2[indices]

        if sampling == "Random_iterative":
            centroid_1 = np.mean(A1_random, 0)
            X = (A1_random - centroid_1).T
        else:
            centroid_1 = np.mean(A1_transformed, 0)
            X = (A1_transformed - centroid_1).T
        
        centroid_2 = np.mean(A2_new, 0)

        ### Covariance matrix
        
        Y = (A2_new - centroid_2 ).T
        S = X @ Y.T

        u , s, vh = np.linalg.svd(S)

        determinant = np.linalg.det(vh.T @ u)

        matrix = np.identity(u.shape[1])
        matrix[-1,-1] = determinant


        R = vh.T @ matrix @ u.T


        t = centroid_2 - R @ centroid_1
        t = np.reshape(t,(3,1))

        tempie = make_homogeneous(R, t)
        total_transformation = total_transformation @ tempie

        iteration += 1

    return total_transformation, RMS_values

def make_transformation_list(frame_sampling_rate, method):


    total_point_cloud = None


    frame_nr = 0


    while frame_nr + frame_sampling_rate < 100 :

        print("Frame number:", frame_nr)

        if frame_nr < 10:

            filename_source = "Data/data/000000000{}.pcd".format(str(frame_nr))

        else:
            filename_source = "Data/data/00000000{}.pcd".format(str(frame_nr))
        

        if frame_nr + frame_sampling_rate < 10:

            filename_target = "Data/data/000000000{}.pcd".format(str(frame_nr + frame_sampling_rate))
        else:
            filename_target = "Data/data/00000000{}.pcd".format(str(frame_nr + frame_sampling_rate))

        pcd = o3d.io.read_point_cloud(filename_target)
        pcd_arr = np.asarray(pcd.points)
        target = pcd_arr[pcd_arr[:, 2] < 2]

        pcd = o3d.io.read_point_cloud(filename_source)
        pcd_arr = np.asarray(pcd.points)
        source = pcd_arr[pcd_arr[:, 2] < 2]

        if frame_nr == 0:
            total_point_cloud = source

        frame_nr += frame_sampling_rate


        if method == "iterative":
            total_transformation, _ = icp_algorithm(source, target , 0.00001, 'All', 10)
        else:
            total_transformation, _ = icp_algorithm(total_point_cloud, target , 0.00001, 'All', 10)


        r, t = split_homogeneous(total_transformation)
                
        transform_total = (r @ total_point_cloud.T + t).T
        total_point_cloud = np.concatenate((transform_total, target), axis = 0)
        source = target

    return total_point_cloud


three_d_point_cloud = make_transformation_list(2,method = 'iterative')



### To visualize the end result
vis_pcd = o3d.geometry.PointCloud()
vis_pcd.points = o3d.utility.Vector3dVector(three_d_point_cloud)
o3d.visualization.draw_geometries([vis_pcd])

### Wrapper to gather results for plots 
def test_wrapper():
    final_dict = {}
    sampling_methods = ['All', 'Random', 'Random_iterative', 'K-means']
    data = [[toy_source, toy_target ], [test_frame_source, test_frame_target], [test_far_frame_source, test_far_frame_target]]
    sampling_size = [10, 30]
    for i, dataset in enumerate(data):
        dataset_dict = {}
        for x in sampling_methods:
            method_dict = {}
            if x in ['All', 'K-means']:
                now = time.time()
                # Add noise
                dataset[0] = np.random.normal(np.mean(dataset[0]), np.std(dataset[0]), size=dataset[0].shape)
                dataset[1] = np.random.normal(np.mean(dataset[1]), np.std(dataset[1]), size=dataset[1].shape)
                _, RMS_values = icp_algorithm(dataset[0] , dataset[1], 0.00001, x)
                end = time.time() - now
                print(dataset)
                print(x)
                print("Time: ", end)
                print("Final RMS: ", RMS_values[-1])
                print("---------")
                method_dict["Time"] = end
                method_dict["RMS"] = RMS_values
            else:
                for size in sampling_size:
                    size_dict = {}
                    now = time.time()
                    _, RMS_values = icp_algorithm(dataset[0], dataset[1], 0.00001, x, size)
                    end = time.time() - now
                    print(dataset)
                    print(x)
                    print("Time: ", end)
                    print("Final RMS: ", RMS_values[-1])
                    print("Sampling percentage: ", size)
                    print("---------")
                    size_dict["Time"] = end
                    size_dict["Sampling_percentage"] = size
                    size_dict["RMS"] = RMS_values
                    method_dict[size] = size_dict
            dataset_dict[x] = method_dict
        final_dict[i] = dataset_dict
        breakpoint()
    return final_dict

### Pickle for storing results
#dictio = test_wrapper()

#with open('noise_results_dict.pickle', 'wb') as handle:
#    pickle.dump(dictio, handle, protocol=pickle.HIGHEST_PROTOCOL)


                
    





