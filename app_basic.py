import cv2 
import sklearn
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.decomposition import PCA
import skimage.morphology
import numpy as np
import time
import os
from flask import *  
app = Flask(__name__)  
 
@app.route('/')  
def upload():  
    return render_template("upload.html")  
 
@app.route('/pca', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        UPLOAD_PCA = 'templates/upload_pca'
        app.config['UPLOAD_PCA'] = UPLOAD_PCA
        image1_path = request.files['file']  
        image2_path = request.files['file1'] 
        image1_path.save(os.path.join(app.config["UPLOAD_PCA"], image1_path.filename))
        image2_path.save(os.path.join(app.config["UPLOAD_PCA"], image2_path.filename))
    
        image1 = cv2.imread(os.path.join(app.config["UPLOAD_PCA"], image1_path.filename))
        image2=cv2.imread(os.path.join(app.config["UPLOAD_PCA"], image2_path.filename))
        out_dir="D:/"
        print(image1)
        print(image2)
        
        
        print('[INFO] Resizing Images ...')
        start = time.time()
        new_size = np.asarray(image1.shape) /5
        new_size = new_size.astype(int) *5
        image1 = cv2.resize(image1, (new_size[0],new_size[1])).astype(int)
        image2 = cv2.resize(image2, (new_size[0],new_size[1])).astype(int)
        end = time.time()
        print('[INFO] Resizing Images took {} seconds'.format(end-start))

        print('[INFO] Computing Difference Image ...')
        start = time.time()
        diff_image = abs(image1 - image2)
        cv2.imwrite(out_dir+'difference.jpg', diff_image)
        end = time.time()
        print('[INFO] Computing Difference Image took {} seconds'.format(end-start))
        diff_image=diff_image[:,:,1]

        print('[INFO] Performing PCA ...')
        start = time.time()
        pca = PCA()
        vector_set, mean_vec=find_vector_set(diff_image, new_size)
        pca.fit(vector_set)
        EVS = pca.components_
        end = time.time()
        print('[INFO] Performing PCA took {} seconds'.format(end-start))

        print('[INFO] Building Feature Vector Space ...')
        start = time.time()
        FVS = find_FVS(EVS, diff_image, mean_vec, new_size)
        components = 3
        end = time.time()
        print('[INFO] Building Feature Vector Space took {} seconds'.format(end-start))

        print('[INFO] Clustering ...')
        start = time.time()
        least_index, change_map = clustering(FVS, components, new_size)
        end = time.time()
        print('[INFO] Clustering took {} seconds'.format(end-start))
        
        
        change_map[change_map == least_index] = 255
        change_map[change_map != 255] = 0
        change_map = change_map.astype(np.uint8)
        kernel = np.asarray(((0,0,1,0,0),
                        (0,1,1,1,0),
                        (1,1,1,1,1),
                        (0,1,1,1,0),
                        (0,0,1,0,0)), dtype=np.uint8)
        cleanChangeMap = cv2.erode(change_map,kernel)               

        print('[INFO] Save Change Map ...')
        cv2.imwrite(os.path.join(app.config["UPLOAD_PCA"], 'ChangeMap.jpg'), cleanChangeMap )

        
        print('[INFO] End Change Detection')
        
        response =  { 'Status' : 'Success', 'ImagePath': os.path.join(app.config["UPLOAD_PCA"], 'ChangeMap.jpg') }
        print(response)
        return jsonify(response) 



def find_vector_set(diff_image, new_size):
 
    i = 0
    j = 0
    vector_set = np.zeros((int(new_size[0] * new_size[1] / 25),25))
    while i < vector_set.shape[0]:
        while j < new_size[0]:
            k = 0
            while k < new_size[1]:
                block   = diff_image[j:j+5, k:k+5]
                feature = block.ravel()
                vector_set[i, :] = feature
                k = k + 5
            j = j + 5
        i = i + 1
 
    mean_vec   = np.mean(vector_set, axis = 0)
    # Mean normalization
    vector_set = vector_set - mean_vec   
    return vector_set, mean_vec



def find_FVS(EVS, diff_image, mean_vec, new):
 
    i = 2
    feature_vector_set = []
 
    while i < new[1] - 2:
        j = 2
        while j < new[0] - 2:
            block = diff_image[i-2:i+3, j-2:j+3]
            feature = block.flatten()
            feature_vector_set.append(feature)
            j = j+1
        i = i+1
 
    FVS = np.dot(feature_vector_set, EVS)
    FVS = FVS - mean_vec
    print ("[INFO] Feature vector space size", FVS.shape)
    return FVS


def clustering(FVS, components, new):
    kmeans = KMeans(components, verbose = 0)
    kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    count  = Counter(output)
 
    least_index = min(count, key = count.get)
    change_map  = np.reshape(output,(new[0] - 4, new[1] - 4))
    return least_index, change_map





if __name__ == '__main__':  
    app.run(debug = True) 