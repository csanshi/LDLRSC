import scipy.io as sio 
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn import metrics
from munkres import Munkres
import os
from sklearn.manifold import TSNE
import numpy as np
from sklearn import cluster
from sklearn.preprocessing import normalize, MinMaxScaler
from PIL import Image


######################################### Load data
def load_data(data_path, shape):
	"""
	return: 
		images: ndarray (n, height, width, 1)
		labels: ndarray (n, 1)
	"""
	raw_data = sio.loadmat(data_path)
	if os.path.basename(data_path) == 'YaleBCrop025.mat':
		images = raw_data['Y'] # (2016, 64, 38)
		images = np.transpose(images) # (38, 64, 2016)
		(num_class, num_image_per_class, dim) = images.shape # (38, 64, 2016)
		images = np.reshape(images, (num_class*num_image_per_class, dim)) # (2432, 2016)
		images = np.reshape(images, (images.shape[0], 42, 48)) # (2432, 42, 48)
		images = np.transpose(images, (0, 2, 1)) # (2432, 48, 42)

		labels = np.zeros(images.shape[0], np.int8)
		# for _class in range(1, num_class+1):
		# 	labels[_class*num_image_per_class:(_class+1)*num_image_per_class] = _class

		for _class in range(0, images.shape[0]):
			labels[_class*num_image_per_class:(_class+1)*num_image_per_class] = _class + 1

	elif os.path.basename(data_path) == 'umist-32-32.mat':
		images = raw_data['img']
		images = np.reshape(images, (-1, shape[1], shape[0])) 
		images = np.transpose(images, (0, 2, 1))
		labels = np.squeeze(raw_data['label']) + 1
	elif os.path.basename(data_path) == 'mnist1000.mat':
		images = raw_data['X']
		images = np.reshape(images, (-1, shape[1], shape[0])) 
		# images = np.transpose(images, (0, 2, 1))
		labels = np.squeeze(raw_data['Y']) + 1
	else:
		images = raw_data['fea']
		images = np.reshape(images, (-1, shape[1], shape[0])) 
		images = np.transpose(images, (0, 2, 1))
		labels = np.squeeze(raw_data['gnd'])

	return images, labels

######################################### Cluster on Coef
def spectralCluster(C, K, d, alpha):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    # C = 0.5*(C + C.T)
    # r = d*K + 1
    # U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
    # U = U[:,::-1]    
    # S = np.sqrt(S[::-1])
    # S = np.diag(S)    
    # U = U.dot(S)    
    # U = normalize(U, norm='l2', axis = 1)       
    # Z = U.dot(U.T)
    # Z = Z * (Z>0)    
    # L = np.abs(Z ** alpha) 
    # L = L/L.max()   
    # L = 0.5 * (L + L.T)    
    # spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
    # spectral.fit(L)
    # grp = spectral.fit_predict(L) + 1
    # return grp, L

    C = 0.5*(C + C.T)
    r = min(d*K + 1, C.shape[0]-1)
    U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
    U = U[:,::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis = 1)
    Z = U.dot(U.T)
    Z = Z * (Z>0)
    L = np.abs(Z ** alpha)
    L = L/L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize',random_state=22)
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L


## scan each column, remain first x largest elements of which sum is greater than [ro * sum of this column]
def thrC(C,ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N,N))
        S = np.abs(np.sort(-np.abs(C),axis=0))
        Ind = np.argsort(-np.abs(C),axis=0)
        for i in range(N):
            cL1 = np.sum(S[:,i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while(stop == False):
                csum = csum + S[t,i]
                if csum > ro*cL1:
                    stop = True
                    Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
                t = t + 1
    else:
        Cp = C

    return Cp

######################################### Compute acc
def compute_acc(gt_labels, labels):
	gt_s,s = gt_labels, labels
# def compute_acc(gt_s,s):
	def best_map(L1,L2):
		#L1 should be the groundtruth labels and L2 should be the clustering labels we got
		Label1 = np.unique(L1)
		nClass1 = len(Label1)
		Label2 = np.unique(L2)
		nClass2 = len(Label2)
		nClass = np.maximum(nClass1,nClass2)
		G = np.zeros((nClass,nClass))
		for i in range(nClass1):
			ind_cla1 = L1 == Label1[i]
			ind_cla1 = ind_cla1.astype(float)
			for j in range(nClass2):
				ind_cla2 = L2 == Label2[j]
				ind_cla2 = ind_cla2.astype(float)
				G[i,j] = np.sum(ind_cla2 * ind_cla1)
		m = Munkres()
		index = m.compute(-G.T)
		index = np.array(index)
		c = index[:,1]
		newL2 = np.zeros(L2.shape)
		for i in range(nClass2):
			newL2[L2 == Label2[i]] = Label1[c[i]]
		return newL2   

	# labels_ = best_map(gt_labels, labels)

	# purity = 0
	# N = gt_labels.shape[0]
	# Label1 = np.unique(gt_labels)
	# Label2 = np.unique(labels_)
	# for label in Label2:
	# 	tempc = [i for i in range(N) if labels[i] == label]
	# 	hist,bin_edges = np.histogram(labels[tempc],Label1)
	# 	purity += max([np.max(hist),len(tempc)-np.sum(hist)])
	# purity /= N

	# num_correct = np.sum(gt_labels == labels_)
	# acc = num_correct / labels_.shape[0]
	# nmi = metrics.normalized_mutual_info_score(gt_labels, labels_)

	c_x = best_map(gt_s,s)
	err_x = np.sum(gt_s[:] != c_x[:])
	missrate = err_x.astype(float) / (gt_s.shape[0])
	NMI = metrics.normalized_mutual_info_score(gt_s, c_x)

	purity = 0
	N = gt_s.shape[0]
	Label1 = np.unique(gt_s)
	nClass1 = len(Label1)
	Label2 = np.unique(c_x)
	nClass2 = len(Label2)
	nClass = np.maximum(nClass1, nClass2)
	for label in Label2:
		tempc = [i for i in range(N) if s[i] == label]
		hist,bin_edges = np.histogram(gt_s[tempc],Label1)
		purity += max([np.max(hist),len(tempc)-np.sum(hist)])
	purity /= N
	return 1-missrate,NMI,purity

	# return acc, nmi, purity


def cluster_and_getACC(Coef, labels, num_class, d, alpha, ro):
	Coef = thrC(Coef, ro)
	predict, L = spectralCluster(Coef, num_class, d, alpha)
	# visualize(L, labels, filep='umist_affinity')
	acc, nmi, purity = compute_acc(gt_labels=labels, labels=predict)
	return acc, nmi, purity

def drawC(C,name='draw_Coef/a.png',norm=False):
    C = np.abs(C)
    C = C * (np.ones_like(C)-np.eye(C.shape[0]))
    if norm:
        C = C / np.sum(C,axis=1,keepdims=True)
    min_max_scaler = MinMaxScaler(feature_range=[0,255])
    CN = min_max_scaler.fit_transform(C)
    CN = CN + 255*np.eye(C.shape[0])
    IC = Image.fromarray(CN).convert('L')
    IC.save(name)
    # IC.show()

def drawDataset():
	Ns = [64, 72, 100, 10, 24]
	paths = ['./Data/YaleBCrop025.mat', './Data/COIL20.mat', './Data/mnist1000.mat', './Data/ORL_32x32.mat', './Data/umist-32-32.mat']
	shapes = [[42, 48], [32, 32], [28, 28], [32, 32], [32, 32]]

	for k in range(5):
		n = Ns[k]
		image_path = paths[k]
		shape = shapes[k]
		imgs = []
		(images, labels) = load_data(image_path, shape=shape)
		# images = tf.cast(images, dtype=tf.float32) # / 255
		for i in range(5):
			tmp = []
			for j in range(10):
				tmp.append(images[i*n+j])
			imgs.append(np.hstack(tmp))
		imgs = np.array(np.vstack(imgs))
		plt.imsave(image_path[:-4] + '.png', imgs, cmap='gray')

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k','orangered','greenyellow','darkviolet']
marks = ['o','+','.']

def visualize(Img,Label,CAE=None,filep=None):
    fig = plt.figure(figsize=(8, 8), dpi=150)
    ax1 = fig.add_subplot(111)
    n = Img.shape[0]
    if CAE is not None:
        bs = CAE.batch_size
        Z = CAE.transform(Img[:bs,:])
        Z = np.zeros([Img.shape[0], Z.shape[1]])
        for i in range(Z.shape[0] // bs):
            Z[i * bs:(i + 1) * bs, :] = CAE.transform(Img[i * bs:(i + 1) * bs, :])
        if Z.shape[0] % bs > 0:
            Z[-bs:, :] = CAE.transform(Img[-bs:, :])
    else:
        Z = Img
    Z_emb = TSNE(n_components=2).fit_transform(Z, Label)
    # print(Z_emb)
    lbs = np.unique(Label)
    for ii in range(lbs.size):
        Z_embi = Z_emb[[i for i in range(n) if Label[i] == lbs[ii]]].transpose()
        # print(Z_embi)
        ax1.scatter(Z_embi[0], Z_embi[1], color=colors[ii % 10], marker=marks[ii // 10], label=str(ii),s=3)
    ax1.legend()
    if filep is not None:
        plt.savefig(filep)
    # plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def gamma(img):
    return np.power(img / 255.0, 1)

# 渲染图像，即将计算出来的该图像的梯度方向和梯度幅值显示出来
def render_gradient( image, cell_gradient):
    cell_size  = 16
    bin_size = 9
    angle_unit = 180// bin_size
    cell_width =  cell_size / 2
    max_mag = np.array(cell_gradient).max()
    for x in range(cell_gradient.shape[0]):
        for y in range(cell_gradient.shape[1]):
            cell_grad = cell_gradient[x][y]
            cell_grad /= max_mag
            angle = 0
            angle_gap = angle_unit
            for magnitude in cell_grad:
                angle_radian = math.radians(angle)
                x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
                y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
                x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
                y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
                cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(abs(magnitude))))
                angle += angle_gap
    return image

# 获取梯度值cell图像，梯度方向cell图像
def div(img, cell_x, cell_y, cell_w):
    cell = np.zeros(shape=(cell_x, cell_y, cell_w, cell_w))
    img_x = np.split(img, cell_x , axis=0)
    for i in range(cell_x):
        img_y = np.split(img_x[i], cell_y, axis=1)
        for j in range(cell_y):
            cell[i][j] = img_y[j]
    return cell

# 获取梯度方向直方图图像，每个像素点有9个值
def get_bins(grad_cell, ang_cell):
    bins = np.zeros(shape=(grad_cell.shape[0], grad_cell.shape[1], 9))
    for i in range(grad_cell.shape[0]):
        for j in range(grad_cell.shape[1]):
            binn = np.zeros(9)
            grad_list = np.int8(grad_cell[i, j].flatten())  # .flatten()为降维函数，将其降维为一维，每个cell中的64个梯度值展平，并转为整数
            ang_list = ang_cell[i, j].flatten()  # 每个cell中的64个梯度方向展平
            ang_list = np.int8(ang_list / 20.0)  # 0-9
            ang_list[ang_list >= 9] = 0
            for m in range(len(ang_list)):
                binn[ang_list[m]] += int(grad_list[m])  # 直方图的幅值
            bins[i][j] = binn

    return bins

# 计算图像HOG特征向量并显示
def hog(img, cell_x, cell_y, cell_w):
    gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # x
    gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # y
    gradient_magnitude = np.sqrt(np.power(gradient_values_x, 2) + np.power(gradient_values_y, 2))
    gradient_angle = np.arctan2(gradient_values_x, gradient_values_y)
    # print(gradient_magnitude.shape, gradient_angle.shape)
    gradient_angle[gradient_angle > 0] *= 180 / 3.14
    gradient_angle[gradient_angle < 0] = (gradient_angle[gradient_angle < 0] + 3.14) * 180 / 3.14
    # plt.imshow()是以图像的大小，显示当前每一个像素点计算出来的梯度方向值
    # plt.imshow(gradient_magnitude ) #显示该图像的梯度大小值 
    # plt.imshow(gradient_angle ) #显示该图像的梯度方向值
    # 该图像的梯度大小值和方向值只能显示一个，如果用 plt.imshow()想要同时显示，则要分区
    # plt.show()
    grad_cell = div(gradient_magnitude, cell_x, cell_y, cell_w)
    ang_cell = div(gradient_angle, cell_x, cell_y, cell_w)
    bins = get_bins(grad_cell, ang_cell)
    # hog_image = render_gradient(np.zeros([img.shape[0], img.shape[1]]), bins)
    # plt.imshow(hog_image, cmap=plt.cm.gray)
    # plt.show()
    feature = []
    for i in range(cell_x - 1):
        for j in range(cell_y - 1):
            tmp = []
            tmp.append(bins[i, j])
            tmp.append(bins[i + 1, j])
            tmp.append(bins[i, j + 1])
            tmp.append(bins[i + 1, j + 1])
            tmp -= np.mean(tmp)
            feature.append(tmp.flatten())
    return np.array(feature).flatten()

def get_hog(img, cell_w=8):
    x = img.shape[0] - img.shape[0] % cell_w #找到离原图像行值最近的能够被8整除的数
    y = img.shape[1] - img.shape[1] % cell_w #找到离原图像列值最近的能够被8整除的数
    resizeimg = cv2.resize(img, (y, x), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("resizeimg",resizeimg)
    cell_x = int(resizeimg.shape[0] // cell_w)  # cell行数
    cell_y = int(resizeimg.shape[1] // cell_w)  # cell列数
    gammaimg = gamma(resizeimg) * 255
    feature = hog(gammaimg, cell_x, cell_y, cell_w)
    return feature
