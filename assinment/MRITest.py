from scipy.io import loadmat
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_multiotsu
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
import cv2
from sklearn.cluster import DBSCAN

data = loadmat('Brain.mat')
T1 = data['T1']
print(type(T1))
label = data['label']
T1_threshold = np.empty_like(label)
#
# 使用numpy.bincount计算每个值的出现次数
counts = np.bincount(label.flatten())

print('pre train的结果')

# 打印结果
for i, count in enumerate(counts):
    print(f"值 {i} 出现的次数: {count}")


#  绘制label标签对应已经分好类的十张图片 注意结果是int类型
def showLabelImg(label, title):
    # 创建一个2x5的子图布局
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # 将图片逐个展示在子图中
    for i, ax in enumerate(axes.flat):
        # 如果图片列表中的图片不足十张，则退出循环
        if i >= 10:
            break
        
        # 获取当前图片
        image = label[:, :, i]
        
        # 在当前子图中展示图片
        im = ax.imshow(image, cmap='jet')
        ax.axis('off')  # 关闭坐标轴
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8)
        cbar.set_label('Label')
        # ax.colorbar()
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    plt.title(title)
    # 显示图像
    plt.show()


# 统一的降噪函数
def reduceNoise(T1):
    # 应用高斯卷积
    T1_convolved = np.empty_like(T1)
    
    # 设定高斯核的标准差
    sigma = 1  # 高斯核的标准差，可以根据需要调整
    
    # 对每个切片应用高斯滤波
    for i in range(T1.shape[2]):
        T1_convolved[:, :, i] = gaussian_filter(T1[:, :, i], sigma=sigma)
    
    # # 可视化第一个切片的原始和卷积后的图像进行对比
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    #
    # # 原始图像
    # axes[0].imshow(T1[:, :, 0], cmap='jet')
    # axes[0].set_title('Original Slice')
    # axes[0].axis('off')
    #
    # # 卷积后的图像
    # axes[1].imshow(T1_convolved[:, :, 0], cmap='jet')
    # axes[1].set_title('Convolved Slice')
    # axes[1].axis('off')
    #
    # plt.show()
    return T1_convolved


# 中值滤波
def median_filter(T1):
    T1_median = np.empty_like(T1)
    filter_size = 5
    for i in range(T1.shape[2]):
        T1_median[:, :, i] = cv2.medianBlur(T1[:, :, i], filter_size)
    return T1_median


# 归一化 或者说 归一到0-255 会丢失大量数据
def process_data(T1):
    T_after = np.empty_like(T1)
    T_after = ((T1 - T1.min()) / (T1.max() - T1.min())) * 255
    return T_after.astype(np.uint8)

def equalizeHist(T1):
    T_result = np.empty_like(T1)
    for i in range(T1.shape[2]):
        term = cv2.equalizeHist(T1[:,:,i])
        T_result[:,:,i]=term
    
    return T_result

# 计算两张图片之间的差距
def ave_MSE(orign, result):
    # 计算差值的平均值
    mse_sum = 0
    for i in range(10):
        mse_sum += MSE(orign[:, :, i], result[:, :, i])
    
    average_mse = mse_sum / 10
    
    print("平均均方误差（MSE）：", average_mse)
    return average_mse


def MSE(orign, result):
    res = np.mean((orign - result) ** 2)
    print('当前的差值为：' + str(res))
    return res


def precision(orign, result):
    matches = orign == result
    
    # 然后，计算布尔数组中True的比例，即分类正确的像素占总像素的比例
    accuracy = np.mean(matches)
    
    print(f'分类准确度: {accuracy:.4f}')


def calculate_f1_score(ground_truth, predictions):
    """
    计算像素级别分类的F1分数。

    :param ground_truth: 真实标签的数组，形状为 (n_images, height, width)。
    :param predictions: 预测标签的数组，形状为 (n_images, height, width)。
    :return: F1分数。
    """
    # 确保输入的形状相同
    assert ground_truth.shape == predictions.shape, "输入数组的形状必须相同。"
    
    # 将多维数组展平
    ground_truth_flat = ground_truth.flatten()
    predictions_flat = predictions.flatten()
    
    # 计算F1分数
    f1 = f1_score(ground_truth_flat, predictions_flat, average='weighted')
    
    return f1


def showHistogram(T1,title):
    # 计算直方图数据
    histogram, bin_edges = np.histogram(T1.flatten(), bins=256, range=(0, T1.max()))
    
    # 绘制直方图
    plt.figure()
    plt.plot(bin_edges[0:-1], histogram)  # bin_edges的数量比histogram多一个，所以需要去掉最后一个
    plt.title(title)
    plt.ylabel("Intensity Value")
    plt.xlabel("Pixels")
    plt.show()


def getThreshold(T1):
    # 使用多阈值Otsu方法计算五个阈值，分割成六个区间
    thresholds = threshold_multiotsu(T1, classes=6)
    
    # 输出计算出的阈值
    print("Calculated thresholds:", thresholds)
    
    return thresholds


def get_threshold_1(T1):
    # 使用多阈值Otsu方法计算五个阈值，分割成六个区间
    thresholds = threshold_multiotsu(T1[:, :, 1], classes=6)
    
    # 输出计算出的阈值
    print("Calculated thresholds:", thresholds)
    
    # 使用计算出的阈值对数据进行分割
    T1_segmented = np.digitize(T1[:, :, 1], bins=thresholds)
    
    # 可视化分割结果的一个示例切片
    plt.figure(figsize=(8, 8))
    plt.imshow(T1_segmented[:, :, 0], cmap='jet')
    plt.colorbar()
    plt.title('Segmented Slice with Multi-Otsu Thresholds')
    plt.axis('off')
    plt.show()


def useThresholds(T1, thresholds):
    # 可视化分割结果
    # 可视化第一个切片的原始和卷积后的图像进行对比
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, ax in enumerate(axes.flat):
        # 使用计算出的阈值对数据进行分割
        T1_segmented = np.digitize(T1[:, :, i], bins=thresholds)
        
        # 如果你想将分割的结果映射到特定的值或类别
        # # 例如，将6个区间（由5个阈值分割成的）映射到
        # class_values = [0, 1, 2, 3, 4, 5]
        # # class_values = [0, 4, 1, 2, 5, 3]
        #
        # # 初始化一个与image形状相同的数组来存放映射后的结果
        # mapped_result = np.zeros_like(T1_segmented)
        #
        # # 对每个类别进行映射
        # for i, value in enumerate(class_values):
        #     mapped_result[T1_segmented == i] = value
        
        T1_threshold[:, :, i] = T1_segmented
        # 在当前子图中展示图片
        im = ax.imshow(T1_segmented, cmap='jet')
        ax.axis('off')  # 关闭坐标轴
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8)
        cbar.set_label('Label')
        # 当前图片的mse记录
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 显示图像
    plt.show()
    
    # 使用numpy.bincount计算每个值的出现次数
    counts = np.bincount(T1_threshold.flatten())
    
    print('阈值切分后的结果')
    
    # 打印结果
    for i, count in enumerate(counts):
        print(f"值 {i} 出现的次数: {count}")
    
    ave_MSE(T1_threshold, label)


# 所有和与之相关的调用放在这里，方便后面查用
def threshold_seg():
    T1_re = reduceNoise(T1)
    showHistogram(T1_re)
    # print('开始计算阈值')
    # getThreshold(T1)
    # print('计算阈值结束')
    # 高斯平滑后的阈值
    thresholds = [54552.81, 121324.016, 186290.6, 238624.78, 285545.1]
    
    # 没有高斯平滑后的阈值
    thresholds_ung = [57478.344, 130283.31, 197340.53, 258649.97, 337202.7]
    
    # thresholds = [54552.81, 121324.016, 186290.6, 285545.1, 320000]
    
    useThresholds(T1, thresholds_ung)
    showLabelImg(label, 'label')
    
    f1 = calculate_f1_score(label, T1_threshold)
    print('f1分数为：' + str(f1))


def kmeans(T1):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    t_K = np.zeros_like(T1)
    for i, ax in enumerate(axes.flat):
        T_seg = T1[:, :, i]
        datak = T_seg.reshape((362 * 434, 1))
        # 使用KMeans算法进行6分类
        kmeans = KMeans(n_clusters=6, random_state=0).fit(datak)
        # 获取分类结果，并将其重新塑形为原始图像的形状
        labels = kmeans.labels_
        segmented_images = labels.reshape((362, 434))
        t_K[:, :, i] = segmented_images
        
        # 在当前子图中展示图片
        im = ax.imshow(segmented_images, cmap='jet')
        ax.axis('off')  # 关闭坐标轴
        # 添加颜色条
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8)
        cbar.set_label('Label')
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 显示图像
    plt.show()
    
    ave_MSE(t_K, label)
    f1 = calculate_f1_score(label, t_K)
    print('f1分数为：' + str(f1))
    showLabelImg(label, 'label')


# 使用kmeans方法
# T1_re = reduceNoise(T1)
# kmeans(T1_re)

# 边缘检测内容
# 定义Prewitt算子
prewitt_kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
prewitt_kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

# 定义Roberts算子
roberts_kernel_x = np.array([[1, 0], [0, -1]])
roberts_kernel_y = np.array([[0, 1], [-1, 0]])


def sobel_filter(T1):
    # 应用Sobel算子
    sobel_x = cv2.Sobel(T1, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(T1, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    return sobel


def prewitt_filter(T1):
    # 应用Prewitt算子
    prewitt_x = cv2.filter2D(T1, -1, prewitt_kernel_x)
    prewitt_y = cv2.filter2D(T1, -1, prewitt_kernel_y)
    prewitt = cv2.magnitude(prewitt_x, prewitt_y)
    return prewitt


def Laplacian_filter(T1):
    # 应用Laplacian算子
    laplacian = cv2.Laplacian(T1, cv2.CV_64F)
    # # 计算绝对值
    # abs_laplacian = np.absolute(laplacian)
    #
    # # 将结果转换为uint8
    # laplacian_8u = np.uint8(abs_laplacian)
    
    return laplacian


def Roberts_filter(T1):
    # 应用Roberts算子
    roberts_x = cv2.filter2D(T1, -1, roberts_kernel_x)
    roberts_y = cv2.filter2D(T1, -1, roberts_kernel_y)
    roberts = cv2.magnitude(roberts_x, roberts_y)
    return roberts


# 展示灰度图
def show_sub_img(T_all):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(T_all[:, :, i], cmap='gray')
        ax.axis('off')  # 关闭坐标轴
        # 添加颜色条
        # cbar = fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8)
        # cbar.set_label('Label')
        
        # 调整子图之间的间距
    plt.tight_layout()
    
    # 显示图像
    plt.show()


# 采用高低阈值1：2的比例，来应用canny算子
def canny_filter(T1):
    # 计算图像的中值
    median_val = np.median(T1)
    
    # 设置低阈值和高阈值，这里我们使用1:2的比例
    lower_threshold = int(max(0, (1.0 - 0.33) * median_val))
    upper_threshold = int(min(255, (1.0 + 0.33) * median_val))
    
    # 应用Canny边缘检测
    edges = cv2.Canny(T1, lower_threshold, upper_threshold)
    return edges


# 闭运算 特别适合于关闭前景对象中的小缺口、连接邻近的对象和平滑对象边界等任务。
def closing_operate(T1):
    # 创建一个结构元素 kernel可以更改
    kernel = np.ones((5, 5), np.uint8)
    
    #  创建一个5x5的矩形核 不同的核有不同的用处
    kernel_rect = np.ones((5, 5), np.uint8)
    
    # 创建一个5x5的椭圆形核
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # 创建一个5x5的十字形核
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    
    # 应用闭运算
    closing = cv2.morphologyEx(T1, cv2.MORPH_CLOSE, kernel)
    return closing


# 分水岭算法
def water_mark(T1):
    # 应用阈值分割来找到可能的区域
    ret, thresh = cv2.threshold(T1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 使用形态学变换去噪声
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # 寻找前景区域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # 寻找未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # 标记标签
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # 为所有的标记加1，确保背景是0
    markers = markers + 1
    
    # 现在标记未知区域为0
    markers[unknown == 255] = 0
    
    # 应用分水岭算法
    markers = cv2.watershed(cv2.cvtColor(T1, cv2.COLOR_GRAY2BGR), markers)
    T1[markers == -1] = [0]  # 边界标记为白色
    
    return markers, T1


def water_mark2(T1, edges):
    # gray = cv2.cvtColor(, cv2.COLOR_BGR2GRAY)
    
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 创建标记图像
    markers = np.zeros_like(T1, dtype=np.uint8)
    # markers = cv2.UMat(markers)
    
    # 为每个轮廓标记不同的标签
    for i, contour in enumerate(contours):
        cv2.drawContours(markers, [contour], -1, i + 1, -1)
    
    # 应用分水岭算法
    markers = cv2.watershed(cv2.cvtColor(T1, cv2.COLOR_GRAY2BGR), markers)
    T1[markers == -1] = [0]  # 边界标记为白色
    
    return markers, T1

def DBSCAN_all_filter(T1):
    res = np.zeros_like(T1)
    for i in range(T1.shape[2]):
        print('现在开始第'+str(i)+'张图的DBSCAN算法')
        res[:,:,i]=DBSCAN_filter(T1[:,:,i])
    return res

def DBSCAN_filter(T1):
    # 将图像转换为一维数组
    data = T1.flatten().reshape(-1, 1)
    
    # 创建 DBSCAN 模型  参数在这里调整 eps的推荐大小为维度*2
    dbscan = DBSCAN(eps=12, min_samples=50000)
    
    # 拟合模型
    dbscan.fit(data)
    
    # 获取每个像素的分类标签
    labels = dbscan.labels_
    
    # 将标签重新组织成与原始图像相同的形状
    segmented_image = labels.reshape(T1.shape)

    cv2.imshow(segmented_image, cmap='jet')
    # cv2.title('分类结果')
    
    return segmented_image

def water_for(T1, t_median):
    markers = np.zeros_like(T1)
    t_water = np.zeros_like(T1)
    for i in range(10):
        marker, t_aw = water_mark2(T1[:, :, i], t_median[:, :, i])
        t_water[:, :, i] = t_aw
        markers[:, :, i] = marker
    return markers, t_water


def region_grow(edge,seed_points):
    # 初始化标记图像
    h, w = edge.shape[:2]
    markers = np.zeros((h, w), np.int32)
    
    # 为floodFill准备掩码，大小需要比原图多2个像素
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # 设置用于floodFill的新值
    new_val = 255
    
    # 对每个种子点进行区域增长
    for i, seed in enumerate(seed_points):  # 假设seed_points包含了选择的种子点坐标
        # 注意：new_val需要为每个区域设定不同的值，以区分不同的区域
        cv2.floodFill(image=edge.astype(np.int32), mask=mask, seedPoint=seed, newVal=i + 1)

    # 可视化分类结果
    # 为了可视化，我们将每个区域标记为不同的灰度值
    img_segmented = np.zeros_like(edge)
    for i in range(1, 7):
        img_segmented[markers == i] = i * 40  # 以不同的灰度值显示不同的区域

    cv2.imshow('Segmented Image', img_segmented)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


T1_after = process_data(T1)
# 直方图均衡
# T1_after = equalizeHist(T1_after)
T1_re = reduceNoise(T1_after)

T1_canny = np.empty_like(T1_re)
T1_canny_close = np.empty_like(T1_re)
for i in range(10):
    # term = Laplacian_filter(T1_re[:, :, i])
    term = sobel_filter(T1_after[:, :, i])
    T1_canny[:, :, i] = term
    # T1_canny_close[:, :, i] = closing_operate(term)

# T1_re2 = reduceNoise(T1_canny)

# show_sub_img(T1_canny)
# t_median = median_filter(T1_canny)
# t_median = median_filter(T1_canny)
# show_sub_img(t_median)
# show_sub_img(label)

# 使用阈值进行分割并比对结果  效果不行
# thresholds = [3, 8, 27, 241, 249]
# useThresholds(t_median, thresholds)


# show_sub_img(T1_canny)
show_sub_img(label)
# showLabelImg(T1_water, 'T1_water')
# showHistogram(T1_canny,'lapulasi')
t_median = median_filter(T1_canny)
# showHistogram(t_median,'t_median')

# (106,1023)=0,0
seed=[(106,1023),(134,804),(219,680),(195,700),(316,688),(302,814)]
modified_data = [(x[0] - 106, 1023-x[1]) for x in seed]

# region_grow(T1_re[:,:,1],modified_data)







# for i in range(10):
#     region_grow(T1[:,:,i],t_median[:,:,i])

# DBSCAN 耗时太长，不好用
# DBSCAN_res = DBSCAN_all_filter(T1_canny)
# showHistogram(DBSCAN_res,title='DBSCAN')
# show_sub_img(DBSCAN_res)
# showLabelImg(DBSCAN_res,'DBSCAN')

# markers, t_after_water = water_for(T1_re, t_median)
# showHistogram(markers,title='markers')
# showHistogram(t_after_water,title="Water")

# f1 = calculate_f1_score(label, markers)
# print('f1分数为：' + str(f1))
