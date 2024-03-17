import glob
import matplotlib.pyplot as plt
from scipy.io import loadmat
import imageio
import sys
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score

loaded_data = np.loadtxt('47-1.txt')
res = loaded_data.astype(np.uint8)
label8 = np.array(Image.open(r'./8label.png'))
# plt.imshow(res, cmap='jet')
#
# # 添加颜色条
# plt.colorbar()
#
# # 显示图像
# plt.show()



def evaluation(imgPredict, imgLabel):
    mask = (imgLabel >= 0) & (imgLabel < 6)
    label = 6 * imgLabel[mask] + imgPredict[mask]
    count = np.bincount(label, minlength=6 ** 2)  # 核心代码
    confusionMatrix = count.reshape(6, 6)
    return confusionMatrix

def classPixelAccuracy(confusionMatrix):
    # return each category pixel accuracy(A more accurate way to call it precision)
    # acc = (TP) / TP + FP
    classAcc = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    
    return classAcc

def meanPixelAccuracy(classAcc):
    meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
    return meanAcc

def meanIntersectionOverUnion(confusionMatrix):
    # Intersection = TP Union = TP + FP + FN
    # IoU = TP / (TP + FP + FN)
    intersection = np.diag(confusionMatrix)  # 取对角元素的值，返回列表
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0) - np.diag(
        confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
    IoU = intersection / union  # 返回列表，其值为各个类别的IoU
    mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
    return mIoU


def MSE(orign, result):
    res = np.mean((orign - result) ** 2)
    print('当前的差值为：' + str(res))
    return res

def f1(ground_truth, predictions):
    # 将多维数组展平
    ground_truth_flat = ground_truth.flatten()
    predictions_flat = predictions.flatten()
    return f1_score(ground_truth_flat, predictions_flat, average='weighted')

def precision(orign, result):
    matches = orign == result
    
    # 然后，计算布尔数组中True的比例，即分类正确的像素占总像素的比例
    accuracy = np.mean(matches)
    
    print(f'分类准确度: {accuracy:.4f}')

matrix = evaluation(res,label8)
print('混淆矩阵')
print(matrix)
print('classPixelAccuracy')
cpa=classPixelAccuracy(matrix)
print(cpa)
print('meanPixelAccuracy')
print(meanPixelAccuracy(cpa))
print('meanIntersectionOverUnion')
print(meanIntersectionOverUnion(matrix))
precision(res,label8)
print('f1score')
print(f1(label8,res))
print('mse')
print(MSE(res,label8))