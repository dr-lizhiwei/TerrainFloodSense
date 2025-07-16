'''
cloud remval and water reconstruction
'''

import numpy as np
import os
import cv2
from PIL import Image
from osgeo import gdal
from collections import Counter
import pandas as pd

from scipy.ndimage import gaussian_filter1d

def water_Rate(Water_Occur_removal, Cloud_Removal, fileName):
    rates = []
    for i in range(0, 101):
        rates.append([])
        # rates[i].append(i)
    '''
    Water Reconstruction
    '''
    cfiles = os.listdir(Cloud_Removal)
    for file in cfiles:
        if file.split('.')[-1] == 'tif' or file.split('.')[-1] == 'png':
            rates[0].append(file.split('.')[0])
            img_dir = os.path.join(Cloud_Removal, file)
            imgMulti = gdal.Open(img_dir)
            image = imgMulti.ReadAsArray()

            # cloud removal
            Cloud_pixels = np.where(image[:, :] == 2)
            image[Cloud_pixels] = 0

            Water_path = os.path.join(Water_Occur_removal, file.split('.')[0] + '.tif')
            waterimg = gdal.Open(Water_path)
            water_img = waterimg.ReadAsArray()
            all_pixels = np.where(water_img[:, :] != 0)

            # Water_Occur = gdal.Open(Water_Occur_path).ReadAsArray()
            # validpix = np.where((Water_Occur >= 0) & (Water_Occur <= 100))
            # Water_Occur = Water_Occur[min(validpix[0]):max(validpix[0]), min(validpix[1]):max(validpix[1])]
            # Water_Occur = np.where(Water_Occur == 128, 0, Water_Occur)
            # Water_Occur = cv2.resize(Water_Occur, (height, width))

            # get (i, j) positions of predicted water pixels
            Seg_pixels = np.where(image[:, :] == 1)
            result = Counter(water_img[Seg_pixels])
            result1 = Counter(water_img[all_pixels])
            dic = dict(result)
            dic1 = dict(result1)
            for key in range(1, 101):
                if key in dic.keys() and key in dic1.keys() and int(dic1[key]) != 0:
                    rates[int(key)].append(int(dic[key]) / int(dic1[key]))
                else:
                    rates[int(key)].append(0)
    # 1）字典按key排序
    # 2）字典转存CSV
    # rates to .csv
    array = np.array(rates)
    pd.DataFrame(array).to_csv(fileName, header=None)
    # [f.write('{0},{1}\n'.format(key, value)) for key, value in rates.items()]

def water_Rate_adj(Water_Occur_removal, Cloud_Removal, fileName):
    rates = []
    for i in range(0, 101):
        rates.append([])
        # rates[i].append(i)
    '''
    Water Reconstruction
    '''
    cfiles = os.listdir(Cloud_Removal)
    for file in cfiles:
        if file.split('.')[-1] == 'tif' or file.split('.')[-1] == 'png':
            rates[0].append(file.split('.')[0])
            img_dir = os.path.join(Cloud_Removal, file)
            imgMulti = gdal.Open(img_dir)
            image = imgMulti.ReadAsArray()

            # cloud removal
            Cloud_pixels = np.where(image[:, :] == 2)
            image[Cloud_pixels] = 0

            Water_path = os.path.join(Water_Occur_removal, file.split('.')[0] + '.tif')
            waterimg = gdal.Open(Water_path)
            water_img = waterimg.ReadAsArray()
            all_pixels = np.where(water_img[:, :] != 0)

            # Water_Occur = gdal.Open(Water_Occur_path).ReadAsArray()
            # validpix = np.where((Water_Occur >= 0) & (Water_Occur <= 100))
            # Water_Occur = Water_Occur[min(validpix[0]):max(validpix[0]), min(validpix[1]):max(validpix[1])]
            # Water_Occur = np.where(Water_Occur == 128, 0, Water_Occur)
            # Water_Occur = cv2.resize(Water_Occur, (height, width))

            # get (i, j) positions of predicted water pixels
            Seg_pixels = np.where(image[:, :] == 1)

            # occresult0 = dict(Counter(water_img[all_pixels]))
            # occresult1 = dict(Counter(water_img[Seg_pixels]))
            # occresult = dict(sorted(occresult1.items()))
            # prearea = 0
            # # for k in range(0, len(occresult.keys()), 2):
            # for k in range(0,  100):
            #     occ = float(list(occresult.keys())[k])
            #     if occresult0[occ] < 10:
            #         continue
            #     area = occresult0[occ] * 0.0009
            #     if abs(prearea - area) < 0.09:
            #         continue
            #     if not k in occresult1.keys():
            #         continue
            #     area1 = occresult1[occ] * 0.0009
            #     rates[int(k)].append(area1/area)
            #     prearea =area
            occresult0 = dict(Counter(water_img[all_pixels]))
            occresult1 = dict(Counter(water_img[Seg_pixels]))
            occresult = dict(sorted(occresult1.items()))
            prearea = 0
            # for k in range(0, len(occresult.keys()), 2):
            for k in range(1, 101):
                area = len(np.where(water_img[all_pixels] > k)[0])
                area1 = len(np.where(water_img[Seg_pixels] > k)[0])
                if area != 0:
                    rates[int(k)].append(area1 / area)
                else:
                    rates[int(k)].append(1)
                # prearea = area
                # water_img = np.where((waterimg > occ), 1, 0)
                # cv2.imwrite(Water_removal_dir + os.sep + str(occ) + '.tif', water_img)  # 保存图片
                # areas.append([str(occ) + '.tif', area])  # 并保存曲线
                # prearea = area

            # result = Counter(water_img[Seg_pixels])
            # result1 = Counter(water_img[all_pixels])
            # dic = dict(result)
            # dic1 = dict(result1)
            # for key in range(1,101):
            #     if key in dic.keys() and key in dic1.keys() and int(dic1[key])!= 0:
            #         rates[int(key)].append(int(dic[key])/int(dic1[key]))
            #     else:
            #         rates[int(key)].append(0)
    # 1）字典按key排序
    # 2）字典转存CSV
    # rates to .csv
    array = np.array(rates)
    pd.DataFrame(array).to_csv(fileName, header=None)
    # [f.write('{0},{1}\n'.format(key, value)) for key, value in rates.items()]


def water_Rate_adjcls(Water_Occur_removal, Cloud_Removal, fileName, Water_Occur_path):
    WO = gdal.Open(Water_Occur_path).ReadAsArray()

    rates = []
    for i in range(0, int(np.max(WO)) + 1):
        rates.append([])
        # rates[i].append(i)
    '''
    Water Reconstruction
    '''
    cfiles = os.listdir(Cloud_Removal)
    for file in cfiles:
        if file.split('.')[-1] == 'tif' or file.split('.')[-1] == 'png':
            rates[0].append(file.split('.')[0])
            img_dir = os.path.join(Cloud_Removal, file)
            image = gdal.Open(img_dir).ReadAsArray()

            # cloud removal
            Cloud_pixels = np.where(image[:, :] == 2)
            image[Cloud_pixels] = 0

            Water_path = os.path.join(Water_Occur_removal, file.split('.')[0] + '.tif')
            waterimg = gdal.Open(Water_path)
            water_img = waterimg.ReadAsArray()
            all_pixels = np.where(water_img[:, :] != 0)

            # Water_Occur = gdal.Open(Water_Occur_path).ReadAsArray()
            # validpix = np.where((Water_Occur >= 0) & (Water_Occur <= 100))
            # Water_Occur = Water_Occur[min(validpix[0]):max(validpix[0]), min(validpix[1]):max(validpix[1])]
            # Water_Occur = np.where(Water_Occur == 128, 0, Water_Occur)
            # Water_Occur = cv2.resize(Water_Occur, (height, width))

            # get (i, j) positions of predicted water pixels
            Seg_pixels = np.where(image[:, :] == 1)

            # occresult0 = dict(Counter(water_img[all_pixels]))
            # occresult1 = dict(Counter(water_img[Seg_pixels]))
            # occresult = dict(sorted(occresult1.items()))
            # prearea = 0
            # # for k in range(0, len(occresult.keys()), 2):
            # for k in range(1, 101):
            #     area = len(np.where(water_img[all_pixels]>k)[0])
            #     area1 = len(np.where(water_img[Seg_pixels]>k)[0])
            #     if area != 0:
            #         rates[int(k)].append(area1 / area)
            #     else:
            #         rates[int(k)].append(1)
            #     # prearea = area
            #     # water_img = np.where((waterimg > occ), 1, 0)
            #     # cv2.imwrite(Water_removal_dir + os.sep + str(occ) + '.tif', water_img)  # 保存图片
            #     # areas.append([str(occ) + '.tif', area])  # 并保存曲线
            #     # prearea = area

            result = Counter(water_img[Seg_pixels])
            result1 = Counter(water_img[all_pixels])
            dic = dict(result)
            dic1 = dict(result1)
            for key in range(1, int(np.max(WO)) + 1):
                if key in dic.keys() and key in dic1.keys() and int(dic1[key]) != 0:
                    rates[int(key)].append(int(dic[key]) / int(dic1[key]))
                else:
                    rates[int(key)].append(0)
    # 1）字典按key排序
    # 2）字典转存CSV
    # rates to .csv
    array = np.array(rates)
    pd.DataFrame(array).to_csv(fileName, header=None)
    # [f.write('{0},{1}\n'.format(key, value)) for key, value in rates.items()]


def water_Rate_with_mean(Water_Occur_removal, Cloud_Removal, fileName, Water_Occur_path):
    WO = gdal.Open(Water_Occur_path).ReadAsArray()

    rates = []
    for i in range(0, int(np.max(WO)) + 1):
        rates.append([])
        # rates[i].append(i)
    '''
    Water Reconstruction
    '''
    cfiles = os.listdir(Cloud_Removal)
    for file in cfiles:
        if file.split('.')[-1] == 'tif' or file.split('.')[-1] == 'png':
            rates[0].append(file.split('.')[0])
            rates[0].append(file.split('.')[0] + "_sum")
            rates[0].append(file.split('.')[0] + "_mean")
            rates[0].append(file.split('.')[0] + "_filter")
            img_dir = os.path.join(Cloud_Removal, file)
            image = gdal.Open(img_dir).ReadAsArray()

            # cloud removal
            Cloud_pixels = np.where(image[:, :] == 2)
            image[Cloud_pixels] = 0

            Water_path = os.path.join(Water_Occur_removal, file.split('.')[0] + '.tif')
            water_img = gdal.Open(Water_path).ReadAsArray()
            all_pixels = np.where(water_img[:, :] != 0)
            # get (i, j) positions of predicted water pixels
            Seg_pixels = np.where(image[:, :] == 1)

            result = Counter(water_img[Seg_pixels])
            result1 = Counter(water_img[all_pixels])
            dic = dict(result)
            dic1 = dict(result1)
            sum = 0
            idx = 0

            for key in range(1, int(np.max(WO)) + 1):
                if key in dic.keys() and key in dic1.keys() and int(dic1[key]) != 0:
                    rate = int(dic[key]) / int(dic1[key])
                    sum += rate
                    idx += 1
                else:
                    rate = 0
                rates[int(key)].append(rate)
                rates[int(key)].append(sum)
                if idx != 0:
                    rates[int(key)].append(sum / idx)
                else:
                    rates[int(key)].append(0)
                # rates[int(key)].append(ratio)

    # 1）字典按key排序
    # 2）字典转存CSV
    # rates to .csv
    array = np.array(rates)
    pd.DataFrame(array).to_csv(fileName, header=None)
    # [f.write('{0},{1}\n'.format(key, value)) for key, value in rates.items()]


def water_Rate_with_filter(Water_Occur_removal, Cloud_Removal, fileName, Water_Occur_path):

    rates = []
    cfiles = os.listdir(Cloud_Removal)
    water_occurrence = gdal.Open(Water_Occur_path).ReadAsArray()
    all_occurrence = Counter(water_occurrence[np.where(water_occurrence>0)])
    sorted_keys = sorted(dict(all_occurrence).keys())

    for file in cfiles:
        if not file.split('.')[-1] == 'tif':
            continue
        occ_row = [file.split('.')[0] + '_occurrence']
        rate_row = [file.split('.')[0]]  # Start a new row with the file name
        filter_row = [file.split('.')[0] + '_filter']  # 滤波数据行名称

        img_dir = os.path.join(Cloud_Removal, file)
        image = gdal.Open(img_dir).ReadAsArray()
        # cloud removal
        Cloud_pixels = np.where(image[:, :] == 2)
        image[Cloud_pixels] = 0

        Water_path = os.path.join(Water_Occur_removal, file.split('.')[0] + '.tif')
        water_img = gdal.Open(Water_path).ReadAsArray()
        # valid pixel
        all_pixels = np.where(water_img[:, :] != 0)
        # get (i, j) positions of predicted water pixels
        Seg_pixels = np.where(image[:, :] == 1)

        result = Counter(water_img[Seg_pixels])
        result1 = Counter(water_img[all_pixels])
        dic = dict(result)
        dic1 = dict(result1)
        temp = 0
        for key in sorted_keys:
            occ_row.append(key)
            if key in dic.keys() and key in dic1.keys():
                rate = int(dic[key]) / int(dic1[key])
                temp = rate
            else:
                rate = temp
            rate_row.append(rate)


        rate_r = pd.Series(rate_row[1:])
        window_size = max(30,int(len(np.unique(water_occurrence))/50)) #30
        print("slide window size:", window_size)
        smoothed_rates = rate_r.rolling(window=window_size, min_periods=1, center=True).mean()
        # smoothed_rates = np.convolve(rate_row[1:], np.ones(window_size) / window_size, mode='same')
        # smoothed_rates = gaussian_filter1d(rate_row[1:], sigma=5)
        # filter_row.append(rate_row[0])
        filter_row.extend(smoothed_rates)

        # 将原始和滤波后的数据行分别添加到 rates 中
        rates.append(occ_row)
        rates.append(rate_row)
        rates.append(filter_row)

    # rates to .csv
    array = np.array(rates).T
    pd.DataFrame(array).to_csv(fileName, header=None)

def water_Rate_Global(Water_Occur_removal, Cloud_Removal, fileName):  # 对比实验采用的阈值计算方法

    rates = []
    for i in range(0, 2):
        rates.append([])
        # rates[i].append(i)
    '''
    Water Reconstruction
    '''
    cfiles = os.listdir(Cloud_Removal)
    for file in cfiles:
        if file.split('.')[-1] == 'tif' or file.split('.')[-1] == 'png':
            rates[0].append(file.split('.')[0])
            img_dir = os.path.join(Cloud_Removal, file)
            imgMulti = gdal.Open(img_dir)
            image = imgMulti.ReadAsArray()

            # cloud removal
            Cloud_pixels = np.where(image[:, :] == 2)
            image[Cloud_pixels] = 0

            Water_path = os.path.join(Water_Occur_removal, file.split('.')[0] + '.tif')
            waterimg = gdal.Open(Water_path)
            water_img = waterimg.ReadAsArray()

            # get (i, j) positions of predicted water pixels
            Seg_pixels = np.where(image[:, :] == 1)
            result1 = Counter(water_img[Seg_pixels])

            dic1 = dict(result1)
            sum = 0
            for key in range(1, int(np.max(water_img)) + 1):
                if key in dic1.keys():
                    sum += int(dic1[key])
            threshold = sum * 0.17 / 100
            count = 0
            for key in range(1, int(np.max(water_img)) + 1):
                # if key in dic1.keys():
                #     count += int(dic1[key])
                if key in dic1.keys() and dic1[key] > threshold:
                    rates[1].append(key)
                    break
                #     rates[int(key)].append(int(dic1[key]))
                # else:
                #     rates[int(key)].append(0)
    # 1）字典按key排序
    # 2）字典转存CSV
    # rates to .csv
    array = np.array(rates)
    pd.DataFrame(array).to_csv(fileName, header=None)
    # [f.write('{0},{1}\n'.format(key, value)) for key, value in rates.items()]

def cloud_percentage(cloud_folder):
    # Rootpath = "D:\_FloodAreasChange"
    # # India1 北方邦
    # StudyArea = "_India"
    # floodday = 132
    # extentI = [1600, 1100, 2600, 2100]
    #
    # StudyArea1 = "_Pakistan"
    # floodday1 = 225
    # extentP = [800, 2000, 1800, 3000]
    #
    # #
    # StudyArea2 = "_Brazil"
    # floodday2 = 245
    # extentB = [1500, 900, 2500, 1900]
    #
    # StudyArea3 = "_USA10"
    # floodday3 = 139
    # extent = [700, 1800, 1800, 2900]  # Michigan

    Rootpath = "D:\_FloodAreasChange"
    StudyArea = "_India"
    floodday = 132
    # extent = [1900, 1100, 2900, 2100]


    # India2 Bardiya
    StudyArea1 = "_Pakistan"
    floodday1 = 225
    extent = [0, 0, 1000, 1000]

    #
    StudyArea2 = "_Brazil"
    floodday2 = 245
    # extent = [1700, 600, 2700, 1600]

    StudyArea3 = "_USA10"
    floodday3 = 139
    extent3 = [700, 1800, 1800, 2900]  # Michigan


    #
    # shape = [3850, 2880]
    shape = [3660, 3661]
    # shape = [2942, 2837]
    # shape = [1862, 3018]
    print(cloud_folder)
    files = os.listdir(cloud_folder)
    for file in files:
        if not file.endswith("tif"):
            continue
        cloud = gdal.Open(os.path.join(cloud_folder,file)).ReadAsArray()
        cloud_rez = cv2.resize(cloud,(shape[0],shape[1]))
        cloud_per = len(np.where(cloud_rez[extent[1]:extent[3],extent[0]:extent[2]]==1)[0])/(abs(extent[1]-extent[3])*abs(extent[0]-extent[2]))
        print(file,cloud_per)

def cloudremove_for_compare(image_path,Water_Occur_path,Water_Occur_adj,cloud_path,cloud_removed,water_removed,Water_Occur_adj_removed):
    image = gdal.Open(image_path).ReadAsArray()
    water_img = gdal.Open(Water_Occur_path).ReadAsArray()
    water_adj = gdal.Open(Water_Occur_adj).ReadAsArray()

    cloud_img = gdal.Open(cloud_path).ReadAsArray()[:,0:2300]
    cloud_resz = cv2.resize(cloud_img, (image.shape[1], image.shape[0]))
    cloud_pix = np.where(cloud_resz[:, :] == 1)

    image[cloud_pix] = 2
    cv2.imwrite(cloud_removed, image)

    water_img[cloud_pix] = 0
    cv2.imwrite(water_removed, water_img)

    water_adj[cloud_pix] = 0
    cv2.imwrite(Water_Occur_adj_removed, water_adj)


