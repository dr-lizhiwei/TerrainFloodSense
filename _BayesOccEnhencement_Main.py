import argparse
from HistogramMatching import histogram_match
import cv2
from osgeo import gdal
import numpy as np
from collections import Counter
from scipy.ndimage import median_filter
import copy


def parse_args():
    parser = argparse.ArgumentParser(description=" ")
    parser.add_argument('-Water_Occur_path', help='path to water occurrence',
                        default=r".\TestDataSet\WaterOccurrence.tif")
    parser.add_argument('-HAND_Path', help='path to HAND index', default=r".\TestDataSet\HAND.tif")
    parser.add_argument('-DEM_Path', help='path to DEM/DSM data', default=r".\TestDataSet\DSM.tif")
    parser.add_argument('-wt', help='weight of HAND (0~1)', default=0.7)
    parser.add_argument('-thr', help='when WO lower than thr, it will be improved (0~100)', default=5)
    parser.add_argument('-classes', help='the interpolation classes of original WO', default=1500)
    parser.add_argument('-Occ_Bayes', help='whether to save the geo-based water occurrence (none/savepath)', default=None)
    parser.add_argument('-Occ_Bayes_Matching', help='savepath of histogram matching water occurrence',
                        default=r".\TestDataSet\WaterOccurrence_bayes_matching.tif")

    args = parser.parse_args()
    return args


def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def combine_DEM_Hand(dempath, handpath, wt):
    filter_size = 7

    dem_array = gdal.Open(dempath).ReadAsArray()
    dem_array = np.where(dem_array == 32767, 0, dem_array)
    # dem_array = dem_array + np.min(dem_array) + 1

    dem_ds = gdal.Open(handpath)
    hand_array = dem_ds.ReadAsArray()
    hand_array = np.where(hand_array == 32767, 0, hand_array)
    # hand_array = cv2.resize(hand_array, (dem_array.shape[1], dem_array.shape[0])).astype(int)
    # hand_array = hand_array+ np.min(hand_array)+1

    hand_array = median_filter(hand_array, size=filter_size)

    normalized_dem = normalize(dem_array)
    normalized_hand = normalize(hand_array)

    weight_hand = wt
    weight_dem = 1 - weight_hand

    risk_score = weight_hand * normalized_hand + weight_dem * normalized_dem

    # risk_score =  dem_array * hand_array
    # normalized_risk = normalize(risk_score)
    # normalized_risk = median_filter(normalized_risk, size=filter_size)

    return median_filter(risk_score, size=filter_size)
    # cv2.imwrite(riskpath,risk_score)


# modified the calculation method for democc, and update the weight calculation
def ReOcc_Weight_Bayes_lowOcc2(handpath, dempath, wt, OccFile, OccRejustPath, n, w=3849, h=2880):
    AdjImg = combine_DEM_Hand(dempath, handpath, wt)
    AdjImg = np.where(AdjImg == 32767, -1, AdjImg)
    dicAdj = dict(Counter(AdjImg[np.where(AdjImg > -1)]))
    print(len(dicAdj.keys()))

    for k in dicAdj.keys():
        dicAdj[k] = []

    w = AdjImg.shape[1]
    h = AdjImg.shape[0]
    WaterOccorg = gdal.Open(OccFile).ReadAsArray()
    WaterOccorg = cv2.resize(WaterOccorg, (w, h))

    # Calculate Occ under HAND conditions
    Occ_0 = np.where(WaterOccorg == 0)  # Get the region where occurrence is 0
    g_array = np.array(Occ_0).transpose(1, 0)
    total_coordinates = g_array.shape[0]  # Total number of zero occurrence pixels
    numw = (WaterOccorg.size - total_coordinates)  # Count the non-zero pixels
    if total_coordinates < numw:
        numw = total_coordinates
    # Retain only numw*0.5 zero pixels and discard the others to balance positive and negative occurrences
    nummask = total_coordinates - numw
    random_coordinates = g_array[
        np.random.choice(total_coordinates, int(nummask), replace=False)]  # Sample random zero pixels to mask
    AdjImg_copy = AdjImg.copy()
    for pnt in random_coordinates:
        AdjImg_copy[pnt[0], pnt[1]] = -1  # Set the selected pixels to -1 to exclude them from further calculation

    # For lower Ha values, the sum will be larger, but as the range increases, Ha will disproportionately increase, leading to more issues in sum calculation
    Rate_PAB2 = dicAdj.copy()
    for ad in Rate_PAB2.keys():
        ha_pix = np.where(AdjImg_copy == ad)
        if len(ha_pix[0]) != 0:
            Rate_PAB2[ad] = sum(WaterOccorg[ha_pix]) / len(
                ha_pix[0])  # Calculate the average occurrence value for each adjacency
        else:
            Rate_PAB2[ad] = 0  # Assign zero if no occurrences are found for the adjacency value

    WaterOcc_Rejust2 = np.array(WaterOccorg.copy(), dtype=np.float32)

    for pts in g_array:
        Ha = AdjImg[pts[0], pts[1]]
        WaterOcc_Rejust2[pts[0], pts[1]] = Rate_PAB2[Ha]

    OccRejustFile2 = OccRejustPath + '_democc.tif'
    print(OccRejustFile2)
    cv2.imwrite(OccRejustFile2, WaterOcc_Rejust2)

    # Modify the range of Occ from 0 to n
    OCC_S = []
    OCC_len = []
    for occ in range(n):
        OCC_S.append(np.where(WaterOccorg == occ))  # Get pixel indices for each occurrence class
        OCC_len.append(len(OCC_S[occ][0]))  # Calculate the number of pixels for each class

    num_pix = min(OCC_len)  # Select the minimum number of pixels across all occurrence classes
    for i in range(n):
        itm = OCC_S[i]
        array = np.array(itm).transpose(1, 0)
        random_coordinates = array[
            np.random.choice(array.shape[0], int(num_pix), replace=False)]  # Sample random pixels
        newpix = random_coordinates.transpose(1, 0)
        itmnew = tuple(np.array(newpix[0])), tuple(np.array(newpix[1]))
        dicOcc = dict(Counter(AdjImg[itmnew]))  # Count the adjacency values at the sampled pixels
        for key in dicAdj.keys():
            if key in dicOcc.keys():
                dicAdj[key].append(dicOcc[key])  # Add the count of adjacency values to the dictionary
            else:
                dicAdj[key].append(0)  # Assign zero if the adjacency value is not found

    Rate_PAB = copy.deepcopy(dicAdj)  # Calculate the confidence (weight) of the original Water Occurrence
    for key, values in dicAdj.items():
        total = sum(values)
        if total != 0:
            Rate_PAB[key] = [value / total for value in values]  # Normalize the weight for each adjacency
        else:
            Rate_PAB[key] = [0] * len(values)

    # Adjust the water occurrence values based on the calculated weight
    WaterOcc_Rejust = np.array(np.where(WaterOccorg >= n, 1, 0), dtype=np.float32)
    Occ_0 = np.where(WaterOcc_Rejust == 0)  # Get the region where occurrence is 0
    g_array = np.array(Occ_0).transpose(1, 0)
    for pts in g_array:
        Ha = AdjImg[pts[0], pts[1]]
        if Ha > -1:
            rate = Rate_PAB[Ha][int(WaterOccorg[pts[0], pts[1]])]  # Get the adjusted rate for each pixel
            WaterOcc_Rejust[pts[0], pts[1]] = rate  # Assign the adjusted rate
        else:
            WaterOcc_Rejust[pts[0], pts[1]] = 0  # Set to zero if the adjacency is invalid

    OccRejustFile = OccRejustPath + '_weight.tif'
    print(OccRejustFile)
    cv2.imwrite(OccRejustFile, WaterOcc_Rejust)  # Save the water occurrence map with adjusted weights

    WaterOcc_Weight = WaterOcc_Rejust * WaterOccorg + (1 - WaterOcc_Rejust) * WaterOcc_Rejust2
    OccRejustFile3 = OccRejustPath + '_nostrech.tif'
    cv2.imwrite(OccRejustFile3, WaterOcc_Weight)


# Use repeated random sampling to alleviate the class imbalance problem
def ReOcc_Weight_Bayes_RandomSample(handpath, dempath, wt, OccFile, OccRejustPath, w=3849, h=2880):
    AdjImg = combine_DEM_Hand(dempath, handpath, wt)
    floodrisk = np.where(AdjImg == 32767, -1, AdjImg)  # Handle invalid data (32767) in adjacency image
    w = AdjImg.shape[1]  # Get the width of the adjacency image
    h = AdjImg.shape[0]  # Get the height of the adjacency image
    WaterOccorg = gdal.Open(OccFile).ReadAsArray()  # Read the original water occurrence data
    WaterOccorg = cv2.resize(WaterOccorg, (w, h))  # Resize the water occurrence image to match adjacency image size

    # Get unique values in the floodrisk map
    unique_values = np.unique(floodrisk)
    # Create a new image to store results
    new_image = np.zeros_like(floodrisk, dtype=np.float32)

    sampling_rounds = 10  # Number of sampling rounds

    seed_number = int(floodrisk.size / unique_values.size * 100)  # Number of pixels to sample
    print("seed_number: ", seed_number)

    for value in unique_values:
        if value == -1:
            continue  # Skip invalid values

        mask = (floodrisk == value)
        pixel_indices = np.where(mask)
        num_pixels = pixel_indices[0].size

        # If there are fewer pixels than seed_number, use all pixels
        if num_pixels <= seed_number:
            mean_wo = WaterOccorg[mask].mean()  # Calculate the mean water occurrence for these pixels
            new_image[mask] = mean_wo  # Assign the mean value to the corresponding pixels in the new image
        else:
            mean_values = []

            # Perform multiple rounds of random sampling
            for _ in range(sampling_rounds):
                # Randomly select seed_number pixels
                sampled_indices = np.random.choice(num_pixels, seed_number, replace=False)
                sampled_pixels = WaterOccorg[pixel_indices[0][sampled_indices], pixel_indices[1][sampled_indices]]
                mean_values.append(
                    sampled_pixels.mean())  # Append the mean water occurrence value for the sampled pixels

            # Calculate the final mean value from all sampling rounds
            final_mean = np.mean(mean_values)
            new_image[mask] = final_mean

        # Calculate the mean and standard deviation of the new image and the original water occurrence data
    mean_new_image = np.mean(new_image)
    std_new_image = np.std(new_image)

    mean_wo = np.mean(WaterOccorg)
    std_wo = np.std(WaterOccorg)

    # Adjust the new image's mean and standard deviation to match the original water occurrence data
    adjusted_image = (new_image - mean_new_image) / std_new_image * std_wo + mean_wo

    cv2.imwrite(OccRejustPath + '_adj_sample_mean.tif', adjusted_image)


# A faster Bayes calculation method
def ReOcc_Weight_Bayes_fast(handpath, dempath, wt, OccFile, OccRejustPath, thr, classes, matching_path):
    AdjImg = combine_DEM_Hand(dempath, handpath, wt)
    floodrisk = np.where(AdjImg == 32767, -1, AdjImg)
    w = AdjImg.shape[1]
    h = AdjImg.shape[0]
    WaterOccorg = gdal.Open(OccFile).ReadAsArray()
    WaterOccorg = cv2.resize(WaterOccorg, (w, h))

    # Get unique values in the floodrisk map
    unique_values = np.unique(floodrisk)
    # Create a new image to store the results
    new_image = np.zeros_like(floodrisk, dtype=np.float32)

    # Calculate the average water occurrence for each unique floodrisk value region
    for value in unique_values:
        # print(count)
        # count+=1
        mask = (floodrisk == value)
        mean_wo = WaterOccorg[mask].mean()
        new_image[mask] = mean_wo

        # Calculate the mean and standard deviation of the new image and the original water occurrence data
    mean_new_image = np.mean(new_image)
    std_new_image = np.std(new_image)

    mean_wo = np.mean(WaterOccorg)
    std_wo = np.std(WaterOccorg)

    # Adjust the new image's mean and standard deviation to match the original water occurrence data
    adjusted_image = (new_image - mean_new_image) / std_new_image * std_wo + mean_wo

    if OccRejustPath != None:
        cv2.imwrite(OccRejustPath, adjusted_image)

    # Histogram matching
    occ_to_Int = histogram_match(adjusted_image, WaterOccorg,thr,classes)
    cv2.imwrite(matching_path, occ_to_Int)

def main():
    # unpack args
    args = parse_args()

    # Bayes adjustment
    handpath = args.HAND_Path
    dempath = args.DEM_Path
    OccFile = args.Water_Occur_path
    OccRejustPath = args.Occ_Bayes
    wt = args.wt
    thr = args.thr
    classes = args.classes
    matching_path = args.Occ_Bayes_Matching
    ReOcc_Weight_Bayes_fast(handpath, dempath, wt, OccFile, OccRejustPath, thr, classes, matching_path)


if __name__ == '__main__':
    main()
