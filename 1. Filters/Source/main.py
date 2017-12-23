# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def help_message():
    print("Usage: [Question_Number] [Input_Options] [Output_Options]")
    print("[Question Number]")
    print("1 Histogram equalization")
    print("2 Frequency domain filtering")
    print("3 Laplacian pyramid blending")
    print("[Input_Options]")
    print("Path to the input images")
    print("[Output_Options]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "[path to input image] " +
          "[output directory]")  # Single input, single output
    print(sys.argv[0] + " 2 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, three outputs
    print(sys.argv[0] + " 3 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, single output


# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================


def histogram_equalization(img_in):

    # Splitting image into three channels
    b, g, r = cv2.split(img_in)

    #calculating histogram for blue, calculating cdf for it
    #and calculating new intesity values
    histb = cv2.calcHist([b], [0], None, [256], [0, 256])
    cdf = np.cumsum(histb)
    cdf = np.around(np.subtract(cdf, np.amin(cdf)))
    cv2.divide(cdf, b.size, cdf)
    cv2.multiply(cdf, 255, cdf)

    #mapping new intesities for blue
    imbnorm = cdf[b.ravel()].reshape(b.shape)

    #calculating histogram for green, calculating cdf for it
    #and calculating new intesity values
    histg = cv2.calcHist([g], [0], None, [256], [0, 256])
    cdf = np.cumsum(histg)
    cdf = np.around(np.subtract(cdf, np.amin(cdf)))
    cv2.divide(cdf, g.size, cdf)
    cv2.multiply(cdf, 255, cdf)

    #mapping new intesities for green
    imgnorm = cdf[g.ravel()].reshape(g.shape)

    #calculating histogram for red, calculating cdf for it
    #and calculating new intesity values
    histr = cv2.calcHist([r], [0], None, [256], [0, 256])
    cdf = np.cumsum(histr)
    cdf = np.around(np.subtract(cdf, np.amin(cdf)))
    cv2.divide(cdf, r.size, cdf)
    cv2.multiply(cdf, 255, cdf)

    #mapping new intesities for red
    imrnorm = cdf[r.ravel()].reshape(r.shape)

    #merge the three channels
    img_out = cv2.merge([imbnorm, imgnorm, imrnorm])

    return True, img_out


def Question1():

    # Read in input images
    input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)

    # Histogram equalization
    succeed, output_image = histogram_equalization(input_image)

    # Write out the result
    output_name = sys.argv[3] + "1.jpg"
    cv2.imwrite(output_name, output_image)

    return True


# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================


def low_pass_filter(img_in):

    #take fourier transform
    dft = cv2.dft(np.float32(img_in), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    #mask high frequencies
    rows, cols = img_in.shape
    crow, ccol = rows / 2, cols / 2
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 10:crow + 10, ccol - 10:ccol + 10] = 1

    #apply mask
    fshift = dft_shift * mask

    #inverse fourier transform
    f_ishift = np.fft.ifftshift(fshift)

    img_out = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    cv2.resize(img_out, img_in.shape, img_out)

    return True, img_out


def high_pass_filter(img_in):

    #take fourier transform
    dft = cv2.dft(np.float32(img_in), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    #mask low frequencies
    rows, cols = img_in.shape
    crow, ccol = rows / 2, cols / 2
    dft_shift[crow - 10:crow + 10, ccol - 10:ccol + 10] = 0

    #take inverse fourier transform
    f_ishift = np.fft.ifftshift(dft_shift)

    img_out = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    return True, img_out


def ft(im, newsize=None):
    dft = np.fft.fft2(np.float32(im), newsize)
    return np.fft.fftshift(dft)


def ift(shift):
    f_ishift = np.fft.ifftshift(shift)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)


def deconvolution(img_in):

    # get gaussian kernel
    gk = cv2.getGaussianKernel(21, 5)
    gk = gk * gk.T

    #take transform of kernel and image
    imf = ft(img_in, (img_in.shape[0], img_in.shape[1]))
    gkf = ft(gk, (img_in.shape[0], img_in.shape[1]))

    #division in frequency domain
    imconvf = imf / gkf

    # Deconvolution result
    img_out = ift(imconvf)
    img_out = cv2.multiply(img_out, 255)

    return True, img_out


def Question2():

    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], 0)
    input_image2 = cv2.imread(sys.argv[3],
                              cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # Low and high pass filter
    succeed1, output_image1 = low_pass_filter(input_image1)
    succeed2, output_image2 = high_pass_filter(input_image1)

    # Deconvolution
    succeed3, output_image3 = deconvolution(input_image2)

    # Write out the result
    output_name1 = sys.argv[4] + "2.jpg"
    output_name2 = sys.argv[4] + "3.jpg"
    output_name3 = sys.argv[4] + "4.jpg"

    cv2.imwrite(output_name1, output_image1)
    cv2.imwrite(output_name2, output_image2)
    cv2.imwrite(output_name3, output_image3)

    return True


# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================


def laplacian_pyramid_blending(img_in1, img_in2):

    #making images same size and recangular
    img_in1 = img_in1[:, :img_in1.shape[0]]
    img_in2 = img_in2[:img_in1.shape[0], :img_in1.shape[0]]

    #gaussian pyramid of first image
    g1 = img_in1.copy()
    gPyr1 = [g1]
    for i in xrange(6):
        g1 = cv2.pyrDown(g1)
        gPyr1.append(g1)

    #gaussian pyramid of second image
    g2 = img_in2
    gPyr2 = [g2]
    for i in xrange(6):
        g2 = cv2.pyrDown(g2)
        gPyr2.append(g2)

    #laplacian pyramid of first image
    lp1 = [gPyr1[5]]
    for i in xrange(5, 0, -1):
        tempGE = cv2.pyrUp(gPyr1[i])
        L = cv2.subtract(gPyr1[i - 1], tempGE)
        lp1.append(L)

    #laplacian pyramid of second image
    lp2 = [gPyr2[5]]
    for i in xrange(5, 0, -1):
        tempGE = cv2.pyrUp(gPyr2[i])
        L = cv2.subtract(gPyr2[i - 1], tempGE)
        lp2.append(L)

    #blending
    LS = []
    for la, lb in zip(lp1, lp2):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
        LS.append(ls)

    #cupsampling and combining
    ls = LS[0]
    for i in xrange(1, 6):
        ls = cv2.pyrUp(ls)
        ls = cv2.add(ls, LS[i])

    img_out = ls  # Blending resultn

    return True, img_out


def Question3():

    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR)

    # Laplacian pyramid blending
    succeed, output_image = laplacian_pyramid_blending(input_image1,
                                                       input_image2)

    # Write out the result
    output_name = sys.argv[4] + "5.jpg"
    cv2.imwrite(output_name, output_image)

    return True


if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) < 4):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])

        if (question_number == 1 and not (len(sys.argv) == 4)):
            help_message()
            sys.exit()
        if (question_number == 2 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number == 3 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
            print("Input parameters out of bound ...")
            sys.exit()

    function_launch = {
        1: Question1,
        2: Question2,
        3: Question3,
    }

    # Call the function
    function_launch[question_number]()
