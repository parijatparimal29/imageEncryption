import argparse
import pywt
import os
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
from scipy.fftpack import dct, idct
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from math import log10, sqrt

def save_as_fp(matrix, filename):
    '''
        Writes the matrix with fp to a .txt file to prevent loss
        input    : numpy.ndarray => 2d matrix where each element is a tuple of r, g, b (or RGBA)
                   string => filename of output txt file
    '''
    file = open(filename, "w", encoding ="utf-8")
    
    #Store the shape of the matrix
    file.write(' '.join([str(elem) for elem in matrix.shape]) + '\n')
    
    #Store the pixel values with floating point
    for row in matrix:
        for tup in row:
            file.write(' '.join([str(elem) for elem in [tup]]) + '\n')
            
    file.close()
    
def read_as_fp(filename):
    '''
        Reads the matrix fp from a .txt file to prevent loss
        input    : string => filename of matrix txt file
    '''
    file = open(filename, "r", encoding ="utf-8")
    textString = file.read().splitlines()
    shape = textString[0].split()
    num_of_rows = int(shape[0])
    num_of_tuples = int(shape[1])
    num_of_elems = int(shape[2])
    tempArr = list()
    for s in textString[1:]:
        s = s.split('[')[-1].split(']')[0]
        for num in s.split():
            tempArr.append(float(num))
    array = np.array(tempArr).reshape(num_of_rows, num_of_tuples, num_of_elems)
    #print(array.dtype.type)
    return array

def handle_input(filename, matrix_mode=0):
    '''
        Converts filename into rgb matrix (or rgba depending on input image)
        input   : string => filename
        ouptut  : [[(r,g,b)]] or [[(r,g,b,a)]] => 2d matrix where each element is a tuple of r, g, b (or RGBA)
    '''
    if matrix_mode:
        new_filename = filename[:filename.rfind('.')+1]+'txt'
        if(os.path.exists(new_filename)):
            return read_as_fp(new_filename)
    return io.imread(filename)/255
    

def handle_output(rgb_matrix, output_filename, matrix_mode=0):
    '''
        Converts rgb (or rgba depending on input image) matrix into image file
        input    : [[(r,g,b)]] or [[(r,g,b,a)]] => 2d matrix where each element is a tuple of r, g, b (or RGBA)
                   string => filename
    '''
    if(len(rgb_matrix.shape)>2 and rgb_matrix.shape[2] == 4):
        rgb_matrix[:,:,3] = 1
    if matrix_mode:
        save_as_fp(rgb_matrix, output_filename[:output_filename.rfind('.')+1]+'txt')
    io.imsave(output_filename,rgb_matrix)

def handle_var_size(placeholder, watermark):
    '''
        Resizes the images based on the size of placeholder and watermark, making them have the same shape
        input    : placeholder and watermark image matrix
    '''
    M = max(placeholder.shape[0], watermark.shape[0])
    N = max(placeholder.shape[1], watermark.shape[1])
    
    placeholder_resized = resize(placeholder, (M, N), anti_aliasing=True)
    watermark_resized = resize(watermark, (M, N), anti_aliasing=True)
    return placeholder_resized, watermark_resized  

def handle_var_resize(image, size):
    '''
        Resizes the images based on the size of placeholder and watermark, making them have the same shape
        input    : placeholder and watermark image matrix
    '''
    image_resized = resize(image, (size, size), 1)
    return image_resized

def rgb_to_grey(rgb_img):
    return rgb2gray(rgb_img)

def apply_by_block(fn, image, block_size=8):
    img_dct = np.empty((len(image[0]), len(image[0])))
    for i in range (0, len(image[0]), block_size):
        for j in range (0, len(image[0]), block_size):
            current_block = image[i:i+block_size, j:j+block_size]
            current_block_dct = fn(fn(current_block.T, norm="ortho").T, norm="ortho")
            img_dct[i:i+block_size, j:j+block_size] = current_block_dct
    return img_dct

def embed_by_block(img_dct, to_hide_array, block_size=8):
    idx = 0
    for i in range (0, len(img_dct), block_size):
        for j in range (0, len(img_dct), block_size):
            if idx < len(to_hide_array):
                current_dct = img_dct[i:i+block_size, j:j+block_size]
                current_dct[5][5] = to_hide_array[idx]
                img_dct[i:i+block_size, j:j+block_size] = current_dct
                idx += 1 
    return img_dct

def extract_by_block(img_dct, block_size=8):
    combined_watermark = []
    for x in range (0, len(img_dct), block_size):
        for y in range (0, len(img_dct), block_size):
            inner_dct = img_dct[x:x+block_size, y:y+block_size]
            combined_watermark.append(inner_dct[5][5])
    return combined_watermark

def get_loss(img1_file, img2_file):
    img1 = handle_input(img1_file)
    img2 = handle_input(img2_file)
    
    #Calculating Mean squared error
    mse = np.sum((img1 - img2)**2) / img1.size
    
    #Calculating Peak Signal-to-Noise Ratio (PSNR)
    psnr = 100
    if mse != 0:
        max_val = 255.0
        psnr = 20*log10(max_val/sqrt(mse))
    return mse, psnr

def shuffle_image(input_rgb, r1, r2):
    rs_row = RandomState(MT19937(SeedSequence(r1)))
    rs_col = RandomState(MT19937(SeedSequence(r2)))
    
    row_size = len(input_rgb)
    col_size = len(input_rgb[0])
    
    shuf_rgb = np.zeros(input_rgb.shape)
    
    rows = list(range(row_size))
    for i in range(row_size):
        row = rs_row.choice(rows)
        rows.remove(row)
        cols = list(range(col_size))
        for j in range(col_size):
            col = rs_col.choice(cols)
            cols.remove(col)
            shuf_rgb[row][col] = input_rgb[i][j]
            
    return shuf_rgb

def inverse_shuffle(input_rgb, r1, r2):
    rs_row = RandomState(MT19937(SeedSequence(r1)))
    rs_col = RandomState(MT19937(SeedSequence(r2)))
    
    row_size = len(input_rgb)
    col_size = len(input_rgb[0])
    
    shuf_rgb = np.zeros(input_rgb.shape)
    
    rows = list(range(row_size))
    for i in range(row_size):
        row = rs_row.choice(rows)
        rows.remove(row)
        cols = list(range(col_size))
        for j in range(col_size):
            col = rs_col.choice(cols)
            cols.remove(col)
            shuf_rgb[i][j] = input_rgb[row][col]
            
    return shuf_rgb

def encrypt_dctdwt(placeholder_image, to_hide_image, encryption_method, output_filename, shuffle, r1, r2, ll):
    '''
        Encrypts to_hide_image into placeholder method using encryption_method and outputs encrypted file as output_filename
        input    : string => placeholder filename
                   string => filename of image to hide
                   string => encryption method chosen
                   string => filename of output image
    '''
    
    #Getting Image matrices
    placeholder_rgb = handle_input(placeholder_image, ll)
    to_hide_rgb = handle_input(to_hide_image, ll)
    
    # Shuffle secret image
    if(shuffle):
        to_hide_rgb = shuffle_image(to_hide_rgb, r1, r2)
    
    #Resizing to correct sizes
    placeholder_rgb = handle_var_resize(placeholder_rgb, 2048)
    to_hide_rgb = handle_var_resize(to_hide_rgb, 128)
    
    #Converting rgb to greyscale
    placeholder_rgb = rgb_to_grey(placeholder_rgb)
    to_hide_rgb = rgb_to_grey(to_hide_rgb)
    
    #Watermarking starts here
    
    #Applying 2D multilevel Discrete Wavelet Transform
    wt_l1 = list(pywt.wavedec2(data = placeholder_rgb, wavelet = 'haar', level = 1))
    
    #Applying DCT 
    dct_l1 = apply_by_block(dct, wt_l1[0], 8)
    
    #Embed watermark into image
    dct_l1 = embed_by_block(dct_l1, to_hide_rgb.ravel(), 8)
    
    #Applying IDCT
    wt_l1[0] = apply_by_block(idct, dct_l1, 8)
    
    #2D multilevel reconstruction 
    watermarked_img = pywt.waverec2(wt_l1, 'haar')
    
    #Watermarking ends here
    
    #Saving the output watermarked image
    #io.imshow(watermarked_img)
    handle_output(watermarked_img, output_filename, ll)
    
def decrypt_dctdwt(encrypted_image, output_filename, shuffle, r1, r2, ll):
    '''
        Encrypts to_hide_image into placeholder method using encryption_method and outputs encrypted file as output_filename
        input    : string => filename of encrypted image
                   string => encryption method chosen
                   string => filename of output image
    '''
    
    #Getting image matrix 
    encrypted_rgb = handle_input(encrypted_image, ll)
    
    #Resizing to correct sizes
    encrypted_rgb = handle_var_resize(encrypted_rgb, 2048)
    
    #Applying 2D multilevel Discrete Wavelet Transform
    wt_l1 = list(pywt.wavedec2(data = encrypted_rgb, wavelet = 'haar', level = 1))
    
    #Applying DCT by block size of 8
    dct_l1 = apply_by_block(dct, wt_l1[0], 8)
    
    #Extracting watermark
    combined_watermark = extract_by_block(dct_l1, 8)
    decrypted_img = np.array(combined_watermark).reshape(128, 128)
    
    # Inverse shuffle to get secret image
    if(shuffle):
        decrypted_img = inverse_shuffle(decrypted_img, r1, r2)
    
    #Saving the output watermark image
    handle_output(decrypted_img, output_filename, ll)

def enc_dct(encryption_parameter, placeholder_image, to_hide_image, encryption_method, output_filename, shuffle, r1, r2, ll):
    '''
        Encrypts to_hide_image into placeholder method using encryption_method and outputs encrypted file as output_filename
        input    : float  => encryption parameter
                   string => placeholder filename
                   string => filename of image to hide
                   string => encryption method chosen
                   string => filename of output image
    '''
    
    placeholder_rgb = handle_input(placeholder_image,ll)
    to_hide_rgb = handle_input(to_hide_image,ll)
    
    # Shuffle secret image
    if(shuffle):
        to_hide_rgb = shuffle_image(to_hide_rgb, r1, r2)
    
    placeholder_rgb, to_hide_rgb = handle_var_size(placeholder_rgb, to_hide_rgb)

    dct_image = dct(placeholder_rgb, norm='ortho') + (encryption_parameter * to_hide_rgb)
    encrypted_rgb = idct(dct_image, norm='ortho')
    
    handle_output(encrypted_rgb, output_filename, ll)
    
def decrypt_dct(encryption_parameter, placeholder_image, encrypted_image, encryption_method, output_filename, shuffle, r1, r2,ll):
    '''
        Encrypts to_hide_image into placeholder method using encryption_method and outputs encrypted file as output_filename
        input    : string => filename of encrypted image
                   string => encryption method chosen
                   string => filename of output image
    '''
    
    placeholder_rgb = handle_input(placeholder_image)
    encrypted_rgb = handle_input(encrypted_image, ll)
   
    decrypted_rgb = (dct(encrypted_rgb, norm='ortho') - dct(placeholder_rgb, norm='ortho'))/encryption_parameter
    
    # Inverse shuffle to get secret image
    if(shuffle):
        decrypted_rgb = inverse_shuffle(decrypted_rgb, r1, r2)
    
    handle_output(decrypted_rgb, output_filename)
    
def encrypt(placeholder_image, to_hide_image, encryption_method, output_filename, shuffle=0, r1=0, r2=0, ll=0):
    '''
        Encrypts to_hide_image into placeholder method using encryption_method and outputs encrypted file as output_filename
        input    : string => placeholder filename
                   string => filename of image to hide
                   string => encryption method chosen
                   string => filename of output image
    '''

    if(encryption_method == 'dct'):
        encryption_parameter = 0.01
        enc_dct(encryption_parameter, placeholder_image, to_hide_image, encryption_method, output_filename, shuffle, r1, r2,ll)
        
    elif(encryption_method == 'dctdwt'):
        encrypt_dctdwt(placeholder_image, to_hide_image, encryption_method, output_filename, shuffle, r1, r2,ll)
        
    else:
        print("Encryption Method not specified or unavailable")
    
def decrypt(placeholder_image, encrypted_image, encryption_method, output_filename, shuffle=0, r1=0, r2=0, ll=0):
    '''
        Encrypts to_hide_image into placeholder method using encryption_method and outputs encrypted file as output_filename
        input    : string => filename of encrypted image
                   string => encryption method chosen
                   string => filename of output image
    '''
    
    if(encryption_method == 'dct'):
        encryption_parameter = 0.01
        decrypt_dct(encryption_parameter, placeholder_image, encrypted_image, encryption_method, output_filename, shuffle, r1, r2,ll)
    
    elif(encryption_method == 'dctdwt'):
        decrypt_dctdwt(encrypted_image, output_filename, shuffle, r1, r2,ll)
        
    else:
        print("Encryption Method not specified or unavailable")

if __name__ == '__main__':
    '''
        Main function => takes input files, output filename, encryption type and action to be perfomed as arguments and 
                         perform encryption and decryption as per the requirements on the input files and return output file.
             
    '''
    
    parser = argparse.ArgumentParser()
    
    # Parse input from command line
    parser.add_argument('--place', help='enter placeholder image filename', default='bunny.png')
    parser.add_argument('--hide', help='enter image filename that needs to be hidden', default='cube.png')
    parser.add_argument('--hidden', help='enter encrypted image filename to decrypt', default='bunny.png')
    parser.add_argument('--en', help='enter encryption method. Example - dct', default='dct')
    parser.add_argument('--action', help='enter action. enc to encrypt and dec to decrypt', default='enc')
    parser.add_argument('--out', help='enter output filename', default='out.png')
    parser.add_argument('--shuffle', help='Shuffle while watermarking 0 for False and 1 for True', default=0)
    parser.add_argument('--r1', help='Enter seed for shuffle', default=0)
    parser.add_argument('--r2', help='Enter seed for shuffle', default=0)
    parser.add_argument('--ll', help='Watermarking - 1 for slow but lossless and 0 for fast with loss', default=0)
    args = parser.parse_args()
    
    # Use arguments to perform requested action
    output_filename = args.out
    placeholder_image = args.place
    encryption_method = args.en
    shuffle = int(args.shuffle)
    r1 = int(args.r1)
    r2 = int(args.r2)
    matrix_mode = int(args.ll)
    
    # Encryption
    if(args.action == 'enc'):
        to_hide_image = args.hide
        encrypt(placeholder_image, to_hide_image, encryption_method, output_filename, shuffle, r1, r2, matrix_mode)
        
    # Decryption    
    if(args.action == 'dec'):
        encrypted_image = args.hidden
        decrypt(placeholder_image, encrypted_image, encryption_method, output_filename, shuffle, r1, r2, matrix_mode)
        
    if(not (args.action == 'enc') and not (args.action == 'dec')):
        print("Invalid Arguments")
    