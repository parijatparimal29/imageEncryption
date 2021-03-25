import argparse
import pywt
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
from scipy.fftpack import dct, idct
import numpy as np

def handle_input(filename):
    '''
        Converts filename into rgb matrix (or rgba depending on input image)
        input   : string => filename
        ouptut  : [[(r,g,b)]] or [[(r,g,b,a)]] => 2d matrix where each element is a tuple of r, g, b (or RGBA)
    '''
    return io.imread(filename)

def rgb_to_grey(rgb_img):
    return rgb2gray(rgb_img)

def handle_output(rgb_matrix, output_filename):
    '''
        Converts rgb (or rgba depending on input image) matrix into image file
        input    : [[(r,g,b)]] or [[(r,g,b,a)]] => 2d matrix where each element is a tuple of r, g, b (or RGBA)
                   string => filename
    '''
    io.imsave(output_filename,rgb_matrix)

def handle_var_size(image, size):
    '''
        Resizes the images based on the size of placeholder and watermark, making them have the same shape
        input    : placeholder and watermark image matrix
    '''
    image_resized = resize(image, (size, size), 1)
    return image_resized

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
            
def encrypt(placeholder_image, to_hide_image, encryption_method, output_filename='watermarked_image.png'):
    '''
        Encrypts to_hide_image into placeholder method using encryption_method and outputs encrypted file as output_filename
        input    : string => placeholder filename
                   string => filename of image to hide
                   string => encryption method chosen
                   string => filename of output image
    '''
    
    #Getting Image matrices
    placeholder_rgb = handle_input(placeholder_image)
    to_hide_rgb = handle_input(to_hide_image)
    
    #Resizing to correct sizes
    placeholder_rgb = handle_var_size(placeholder_rgb, 2048)
    to_hide_rgb = handle_var_size(to_hide_rgb, 128)
    
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
    handle_output(watermarked_img, output_filename)

def decrypt(encrypted_image, encryption_method, output_filename = 'watermark.png'):
    '''
        Encrypts to_hide_image into placeholder method using encryption_method and outputs encrypted file as output_filename
        input    : string => filename of encrypted image
                   string => encryption method chosen
                   string => filename of output image
    '''
    #Getting image matrix 
    encrypted_rgb = handle_input(encrypted_image)
    
    #Resizing to correct sizes
    encrypted_rgb = handle_var_size(encrypted_rgb, 2048)
    
    #Applying 2D multilevel Discrete Wavelet Transform
    wt_l1 = list(pywt.wavedec2(data = encrypted_rgb, wavelet = 'haar', level = 1))
    
    #Applying DCT by block size of 8
    dct_l1 = apply_by_block(dct, wt_l1[0], 8)
    
    #Extracting watermark
    combined_watermark = extract_by_block(dct_l1, 8)
    decrypted_img = np.array(combined_watermark).reshape(128, 128)
    
    #Saving the output watermark image
    #io.imshow(decrypted_img)
    handle_output(decrypted_img, output_filename)

if __name__ == '__main__':
    '''
        Main function => takes input files, output filename, encryption type and action to be perfomed as arguments and 
                         perform encryption and decryption as per the requirements on the input files and return output file.
             
    '''
    
    parser = argparse.ArgumentParser()
    
    # Parse input from command line
    parser.add_argument('--place', help='enter placeholder image filename', default='bunny.png')
    parser.add_argument('--hide', help='enter image filename that needs to be hidden', default='cube.png')
    parser.add_argument('--infile', help='enter encrypted image filename to decrypt', default='bunny.png')
    parser.add_argument('--en', help='enter encryption method. Example - dct', default='dct')
    parser.add_argument('--action', help='enter action. enc to encrypt and dec to decrypt', default='enc')
    parser.add_argument('--out', help='enter output filename', default='out.png')
    args = parser.parse_args()
    
    # Use arguments to perform requested action
    output_filename = args.out
    
    # Encryption
    if(args.action == 'enc'):
        placeholder_image = args.place
        to_hide_image = args.hide
        encryption_method = args.en
        encrypt(placeholder_image, to_hide_image, encryption_method, output_filename)
        
    # Decryption    
    if(args.action == 'dec'):
        placeholder_image = args.place
        encrypted_image = args.infile
        encryption_method = args.en
        decrypt(placeholder_image, encrypted_image, encryption_method, output_filename)
        
    if(not (args.action == 'enc') and not (args.action == 'dec')):
        print("Invalid Arguments")
    