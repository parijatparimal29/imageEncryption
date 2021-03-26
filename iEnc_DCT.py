import argparse
from skimage import io
from scipy.fftpack import dct, idct
import numpy as np

def handle_input(filename):
    '''
        Converts filename into rgb matrix (or rgba depending on input image)
        input   : string => filename
        ouptut  : [[(r,g,b)]] or [[(r,g,b,a)]] => 2d matrix where each element is a tuple of r, g, b (or RGBA)
    '''
    
    return io.imread(filename)/255
    

def handle_output(rgb_matrix, output_filename):
    '''
        Converts rgb (or rgba depending on input image) matrix into image file
        input    : [[(r,g,b)]] or [[(r,g,b,a)]] => 2d matrix where each element is a tuple of r, g, b (or RGBA)
                   string => filename
    '''
    
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

def encrypt(placeholder_image, to_hide_image, encryption_method, output_filename):
    '''
        Encrypts to_hide_image into placeholder method using encryption_method and outputs encrypted file as output_filename
        input    : string => placeholder filename
                   string => filename of image to hide
                   string => encryption method chosen
                   string => filename of output image
    '''
    
    placeholder_rgb = handle_input(placeholder_image)
    to_hide_rgb = handle_input(to_hide_image)
    placeholder_rgb, to_hide_rgb = handle_var_size(placeholder_rgb, to_hide_rgb)

    if(encryption_method == 'dct'):
        encryption_parameter = 0.01
    
        dct_image = dct(placeholder_rgb, norm='ortho') + (encryption_parameter * to_hide_rgb)
        encrypted_rgb = idct(dct_image, norm='ortho')
    
        handle_output(encrypted_rgb, output_filename)
    
def decrypt(placeholder_image, encrypted_image, encryption_method, output_filename):
    '''
        Encrypts to_hide_image into placeholder method using encryption_method and outputs encrypted file as output_filename
        input    : string => filename of encrypted image
                   string => encryption method chosen
                   string => filename of output image
    '''
    
    placeholder_rgb = handle_input(placeholder_image)
    encrypted_rgb = handle_input(encrypted_image)
    
    if(encryption_method == 'dct'):
        encryption_parameter = 0.01
    
        decrypted_rgb = (dct(encrypted_rgb, norm='ortho') - dct(placeholder_rgb, norm='ortho'))/encryption_parameter
        handle_output(decrypted_rgb, output_filename)
    


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
    