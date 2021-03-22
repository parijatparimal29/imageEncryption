import argparse
from PIL import Image
import numpy as np

def handle_input(filename):
    '''
        Converts filename into rgb matrix (or rgba depending on input image)
        input   : string => filename
        ouptut  : [[(r,g,b)]] or [[(r,g,b,a)]] => 2d matrix where each element is a tuple of r, g, b (or RGBA)
    '''
    
    img = Image.open(filename)
    rgb_matrix = np.array(img)
    return rgb_matrix
    

def handle_output(rgb_matrix, output_filename):
    '''
        Converts rgb (or rgba depending on input image) matrix into image file
        input    : [[(r,g,b)]] or [[(r,g,b,a)]] => 2d matrix where each element is a tuple of r, g, b (or RGBA)
                   string => filename
    '''
    
    img = Image.fromarray(rgb_matrix)
    img.save(output_filename)
    

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
    
    encrypted_rgb = None
    # -todo- Perform encryption based on encryption method and update encrypted_rgb
    if(encrypted_rgb is None):
        handle_output(to_hide_rgb, output_filename)
    else:
        handle_output(encrypted, output_filename)
    
def decrypt(encrypted_image, encryption_method, output_filename):
    '''
        Encrypts to_hide_image into placeholder method using encryption_method and outputs encrypted file as output_filename
        input    : string => filename of encrypted image
                   string => encryption method chosen
                   string => filename of output image
    '''
    
    encrypted_rgb = handle_input(encrypted_image)
    
    decrypted_rgb = None
    # -todo- Perform decryption based on encryption method and update decrypted_rgb
    
    handle_output(decrypted_rgb, output_filename)
    
    

if __name__ == '__main__':
    '''
        Main function => takes input files, output filename, encryption type and action to be perfomed as arguments and 
                         perform encryption and decryption as per the requirements on the input files and return output file.
             
    '''
    
    parser = argparse.ArgumentParser()
    
    # Parse input from command line
    parser.add_argument('--in1', help='enter placeholder image filename', default='bunny.png')
    parser.add_argument('--in2', help='enter image filename that needs to be hidden', default='cube.png')
    parser.add_argument('--infile', help='enter encrypted image filename to decrypt', default='bunny.png')
    parser.add_argument('--en', help='enter encryption method. Example - dct', default='dct')
    parser.add_argument('--action', help='enter action. enc to encrypt and dec to decrypt', default='enc')
    parser.add_argument('--out', help='enter output filename', default='out.png')
    args = parser.parse_args()
    
    # Use arguments to perform requested action
    output_filename = args.out
    
    # Encryption
    if(args.action == 'enc'):
        placeholder_image = args.in1
        to_hide_image = args.in2
        encryption_method = args.en
        encrypt(placeholder_image, to_hide_image, encryption_method, output_filename)
        
    # Decryption    
    if(args.action == 'dec'):
        encrypted_image = args.infile
        encryption_method = args.en
        decrypt(encrypted_image, encryption_method, output_filename)
        
    if(not (args.action == 'enc') and not (args.action == 'dec')):
        print("Invalid Arguments")
    