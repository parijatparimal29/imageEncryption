import argparse

def handle_input(filename):
    
    rgb_matrix = None
    # -todo- convert input file to rgb matrix 
    return rgb_matrix
    

def handle_output(rgb_matrix, output_filename):
    # -todo- convert rgb matrix to output file
    

def encrypt(placeholder_image, to_hide_image, encryption_method, output_filename):
    placeholder_rgb = handle_input(placeholder_image)
    to_hide_rgb = handle_input(to_hide_image)
    
    encrypted_rgb = None
    # -todo- Perform encryption based on encryption method and update encrypted_rgb
    
    handle_output(encrypted_rgb, output_filename)
    
def decrypt(encrypted_image, encryption_method, output_filename):
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
    parser.add_argument('--in1', help='enter placeholder image filename', default='input.png')
    parser.add_argument('--in2', help='enter image filename that needs to be hidden', default='input.png')
    parser.add_argument('--in', help='enter encrypted image filename to decrypt', default='input.png')
    parser.add_argument('--en', help='enter encryption method. Example - dct', default='dct')
    parser.add_argument('--action', help='enter action. enc to encrypt and dec to decrypt', default='enc')
    parser.add_argument('--out', help='enter output filename', default='output')
    args = parser.parse_args()
    
    # Use arguments to perform requested action
    output_filename = args.out
    
    # Encryption
    if(args.action =='enc'):
        placeholder_image = args.in1
        to_hide_image = args.in2
        encryption_method = args.en
        encrypt(placeholder_image, to_hide_image, encryption_method, output_filename)
        
    # Decryption    
    if(args.action =='dec'):
        encrypted_image = args.in
        encryption_method = args.en
        decrypt(encrypted_image, encryption_method, output_filename)
    