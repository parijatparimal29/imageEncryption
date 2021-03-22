# imageEncryption

Watermarking based image encryption/decryption system using discrete cosine transformations and running via command line. 

Steps:

1. Upload an image A (an image that will be a placeholder)
2. Upload an image B (an image that we want to protect)
3. Embed image B into A to generate image C by using Discrete Cosine Transform (DCT)
4. Image C can now be distributed to trusted parties
5. Trusted party can decrypt image B from image C

How to run:

To encrypt an image into a placeholder image:

$python iEnc.py --in1 <placeholder_filename> --in2 <hide_image_filename> --en <encryption_method> --action enc --out <output_filename>$

Example:
python iEnc.py --in1 placeholder.png --in2 hide.png --en dct --action enc --out out.png

To decrypt an image and get the original image:
python iEnc.py --infile <encrypted_image_filename> --en <encryption_method> --action dec --out <output_filename>

Example:
python iEnc.py --infile hidden.png --en dct --action dec --out out.png

Note: Include path for filenames if the filename not in same folder as iEnc.py
