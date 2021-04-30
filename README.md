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
```bash
python iEnc.py --place <placeholder_filename> --hide <hide_image_filename> --en <encryption_method> --action enc --out <output_filename>
```
Example:
``` bash
python iEnc.py --place placeholder.png --hide hide.png --en dct --action enc --out hidden.png
```
To decrypt an image and get the original image:
```bash
python iEnc.py --place <placeholder_filename> --hidden <encrypted_image_filename> --en <encryption_method> --action dec --out <output_filename>
```
Example:
```bash
python iEnc.py --place placeholder.png --hidden hidden.png --en dct --action dec --out out.png
```
Note: Include path for filenames if the filename not in same folder as iEnc.py
<br>
=> encryption using only dct.
```bash
--en dct 
```
=> encryption using combination of dct and dwt.
```bash
--en dctdwt 
``` 

Optional arguments:

To add a shuffle algorithm on top of other chosen algorithm, use the below argument:
```bash
--shuffle 1 --r1 <integer-32> --r2 <integer-32>
```
r1 and r2 are optional arguments, these are seed values for random numbers used in shuffle algorithm. Recovering the secret image is only possible if shuffle, r1 and r2 values are same as was used while encryption. These r1 and r2 values could be secret tokens that can be used in integration with other security mechanisms.

To make the conversions lossless, use the below argument:
```
--ll 1
```
Note: The lossless algorithm is slower and consumes more memory. Also, the shuffle algorithm also slows down the encryption and decryption. Hence, usage of these options are only advised if required.


