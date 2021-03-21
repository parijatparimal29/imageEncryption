# imageEncryption

Watermarking based image encryption/decryption system using discrete cosine transformations and running via command line. 

Steps:

1. Upload an image A (an image that will be a placeholder)
2. Upload an image B (an image that we want to protect)
3. Embed image B into A to generate image C by using Discrete Cosine Transform (DCT)
4. Image C can now be distributed to trusted parties
5. Trusted party can decrypt image B from image C
