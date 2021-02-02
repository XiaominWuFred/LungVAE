# lungMaskGen

Please add data sets:  
images/  
masks/  
train.csv  

under the following path to run the software package:
lungMaskGen/pairImg/

How to use:  
case 1: dnoise VAE for generating lung mask:  
python3 main.py --model dvae  
Output saved in lungMaskGen/dvaeOut  

case 2: Vae + Unet for generating lung mask:  
python3 main.py --model vaeUnet  
Output saved in lungMaskGen/vis  

case 3: Unet for generating lung mask:  
python3 main.py --model unet  
Output saved in lungMaskGen/vis  

if want to test blocked input, add --blockCustom after each case.