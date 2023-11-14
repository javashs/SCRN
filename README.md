# SCRN
Swin transformer for simultaneous denoising and interpolation of seismic data

Step 1: Download the dataset. 
Run the download_data.py file under the util file. If the download speed is slow, please go to this URL (https://wiki.seg.org/wiki) to download.
Step 2: Split the dataset. 
Run the get_patch.py file under the util file. Split segy files into patches and save them in npy data format.
Part 3: Training patches data. 
Please run the train.py file directly.
Part 4: Test the model. 
Choose an optimal model to use on the test data. Please run the test.py file directly.

model: models of network design.
test_data: synthetic data for the test.
train_data: synthetic data for training is obtained from https://wiki.seg.org/wiki/Open_data.
trained_model: trained model.
util: tools for training and testing.
test.py: test code.
train.py: training code.
Due to the sensitive nature of the commercial datasets, the raw data would remain confidential and would not be shared.
