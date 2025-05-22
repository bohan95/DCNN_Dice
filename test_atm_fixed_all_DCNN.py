import os
import torch
import numpy as np
import h5py
import torch.nn as nn
import torch.utils.data as utils
from torch.autograd import Variable

# Custom module imports
import RESNET152_ATT_naive
from Util import preprocess_fiber_input

# Load .mat function
def loadmat(filename):
    with h5py.File(filename, 'r') as data:
        if 'Whole_tracks' not in data:
            raise KeyError(f"❌ Error: 'Whole_tracks' doesn't exist in {filename}!")
        
        whole_tracks = data['Whole_tracks']
        if 'count' not in whole_tracks or 'data' not in whole_tracks:
            raise KeyError(f"❌ Error: 'Whole_tracks' is incomplete in {filename}, included: {list(whole_tracks.keys())}")

        count = int(whole_tracks['count'][()].item())
        track = [np.transpose(data[whole_tracks['data'][i].item()][:]).astype(np.float32) for i in range(count)]
    
    return {'tracks': {'count': count, 'data': track}}

# Softmax function
def mySoftmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)[:, np.newaxis]
    return e_x / div

# Flip fibers
def udflip_x_only(X_nparray, shuffle=True):
    if X_nparray.shape[2] == 4:
        if np.std(X_nparray[:, 0, :]) > np.std(X_nparray[:, -1, :]):
            print("Detected special info in the first column, swapping...")
            X_nparray = np.concatenate((X_nparray[:, 1:, :], X_nparray[:, 0:1, :]), axis=1)
    
    X_flipped = np.flip(X_nparray, axis=2)  
    X_aug = np.vstack((X_nparray, X_flipped))

    if shuffle:
        shuffle_idx = np.random.permutation(X_aug.shape[0])
        return X_aug[shuffle_idx]
    else:
        return X_aug

# Test function
def test(model, tst_loader, device):
    model.eval()
    logit, attVec = [], []
    for data in tst_loader:
        with torch.no_grad():
            data = data[0]  # Unpack
            data = Variable(data.to(device))
            data_processed = preprocess_fiber_input(data, device=device, net_type='no_roi')
            output, _, *_ = model(data_processed)
            logit.append(output.data.cpu().numpy())
            attVec.append(np.zeros((output.shape[0], 1)))  # No attention vector, just zeros
    return logit, attVec

# Main
input_folder = "../Testing_Set"
output_folder = "../Testing_Set_Results_no_roi"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Model paths
modelpath = './save_small/focal_loss_and_center_loss_no_roi.model'
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# Model parameters
NCLASS = 15
args_test_batch_size = 10000

# Initialize model
model = RESNET152_ATT_naive.resnet18(num_classes=NCLASS, input_ch=3)
model.to(device)

# Load model weights
model.load_state_dict(torch.load(modelpath, map_location=device))
model.eval()

# Process all .mat files
for filename in os.listdir(input_folder):
    if filename.endswith(".mat") and filename.find("label") == -1:
        matpath = os.path.join(input_folder, filename)
        print(f"📌 Processing: {matpath}")

        mat = loadmat(matpath)
        X_test = np.asarray(mat['tracks']['data']).astype(np.float32)
        X_test_original = np.transpose(X_test, (0, 2, 1))

        X_test = udflip_x_only(X_test_original, shuffle=False)
        X_test_np = X_test.copy()
        X_test = torch.from_numpy(X_test)

        kwargs = {'num_workers': 1, 'pin_memory': True}
        tst_set = utils.TensorDataset(X_test)
        tst_loader = utils.DataLoader(tst_set, batch_size=args_test_batch_size, shuffle=False, **kwargs)

        with torch.no_grad():
            logit, attVec = test(model, tst_loader, device)
            attVec = np.vstack(attVec).astype(np.float32)
            prob = np.exp(np.vstack(logit)).astype(np.float32)
            membership = np.argmax(prob, axis=1).reshape((-1, 1)).astype(np.float32)
            maxprob = np.amax(prob, axis=1).reshape((-1, 1)).astype(np.float32)

            output_max = np.zeros((X_test_np.shape[0], 7), dtype=np.float32)
            output_max[:, 0] = np.arange(1, X_test_np.shape[0] + 1)
            for i in range(X_test_np.shape[0]):
                output_max[i, 1:4] = X_test_np[i, 0:3, 0]
                output_max[i, 4] = X_test_np[i, 3, 0]

            output_max = np.hstack((output_max, prob, membership, maxprob))

            output_txt_path = os.path.join(output_folder, filename.replace('.mat', '.txt'))
            np.savetxt(output_txt_path, output_max, fmt='%.4e')

            for i in range(NCLASS):
                submat = output_max[np.where(output_max[:, -2] == i)]
                fiberIndex = submat[:, 0].reshape((-1, 1))
                fiberProb = submat[:, -1].reshape((-1, 1))
                fiberAtm = attVec[np.where(output_max[:, -2] == i)]

                np.savetxt(output_txt_path.replace('.txt', f'_{i:02}_fiberindex.txt'), fiberIndex, fmt='%d')
                np.savetxt(output_txt_path.replace('.txt', f'_{i:02}_fiberprob.txt'), fiberProb, fmt='%.4e')
                np.savetxt(output_txt_path.replace('.txt', f'_{i:02}_fiberatm.txt'), fiberAtm, fmt='%.4e')

            print(f"✅ Results saved for {filename}!")

print("🎉 All files processed!")
