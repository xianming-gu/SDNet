import argparse

parser = argparse.ArgumentParser(description='MyOption')

parser.add_argument('--DEVICE', type=str, default='cuda:0')
parser.add_argument('--epoch', type=int, default=300)  # 1000 800 600
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--seed', type=int, default=3407)

parser.add_argument('--dir_train', type=str, default='./MyDatasets/SPECT-MRI/train/')  # CT PET SPECT
parser.add_argument('--dir_test', type=str, default='./MyDatasets/SPECT-MRI/test/')  # CT PET SPECT

parser.add_argument('--img_type1', type=str, default='SPECT/')  # CT PET SPECT
parser.add_argument('--img_type2', type=str, default='MRI/')

parser.add_argument('--model_save_path', type=str, default='./modelsave/SPECT/')  # CT PET SPECT
parser.add_argument('--model_save_name', type=str, default='MyModel.pth')
parser.add_argument('--temp_dir', type=str, default='./temp/SPECT-MRI')  # CT PET SPECT

parser.add_argument('--img_save_dir', type=str, default='result/SPECT-MRI')  # CT PET SPECT

args = parser.parse_args()
