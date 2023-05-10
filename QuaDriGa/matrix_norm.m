close all
clear all



H_r = hdf5read('./channel_sequences.hdf5', '/H_r');
H_i = hdf5read('./channel_sequences.hdf5', '/H_i');
H = H_r + H_i * 1i;
old_H_size = size(H)

H_before_norm = H(1,4,4,4,3)

H = H ./ sqrt(sum(abs(H).^2,[3 4])./ prod(size(H,2:4)));

New_H_size = size(H)
H_after_norm = H(1,4,4,4,3)

H = permute(squeeze(mean(H(:,:,:,:,3),4)),[2,1]);
H_after_mean = H(4,4)
final_H_size = size(H)


