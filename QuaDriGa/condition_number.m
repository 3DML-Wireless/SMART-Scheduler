clear
close all;

H_r = hdf5read('./1_out_3.hdf5', '/H_r');
H_i = hdf5read('./1_out_3.hdf5', '/H_i');
H = H_r+ H_i * 1i;
size(H)

CN = zeros(400);
R = zeros(400);

for i = 1:400
    CN(i) = cond(squeeze(H(i,:,:)),2);
    R(i) = rank(squeeze(H(i,:,:)));
end

plot(CN);
