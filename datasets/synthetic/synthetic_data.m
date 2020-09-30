clc;clear
% synthetic data generation
I = 20; I1 = I; I2 = I; I3 = I; R = 2;
%% synthetic_low-CP-rank
A1 = rand(I,R);
A2 = rand(I,R);
A3 = rand(I,R);

CP_syn = zeros(I1,I2,I3);
for i = 1:R
    temp = A1(:,i)*A2(:,i)';
    for j = 1:I
        CP_syn(:,:,j) = CP_syn(:,:,j) + temp * A3(j,i);
    end
end
clear A1 A2 A3 i j temp

%% synthetic_low-n-rank
core = tensor(rand(R,R,R),[R,R,R]);
U = {rand(I,R),rand(I,R),rand(I,R)};
Tucker_syn = ttensor(core,U);
clear core U;

%% synthetic_low-tubal-rank
A = rand(I,R,I);
B = rand(R,I,I);
tubal_syn = tprod(A,B);
clear A B