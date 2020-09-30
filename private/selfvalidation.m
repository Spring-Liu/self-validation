function score = selfvalidation(W,S,tensor,tensor_hat)
if nargin < 4
    tensor_hat = tensor;
end

ack = double(tensor_hat);

iter = 1;
for i = 1:S(1)
    for j = 1:S(2)
        Q = zeros(S);
        for a = 1:S(1)
            for b = 1:S(2)
                Q(a,b,:) = W(i,j,:);
            end
        end
        T12 = ack .* Q;
        t = 1; T12_hat = [];
        for k = 1:S(3)
            if sum(T12(:,:,k)) > 0
                T12_hat(:,:,t) = T12(:,:,k);
                t = t+1;
            end
        end
        X_hat{iter} = T12_hat;
        iter = iter + 1;
        clear T12_hat;
    end
end

for i = 1:S(1)
    for j = 1:S(3)
        Q = zeros(S);
        for a = 1:S(1)
            for b = 1:S(3)
                Q(a,:,b) = W(i,:,j);
            end
        end
        T13 = ack .* Q;
        t = 1;T13_hat = [];
        for k = 1:S(2)
            if sum(T13(:,k,:)) > 0
                T13_hat(:,t,:) = T13(:,k,:);
                t = t+1;
            end
        end
        X_hat{iter} = T13_hat;
        iter = iter + 1;
        clear T13_hat;
    end
end

for i = 1:S(2)
    for j = 1:S(3)
        Q = zeros(S);
        for a = 1:S(2)
            for b = 1:S(3)
                Q(:,a,b) = W(:,i,j);
            end
        end
        T23 = ack .* Q;
        t = 1;T23_hat = [];
        for k = 1:S(1)
            if sum(T23(k,:,:)) > 0
                T23_hat(t,:,:) = T23(k,:,:);
                t = t+1;
            end
        end
        X_hat{iter} = T23_hat;
        iter = iter + 1;
        clear T23_hat;
    end
end

for it = 1:iter-1
    X_hat_1 = Unfold(X_hat{1,it},size(X_hat{1,it}),1);
    X_hat_2 = Unfold(X_hat{1,it},size(X_hat{1,it}),2);
    X_hat_3 = Unfold(X_hat{1,it},size(X_hat{1,it}),3);
    
    rank_n(it) = 1/3 * sum( sum(svd(X_hat_1)) + sum(svd(X_hat_2)) + sum(svd(X_hat_3)));
end

X_1 = Unfold(ack,size(ack),1);
X_2 = Unfold(ack,size(ack),2);
X_3 = Unfold(ack,size(ack),3);
rank_tucker_syn = 1/3 * sum( sum(svd(X_1)) + sum(svd(X_2)) + sum(svd(X_3)));

score = min(rank_n./rank_tucker_syn);