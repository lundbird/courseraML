function k = chooseK(S,alpha)
Sdiag = diag(S);
Stot = sum(Sdiag);
q=1;
k=1;
while q>alpha
    k=k+1;
    q = 1-sum(Sdiag(1:k))/Stot;
end
end


