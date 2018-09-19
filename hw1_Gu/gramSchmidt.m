function Q=gramSchmidt(N)
    Q = gramSchmidtRec(N,N);
    
function Q=gramSchmidtRec(n,N)
    v = rand(N,1)*2-1;
    if n>1
        M = gramSchmidtRec(n-1,N);
        v = v - M*M'*v;
        while all(v==0) % rare case if random vector sits on previous plane
            v = rand(N,1)*2-1;
            v = v - M*M'*v;
        end
        Q = [M,v/norm(v)];
    else
        Q=v/norm(v);
    end