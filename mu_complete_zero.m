
% 
% mu: marginal dual variable
% E: corresponding edge list
% Enew: all edges SORTED
% 
function [ muNew ] = mu_complete(mu, E, Enew, C)

if 1==0
    mu=dlmread('tmp');
    E=dlmread('tmpe');
    Enew=E;
    Enew=[Enew;[1,2];[3,5]];
end

    % rearrange head and tail for E
    E=[E,min(E')',max(E')'];E=E(:,3:4);
    % rearrange head and tail for Enew
    Enew=[Enew,min(Enew')',max(Enew')'];Enew=Enew(:,3:4);
    % sort Enew again in case it's not sorted
    Enew=sortrows(Enew,[1,2]);
    % sort mu
    Eadd=reshape(reshape(repmat(E,1,4)',1,8*size(E,1))',2,4*size(E,1))';
    mu=sortrows([Eadd,repmat([1;2;3;4],size(Eadd,1)/4,1),mu],[1,2,3]);
    % sort E
    E=sortrows(E,[1,2]);
    % complete mu matrix
    % TODO: current solution using loop
    Eadd=[];
    B=[]; % equation in Y part
    j=0;
    for i=1:size(Enew,1)
        if find(E(:,1)==Enew(i,1) & E(:,2)==Enew(i,2))
            continue
        end
        j=j+1;
        Eadd=[Eadd;Enew(i,:)];
        if 1==0
        % i
        k=find(mu(:,1)==Enew(i,1));
        if size(k,1)~=0 % i head
            B=[B;mu(k(1,1),4:size(mu,2))+mu(k(2,1),4:size(mu,2))]; % -
            B=[B;mu(k(3,1),4:size(mu,2))+mu(k(4,1),4:size(mu,2))]; % +
        else            % i tail
            k=find(mu(:,2)==Enew(i,1));
            B=[B;mu(k(1,1),4:size(mu,2))+mu(k(3,1),4:size(mu,2))]; % -
            B=[B;mu(k(2,1),4:size(mu,2))+mu(k(4,1),4:size(mu,2))]; % +
        end
        % j
        k=find(mu(:,1)==Enew(i,2));
        if size(k,1)~=0 % j head
            B=[B;mu(k(1,1),4:size(mu,2))+mu(k(2,1),4:size(mu,2))]; % -
            B=[B;mu(k(3,1),4:size(mu,2))+mu(k(4,1),4:size(mu,2))]; % +
        else            % j tail
            k=find(mu(:,2)==Enew(i,2));
            B=[B;mu(k(1,1),4:size(mu,2))+mu(k(3,1),4:size(mu,2))]; % -
            B=[B;mu(k(2,1),4:size(mu,2))+mu(k(4,1),4:size(mu,2))]; % +
        end
        end
    end
    B=repmat(repmat(zeros(1,size(mu,2)-3),4,1),j,1);
    Eadd=reshape(reshape(repmat(Eadd,1,4)',1,8*size(Eadd,1))',2,4*size(Eadd,1))';
    % solve linear system
    A=[1,1,0,0;0,0,1,1;1,0,1,0;0,1,0,1];
    sol=[];

    solver=5;
    if solver==1
        for i=1:(size(Enew,1)-size(E,1))           
            solE=[];
            for j=(1:size(mu,2)-3)
                 solE=[solE,lsqnonneg(A,B(((i-1)*4+1):((i-1)*4+4),j))];
            end
            sol=[sol;solE];

        end
    elseif solver==2
        for j=(1:size(mu,2)-3)
            sol=[sol,lsqnonneg(kron(diag(diag(ones(size(B,1)/4,size(i,1)/4))),A),B(:,j))];
        end
    elseif solver==3
        for j=(1:size(mu,2)-3)
            sol=[sol,pinv(kron(diag(diag(ones(size(B,1)/4,size(B,1)/4))),A))*B(:,j)];
        end
    elseif solver==4
        for i=1:(size(Enew,1)-size(E,1))           
            solE=[];
            for j=(1:size(mu,2)-3)
                 solE=[solE,pinv(A)*B(((i-1)*4+1):((i-1)*4+4),j)];
            end
            sol=[sol;solE];
        end
    elseif solver==5
        sol=zeros(size(Eadd,1),size(B,2));
    end
    
    % combine results
    muNew=sortrows([mu;[Eadd,repmat([1;2;3,;4],size(Eadd,1)/4,1),sol]],[1,2,3]);
    muNew=muNew(:,4:size(muNew,2));
end

