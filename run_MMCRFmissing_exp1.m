
function rtn = runMMCRFmissing(inname)

% add path of libsvm
addpath '~/softwares/libsvm-3.12/matlab/'
addpath '../../ensemble_mmcrf/shared_scripts/'

if nargin ==0
    names={'emotions','yeast','scene','enron','cal500','fp','cancer','medical','toy10','toy50'}
else
    names={inname}
end

for name=names
[sta,comres]=system('hostname');
if strcmp(comres(1:4),'dave')
    X=dlmread(sprintf('/fs/group/urenzyme/workspace/data/%s_features',name{1}));
    Y=dlmread(sprintf('/fs/group/urenzyme/workspace/data/%s_targets',name{1}));
else
    X=dlmread(sprintf('../../ensemble_mmcrf/shared_scripts/test_data/%s_features',name{1}));
    Y=dlmread(sprintf('../../ensemble_mmcrf/shared_scripts/test_data/%s_targets',name{1}));
end

rand('twister', 0);

%------------
%
% preparing     
%
%------------
% example selection with meaningful features
Xsum=sum(X,2);
X=X(find(Xsum~=0),:);
Y=Y(find(Xsum~=0),:);
% label selection with two labels
Yuniq=[];
for i=1:size(Y,2)
    if size(unique(Y(:,i)),1)>1
        Yuniq=[Yuniq,i];
    end
end
Y=Y(:,Yuniq);

% feature normalization (tf-idf for text data, scale and centralization for other numerical features)
if or(strcmp(name{1},'medical'),strcmp(name{1},'enron')) 
    X=tfidf(X);
elseif ~(strcmp(name{1}(1:2),'to'))
    X=(X-repmat(min(X),size(X,1),1))./repmat(max(X)-min(X),size(X,1),1);
end

% change Y from -1 to 0: labeling (0/1)
Y(Y==-1)=0;

% length of x and y
Nx = length(X(:,1));
Ny = length(Y(1,:));

% stratified cross validation index
nfold = 3;
Ind = getCVIndex(Y,nfold);

% performance
perf=[];

% get dot product kernels from normalized features or just read precomputed kernels
if or(strcmp(name{1},'fp'),strcmp(name{1},'cancer'))
    K=dlmread(sprintf('/fs/group/urenzyme/workspace/data/%s_kernel',name{1}));
else
    K = X * X'; % dot product
    K = K ./ sqrt(diag(K)*diag(K)');    %normalization diagonal is 1
end

%------------
%
% parameter selection
% 
%------------
% parameter selection
mmcrf_cs=[50]%,10,5,1,0.5,0.1,0.01];
mmcrf_gs=[0.8]%,0.7,0.6];
if 1==0
Isel = randsample(1:size(K,2),ceil(size(K,2)*.2));
IselTrain=Isel(1:ceil(numel(Isel)/3*2));
IselTest=Isel((ceil(numel(Isel)/3*2+1)):numel(Isel));
selRes=zeros(numel(mmcrf_gs),numel(mmcrf_cs));
for i=1:numel(mmcrf_gs)
for j=1:numel(mmcrf_cs)
    global Kx_tr;
    global Kx_ts;
    global Y_tr;
    global Y_ts;
    global E;
    global debugging;
    global params;
    % set parameters
    params.mlloss = 0;	% assign loss to microlabels or edges
    params.profiling = 1;	% profile (test during learning)
    params.epsilon = mmcrf_gs(i); %0.6;	% stopping criterion: minimum relative duality gap
    params.C =mmcrf_cs(j);		% margin slack
    params.max_CGD_iter = 1;		% maximum number of conditional gradient iterations per example
    params.max_LBP_iter = 2;		% number of Loopy belief propagation iterations
    params.tolerance = 1E-10;		% numbers smaller than this are treated as zero
    params.filestem = sprintf('tmp_%s',name{1});		% file name stem used for writing output
    params.profile_tm_interval = 10;	% how often to test during learning
    params.maxiter = 5;		% maximum number of iterations in the outer loop
    params.verbosity = 1;
    params.debugging = 0;
    % random seed
    rand('twister', 0);
    % generate random graph
    Nnode=size(Y,2);
    E=randTreeGenerator(Nnode); % generate
    E=[E,min(E')',max(E')'];E=E(:,3:4); % arrange head and tail
    E=sortrows(E,[1,2]); % sort by head and tail
    % running
    Ypred = [];
    YpredVal = [];
    % nfold cross validation
    Itrain = IselTrain;
    Itest  = IselTest;
    Kx_tr = K(Itrain,Itrain);
    Kx_ts = K(Itest,Itrain)';
    Y_tr = Y(Itrain,:); Y_tr(Y_tr==0)=-1;
    Y_ts = Y(Itest,:); Y_ts(Y_ts==0)=-1;
    % running
    rtn = learn_MMCRFmissing;
    % collecting results
    load(sprintf('Ypred_%s.mat', params.filestem));
    Ypred = Ypred_ts;
    selRes(i,j) = sum(sum((Ypred>=0)==Y(IselTest,:)))
end
end
mmcrf_c=mmcrf_cs(find(max(selRes,[],1)==max(max(selRes,[],1))));
mmcrf_g=mmcrf_gs(find(max(selRes,[],2)==max(max(selRes,[],2))));
if numel(mmcrf_c) >1
    mmcrf_c=mmcrf_c(1);
end
if numel(mmcrf_g) >1
    mmcrf_g=mmcrf_g(1);
end
selRes
mmcrf_c
mmcrf_g

pa=[mmcrf_cs;selRes];
pa=[[0,mmcrf_gs]',pa]
dlmwrite(sprintf('../parameters/%s_parammcrftree',name{1}),pa)
end



mmcrf_c=mmcrf_cs(1)
mmcrf_g=mmcrf_gs(1)



%------------
%
% predict with missing (exp1)
%
%------------
global Kx_tr;
global Kx_ts;
global Y_tr;
global Y_ts;
global E;
global debugging;
global params;

% set parameters
params.mlloss = 0;	% assign loss to microlabels or edges
params.profiling = 1;	% profile (test during learning)
params.epsilon = mmcrf_g; %0.6;	% stopping criterion: minimum relative duality gap
params.C =mmcrf_c ;		% margin slack
params.max_CGD_iter = 1;		% maximum number of conditional gradient iterations per example
params.max_LBP_iter = 2;		% number of Loopy belief propagation iterations
params.tolerance = 1E-10;		% numbers smaller than this are treated as zero
params.filestem = sprintf('tmp_%s',name{1});		% file name stem used for writing output
params.profile_tm_interval = 10;	% how often to test during learning
params.maxiter = 5;		% maximum number of iterations in the outer loop
params.verbosity = 1;
params.debugging = 3;
% random seed
rand('twister', 0);
% generate random graph
Nrep=1;
muList=cell(Nrep,1);
Nnode=size(Y,2);
Elist=cell(Nrep,1);
for i=1:Nrep
    E=randTreeGenerator(Nnode); % generate
    E=[E,min(E')',max(E')'];E=E(:,3:4); % arrange head and tail
    E=sortrows(E,[1,2]); % sort by head and tail
    Elist{i}=E; % put into cell array
end
% running
perfPer_prev=[];

for i=1:size(Elist,1)
    E = Elist{i};
    
    for k=1%nfold
        Itrain=find(Ind~=k);
        Itest=find(Ind==k);
        for P_missing=0.00:0.02:0.1
            
            perfPer=[];
            
            gKx_tr = K(Itrain,Itrain);
            gKx_ts = K(Itest,Itrain)';
            gY_tr = Y(Itrain,:); gY_tr(gY_tr==0)=-1;
            gY_ts = Y(Itest,:); gY_ts(gY_ts==0)=-1;
            
            % missing at random
            NtrP=0.4;
            M_rd=reshape(randsample([0,1],(size(gY_tr,1)-round(size(gY_tr,1)*NtrP))*size(gY_tr,2),true,[P_missing,1-P_missing]),...
                size(gY_tr,1)-round(size(gY_tr,1)*NtrP),size(gY_tr,2));
            gY_tr(round(size(gY_tr,1)*NtrP+1):size(gY_tr,1),:)=gY_tr(round(size(gY_tr,1)*NtrP+1):size(gY_tr,1),:) .* M_rd;
            
            if P_missing==0
                start=0.05;
            else
                start=NtrP;
            end
            
            for per=start:0.05:1
                Iper=1:round(size(gY_tr,1)*per);
                Kx_tr=gKx_tr(Iper,Iper);
                Kx_ts=gKx_ts(Iper,:);
                Y_tr=gY_tr(Iper,:);
                Y_ts=gY_ts;
                
                % running on x% data
                rtn = learn_MMCRFmissing;
                % collect results
                load(sprintf('Ypred_%s.mat', params.filestem));
                [acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance((Y_ts==1),(Ypred_ts==1));
                perfPer=[perfPer;[acc,vecacc,pre,rec,f1,auc1,auc2]]
            end
            if P_missing==0
                perfPer_prev=perfPer;
            else
                perfPer=[perfPer_prev(1:7,:);perfPer];
            end
            
            % save results
            dlmwrite(sprintf('../results/%s_%.2f_perfMissing',name{1},P_missing),perfPer)
        end
    end
    plot(perfPer(:,1))
end




end

%rtn = [];
end




