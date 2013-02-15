
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

if 1==0
%------------
%
% SVM, single label        
%
%------------
Ypred = [];
YpredVal = [];
% iterate on targets (Y1 -> Yx -> Ym)
for i=1:Ny
    % nfold cross validation
    Ycol = [];
    YcolVal = [];
    for k=1:nfold
        Itrain = find(Ind ~= k);
        Itest  = find(Ind == k);
        % training & testing with kernel
        if strcmp(name{1}(1:2),'to')
                svm_c=0.01;
        elseif strcmp(name{1},'cancer')
                svm_c=5
        else
                svm_c=0.5;
        end
        model = svmtrain(Y(Itrain,i),[(1:numel(Itrain))',K(Itrain,Itrain)],sprintf('-b 1 -q -c %.2f -t 4',svm_c));
        [Ynew,acc,YnewVal] = svmpredict(Y(Itest,k),[(1:numel(Itest))',K(Itest,Itrain)],model,'-b 1');
        [Ynew] = svmpredict(Y(Itest,k),[(1:numel(Itest))',K(Itest,Itrain)],model);
        Ycol = [Ycol;[Ynew,Itest]];
        if size(YnewVal,2)==2
            YcolVal = [YcolVal;[YnewVal(:,abs(model.Label(1,:)-1)+1),Itest]];
        else
            YcolVal = [YcolVal;[zeros(numel(Itest),1),Itest]];
        end
    end
    Ycol = sortrows(Ycol,size(Ycol,2));
    Ypred = [Ypred,Ycol(:,1)];
    YcolVal = sortrows(YcolVal,size(YcolVal,2));
    YpredVal = [YpredVal,YcolVal(:,1)];
end
% performance of svm
[acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y,Ypred,YpredVal);
perf=[perf;[acc,vecacc,pre,rec,f1,auc1,auc2]];perf

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
% MMCRFmissing      
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
perfRand=[];
perfValEns=[];
perfBinEns=[];
Yenspred=zeros(size(Y));
YenspredBin=zeros(size(Y));
YenspredVal=zeros(size(Y));
for i=1:size(Elist,1)
    E = Elist{i};
    Ypred = [];
    YpredVal = [];
    % nfold cross validation
    for k=1:nfold
        Itrain = find(Ind ~= k);
        Itest  = find(Ind == k);
        Kx_tr = K(Itrain,Itrain);
        Kx_ts = K(Itest,Itrain)';
        Y_tr = Y(Itrain,:); Y_tr(Y_tr==0)=-1;%mn=9;Y_tr(100,1:mn)=repmat(0,1,mn);
        Y_ts = Y(Itest,:); Y_ts(Y_ts==0)=-1;
        % running
        rtn = learn_MMCRFmissing;return
        % save margin dual mu
        muList{(i-1)*nfold+k}=rtn;
        % collecting results
        load(sprintf('Ypred_%s.mat', params.filestem));
        Ypred = [Ypred;[Ypred_ts,Itest]];
        YpredVal = [YpredVal;[Ypred_ts_val,Itest]];
    end
    YpredVal = sortrows(YpredVal,size(YpredVal,2));
    YpredVal = YpredVal(:,1:size(Y,2));
    YenspredVal = YenspredVal+YpredVal;
    Ypred = sortrows(Ypred,size(Ypred,2));
    Ypred = Ypred(:,1:size(Y,2));
    YenspredBin = YenspredBin+Ypred;
    
    % auc & roc random model
    [acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y,(Ypred==1),YpredVal);
    perfRand=[perfRand;[acc,vecacc,pre,rec,f1,auc1,auc2]];
    
    % auc & roc ensemble val model
    [acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y,YenspredVal>=0,YenspredVal);
    perfValEns=[perfValEns;[acc,vecacc,pre,rec,f1,auc1,auc2]];
    
    % auc & roc ensemble bin model
    [acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y,YenspredBin>=0);
    perfBinEns=[perfBinEns;[acc,vecacc,pre,rec,f1,auc1,auc2]];
end
YenspredVal=YenspredVal/Nrep;
Yenspred = (YenspredVal>=0);

% performance of Random Model
perf=[perf;mean(perfRand,1)];
% performance of Bin ensemble
[acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y,YenspredBin>=0);
perf=[perf;[acc,vecacc,pre,rec,f1,auc1,auc2]];perf
dlmwrite(sprintf('../predictions/%s_predBinTreeEns',name{1}),YenspredBin>=0)
% performance of Val ensemble
[acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y,Yenspred,YenspredVal);
perf=[perf;[acc,vecacc,pre,rec,f1,auc1,auc2]];perf
dlmwrite(sprintf('../predictions/%s_predValTreeEns',name{1}),Yenspred)







return










%------------
%
% wensMMCRF      
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
params.debugging = 0;

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
perfRand=[];
perfValEns=[];
perfBinEns=[];
Yenspred=zeros(size(Y));
YenspredBin=zeros(size(Y));
YenspredVal=zeros(size(Y));
for i=1:size(Elist,1)
    E = Elist{i};
    Ypred = [];
    YpredVal = [];
    % nfold cross validation
    for k=1:nfold
        Itrain = find(Ind ~= k);
        Itest  = find(Ind == k);
        Kx_tr = K(Itrain,Itrain);
        Kx_ts = K(Itest,Itrain)';
        Y_tr = Y(Itrain,:); Y_tr(Y_tr==0)=-1;
        Y_ts = Y(Itest,:); Y_ts(Y_ts==0)=-1;
        % running
        rtn = learn_MMCRF;
        % save margin dual mu
        muList{(i-1)*nfold+k}=rtn;
        % collecting results
        load(sprintf('Ypred_%s.mat', params.filestem));
        Ypred = [Ypred;[Ypred_ts,Itest]];
        YpredVal = [YpredVal;[Ypred_ts_val,Itest]];
    end
    YpredVal = sortrows(YpredVal,size(YpredVal,2));
    YpredVal = YpredVal(:,1:size(Y,2));
    YenspredVal = YenspredVal+YpredVal;
    Ypred = sortrows(Ypred,size(Ypred,2));
    Ypred = Ypred(:,1:size(Y,2));
    YenspredBin = YenspredBin+Ypred;
    
    % auc & roc random model
    [acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y,(Ypred==1),YpredVal);
    perfRand=[perfRand;[acc,vecacc,pre,rec,f1,auc1,auc2]];
    
    % auc & roc ensemble val model
    [acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y,YenspredVal>=0,YenspredVal);
    perfValEns=[perfValEns;[acc,vecacc,pre,rec,f1,auc1,auc2]];
    
    % auc & roc ensemble bin model
    [acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y,YenspredBin>=0);
    perfBinEns=[perfBinEns;[acc,vecacc,pre,rec,f1,auc1,auc2]];
end
YenspredVal=YenspredVal/Nrep;
Yenspred = (YenspredVal>=0);

% performance of Random Model
perf=[perf;mean(perfRand,1)];
% performance of Bin ensemble
[acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y,YenspredBin>=0);
perf=[perf;[acc,vecacc,pre,rec,f1,auc1,auc2]];perf
dlmwrite(sprintf('../predictions/%s_predBinTreeEns',name{1}),YenspredBin>=0)
% performance of Val ensemble
[acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y,Yenspred,YenspredVal);
perf=[perf;[acc,vecacc,pre,rec,f1,auc1,auc2]];perf
dlmwrite(sprintf('../predictions/%s_predValTreeEns',name{1}),Yenspred)


%------------
%
% mdensMMCRF      
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
params.maxiter = 10;		% maximum number of iterations in the outer loop
params.verbosity = 1;
params.debugging = 0;
perfMadEns=[];
for i=1:size(Elist,1)
    % get new E
    Enew = [];
    for j=1:i
        Enew=[Enew;Elist{j}];
    end
    Enew=unique(Enew,'rows');
    E=Enew;
    Ypred = [];
    YpredVal = [];
    % nfold cross validation
    for k=1:nfold
        % training testing label
        Itrain = find(Ind ~= k);
        Itest  = find(Ind == k);
        % training testing kernel
        Kx_tr = K(Itrain,Itrain);
        Kx_ts = K(Itest,Itrain)';
        % training and testing target
        Y_tr = Y(Itrain,:); Y_tr(Y_tr==0)=-1;
        Y_ts = Y(Itest,:); Y_ts(Y_ts==0)=-1;
        % ensemble marginal dual
        muNew=zeros(4*size(Enew,1),size(Kx_tr,1));
        for j=1:i 
            muNew=muNew+mu_complete_zero(muList{(j-1)*nfold+k},Elist{j},Enew,params.C);
            % muNew=muNew+mu_complete_constrainted(muList{(i-1)*nfold+k},Elist{i},Enew,params.C);
            % muNew=muNew+onestep_inference(mu_complete_zero(muList{(i-1)*nfold+k},Elist{i},Enew,params.C));
        end
        muNew=muNew/Nrep;
        % running given mu and Enew
        params.mu=muNew;
        rtn = learn_ENSMMCRF;
        % collecting results
        load(sprintf('Ypred_%s.mat', params.filestem));
        Ypred = [Ypred;[Ypred_ts,Itest]];
        YpredVal = [YpredVal;[Ypred_ts_val,Itest]];
    end
    YpredVal = sortrows(YpredVal,size(YpredVal,2));
    YpredVal = YpredVal(:,1:size(Y,2));
    Ypred = sortrows(Ypred,size(Ypred,2));
    Ypred = Ypred(:,1:size(Y,2));

    % performance of Md ensemble
    [acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y,Ypred>=0,Ypred);
    perfMadEns=[perfMadEns;[acc,vecacc,pre,rec,f1,auc1,auc2]];
end

% auc & roc
[acc,vecacc,pre,rec,f1,auc1,auc2]=get_performance(Y,Ypred>=0,Ypred);
perf=[perf;[acc,vecacc,pre,rec,f1,auc1,auc2]];perf
dlmwrite(sprintf('../predictions/%s_predMadTreeEns',name{1}),Ypred>=0)


% plot data with true labels
hFig = figure('visible','off');
set(hFig, 'Position', [500,500,1200,500])
subplot(3,5,1);plot(perfBinEns(:,1));title('Bin accuracy');
subplot(3,5,2);plot(perfBinEns(:,2));title('multilabel accuracy');
subplot(3,5,3);plot(perfBinEns(:,5));title('F1');
subplot(3,5,4);plot(perfBinEns(:,6));title('AUC');
subplot(3,5,5);plot(perfBinEns(:,7));title('AUC2');

subplot(3,5,6);plot(perfValEns(:,1));title('Val accuracy');
subplot(3,5,7);plot(perfValEns(:,2));title('multilabel accuracy');
subplot(3,5,8);plot(perfValEns(:,5));title('F1');
subplot(3,5,9);plot(perfValEns(:,6));title('AUC');
subplot(3,5,10);plot(perfValEns(:,7));title('AUC2');

subplot(3,5,11);plot(perfMadEns(:,1));title('Mad accuracy');
subplot(3,5,12);plot(perfMadEns(:,2));title('multilabel accuracy');
subplot(3,5,13);plot(perfMadEns(:,5));title('F1');
subplot(3,5,14);plot(perfMadEns(:,6));title('AUC');
subplot(3,5,15);plot(perfMadEns(:,7));title('AUC2');
print(hFig, '-depsc',sprintf('../plots/%s_TreeEns.eps',name{1}));
% save results
dlmwrite(sprintf('../results/%s_perfTreeEns',name{1}),perf)
dlmwrite(sprintf('../results/%s_perfRandTreeEns',name{1}),perfRand)
dlmwrite(sprintf('../results/%s_perfValTreeEnsProc',name{1}),perfValEns)
dlmwrite(sprintf('../results/%s_perfBinTreeEnsProc',name{1}),perfBinEns)
dlmwrite(sprintf('../results/%s_perfMadTreeEnsProc',name{1}),perfMadEns)

end

%rtn = [];
end




