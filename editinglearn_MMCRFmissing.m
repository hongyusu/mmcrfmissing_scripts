% Structured classification algorithm for output spaces possessing an
% associative conditional random field structure
% (c) Juho Rousu, 2004-2010
% 
% 
% * Kernelized input
% * Explicit (non-kernelized) output with graph structure
% * Conditional gradient optimization
% * Loopy belief propagation for approximate inference 
%   (used in gradient computations and prediction)
% 


% Debug log
% learning_MMCRFmissing:
%               1. while condition
%               2. get previous optimal mu
%               3. continuous search after get optimal mu
% optimize_x:
%               1. G0>Gmax, keep previous solution, change from
%               mu_l=zero(mu_x) to mu_l=mu_x
%               1. G0>Gmax, keep previous solution, change from
%               Kxx_mu_l=zero(Kxx_mu_x) to Kxx_mu_l=Kxx_mu_x



function rtn=learn_MMCRFmissing
    % Input data assumed by the algorithm
    global Kx_tr; % X-kernel, assume to be positive semidefinite and normalized (Kx_tr(i,i) = 1)
    global Y_tr; % Y-data: assumed to be class labels encoded {-1,+1}
    global E; % edges of the Markov network e_i = [E(i,1),E(i,2)];
    global params; % parameters use by the learning algorithm
    global loss; % losses associated with different edge labelings
    global mu; % marginal dual varibles: these are the parameters to be learned
    global m; % number of training instances
    global l; % number of labels
    global Ye; % Denotes the edge-labelings 1 <-- [-1,-1], 2 <-- [-1,+1], 3 <-- [+1,-1], 4 <-- [+1,+1]
    global IndEdgeVal; % IndEdgeVal{u} = [Ye == u] 
    global Kmu; % Kx_tr*mu
    global primal_ub;
    global profile;
    global obj;
    global opt_round;
    
    optimizer_init;
    
    global Rmu;
    global Smu;
    
    profile_init;

    l = size(Y_tr,2);
    m = size(Kx_tr,1);
    mu = zeros(4*size(E,1),m);
    loss = compute_loss_vector(Y_tr,params.mlloss);
    % Matrices for speeding up gradient computations
    Ye = reshape(loss == 0,4,size(E,1)*m);
    for u = 1:4
        IndEdgeVal{u} = sparse(reshape(Ye(u,:)~=0,size(E,1),m));
    end
    Ye = reshape(Ye,4*size(E,1),m);
    Kxx_mu_x = zeros(4*size(E,1),m);
    Kmu = zeros(numel(mu),1);

    print_message('Starting descent...',0);
    obj = 0;
    primal_ub = Inf;
    iter = 0;
    opt_round = 0;
    profile_update;
    compute_duality_gap;
    profile.n_err_microlbl_prev=profile.n_err_microlbl;
    progress_made = 1;
    
    prev_mu=0;
    prev_obj=0;
    prev_Kxx_mu_x=0;
    prev_Rmu=0;
    prev_Smu=0;
    
    params.maxiter=1;
    
    while (primal_ub - obj >= params.epsilon*obj & ... % satisfy duality gap
            profile.n_err_microlbl <= profile.n_err_microlbl_prev & ... % decrease training microlabel error
            progress_made == 1 & ...   % make progress
            opt_round <= params.maxiter ... % within iteration limitation
            )
        
        opt_round = opt_round + 1;
        progress_made = 0; 
        
        prev_mu=mu;
        prev_obj=obj;
        prev_Kxx_mu_x=Kxx_mu_x;
        prev_Rmu=Rmu;
        prev_Smu=Smu;
        
        print_message('Conditional gradient optimization...',3)
        for x = 1:m
            % obtain initial gradient for index-x
            Kmu_x = compute_Kmu_x(x,Kx_tr(:,x));
            % conditional gradient optimization on index-x
            [mu(:,x),Kxx_mu_x(:,x),obj,x_iter] = optimize_x(x, obj, mu(:,x), Kmu_x, Kxx_mu_x(:,x),loss(:,x),Ye(:,x),params.C,params.max_CGD_iter);
            %obj0 = mu(:)'*loss(:) - (mu(:)'*reshape(compute_Kmu(Kx_tr),4*size(E,1)*m,1))/2;
            iter = iter + x_iter;
            profile.iter = iter;
        end
        
		progress_made =  obj > prev_obj; 
        
		print_message('Duality gap and primal upper bound',3);
		compute_duality_gap;
		profile.next_profile_tm = 0;
        profile_update;
        
    end     % end while
    %ddd
    if opt_round <= params.maxiter
        mu=prev_mu;
        obj=prev_obj;
        Rmu=prev_Rmu;
        Smu=prev_Smu;
        Kxx_mu_x=prev_Kxx_mu_x;
    end
    % continue searching 
    if 1==0
        opt_mu=0;
        ts_err=1e10;
        tr_err=1e10;
        iter=0;
        for x = 1:m
            % obtain initial gradient for index-x
            Kmu_x = compute_Kmu_x(x,Kx_tr(:,x));            
            % conditional gradient optimization on index-x
            [mu(:,x),Kxx_mu_x(:,x),obj,x_iter] = optimize_x(x, obj, mu(:,x), Kmu_x, Kxx_mu_x(:,x),loss(:,x),Ye(:,x),params.C,params.max_CGD_iter);
            profile_update;
            if tr_err>profile.n_err_microlbl
                tr_err = profile.n_err_microlbl;
                ts_err = profile.n_err_microlbl_ts;
                opt_mu=mu;
            end
        end
        mu=opt_mu;
    end
    profile_update;
    
    rtn = mu;
end

    
function Kmu_x = compute_Kmu_x(x,Kx)
    global E;
    global IndEdgeVal;
    global Rmu;
    global Smu;
    global term12;
    global term34;
    global m;
    
    % For speeding up gradient computations: 
    % store sums of marginal dual variables, distributed by the
    % true edge values into Smu
    % store marginal dual variables, distributed by the
    % pseudo edge values into Rmu
    
    if isempty(Rmu)
        Rmu = cell(1,4);
        Smu = cell(1,4);
        for u = 1:4
            Smu{u} = zeros(size(E,1),m);
            Rmu{u} = zeros(size(E,1),m);
        end
    end

    for u = 1:4
        Ind_te_u = full(IndEdgeVal{u}(:,x));   
        H_u = Smu{u}*Kx-Rmu{u}*Kx;
        term12(1,Ind_te_u) = H_u(Ind_te_u)';
        term34(u,:) = -H_u';
    end
    Kmu_x = reshape(term12(ones(4,1),:) + term34,4*size(E,1),1);
end
 
    
function compute_duality_gap
    global E;
    global m;
    global params;
    global mu;
    global Kmu;
    global loss;
    global obj;
    global primal_ub;
    global duality_gap;
    global opt_round;
    l_siz = size(loss);
    loss = reshape(loss,4,size(E,1)*m);
    kmu_siz = size(Kmu);
    Kmu = reshape(Kmu,4,size(E,1)*m);
    gradient = loss - Kmu;
    mu_siz = size(mu);
    mu = reshape(mu,4,size(E,1)*m); gradient = reshape(gradient,4,size(E,1)*m);
    dgap = Inf; LBP_iter = 1;Gmax = -Inf;
    while LBP_iter <= size(E,1)
        LBP_iter = LBP_iter*2; % no of iterations = diameter of the graph
        [Ymax,YmaxVal,G] = max_gradient_labeling(gradient,LBP_iter); 
        Gmax = max(Gmax,G);

        duality_gap = params.C*max(Gmax,0) - sum(reshape(sum(gradient.*mu),size(E,1),m),1)';
        dgap = sum(duality_gap);

        if obj+dgap < primal_ub+1E-6
            break;
        end
    end
    %primal_ub = min(obj+dgap,primal_ub);
    if primal_ub == Inf
         primal_ub = obj+dgap;
    else
         primal_ub = (obj+dgap)/min(opt_round,10)+primal_ub*(1-1/min(opt_round,10)); % averaging over a few last rounds
    end
    loss= reshape(loss,l_siz);
    Kmu = reshape(Kmu,kmu_siz);
    mu = reshape(mu,mu_siz);
end

function printme(x,str)
if x==32
    fprintf('%s\n\n',str);
end
end

% Conditional gradient optimizer for a single example
% mu_x, Kxx_mu_x -> a column in the matrix
function [mu_x,Kxx_mu_x,obj,iter] = optimize_x(x,obj,mu_x,Kmu_x,Kxx_mu_x,loss_x,te_x,C,maxiter)
    global E;
    global Rmu;
    global Smu;
    global IndEdgeVal;
    global params;
    global Y_tr;
    iter = 0;
    while iter < maxiter
        % calculate gradient for current example
        gradient =  loss_x - Kmu_x;
        printme(x,'loss_x');printme(x,sprintf('%.2f %.2f %.2f %.2f \n',loss_x'));
        printme(x,'mu_x');printme(x,sprintf('%.2f %.2f %.2f %.2f \n',mu_x'));
        printme(x,'Kmu_x');printme(x,sprintf('%.2f %.2f %.2f %.2f \n',Kmu_x'));
        printme(x,'gradient');printme(x,sprintf('%.2f %.2f %.2f %.2f \n',gradient'));
        % terminate if gradient is too small
        if norm(gradient) < params.tolerance
            break;
        end
        % find maximum gradient labeling, Ymax-labeling, Gmax-global maxima
        % under gradient labeling
        [Ymax,YmaxVal,Gmax] = max_gradient_labeling(gradient);
        % gradient towards zero, current maxima
        G0 = -mu_x'*gradient;
        printme(x,'Ymax');printme(x,sprintf('%.2f ',Ymax'));
        printme(x,sprintf('Gmax %.2f G0 %.2f',Gmax,G0));
        printme(x,'mu_x');printme(x,sprintf('%.2f %.2f %.2f %.2f \n',mu_x));
                
        % convert labeling to update direction
        Umax_e = 1+2*(Ymax(:,E(:,1))>0) + (Ymax(:,E(:,2)) >0);
        mu_1 = zeros(size(mu_x));
       
        
        if Gmax >=G0% max(params.tolerance,G0) % keep current solution
			for u = 1:4
		        mu_1(4*(1:size(E,1))-4 + u) = C*(Umax_e == u);
            end
			if sum(mu_1) > 0
			    smu_1_te = sum(reshape(mu_1.*te_x,4,size(E,1)),1);
			    smu_1_te = reshape(smu_1_te(ones(4,1),:),length(mu_x),1);
			    kxx_mu_1 = ~te_x*C+mu_1-smu_1_te;
    			%kxx_mu_1 = ones(size(te_x))*C-te_x*C-smu_1_te+mu_1;
			else
	    		kxx_mu_1 = zeros(size(mu_x));
			end
			Kmu_1 = Kmu_x + kxx_mu_1 - Kxx_mu_x;
            printme(x,'Kmu_1');printme(x,sprintf('%.2f %.2f %.2f %.2f \n ',Kmu_1'));
        else % G0>Gmax, no change
            if G0 < params.tolerance
                printme(x,'skip')
                break;
            else % keep last solution
                printme(x,'last')
                %kxx_mu_1 = zeros(size(mu_x));
                kxx_mu_1 = Kxx_mu_x;
                %mu_1 = zeros(size(mu_x));
                mu_1=mu_x;
                Kmu_1 = Kmu_x + kxx_mu_1 - Kxx_mu_x;
            end
        end
        d_x = mu_1 - mu_x;
        printme(x,'d_x');printme(x,sprintf('%.2f %.2f %.2f %.2f \n',d_x));
        Kd_x = Kmu_1 - Kmu_x;
        printme(x,'Kd_x');printme(x,sprintf('%.2f %.2f %.2f %.2f \n',Kd_x));
        l = gradient'*d_x;
        q = d_x'*Kd_x;
        alpha = min(l/q,1);
        
        delta_obj = gradient'*d_x*alpha - alpha^2/2*d_x'*Kd_x;
        printme(x,sprintf('q %.2f l/q %.2f alpha %.2f delta_obj %.2f obj %.2f',q,l/q,alpha,delta_obj,obj));
        if or(delta_obj <= 0,alpha <= 0)
            printme(x,'obj=0');
            break;
        end
        
        printme(x,'d_x*alpha');printme(x,sprintf('%.2f %.2f %.2f %.2f \n ',d_x*alpha'));
        mu_x = mu_x + d_x*alpha;
        Kmu_x = Kmu_x + Kd_x*alpha;
        obj = obj + delta_obj;
        Kxx_mu_x = (1-alpha)*Kxx_mu_x + alpha*kxx_mu_1;
        iter = iter + 1;
        printme(x,'mu_x');printme(x,sprintf('%.2f %.2f %.2f %.2f \n',mu_x))
    end
    % For speeding up gradient computations: 
    % store sums of marginal dual variables, distributed by the
    % true edge values into Smu
    % store marginal dual variables, distributed by the
    % pseudo edge values into Rmu
    mu_x = reshape(mu_x,4,size(E,1));
    for u = 1:4
        Smu{u}(:,x) = (sum(mu_x)').*IndEdgeVal{u}(:,x);
        Rmu{u}(:,x) = mu_x(u,:)';
    end
    mu_x = reshape(mu_x,4*size(E,1),1);
end


% Complete gradient
function Kmu = compute_Kmu(Kx,mu0)
    global E;
    global mu;
    global IndEdgeVal;
    global params;

    if nargin < 2
        mu0 = mu;
    end
    m_oup = size(Kx,2);
    m = size(Kx,1);
    if  0 %and(params.debugging, nargin == 2)
        for x = 1:m
           Kmu(:,x) = compute_Kmu_x(x,Kx(:,x));
        end
        Kmu = reshape(Kmu,4,size(E,1)*m);
    else
        mu_siz = size(mu0);
        mu0 = reshape(mu0,4,size(E,1)*m);
        Smu = reshape(sum(mu0),size(E,1),m);
        term12 =zeros(1,size(E,1)*m_oup);
        Kmu = zeros(4,size(E,1)*m_oup);
        for u = 1:4
            IndEVu = full(IndEdgeVal{u});    
            Rmu_u = reshape(mu0(u,:),size(E,1),m);
            H_u = Smu.*IndEVu;
            H_u = H_u - Rmu_u;
            Q_u = H_u*Kx;
            term12 = term12 + reshape(Q_u.*IndEVu,1,m_oup*size(E,1));
            Kmu(u,:) = reshape(-Q_u,1,m_oup*size(E,1));
        end
        for u = 1:4
            Kmu(u,:) = Kmu(u,:) + term12;
        end
    end
    %mu = reshape(mu,mu_siz);
end


function profile_update
    global params;
    global profile;
    global E;
    global Ye;
    global Y_tr;
    global Kx_tr;
    global Y_ts;
    global Kx_ts;
    global Y_pred;
    global Y_predVal;
    global mu;
    global obj;
    global primal_ub;
    m = size(Ye,2);
    tm = cputime;
    print_message(sprintf('alg: M3LBP tm: %d  iter: %d obj: %f mu: max %f min %f dgap: %f',...
    round(tm-profile.start_time),profile.iter,obj,max(max(mu)),min(min(mu)),primal_ub-obj),5,sprintf('%s.log',params.filestem));
    if params.profiling
        profile.next_profile_tm = profile.next_profile_tm + params.profile_tm_interval;
        profile.n_err_microlbl_prev = profile.n_err_microlbl;

        [Ypred_tr,Ypred_tr_val] = compute_error(Y_tr,Kx_tr);
        profile.microlabel_errors = sum(abs(Ypred_tr-Y_tr) >0,2);
        profile.n_err_microlbl = sum(profile.microlabel_errors);
        profile.p_err_microlbl = profile.n_err_microlbl/numel(Y_tr);
        profile.n_err = sum(profile.microlabel_errors > 0);
        profile.p_err = profile.n_err/length(profile.microlabel_errors);

        [Ypred_ts,Ypred_ts_val] = compute_error(Y_ts,Kx_ts);
        profile.microlabel_errors_ts = sum(abs(Ypred_ts-Y_ts) > 0,2);
        profile.n_err_microlbl_ts = sum(profile.microlabel_errors_ts);
        profile.p_err_microlbl_ts = profile.n_err_microlbl_ts/numel(Y_ts);
        profile.n_err_ts = sum(profile.microlabel_errors_ts > 0);
        profile.p_err_ts = profile.n_err_ts/length(profile.microlabel_errors_ts);

        print_message(sprintf('td: %d err_tr: %d (%3.2f) ml.loss tr: %d (%3.2f) err_ts: %d (%3.2f) ml.loss ts: %d (%3.2f) obj: %d',...
        round(tm-profile.start_time),profile.n_err,profile.p_err*100,profile.n_err_microlbl,profile.p_err_microlbl*100,round(profile.p_err_ts*size(Y_ts,1)),profile.p_err_ts*100,sum(profile.microlabel_errors_ts),sum(profile.microlabel_errors_ts)/numel(Y_ts)*100, obj),0,sprintf('%s.log',params.filestem));
        %print_message(sprintf('%d here',profile.microlabel_errors_ts),4);

        sfile = sprintf('Ypred_%s.mat',params.filestem);
        save(sfile,'Ypred_tr','Ypred_ts','params','Ypred_ts_val');
        Ye = reshape(Ye,4*size(E,1),m);
    end
end


function [Ypred,YpredVal] = compute_error(Y,Kx) 
    global profile;
    global Ypred;
    global YpredVal
    
    if isempty(Ypred)
        Ypred = zeros(size(Y));
    end
    w_phi_e = compute_w_phi_e(Kx);
    [Ypred,YpredVal] = max_gradient_labeling(w_phi_e);
end


function w_phi_e = compute_w_phi_e(Kx)
    global E;
    global m;
    global Ye;
    global mu;

    Ye_siz = size(Ye);
    Ye = reshape(Ye,4,size(E,1)*m);   
    mu_siz = size(mu);
    mu = reshape(mu,4,size(E,1)*m);
    m_oup = size(Kx,2);

    if isempty(find(mu,1))
        w_phi_e = zeros(4,size(E,1)*m_oup);
    else  
        w_phi_e = sum(mu);
        w_phi_e = w_phi_e(ones(4,1),:);
        w_phi_e = Ye.*w_phi_e;
        w_phi_e = w_phi_e-mu;
        w_phi_e = reshape(w_phi_e,4*size(E,1),m);
        w_phi_e = w_phi_e*Kx;
        w_phi_e = reshape(w_phi_e,4,size(E,1)*m_oup);
    end
    mu = reshape(mu,mu_siz);
    Ye = reshape(Ye,Ye_siz);
end


function [Ymax,YmaxVal,Gmax] = max_gradient_labeling(gradient,max_iter)
    % gradient is length 4*|E| column vector containing the gradient for each edge-labeling 
    global E;
    global MBProp; % 2|E|x2|E| direction-specific adjacency matrix
    global MBPropEdgeNode;
    global params;
    if params.debugging == 1
        [Ymax,Gmax] = max_gradient_labeling_brute_force(gradient);
    else
        ineg = 1;
        ipos = 2;
        if isempty(MBProp)
            [MBProp,MBPropEdgeNode] = buildBeliefPropagationMatrix(E);
        end
        if nargin < 2
            max_iter = params.max_LBP_iter;
        end
        m = numel(gradient)/(4*size(E,1));
        g_siz = size(gradient);
        gradient = reshape(gradient,4,size(E,1)*m);
        
        % Edge-labeling specific gradient matrices m x |E|
        Gnn = reshape(gradient(1,:),size(E,1),m)'; % edge-gradients for labeling [-1,-1]
        Gnp = reshape(gradient(2,:),size(E,1),m)'; % edge-gradients for labeling [-1,+1]
        Gpn = reshape(gradient(3,:),size(E,1),m)'; % edge-gradients for labeling [+1,-1]
        Gpp = reshape(gradient(4,:),size(E,1),m)'; % edge-gradients for labeling [+1,+1]
        
        % SumMsg_*_*: mx|E| matrices storing the sums of neighboring messages from
        % the head and tail of the edge, respectively, on the condition that
        % the head (resp. tail) is labeled with -1 --> neg or +1 --> pos.
        SumMsg_head_neg = zeros(m,size(E,1));
        SumMsg_head_pos = zeros(m,size(E,1));
        SumMsg_tail_neg = zeros(m,size(E,1));
        SumMsg_tail_pos = zeros(m,size(E,1));
        
        iTail = 1:size(E,1);
        iHead = size(E,1)+iTail;
        
        % Iterate until messages have had time to go accros the whole graph: at
        % most this takes O(|E|) iterations (i.e. when the graph is a chain)
        for iter = 1:max_iter
            % find max-gradient configuration and propage gradient value over the edge
            Msg_head_neg = max(SumMsg_tail_pos+Gpn,SumMsg_tail_neg+Gnn);
            Msg_head_pos = max(SumMsg_tail_pos+Gpp,SumMsg_tail_neg+Gnp);
            Msg_tail_neg = max(SumMsg_head_pos+Gnp,SumMsg_head_neg+Gnn);
            Msg_tail_pos = max(SumMsg_head_pos+Gpp,SumMsg_head_neg+Gpn);
            
            % Sum up gradients of consistent configurations and propage to neighboring
            % edges
            SumMsg_tail_neg = [Msg_tail_neg,Msg_head_neg]*MBProp(:,iTail);
            SumMsg_tail_pos = [Msg_tail_pos,Msg_head_pos]*MBProp(:,iTail);
            SumMsg_head_neg = [Msg_tail_neg,Msg_head_neg]*MBProp(:,iHead);
            SumMsg_head_pos = [Msg_tail_pos,Msg_head_pos]*MBProp(:,iHead);
        end
        
        % find out the labeling: sum up the edge messages coming towards each node
        M_max1 = [Msg_tail_neg,Msg_head_neg]*MBPropEdgeNode;
        M_max2 = [Msg_tail_pos,Msg_head_pos]*MBPropEdgeNode;
        % pick the label of maximum message value
        Ymax = (M_max1 <= M_max2)*2-1;
        % get predicted value
        YmaxVal = (M_max2 - M_max1);

        normModel=1;
        if normModel==1 % normalize by edge degree
            NodeDegree = ones(size(YmaxVal,2),1);
            for v = 1:size(YmaxVal,2)
                NodeDegree(v) = sum(E(:) == v);
            end
            YmaxVal=YmaxVal./repmat(NodeDegree',size(YmaxVal,1),1);
        end
        if normModel==2 % normailze into unit vector length
            if size(YmaxVal,1) > 1
                YmaxValNorm=[];
                for i=1:size(YmaxVal,1)
                    YmaxValNorm=[YmaxValNorm;norm(YmaxVal(i,:))];
                end
                YmaxVal=YmaxVal./repmat(YmaxValNorm,1,size(YmaxVal,2));
            end
        end

        if nargout > 2
            % find out the max gradient for each example: pick out the edge labelings
            % consistent with Ymax
            Umax(1,:) = reshape(and(Ymax(:,E(:,1)) == -1,Ymax(:,E(:,2)) == -1)',1,size(E,1)*m);
            Umax(2,:) = reshape(and(Ymax(:,E(:,1)) == -1,Ymax(:,E(:,2)) == 1)',1,size(E,1)*m);
            Umax(3,:) = reshape(and(Ymax(:,E(:,1)) == 1,Ymax(:,E(:,2)) == -1)',1,size(E,1)*m);
            Umax(4,:) = reshape(and(Ymax(:,E(:,1)) == 1,Ymax(:,E(:,2)) == 1)',1,size(E,1)*m);
            % sum up the corresponding edge-gradients
            Gmax = reshape(sum(gradient.*Umax),size(E,1),m);
            Gmax = reshape(sum(Gmax,1),m,1);
        end
        gradient = reshape(gradient,g_siz);
    end
end


% Construct a matrix containing the neighborhood information of the edges.
% The matrix consists of four blocks, corresponding to the edges that merge 
% (e(2) = e'(2)), branch (e(1) = e'(1)), form a chain forward (e(2) =
% e'(1)) or backward (e(1) = e'(2))
function [MBProp,MBPropEdgeNode] = buildBeliefPropagationMatrix(E)
    MBProp = zeros(size(E,1)*2); % for edge to edge propagation
    MBPropEdgeNode = zeros(size(E,1)*2,max(max(E))); % for edge to node propagation

    numEdges = size(E,1);

    iTail = 1:numEdges;
    iHead = iTail+numEdges;

    for node = 1:max(max(E))

      eTail = find(E(:,1) == node);
      eHead = find(E(:,2) == node);

      % Edges that meet node with their tail
      MBPropEdgeNode(iTail(eTail),node) = 1;
      % Edges that meet node with the head
      MBPropEdgeNode(iHead(eHead),node) = 1;

      % Matrix block for progating messages from edges that meet with their
      % tails at node (eTail); 
      Link = MBProp(iTail,iTail); 
      Link(eTail,eTail) = 1;
     % remove diagonal; we do not propage messages back to self 
      MBProp(iTail,iTail) = Link-diag(diag(Link));

      % Matrix block for progating messages via a backward chain (eTail meeting eHead) at node;
      % messages will go from iTail to iTail (excluding self loops)
      Link = MBProp(iTail,iHead);
      Link(eTail,eHead) = 1;
      % remove diagonal; in case there are self loops e = (v,v) in the graph
      MBProp(iTail,iHead) = Link-diag(diag(Link));

      % Matrix block for progating messages from edges that meet with their
      % heads at node (eHead)
      Link = MBProp(iHead,iHead);
      Link(eHead,eHead) = 1; 
      % remove diagonal; we do not propage messages back to self
      MBProp(iHead,iHead) = Link-diag(diag(Link));

      % Matrix block for progating messages  via a forward chain (eHead meeting
      % eTail) at node;
      Link = MBProp(iHead,iTail);
      Link(eHead,eTail) = 1;
      % remove diagonal; in case there are self loops e = (v,v) in the graph
      MBProp(iHead,iTail) = Link-diag(diag(Link));

    end
end


function loss = compute_loss_vector(Y,scaling)
    global E;
    global m;
    print_message('Computing loss vector...',0);
    loss = ones(4,m*size(E,1));
    Te1 = Y(:,E(:,1))'; % the label of edge tail
    Te2 = Y(:,E(:,2))'; % the label of edge head
    NodeDegree = ones(size(Y,2),1);
    if scaling == 1 % rescale to microlabels by dividing node loss among the adjacent edges
        for v = 1:size(Y,2)
            NodeDegree(v) = sum(E(:) == v);
        end
    end
    NodeDegree = repmat(NodeDegree,1,m);
    u = 0;
    for u_1 = [-1, 1]
        for u_2 = [-1, 1]
            u = u + 1;
            %loss(u,:) = reshape((Te1 ~= u_1)./NodeDegree(E(:,1),:)+(Te2 ~= u_2)./NodeDegree(E(:,2),:),m*size(E,1),1);
            loss(u,:) = reshape(((Te1 ~= u_1)-(Te1 == 0)/2)./NodeDegree(E(:,1),:)+((Te2 ~= u_2)-(Te2 == 0)/2)./NodeDegree(E(:,2),:),m*size(E,1),1); % x-x:0,x-y:1,x-o:0.5
        end
    end
    loss = reshape(loss,4*size(E,1),m);
end


function profile_init
    global profile;
    profile.start_time = cputime;
    profile.next_profile_tm = profile.start_time;
    profile.n_err = 0;
    profile.p_err = 0; 
    profile.n_err_microlbl = 0; 
    profile.p_err_microlbl = 0; 
    profile.n_err_microlbl_prev = 0;
    profile.microlabel_errors = [];
    profile.iter = 0;
    profile.err_ts = 0;
end

function optimizer_init
    clear global MBProp;
    clear global MBPropEdgeNode;
    clear global Rmu;
    clear global Smu;
    clear global term12;
    clear global term34;
end

function print_message(msg,verbosity_level,filename)
    global params;
    if params.verbosity >= verbosity_level
        fprintf('%s: %s\n',datestr(clock),msg);
        if nargin == 3
            fid = fopen(filename,'a');
            fprintf(fid,'%s: %s\n',datestr(clock),msg);
            fclose(fid);
        end
    end
end





