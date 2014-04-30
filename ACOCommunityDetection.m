%
% Tom de Ruijter - deruijter.tom@gmail.com
% April 2014
%
% Implements the Community Detection Algorithm proposed in 'Community
% Detection Using Ant Colony Optimization', by Hongao, Zuren and Zhigang.
%
% This software is free to use by anyone for any purpose.
% I am not responsible for any consequences caused by this software.
%

% Applies Ant based communty detection on an adjacency matrix A.
% Returns a membership vector C.
function [C,score,nmi] = ACOCommunityDetection (A, measure, heuristic, varargin)
    % A - Adjacency matrix representing a graph
    % measure - Function handle of method used as maximizer

    assert(size(A,1) == size(A,2));

    % Parameters
    N = 200;                % Iterations
    n_a = 30;               % Ant count
    ro = 0.8;               % Pheromone dispersal
    e = 0.005;              % Pheromone range scaling
    a = 1.0;                % Pheromone weight
    b = 2.0;                % Heuristic weight
    t = 10;                 % Starting pheromone value
    tau = 0.15;             % Teleportation probability

    % Initializing trail and heuristic
    fprintf('Initializing trails and heuristic\n');
    tic();
    T = full(double(A~=0) .* t);
    T(logical(eye(size(T)))) = 0;               % Set diagonal to zero
%     H = 1.0 ./ (1 + exp(-corrcoef(A)));         % Sigmoidal Pearson correlation
    H = 1.0 ./ (1 + exp(-feval(heuristic,A)));

    H(A==0) = 0;                                % Remove non-connected values
    t = toc();
    
    % Initialize optional variables for goal function
    vars = {};
    if strcmp(measure,'Infomap')
       vars.ergodic = ErgodicProbability(A,tau);
       vars.tau = tau;
    end
    fprintf('Done initializing: %f seconds\n',t);
    
    % Main loop
    dim = length(A);
    C_sgb = zeros(1,dim);                       % Global best solution
    
    if strcmp(measure,'Infomap')
        M_sgb = Inf;
    else
         M_sgb = 0;
    end
    tic();
    for it = 1:N
        if mod(it,10) == 0
            fprintf('Iteration %d.\n',it);
        end
        s_ib = ones(1,dim) * (-1);              % Iteration best solution
        if strcmp(measure,'Infomap')
            M_sib = Inf;
        else
             M_sib = 0;
        end
        P = ProbabilityMap(A,T,H,a,b);
        for ant = 1:n_a
%             fprintf('-Ant %d\n',ant);
            [s,C] = ConstructTrail(P);          % Construct trail for a single ant
            M_s = Goal(A,C,measure,vars);
            
            if (strcmp(measure,'Modularity') && (M_s > M_sib)) || (strcmp(measure,'Infomap') && (M_s < M_sib))
                s_ib = s;
                M_sib = M_s;
            end
            if (strcmp(measure,'Modularity') && (M_s > M_sgb)) || (strcmp(measure,'Infomap') && (M_s < M_sgb))
                M_sgb = M_s;
                C_sgb = C;
            end
            
        end
        t_max = Goal(A,C_sgb,measure,vars) / (1 - ro);
        if(t_max <= 0.05 || t_max >= 1.0)
            t_max = 0.5;
        end
        t_min = t_max * e;
        T = UpdateTrails(T,t_max,t_min,s_ib,M_sib,ro);
    end
    t = toc();
    % Return the trail with the best goal score
    fprintf('Run finished in %f seconds.\n',t);
    fprintf('Highest score achieved: %f\n',M_sgb);
    
    % Prepare function output
    C = C_sgb;
    score = M_sgb;
    nmi = -1;
    
    if(nargin > 3)
        S = VecToMat(C_sgb); 
        S_true = VecToMat(varargin{1}); 
        nmi = NormalizedMutualInformation(S_true,S);
        fprintf('Normalized Mutual information: %f\n',nmi);
    end
    
end

% To maximze goal function
function [Score] = Goal (A,c,measure,vars)

    % Construct binary membership matrix out of membership vector.
    S = VecToMat(c);
    
    if strcmp(measure, 'Modularity')
        Score = Modularity(A,S);
    else strcmp(measure, 'Infomap');
        Score = Infomap(A,S,vars.ergodic,vars.tau);
    end

end

% === GOAL FUNCTIONS === %

% Infomap measure
function [I] = Infomap(A,S,p_ergodic,tau)

    num_cats = size(S,2);
	q = zeros(num_cats,1);
    p_modules = p_ergodic' * S;
	e = 0;			% Sum of exit entropies
	s = 0;			% Sum of movement entropies
    
	% Loop over all modules
    % TODO: Write this into matrix notation.
	for m = 1:num_cats
        q(m) = ExitProbability(A,S(:,m),p_ergodic,tau);
		e = e + plogp(q(m));
		s = s + plogp(q(m) + p_modules(m));
	end
    Q = sum(q);		% Sum of cluster change probabilities
	% Construct final solution
	I = plogp(Q) - 2 * e + s;
    if I <= 0
        I = Inf;
    end
    
end

% Calculates the Newman-Girvan modularity
function [Q] = Modularity (A,S)
    % Matrix formulation of modularity, as explained on 
    % http://en.wikipedia.org/wiki/Modularity_(networks)#Matrix_formulation
    m = nnz(A);
    K = diag(A*A');
    B = A - ((K*K')/(2*m));
    Q = trace(S'*B*S)/(2*m);
end

% === HEURISTICS === %

% Pearson Correlation
function [C] = PearsonCorrelation(A)
    % Only works for binary graphs
    
    n = length(A);
    C = zeros(size(A));
    
    sums = sum(A);
    mu = sums ./ n;
    sigma = sums .* (1-mu).^2 + (n-sums-1) .* (-mu).^2;
    sigma = sqrt(sigma ./ n);

    for i = 1:n
        for j = 1:n
            for l =1:n
                tmp = (A(i,l) - mu(i)) .* (A(j,l) - mu(j));
                C(i,j) = C(i,j) + tmp;
            end
            C(i,j) = C(i,j) ./ (n .* sigma(i) .* sigma(j));     
        end
    end
end

% Edge-randomized probability
function [P] = EdgeRandomizedProbability(A)

    K = diag(A*A');
    m = nnz(A);
    P = ((K*K')/(2*m));

end

% Target vertex degree
function [D] = VertexDegree(A)

    K = diag(A*A');
    D = repmat(K,1,length(A));
    
end

% Vertex deree distance
function [D] = VertexDegreeDistance(A)
    
    D = zeros(size(A));
    K = diag(A*A');
    n = length(A);
    for i = 1:n
        for j = 1:n
            D(i,j) = K(j) - K(i);
        end
    end

end

% === EVALUATION === %

% NMI measure
function [I] = NormalizedMutualInformation (X,Y)
    I = (2.*MutualInformation(X,Y)) ./ (Entropy(X) + Entropy(Y));
end

% Given two module assignment matrices, calculates their MI.
function [I] = MutualInformation (X,Y)
    n = size(X,1);
    Px = sum(X)./n;
    Py = sum(Y)./n;
    n_x = length(Px);
    n_y = length(Py);
    Pxy = zeros(n_x,n_y);

    I = 0;
    for k = 1:n_x
        for l = 1:n_y
             Pxy(k,l) = sum(X(:,k) & Y(:,l)) ./ n;
             norm = Px(k) .* Py(l);
          
             if norm ~= 0
                 P = Pxy(k,l) ./ norm;
                 if (P >= 1e-15)
                     I = I + (Pxy(k,l) .* log(P));
                 end
             end
        end
    end
end

% Entropy measure
function [H] = Entropy(S)
    P = sum(S) ./ size(S,1);
    H = arrayfun(@(x) plogp(x),P);
    H = - sum (H);
end

% === UTILITY === %

% Converts an index vector into a binary matrix
function [S] = VecToMat(c)
    dim = length(c);
    num_cats = max(c);
    S = zeros(dim,num_cats);
    idx = sub2ind([dim,num_cats],1:dim,c);
    S(idx)=1;
end

% Calculates the ergodic node visit frequency of a graph A
% by using the power method.
function [p_ergodic] = ErgodicProbability (A,tau)
	n = length(A);
	p = ones(n,1) / n;

	K_inv = diag(1 ./ sum(A));
	M = A' * K_inv;
	W = tau .* ( ones(n,n) ./ n ) + (1-tau) .* M;

    prev = 0;
    it = 0;
	while (norm(p - prev) > 1e-4 && it < 10)
		prev = p;
        p = W * p;
        it = it+1;
	end
	p_ergodic = p;
end

% Calculates the probability a random step moves outside of a module
function [q] = ExitProbability (A,c,p,tau)
    % Adjacency matrix A
    % Node coloring vector c for *one* module
    % Ergodic node probabilities p

    n = length(c);
    n_i = sum(c);

    a = c==1;      % indexes of members of cluster i.
    b = c~=1;      % index of non-members of cluster i.

    module = p(a);          % Ergodic probs of vertices within module
    exits = A(a,b);         % Outgoing weights of the specified module
    migrate_prob = 0;

    for k = 1:length(module)
        migrate_prob = migrate_prob + sum(module(k) .* exits(k,:));
    end    

    q = tau.*((n-n_i)./(n-1)).*(p'*c) + (1-tau).*(migrate_prob);

end

% Negative entropy measure
function [H] = plogp(x)
	if (x < 1e-15)
		H = 0.0;
	else
		H = x * log(x);
	end
end

% Performs a depth-first search to construct membership vectors
function [C] = Membership(s)

dim = length(s);
visited = zeros(dim,1);
C = zeros(1,dim);
c = 1;
while (dim > nnz(C))
	stack = find(visited==0, 1, 'first');  					% Pick first unvisited node
    [C,visited] = Search(s,C,c,stack,visited);
    c = c+1;
end

end

function [C,visited] = Search(s,C,c,stack,visited)

    while (~isempty(stack))
        v = stack(1); stack(1) = [];						% Pop first element from queue
        visited(v) = 1;             						% Add element to visited
        C(v) = c;
        % Below list may contain duplicates, which is not a problem.
        neighbours = Neighbours (s,v,visited);
        stack = [neighbours,stack];							%#ok<AGROW>		
    end

end

function [neighbours] = Neighbours (s,v,visited)
    neighbours = [s(v), find(s==v)'];                   % Expand neighbours
    neighbours = neighbours(visited(neighbours) == 0);	% Filter out visited neighbours
end

% === CORE === %

% Calculates the ant-based probability for every vertex in the graph
function [P] = ProbabilityMap (A,T,H,a,b)

    dim = length(A);
    P = T.^a .* H.^b;
    
    for i = 1:dim
        norm = 0;
        for j = 1:dim
            norm = norm + T(i,j).^a .* H(i,j).^b;
        end
%         norm = sum( T(i,:).^a .* H(i,:).^b );
        P(i,:) = P(i,:) ./ norm;        
    end
    
end

% Given the probability map, samples a trail S with node coloring C.
function [S,C] = ConstructTrail (P)

    N = length(P);
    [S,~] = find(mnrnd(1,full(P))');
    
    C = zeros(1,N);
    c = 1;

    for v = 1:N
        s = S(v);
        if C(s) ~= 0
            if C(v) ~= 0
                if C(s) == C(v)
                    continue;
                else
                    C(logical(C == C(s))) = C(v);
                end
            else
                C(v) = C(s);
            end

        else
            if C(v) ~= 0
                C(s) = C(v);
            else
                C(v) = c;
                C(s) = c;
                c = c + 1;
            end
        end
    end
    
end

% Updates the pheromone on all edges and scales them within the interval [t_min,t_max]
function [T] = UpdateTrails (T,t_max,t_min,s,M,ro)

    % Update pheromone


    % Scale T within interval
%     dT = diff( [t_min,t_max] );
%     T =  T - min( T(:));            % set range of A between [0, inf)
%     T =  T ./ max( T(:)) ;          % set range of A between [0, 1]
%     T =  T .* dT ;                  % set range of A between [0, dRange]
%     T =  T + t_min;                 % shift range of A to R


    % Threshold T within interval
    T(logical(T > t_max)) = t_max;
    T(logical(T < t_min)) = t_min;

    T = T .* ro;
%     length(s)
    for i=1:length(s)
        T(i,s(i)) = T(i,s(i)) + M;
        if i~=s(s(i)), T(s(i),i) = T(s(i),i) + M; end
    end
    
end