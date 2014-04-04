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
function [C] = ACOCommunityDetection (A)

assert(size(A,1) == size(A,2));

% Parameters
N = 100;
n_a = 30;
ro = 0.8;
e = 0.005;
a = 1.0;
b = 2.0;
t = 10;

% Initializing trail and heuristic
T = double(A~=0) .* t;
T(logical(eye(size(T)))) = 0;               % Set diagonal to zero
H = 1.0 / (1 + exp(corrcoef(A)));           % Sigmoidal Pearson correlation
H(logical(T==0)) = 0;                       % Remove non-connected values

% Main loop
dim = length(A);
s_gb = ones(dim,1) * (-1);                  % Global best solution
M_sgb = 0;
for it = 1:N
    s_ib = ones(dim,1) * (-1);              % Iteration best solution
    M_sib = 0;
    S = cell(n_a,1);
    P = ProbabilityMap(T,H,a,b);
    for ant = 1:n_a
        [s,~] = find(mnrnd(1,P)');          % Construct trail for a single ant
        S{ant,1} = s;
        M_s = Modularity(A,s);
        if (M_s > M_sib)
            s_ib = s;
            M_sib = M_s;
        end
        if (M_s > M_sgb)
            s_gb = s;
            M_sgb = M_s;
        end
    end
    t_max = Modularity(A,s_gb) / (1 - ro);
    t_min = t_max * e;
    T = UpdateTrails(T,t_max,t_min,s_ib,M_sib,ro);
end

% Return the trail with the best modularity
C = Membership(s_gb);
end


% Calculates the Newman-Girvan modularity
function [Q] = Modularity (A,s)

m = nnz(A);
dim = length(A);
sum = 0;
C = Membership(s);
for i = 1:dim;
    for j = 1:dim;
        % return 0 if i and j are in different modules
        if C(i) ~= C(j)
            continue;
        end
        k_i = nnz(A(i,:));
        k_j = nnz(A(j,:));
        sum = sum + (A(i,j) - (k_i*k_j)/(2*m));
    end
end
Q = sum / (2*m);
end

% Performs a bread-first search to construct membership vectors
function [C] = Membership(s)

dim = length(s);
visited = zeros(dim,1);
count = 0;                                                      % i == nnz(C)
C = zeros(1,dim);
c = 0;
while (dim > count)
	queue = find(visited==0, 1, 'first');  					% Pick first unvisited node
    c = c+1;
	while (~isempty(queue))
        v = queue(1); queue(1) = [];						% Pop first element from queue
        visited(v) = 1; count=count+1;						% Add element to visited
        C(v) = c;
        neighbours = [s(v), find(s==v)];					% Expand neighbours
        neighbours = neighbours(visited(neighbours) == 0);	% Filter out visited neighbours
        queue = [neighbours,queue];							%#ok<AGROW>		
	end
end

end

% Calculates the ant-based probability for every vertex in the graph
function [P] = ProbabilityMap (T,H,a,b)

dim = length(T);
P = (T.^a .* H.^b);
for i = 1:dim
    norm = sum( T(i,:).^a .* H(i,:).^b );
    P(i,:) = P(i,:) ./ norm;
end

end

% Updates the pheromone on all edges and scales them within the interval [t_min,t_max]
function [T] = UpdateTrails (T,t_max,t_min,s,M,ro)

% Update pheromone
T = T * ro;
for i=1:length(s)
    T(i,s(i)) = T(i,s(i)) + M;
    T(s(i),i) = T(s(i),i) + M;
end

% Scale T within interval [t_min,t_max]
dT = diff( [t_min,t_max] );
T =  T - min( T(:));            % set range of A between [0, inf)
T =  T ./ max( T(:)) ;          % set range of A between [0, 1]
T =  T .* dT ;                  % set range of A between [0, dRange]
T =  T + t_min;                 % shift range of A to R

end
