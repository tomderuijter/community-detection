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
H = 1.0 ./ (1 + exp(corrcoef(A)));           % Sigmoidal Pearson correlation
H(logical(T==0)) = 0;                       % Remove non-connected values

% Main loop
dim = length(A);
C_sgb = zeros(1,dim);
M_sgb = 0;
for it = 1:N
    s_ib = ones(1,dim) * (-1);              % Iteration best solution
    M_sib = 0;
    P = ProbabilityMap(T,H,a,b);
    for ant = 1:n_a
        [s,C] = ConstructTrail(P);          % Construct trail for a single ant
        M_s = Modularity(A,C);
        if (M_s > M_sib)
            s_ib = s;
            M_sib = M_s;
        end
        if (M_s > M_sgb)
            M_sgb = M_s;
            C_sgb = C;
        end
    end
    t_max = Modularity(A,C_sgb) / (1 - ro);
    t_min = t_max * e;
    T = UpdateTrails(T,t_max,t_min,s_ib,M_sib,ro);
end

% Return the trail with the best modularity
C = C_sgb;
end

% Calculates the Newman-Girvan modularity
function [Q] = Modularity (A,C)

m = nnz(A);
dim = length(C);

% Construct membership matrix out of vector.
num_cats = max(C);
S = zeros(dim,num_cats);
% dim
% num_cats
% C
idx = sub2ind([dim,num_cats],1:dim,C);
S(idx)=1;

K = diag(A*A');
B = A - ((K*K')/(2*m));
Q = trace(S'*B*S)/(2*m);

end

% Given the probability map, samples a trail S with node coloring C.
function [S,C] = ConstructTrail(P)

N = length(P);
[S,~] = find(mnrnd(1,P)');
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
