% AP算法作为模糊网络聚类过程
clear;clc;
format long
%--------[iris_matrix,txt_iris,raw_iris] = xlsread('iris.csv');                %12  14
[iris_raw, iris_regulation] = get_IrisData_GNF();
% load('Isolet.mat')           %114 115
% load('jaffe.mat')            %31  34
% load('lung_discrete.mat')    %12  13
% load('madelon.mat')          %96  104
% load('ORL.mat')              %72  73
sample_index = randperm(size(iris_raw, 1));
temp_matrix = iris_raw(:, 1:4);
% temp_matrix = [temp_matrix, -1*ones(size(iris_raw, 1), 1)];
temp_matrix = [temp_matrix, iris_raw(:, 5:7)];
iris_matrix = temp_matrix(sample_index(1:90), :);
[l,n]=size(iris_matrix);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              

% for i = 1:l
	% if ( strcmp(txt_iris{i}, 'Iris-setosa')==1 )
		% iris_matrix(i, 5:7) = [0, 0, 1];	% 便于表示，将类别表示为1 ， 2， 3
	% elseif ( strcmp(txt_iris{i}, 'Iris-versicolor') == 1)
		% iris_matrix(i, 5:7) = [0,1,0];
	% else
		% iris_matrix(i, 5:7) = [1,0,0];
	% end
% end
		

s=zeros([l,l]);

for i=1:l
    for j=1:l
        s(i,j)=-norm(iris_matrix(i,:)-iris_matrix(j,:));
    end
end
% p_median = median(median(s));
% p_mean=mean(mean(s));
% p_min=min(min(s));


% p1=p_median*ones(1,l);
% p2=p_mean*ones(1,l);
% p3=p_min*ones(1,l);
% 
% p1=p1';
% p2=p2';
% p3=p3';
% 
% for i=1:l
%     s(i,i)=p2(i);
% end
% for i = 70:0.01:108
	parameter = -85;
	p1= parameter *ones(1,l);
	
	[Num_mid,mid,idx] = apcluster(s, p1, iris_matrix);
% [idx,netsim,dpsim,expref,R,A] = final_apcluster(s,p1);
% [idx,netsim,dpsim,expref,R,A] = origin_apcluster(s,p2);
% [idx,netsim,dpsim,expref,R,A] = self_apcluster(s,p2);
% [idx,netsim,dpsim,expref,R,A] =oo_apcluster(s,p2);
% length(unique(idx))
%-----------------------------------------------以下为wfy 寻找合适参数的过程
% 	if(size(Num_mid, 1) == 3)
% 		disp('聚为 3 类')
% 		parameter
% 	else
% 		size(Num_mid)
% 	end
% end
[temp, I] = max(iris_matrix(:, 5:7), [], 2);
sum(I==idx)
disp('s')