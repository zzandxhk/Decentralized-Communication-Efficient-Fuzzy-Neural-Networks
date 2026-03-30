function  [idx, C, a_membership, width_membership] = kmeans_value_parameter(number_rules, number_classes)
    
    % get the value of sample form workplace
    train_sample_input = evalin('base', 'train_sample_input;');
    number_feature = evalin('base', 'number_feature;');
    
    train_sample = [train_sample_input]';

    % Below is to ues algorithm K-means; 
    % Arrange as sample * feature;
    % idx means the class each sample belongs to 
    % C means the center node of each classes clustered
    [idx, C] = kmeans(train_sample, number_rules, 'dist','sqEuclidean','rep',4);   
    
    
    % Use the cluster center as the center of membership function
    a_membership = C';     
    % Get the value of width from the distance between the center and each node
    
    for k = 1: number_rules
        tempCluster = train_sample(idx==k,1:number_feature); % 取出第 K 类的所有输入，临时变量
        tempDiff = tempCluster - repmat(C(k,:),size(tempCluster,1),1); %做差，计算第 K 类所有输入分别减去其对于的中心，用于计算到中心点的距离
        middle = tempDiff.*tempDiff;    %%% 距离计算公式，平方开根号. middle 表示距离平方的结果，sqrt 表示开根号的结果。
        
        tempdist = sqrt(middle); %%%% sqrt 表示开根号的结果,输入维数的用处。
        
        %%%%% 求每一列的标准差,如何存储矩阵。
%         tempdist=1./tempdist;
        width_membership(:,k) = std(tempdist);  %%%% 对 tempdist 的每一列求标准差，当 k 为其中一类时，
    end
   
end
    