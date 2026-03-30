
%    cluster_method == 1: giving the center and width of membership function
%                         with k-means
%    cluster_method == 2: giving the center and width of membership
%                         function with AP algorithm

%    is_nonstationary ==1: with nonstationary process




clc
clear 
close all

load 'dataset\Vehicle.mat'
% 
% details = '归一化范围 input[-1, 1]，output(0 | 1)，inputEx在input的基础上添加偏置系数 1';
% 
% input = DRD.data';
% inDim = size(input,1);
% instanceNum = size(input, 2);
% inputEx = [input; ones(1, instanceNum)];
% name = 'DRD';
% outDim = max(DRD.label);
% output = zeros(outDim, instanceNum);
% 
% for i = 1:instanceNum
%     output(DRD.label(i), i) = 1;
% end
% rawData = [DRD.data, DRD.label]';
% clear DRD
% save('G:\期刊论文\codenew\code\classify\dataset\DRD.mat')







% load('dataset\DRD.mat')
isNorm = 'y';    % 'n' is not normalized, 'y' is normalized
clear_useless_information_classify(isNorm);
disp('对于不同的数据集 mat文件，可能命名不同导致报错')

tic

cluster_method = 1;
is_nonstationary = 1;
num_nonstationary = 30;
disturb_coff = [0.1,0.3];    % 非平稳的扰动系数
set_grad = 0;           % whether to use conjugate   True:1 || False: 0

% set the hyperparameter
rate_for_train = 0.8;   % the number of sample used to be train set
MaxEpoches = 1000;      % the maximum number of epoch of the iteration
% learn_rate_GNF = 0.07;  % learning rate 
ap_initial = -5;       % AP初始值

FSglobal_train_acc = zeros(1,10);
FSglobal_test_acc = zeros(1,10);
NFSglobal_train_acc = zeros(1,10);
NFSglobal_test_acc = zeros(1,10);


for i=1:1    %调参用
    
%     FSglobal_train_acc(i) = 0;
%     FSglobal_test_acc(i) = 0;
%     NFSglobal_train_acc(i) = 0;
%     NFSglobal_test_acc(i) = 0;
    
    
    for parameter_loop = 1:10
        clear_information_classify();    
        
        % 随机选择训练样本
        [train_sample_input, train_sample_output, test_sample_input, test_sample_output, number_train_sample, number_test_sample] = ...
            select_train_sample(input_sample, output_sample);
        

%%%%%%%%%%%%%%%%%↓↓↓↓↓实现聚类过程↓↓↓↓↓%%%%%%%%%%%%%%%%%%%%%
        if(cluster_method == 1)
            %   K-means 
            number_rules = number_classes;   % the number of the fuzzy rules, we set it the number of the classes here;
            [idx, C, a_membership, width_membership] = kmeans_value_parameter(number_rules, number_classes);
            link_weights_rule2output = rand(number_classes,number_rules);
            %   AP
        elseif(cluster_method == 2)
            [C_index,C,idx, a_membership, width_membership, link_weights_rule2output] = ap_value_parameter(number_classes, ap_initial);
            number_rules = length(C_index);             
        end
%%%%%%%%%%%%%%%%%↑↑↑↑↑实现聚类过程↑↑↑↑↑%%%%%%%%%%%%%%%%%%%%%        
        
%%%%%%%%%%%%%%%%%↓↓↓↓↓一型模糊神经网络更新过程↓↓↓↓↓%%%%%%%%%%%%%%%%%%%%%        
        learn_rate_GNF = 0.15;  % learning rate
        is_update_memb = 1;
        is_update_width = 1;
        is_update_w = 1;
        [a_membership, width_membership, link_weights_rule2output, Error] = ...
            decent_grad_method_anfnn_classify( a_membership, is_update_memb, width_membership, is_update_width, link_weights_rule2output,is_update_w, 1);
       
        %鲁棒性测试：中心、宽度、权重加噪声    
%       a_membership = a_membership + normrnd(0, 0.04, size(a_membership,1), size(a_membership,2));
%       width_membership = width_membership + normrnd(0, 0.04, size(width_membership,1), size(width_membership,2));
%       link_weights_rule2output = link_weights_rule2output + normrnd(0, 0.04, size(link_weights_rule2output,1), size(link_weights_rule2output,2));
        
        
        [train_label_output_layer, test_label_output_layer] = print_label_output_layer();
        [currentTrainAcc, currentTestAcc] = print_result_on_dataset_classify(train_label_output_layer, test_label_output_layer);
        
        FSglobal_train_acc(parameter_loop) = currentTrainAcc;
        FSglobal_test_acc(parameter_loop) = currentTestAcc;
        
        
%%%%%%%%%%%%%%%%%↑↑↑↑↑一型模糊神经网络更新过程↑↑↑↑↑%%%%%%%%%%%%%%%%%%%%% 
        %非平稳模糊系统：投票机制决定结果
        if is_nonstationary == 1 && num_nonstationary >=1
            allModel_train_label = zeros(num_nonstationary,number_train_sample); %保存num_nonstationary个FNN的标签输出结果，训练过程
            allModel_test_label = zeros(num_nonstationary,number_test_sample); %保存num_nonstationary个FNN的标签输出结果，测试过程
            allModel_train_label(1,:) = train_label_output_layer;
            allModel_test_label(1,:) = test_label_output_layer;
          
            
%%%%%%%%%%%%%%%%%↓↓↓↓↓非平稳模糊神经网络更新过程↓↓↓↓↓%%%%%%%%%%%%%%%%%%%%%                
            %记录第一次更新完的中心宽度权重
            a_membership_1 = a_membership;   
            width_membership_1 = width_membership;
            link_weights_rule2output_1 = link_weights_rule2output;
            learn_rate_GNF = 0.1;  % learning rate
            is_update_w = 1;
            is_update_memb = 0;
            is_update_width = 1;
            for i_nonstationary = 1:num_nonstationary
                
%                随机选择百分之70的单独模糊神经网络更新
                if unifrnd(0,1) < 0.8
                    is_update_w = 1;
                    is_update_memb = 0;
                    is_update_width = 1; 
                end
                
                [a_membership, width_membership, link_weights_rule2output, Error] = ...
                    decent_grad_method_anfnn_classify( a_membership_1, is_update_memb, width_membership_1, is_update_width, link_weights_rule2output_1,is_update_w, i_nonstationary);
                
            %鲁棒性测试：中心、宽度、权重加噪声    
%               a_membership = a_membership + normrnd(0, 0.05, size(a_membership,1), size(a_membership,2));
%               width_membership = width_membership + normrnd(0, 0.04, size(width_membership,1), size(width_membership,2));
%               link_weights_rule2output = link_weights_rule2output + normrnd(0, 0.04, size(link_weights_rule2output,1), size(link_weights_rule2output,2));
                

                [train_label_output_layer, test_label_output_layer] = print_label_output_layer();
                allModel_train_label(i_nonstationary,:) = train_label_output_layer;
                allModel_test_label(i_nonstationary,:) = test_label_output_layer;
            end
            
            % 随机选取百分之70的网络输出结果
            Mark = randperm(num_nonstationary);
            Vector_mark = Mark(:,1:num_nonstationary * 0.7);
            allModel_train_label = allModel_train_label(Vector_mark,:);
            
            
            
            train_label_output_layer = mode(allModel_train_label);
            test_label_output_layer = mode(allModel_test_label);
            [currentTrainAcc, currentTestAcc] = print_result_on_dataset_classify(train_label_output_layer, test_label_output_layer);
        end
        
        NFSglobal_train_acc(parameter_loop) = currentTrainAcc;
        NFSglobal_test_acc(parameter_loop) = currentTestAcc;
%%%%%%%%%%%%%%%%%↑↑↑↑↑非平稳模糊神经网络更新过程↑↑↑↑↑%%%%%%%%%%%%%%%%%%%%%   
       
        
    end
    
 
end


result = [FSglobal_train_acc; FSglobal_test_acc; NFSglobal_train_acc; NFSglobal_test_acc]';

% save result_classify.mat
% find(global_test_acc == max(global_test_acc))
mean(result)
std(result)
toc