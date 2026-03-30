function [a_membership, width_membership, link_weights_rule2output, Error] = decent_grad_method_anfnn_classify( a_membership,is_update_memb, width_membership, is_update_width, link_weights_rule2output,is_update_w, i_nonstationary )
% Below is to introduce the variable in workplace
MaxEpoches = evalin('base', 'MaxEpoches;');
learn_rate_GNF = evalin('base', 'learn_rate_GNF;');
number_classes = evalin('base', 'number_classes;');
number_feature = evalin('base', 'number_feature;');
number_rules = evalin('base', 'number_rules;');
train_sample_input = evalin('base', 'train_sample_input;');
train_sample_output = evalin('base', 'train_sample_output;');
test_sample_input = evalin('base', 'test_sample_input;');
test_sample_output = evalin('base', 'test_sample_output;');
number_train_sample = evalin('base', 'number_train_sample;');
number_test_sample = evalin('base', 'number_test_sample;');

%% 扰动项
disturb_coff = evalin('base', 'disturb_coff;');
a_membership = disturb_coff(1)*sin(disturb_coff(2)*(i_nonstationary-1)) + a_membership; 

%% 梯度更新过程
% set the variable for train and test
train_node_rule_layer = ones(number_rules, number_train_sample);
train_node_output_layer = ones(number_classes, number_train_sample);
test_node_rule_layer = ones(number_rules, number_test_sample);
test_node_output_layer = ones(number_classes, number_test_sample);
Epoches = 1;
train_node_input_layer = train_sample_input;
test_node_input_layer = test_sample_input;
Error = zeros(1,MaxEpoches);
delta_a_membership = 0;
delta_width_membership = 0;
delta_link_weights_rule2output = 0;

while(Epoches <= 1000)
    %计算目标函数
    for loop_for_sample = 1:number_train_sample 
        
        %Dropout 防止过拟合
%         Drop_vector=ones(size(train_node_input_layer,1),1);
%         for Drop_Point = 1: size(train_node_input_layer,1)
%             Drop_vector(Drop_Point,1) =  Drop_vector(Drop_Point,1) * randsample([0,1],1,true,[0.05,0.95]);
%         end
%         train_node_membership_layer = exp(- ( (train_node_input_layer(:, loop_for_sample) - a_membership).^2 ) .* (2 * (width_membership .^ 2)) .* Drop_vector );
        train_node_membership_layer = exp(- ( (train_node_input_layer(:, loop_for_sample) - a_membership).^2 ) .* ( (width_membership .^ 2)) );
        
        note = ones(1, number_rules);
        
        
        for i = 1:number_feature
            note = train_node_membership_layer(i, :) .* note;
        end
        train_node_rule_layer(:, loop_for_sample) = note';
        
        train_node_output_layer(:, loop_for_sample) =  link_weights_rule2output * train_node_rule_layer(:, loop_for_sample);
    end
    Error(Epoches) = sum(sum((train_node_output_layer - train_sample_output).^2));
    
    if is_update_w == 1
        %对权值求梯度
        delta_link_weights_rule2output = (train_node_output_layer- train_sample_output) * train_node_rule_layer';
        delta_link_weights_rule2output = delta_link_weights_rule2output / number_train_sample;
    end
    
    if is_update_memb == 1
        % 计算梯度
        for i = 1: number_train_sample

            % 对中心
            temp_c = ( link_weights_rule2output' * (link_weights_rule2output * train_node_rule_layer(:, i) - ...
                train_sample_output(:, i) ) )' .* (2 * train_node_rule_layer(:, i)' ).* width_membership .* width_membership .* ...
                (train_node_input_layer(:, i) - a_membership );
            delta_a_membership = delta_a_membership + temp_c;
        end
        delta_a_membership = delta_a_membership / number_train_sample;
    end
    
    if is_update_width == 1
        % 计算宽度梯度
        for i = 1: number_train_sample
            % 对宽度
            temp_width = -( link_weights_rule2output' * (link_weights_rule2output * train_node_rule_layer(:, i) - ...
                train_sample_output(:, i) ) )' .* (2 * train_node_rule_layer(:, i)' ) .* (train_node_input_layer(:, i) - a_membership) .* ...
                (train_node_input_layer(:, i) -  a_membership) .* width_membership;
            delta_width_membership = delta_width_membership + temp_width;
        end
        delta_width_membership = delta_width_membership / number_train_sample;      
    end  
    
    
    
    % 计算共轭
    %     if(Epoches == 1)
    %         G_a{Epoches} = delta_a_membership;
    %         G_width{Epoches} = delta_width_membership;
    %         G_link_weights{Epoches} = delta_link_weights_rule2output;
    %         g_a{Epoches} = reshape(G_a{Epoches}, size(G_a{Epoches}, 1)*size(G_a{Epoches}, 2), 1);
    %         g_width{Epoches} = reshape(G_width{Epoches}, size(G_width{Epoches}, 1)*size(G_width{Epoches}, 2), 1);
    %         g_link_weights{Epoches} = reshape(G_link_weights{Epoches}, size(G_link_weights{Epoches}, 1)*size(G_link_weights{Epoches}, 2), 1);
    %         %
    %         D_a{Epoches} = -G_a{Epoches};
    %         D_width{Epoches} = -G_width{Epoches};
    %         D_link_weights{Epoches} = -G_link_weights{Epoches};
    %         d_a{Epoches} =- g_a{Epoches};
    %         d_width{Epoches} =- g_width{Epoches};
    %         d_link_weights{Epoches} = -g_link_weights{Epoches};
    %     end
    %     if(Epoches > 1)
    %         % G
    %         G_a{Epoches} = delta_a_membership;
    %         G_width{Epoches} = delta_width_membership;
    %         G_link_weights{Epoches} = delta_link_weights_rule2output;
    %
    %
    %         [D_a, beta_a] = cal_conjugate_classify(D_a, G_a, Epoches); % to calculation conjugate coefficient
    %         [D_width, beta_width] = cal_conjugate_classify(D_width, G_width, Epoches);
    %         [D_link_weights, beta_link] = cal_conjugate_classify(D_link_weights, G_link_weights, Epoches);
    %
    % %        g_width{Epoches} = reshape(G_width{Epoches}, size(G_width{Epoches}, 1)*size(G_width{Epoches}, 2), 1);
    % %        g_link_weights{Epoches} = reshape(G_link_weights{Epoches}, size(G_link_weights{Epoches}, 1)*size(G_link_weights{Epoches}, 2), 1);
    % %
    %         %beta_a
    % %        beta_width
    % %         beta_link
    %
    %     end
    
    
    
    % 下面更新梯度
    a_membership = a_membership - learn_rate_GNF * delta_a_membership;
    width_membership = width_membership - learn_rate_GNF * delta_width_membership;
    link_weights_rule2output = link_weights_rule2output - learn_rate_GNF * delta_link_weights_rule2output;
    
    Epoches = Epoches + 1;
end
Error = Error(1:Epoches-1);
