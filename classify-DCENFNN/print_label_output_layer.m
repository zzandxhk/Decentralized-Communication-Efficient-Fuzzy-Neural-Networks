function [train_label_output_layer, test_label_output_layer] = print_label_output_layer()

% Introduce the global variable from workplace;
a_membership = evalin('base', 'a_membership;');
width_membership = evalin('base', 'width_membership;');
link_weights_rule2output = evalin('base', 'link_weights_rule2output;');
number_feature = evalin('base', 'number_feature;');
number_rules = evalin('base', 'number_rules;');
number_classes = evalin('base', 'number_classes;');

train_node_input_layer = evalin('base', 'train_sample_input;');
number_train_sample = evalin('base', 'number_train_sample;');
train_node_rule_layer = zeros(number_rules, number_train_sample);
train_node_output_layer = zeros(number_classes, number_train_sample);

test_node_input_layer = evalin('base', 'test_sample_input;');
number_test_sample = evalin('base', 'number_test_sample;');
test_node_rule_layer = zeros(number_rules, number_test_sample);
test_node_output_layer = zeros(number_classes, number_test_sample);

for loop_for_sample = 1:number_train_sample   % batch process
    train_node_membership_layer = exp(- ( (train_node_input_layer(:, loop_for_sample) - a_membership).^2 ) .* ( (width_membership .^ 2)) );
    note_train = ones(1, number_rules);
    for i = 1:number_feature
        note_train = train_node_membership_layer(i, :) .* note_train;
    end
    train_node_rule_layer(:, loop_for_sample) = note_train';
    
    train_node_output_layer(:, loop_for_sample) =  link_weights_rule2output * train_node_rule_layer(:, loop_for_sample);
end

for loop_for_sample = 1:number_test_sample   % batch process
    test_node_membership_layer = exp(- ( (test_node_input_layer(:, loop_for_sample) - a_membership).^2 ) .* ( (width_membership .^ 2)) );
    note_test = ones(1, number_rules);
    for i = 1:number_feature
        note_test = test_node_membership_layer(i, :) .* note_test;
    end
    test_node_rule_layer(:, loop_for_sample) = note_test';
    
    test_node_output_layer(:, loop_for_sample) =  link_weights_rule2output * test_node_rule_layer(:, loop_for_sample);
end

[~,train_label_output_layer] = max(train_node_output_layer, [], 1);
[~,test_label_output_layer] = max(test_node_output_layer, [], 1);
end