function [currentTrainAcc, currentTestAcc] = print_result_on_dataset_classify(train_label_output_layer, test_label_output_layer)

number_train_sample = evalin('base', 'number_train_sample;');
number_test_sample = evalin('base', 'number_test_sample;');
train_sample_output = evalin('base', 'train_sample_output;');
test_sample_output = evalin('base', 'test_sample_output;');

fprintf('Train_acc')
[~, Result_Train_Idea] = max(train_sample_output, [], 1);
currentTrainAcc = sum(train_label_output_layer == Result_Train_Idea) / number_train_sample

fprintf('Test_acc')
[~, Result_Test_Idea] = max(test_sample_output, [], 1);
currentTestAcc = sum(test_label_output_layer == Result_Test_Idea) / number_test_sample

end