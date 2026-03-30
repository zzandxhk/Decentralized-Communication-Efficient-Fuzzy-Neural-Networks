function clear_useless_information_classify(isNorm)
    
	if(isNorm == 'n')
		evalin('base', 'input_sample = input;');		% rename the data item
		evalin('base', 'inDim;')
	else
		evalin('base', 'input_sample = inputEx;');		% rename the data item
		evalin('base', 'inDim = inDim + 1;')
	end
	  
    evalin('base', 'output_sample = output;');
    evalin('base', 'total_sample = instanceNum;');
    evalin('base', 'number_feature = inDim;')
    evalin('base', 'number_classes = outDim;')
    evalin('base', 'clear details')					% clear the unuseful information
    evalin('base', 'clear inputEx')
    evalin('base', 'clear input')
    evalin('base', 'clear output')
    evalin('base', 'clear name')
    evalin('base', 'clear instanceNum')
    evalin('base', 'clear rawData')
    evalin('base', 'clear inDim')
    evalin('base', 'clear outDim')
end