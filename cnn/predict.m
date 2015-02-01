function preds = predict(opttheta,images,labels,varargin)
    % call cnnCost() on mini-batches, for stupid memory reasons...
    numImages = size(labels, 1);
    assert(numImages == size(images, 3));

    %batchSize = 256; % KNOW this size won't crash, from SGD
    %batchSize = 10000;%2048; % increased manually to find the crash limit. is bigger really better, though?
    %batchSize = 2500; % MATLAB (32-bit) approximate crash limit
    batchSize = 1000; % to be safe. training takes ~ 1h! don't want to crash stupidly on testing (again)
    preds = [];
    for start=1:batchSize:numImages
        batch = start:start+batchSize-1;
        [unused_, unused_, batchPreds] = cnnCost(opttheta, images(:,:,batch), labels(batch), varargin{:});
        preds = [preds; batchPreds];
    end
end
