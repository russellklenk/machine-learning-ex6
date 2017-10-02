function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
tryCount = 8;
tryVals  =[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
minError = 99999999999; % some large value
for i=1:tryCount
    for j=1:tryCount
        % train the SVM with the current tryC and trySigma.
        % printf("Training model with C = %f and sigma = %f.\n", tryVals(i), tryVals(j));
        model = svmTrain(X, y, tryVals(i), @(x1, x2) gaussianKernel(x1, x2, tryVals(j)));
        % compute predictions for the cross validation set.
        predictions = svmPredict(model, Xval);
        % compute the cross-validation error.
        cvError = mean(double(predictions ~= yval));
        % printf("Computed cross-validation error is %f.\n", cvError);
        % if this attempt was the best so far, save C and sigma.
        if (cvError < minError)
            C = tryVals(i);
            sigma = tryVals(j);
            minError = cvError;
        end
    end
end

% printf("Returning C = %f and sigma = %f.\n", C, sigma);






% =========================================================================

end
