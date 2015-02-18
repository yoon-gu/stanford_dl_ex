function [acc, train_acc] = metarun()
    % a quickie wrapper function that will run stlExercise multiple times, saving results to disk each time.
    % purpose: to get error bars over multiple runs (each ~45 min in 64-bit Octave, since 32-bit MATLAB crashes/doesn't converge)

    addpath ../common % for isOctave()
    assert(isOctave(), 'RUN_RICA requires Octave! this simplifies my save() syntax too.')
    
    for r=1:10
        
        %%% the main event %%%
        stlExercise; % hmm, problem: stlExercise calls `clear all`...       
        
        % horrible workaround because stlExercise's `clear all` kills ALL variables in Octave 3.2.4
        if exist('acc.txt') % have to hard-code this name! it'll get cleared otherwise, even for local functions
            load 'acc.txt'
        else
            acc = [];
            train_acc = [];
        end
        
        % accumulate results of current run
        acc = [acc, mean(pred(:) == testLabels(:))];
        train_acc = [train_acc, mean(train_pred(:) == trainLabels(:))];
        save('-text', 'acc.txt', 'acc', 'train_acc');
        
        % also save the trained RICA weights, which are costly. 
        % what the hell - save the whole damn workspace, besides the massive data structures
        % can conceivably use as 'load state' for further tinkering
        run_number = length(acc); % don't have to save r to disk!
        clear mnist*;
        clear patches;
        clear train*
        clear test*
        clear unlabeled*;
        if exist('x'); clear x; end
        
        % the important part here is to save parameters, not data.
        save('-mat', sprintf('run%02d.mat', run_number));
        
    end
    
    load('acc.txt') % because clear train* will kill these train_acc
    fprintf('Training accuracy: %g +/- %g\n', mean(train_acc), std(train_acc) / sqrt(run_number));
    fprintf('Test accuracy: %g +/- %g\n', mean(acc), std(acc) / sqrt(run_number));
    
    %delete '~preds.txt' % hmm, leave it be?? be careful on multiple runs then...
end
