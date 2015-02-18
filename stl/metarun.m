function [preds, train_preds] = metarun()
    % a quickie wrapper function that will run stlExercise multiple times, saving results to disk each time.
    % purpose: to get error bars over multiple runs (each ~45 min in 64-bit Octave, since 32-bit MATLAB crashes/doesn't converge)

    addpath ../common % for isOctave()
    assert(isOctave(), 'RUN_RICA requires Octave! this simplifies my save() syntax too.')
    
    for r=1:10
        
        %%% the main event %%%
        stlExercise; % hmm, problem: stlExercise calls `clear all`...       
        
        % horrible workaround because stlExercise's `clear all` kills ALL variables in Octave 3.2.4
        if exist('preds.txt') % have to hard-code this name! it'll get cleared otherwise, even for local functions
            load 'preds.txt'
        else
            preds = [];
            train_preds = [];
        end
        
        % accumulate results of current run
        preds = [preds, pred];
        train_preds = [train_preds, train_pred];
        save('-text', 'preds.txt', 'preds', 'train_preds');
        
        % also save the trained RICA weights, which are costly. 
        % what the hell - save the whole damn workspace, besides the massive data structures
        % can conceivably use as 'load state' for further tinkering
        run_number = length(preds); % don't have to save r to disk!
        clear mnistData;
        clear unlabeledData;
        save('-mat', sprintf('run%02d.mat', run_number));
        
    end
    
    fprintf('Training accuracy: %g +/- %g\n', mean(train_preds), std(train_preds) / sqrt(run_number));
    fprintf('Test accuracy: %g +/- %g\n', mean(preds), std(preds) / sqrt(run_number));
    
    %delete '~preds.txt' % hmm, leave it be?? be careful on multiple runs then...
end
