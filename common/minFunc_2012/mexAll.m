% minFunc
fprintf('Compiling minFunc files...\n');

% nicsurpatanu's fix at https://github.com/amaas/stanford_dl_ex/issues/3
mex -o minFunc/compiled/mcholC.mex minFunc/mex/mcholC.c
mex -o minFunc/compiled/lbfgsC.mex minFunc/mex/lbfgsC.c
mex -o minFunc/compiled/lbfgsAddC.mex minFunc/mex/lbfgsAddC.c
mex -o minFunc/compiled/lbfgsProdC.mex minFunc/mex/lbfgsProdC.c
