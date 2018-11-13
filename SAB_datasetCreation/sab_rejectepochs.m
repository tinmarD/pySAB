% SAB_REJECTEPOCHS
%   The script load a SAB eeglab dataset (EEGrec or EEGenc), and call eeglab
%   gui to manually remove the bad epochs, it also update the hits,
%   correctRejects, omissions, falseAlarms and reaction time vectors by
%   deleting the indices correponding to bad epochs.
%   This script should be used after sab_getdatasets
% 
%   See also sab_getdatasets
%
%   Author(s): Martin Deudon (2016)



%% Parameters
datasetInDirPath    = 'C:\Users\deudon\Desktop\SAB\_Data\003_JG\datasets_600ms';
recOn               = 1;
encOn               = 0;

%% Create output directory
if strcmp(datasetInDirPath(end),filesep); datasetInDirPath=datasetInDirPath(1:end-1); end;
datasetOutDirPath   = [datasetInDirPath,'_epochRej'];   
createuniquedir(datasetOutDirPath);

%% Load data
if recOn; load(fullfile(datasetInDirPath,'EEGrec')); end;
if encOn; load(fullfile(datasetInDirPath,'EEGenc')); end;
load(fullfile(datasetInDirPath,'hits'));
load(fullfile(datasetInDirPath,'correctRejects'));
load(fullfile(datasetInDirPath,'omissions'));
load(fullfile(datasetInDirPath,'falseAlarms'));
load(fullfile(datasetInDirPath,'reactionTimes'));

%% Epoch rejection
%- Recognition dataset
if recOn
    eeglab;
    EEG     = EEGrec;
    ALLEEG  = EEG;
    %- Call pop_eegplot for manual epoch rejection and wait for the end of the
    %selection of bad epochs
    pop_eegplot(EEG, 1, 0, 0);
    uiwait();
    %- Get the vector of rejected epochs indices
    epochRejectInd  = ALLEEG(end).reject.rejmanual;
    %- Reject the epoch in EEGrec
    EEGrec          = pop_rejepoch(EEGrec, epochRejectInd, 1);
    %- Quit eeglab gui
    close(gcf);
    clear global EEG ALLEEG LASTCOM CURRENTSET;
    %- Update response vectors and reaction time vector
    hits(epochRejectInd)            = [];
    correctRejects(epochRejectInd)  = [];
    omissions(epochRejectInd)       = [];
    falseAlarms(epochRejectInd)     = [];
    reactionTimes(epochRejectInd)   = [];
    %- Save the changes
    save(fullfile(datasetOutDirPath,'hits'),'hits');
    save(fullfile(datasetOutDirPath,'correctRejects'),'correctRejects');
    save(fullfile(datasetOutDirPath,'omissions'),'omissions');
    save(fullfile(datasetOutDirPath,'falseAlarms'),'falseAlarms');
    save(fullfile(datasetOutDirPath,'reactionTimes'),'reactionTimes');
    save(fullfile(datasetOutDirPath,'EEGrec'),'EEGrec');
    disp(['Saving updated dataset and response vectors in ',datasetOutDirPath]);
end
%- Encodage dataset
if encOn
    eeglab;
    EEG     = EEGenc;
    ALLEEG  = EEG;
    %- Call pop_eegplot for manual epoch rejection and wait for the end of the
    %selection of bad epochs
    pop_eegplot(EEG, 1, 0, 1);
    uiwait();
    %- Get the vector of rejected epochs indices
    epochRejectInd  = ALLEEG(end).reject.rejmanual;
    EEGenc          = ALLEEG(2);
    %- Save the changes
    save(fullfile(datasetOutDirPath,'EEGenc'),'EEGenc');
    disp(['Saving updated dataset and response vectors in ',datasetOutDirPath]);
end
