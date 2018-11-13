function [EEGenc, EEGrec, hits, correctRejects, falseAlarms, omissions, reactionTimes] = ...
    sab_getdatasets(varargin) 
% [EEGenc, EEGrec, hits,  correctRejects, falseAlarms, omissions, 
%  reactionTimes] = SAB_GETDATASETS() 
%   Creates and returns encodage and recognition datasets, the hits and
%   correct_rejects indices and the reaction times for hits (rt).
%   
%   The user will first be asked for the directories containing the eeg files 
%   (in .edf format) and the eprime files (exported in text format - see
%   wiki). Requires EEGLAB.
%
%   This script works for both Macro and micro recordings. The script asks
%   the user the type of data.
%
%   Output values:
%       - EEGenc    : EEGLAB dataset containing all the epochs of the
%       encoding phase
%       - EEGrec    : EEGLAB dataset containing all the epochs of the
%       recognition phase (both correct and incorrect responses)
%       - hits      : binary vector equal to 1 if the recognition epoch 
%       is a hit, 0 otherwise.
%       - correctRejects : binary vector equal to 1 if the recognition 
%       epoch is a correct reject, 0 otherwise.
%       - falseAlarms    : binary vector equal to 1 if the recognition 
%       epoch is a false alarm, 0 otherwise.
%       - omissions      : binary vector equal to 1 if the recognition 
%       epoch is a false alarm reject, 0 otherwise.
%       - reactionTimes : double vector containing the reaction time 
%       if the epoch is a hit, 0 otherwise
% 
%   Optional inputs:
%       SAB_GETDATASETS ('rec_only') : will only create the dataset for 
%       REC files.
%       SAB_GETDATASETS ('nosave')   : will not save the datasets and
%       vectors. 
%
%   The temporal limits of the epochs must be set inside the function
%   (EPOCH_TIME_BEFORE_STIM and EPOCH_TIME_AFTER_STIM variables)
% 
%   See also sab_getpatharrays, sab_rejectepochs
%
%   Author(s): Martin Deudon (2016)

EEGenc          = [];
EEGrec          = [];
hits            = [];
correctRejects  = [];
omissions       = [];
falseAlarms     = [];
reactionTimes   = [];
imageNames      = [];

rec_only        = 0;
save_mat        = 1;
if nargin>0 
    if sum(strcmp(varargin,'rec_only'))==1; rec_only = 1; end;
    if sum(strcmp(varargin,'nosave'))==1;   save_mat = 0; end;
end


%% Function variables :
EPOCH_TIME_BEFORE_STIM      = 0.2; % time to take before stim to create epoch
EPOCH_TIME_AFTER_STIM       = 0.8; % time to take after stim to create epoch

TRIGGER_THRESHOLD_MACRO     = 100;  % a trigger means the "trigger signal" is overs this threshold
TRIGGER_THRESHOLD_MICRO     = 1000; % a trigger means the "trigger signal" is overs this threshold
% EPRIME_FIELD_NAME_ROW       = 1;
% EPRIME_DATA_START_ROW       = 2;

% For REC
EPRIME_ONSET_TIME_STR       = 'reponseReco1.OnsetTime';     % Time of stimulus (image appears)
EPRIME_RT_TIME_STR          = 'reponseReco1.RTTime';        % Time of response (good or bad)
EPRIME_RT_STR               = 'reponseReco1.RT';            % Reaction time 
EPRIME_CRESP_STR            = 'reponseReco1.CRESP';         % Correct response (1 or 0)
EPRIME_RESP_STR             = 'reponseReco1.RESP';          % Subject response (1 or 0)
EPRIME_IMAGE_STR            = 'image';                      % Image filename
% For ENC
EPRIME_ONSET_TIME_STR_ENC   = 'StimEnco1.OnsetTime';        % Time of stimulus (image appears)
EPRIME_IMAGE_STR_ENC        = 'image';                      % Image filename

NB_STIM_EVENT_PER_ENC_FILE  = 30;
NB_STIM_EVENT_PER_REC_FILE  = 60;


%% files managment
% First ask the user for the .edf file directory (both ENC and REC) and for
% the eprime text file (see wiki for conversion?) directory 

eeg_dir_path    = uigetdir ('.','Select EEG (.edf) folder');
if isnumeric(eeg_dir_path); return; end;
eprime_dir_path = uigetdir ('.','Select Eprime text files folder');
if isnumeric(eprime_dir_path); return; end;

%- Ask user if its micro or macro data
macroData = questdlg('Is it MACRO data ?');
macroData = strcmp(macroData,'Yes');
if macroData
    TRIGGER_THRESHOLD       = TRIGGER_THRESHOLD_MACRO;
else
    TRIGGER_THRESHOLD       = TRIGGER_THRESHOLD_MICRO;
end

%- try to get patient number
eeg_patientnb    = cell2mat(regexp(eeg_dir_path,'\\p\d+','match'));
eprime_patientnb = cell2mat(regexp(eprime_dir_path,'\\p\d+','match'));
if ~isempty(eeg_patientnb) && ~isempty(eprime_patientnb) && ...
        str2double(regexp(eeg_patientnb,'\d+','match')) ~= str2double(regexp(eprime_patientnb,'\d+','match'))
    msgbox ('Warning: EEG directory path and Eprime file seem to point to 2 different subjects');
    warning ('EEG directory path and Eprime file seem to point to 2 different subjects');
    warning (eeg_dir_path);
    warning (eprime_dir_path);
end

edf_files_struct    = rdir([eeg_dir_path,filesep '**',filesep '*.edf']);
[edf_enc_path_array, edf_rec_path_array] = sab_getpatharrays (edf_files_struct);
eprime_files_struct = rdir([eprime_dir_path,filesep '**', filesep '*.txt']);
[eprime_enc_path_array, eprime_rec_path_array] = sab_getpatharrays (eprime_files_struct);

% Display what files has been identified
nb_edf_enc_files    = length (edf_enc_path_array);
nb_edf_rec_files    = length (edf_rec_path_array);
nb_eprime_enc_files = length (eprime_enc_path_array);
nb_eprime_rec_files = length (eprime_rec_path_array);

if nb_edf_enc_files~=nb_eprime_enc_files && rec_only==0
	warning ('The number of .edf ENC files is different from the number of .txt eprime ENC files. Will only consider REC files');
    rec_only = 1;
end 
if nb_edf_rec_files~=nb_eprime_rec_files
	error ('The number of .edf REC files is different from the number of .txt eprime REC files');
end

disp([num2str(nb_edf_enc_files),' ENC edf files found.']);
disp([num2str(nb_edf_rec_files),' REC edf files found.']);
disp([num2str(nb_eprime_enc_files),' ENC eprime files found.']);
disp([num2str(nb_eprime_rec_files),' REC eprime files found.']);

if rec_only == 0
    disp('ENC files correspondence:');
    for i=1:length(edf_enc_path_array)
        path_edf_i      = cell2mat(edf_enc_path_array(i));
        temp_edf_i      = regexp (path_edf_i,filesep);
        name_edf_i      = path_edf_i (temp_edf_i(end)+1:end);
        path_eprime_i   = cell2mat(eprime_enc_path_array(i));
        temp_eprime_i   = regexp (path_eprime_i,filesep);
        name_eprime_i   = path_eprime_i (temp_eprime_i(end)+1:end);
        disp([name_edf_i,' <------> ',name_eprime_i]);
    end
end
disp('REC files correspondence:');
for i=1:length(edf_rec_path_array)
    path_edf_i      = cell2mat(edf_rec_path_array(i));
    temp_edf_i      = regexp (path_edf_i,filesep);
    name_edf_i      = path_edf_i (temp_edf_i(end)+1:end);
    path_eprime_i   = cell2mat(eprime_rec_path_array(i));
    temp_eprime_i   = regexp (path_eprime_i,filesep);
    name_eprime_i   = path_eprime_i (temp_eprime_i(end)+1:end);
    disp([name_edf_i,' <------> ',name_eprime_i]);
end

% Try to guess trigger size (trigger doubled or not) by reading first REC file
EEG_temp        = pop_biosig (cell2mat(edf_rec_path_array(i)));
trigger_ind     = abs(EEG_temp.data(EEG_temp.nbchan,:)) > TRIGGER_THRESHOLD;
trigger_ind     = [trigger_ind,0]==1 & [0,trigger_ind]==0;
t_triggers      = EEG_temp.times (trigger_ind);
trig_time_diff  = sort(diff(t_triggers));
if trig_time_diff (round(length(trig_time_diff)/3))<10
    trigger_size_guess = 2;
else
    trigger_size_guess = 1;
end


% Ask the user if the triggers are doubled or not
TRIGGER_SIZE    = inputdlg ('Size of the trigger (2 if doubled): ', 'Are the triggers doubled ?', 1, {num2str(trigger_size_guess)});
if isempty(TRIGGER_SIZE); 
    TRIGGER_SIZE = trigger_size_guess;
else
    TRIGGER_SIZE    = str2double(cell2mat(TRIGGER_SIZE));
end

enc_channels_removed    = [];

%% ENCodage Files
% Create EEGLAB datasets variables
EEGenc = []; EEGrec = [];
ALLEEG = [];
bad_channel_ind = [];


if rec_only==0 && nb_edf_enc_files ~= 0
    disp ('Encodage Files :');
    disp (['|',blanks(nb_edf_enc_files),'|']);
    for i=1:nb_edf_enc_files
        EEG = pop_biosig (cell2mat(edf_enc_path_array(i)));
        EEG.setname = ['SAB_ENC_',num2str(i)];
    
        trigger_ind     = abs(EEG.data(EEG.nbchan,:)) > TRIGGER_THRESHOLD;
        trigger_ind     = [trigger_ind,0]==1 & [0,trigger_ind]==0;
        t_triggers_i    = EEG.times (trigger_ind);
        trig_time_diff  = sort(diff(t_triggers_i));        % Get the times of the triggers (must be last channel of the EEG)
        % Try to detect if triggers are doubled or not by looking at the
        % time difference between 2 triggers
        if trig_time_diff (round(length(trig_time_diff)/3))<10 % Inferior to 10ms
            if TRIGGER_SIZE==1; 
                warning (['Triggers seem to be doubled in the file : ',cell2mat(edf_enc_path_array(i))]);
                figure; plot(EEG.times/1000,EEG.data(EEG.nbchan,:));
            end
        else
            if TRIGGER_SIZE==2; 
                warning (['Triggers seem NOT to be doubled in the file : ',cell2mat(edf_enc_path_array(i))]);
                figure; plot(EEG.times/1000,EEG.data(EEG.nbchan,:));
            end
        end
        
        disp (['|',repmat('-',1,i),blanks(nb_edf_enc_files-i),'|']);
        
        % Read associated eprime text file
        fid_eprime_enc_i    = fopen(cell2mat(eprime_enc_path_array(i)));
        % Get columns names (first line) % Stim time and image name
        field_names         = textscan(fid_eprime_enc_i,'%s%s',1,'delimiter','\t');
        field_names         = [field_names{:}];
        % Get the rest of the file
        eprime_enc_data_i   = textscan(fid_eprime_enc_i,'%s%s','delimiter','\t');
        eprime_enc_data_i   = [eprime_enc_data_i{:}];
        fclose(fid_eprime_enc_i);
        % Get order of columns
        onset_time_col_enc  = strcmpi(field_names,EPRIME_ONSET_TIME_STR_ENC)*(1:length(field_names)).';
        imagename_col_enc   = strcmpi(field_names,EPRIME_IMAGE_STR_ENC)     *(1:length(field_names)).';
        
        if onset_time_col_enc == 0
            error ('Did you forget the Onset Time field in the eprime file (ENC) ?');
        elseif imagename_col_enc == 0
            warning ('Did you forget the Image Name field in the eprime file (ENC)?');
        end        
        
        % Delete the lines whith no information (first 3 starting triggers and some
        % at the end (?))        
        empty_lines_ind     = strcmp(eprime_enc_data_i(:,onset_time_col_enc),'');
        eprime_enc_data_i   = eprime_enc_data_i(~empty_lines_ind,:);
        
        delay = -1; 
        j = 1;
        while delay <= 0
            % Calcul the delay in ms on the first image (which time appear on both the
            % eprime file and in the eeg file (trigger)
            delay = str2double(eprime_enc_data_i(j,onset_time_col_enc)) - t_triggers_i(3*TRIGGER_SIZE+1);
            j = j+1;
        end
        delay = round(delay);
        
        % Get the times of stimulus
        stim_events_i   = str2double(eprime_enc_data_i (:,onset_time_col_enc));
        stim_events_i   = stim_events_i (stim_events_i > 0);
        stim_events_i   = stim_events_i - delay;
        % Supposed to have 30 stim_events for each enc file - check that
        if length (stim_events_i) ~= NB_STIM_EVENT_PER_ENC_FILE
            warning (['An abnormal number of stimulus was detected for encodage: ', num2str(length(stim_events_i)), ...
                ' stimulus detected - ', num2str(NB_STIM_EVENT_PER_ENC_FILE) ,' were expected. In file : ',  ...
                cell2mat(eprime_enc_path_array(i)) ]);
        end

        if ~isempty(EEG.event)
            nb_events_i = length(EEG.event);
            event_i     = EEG.event(1);
        else
            nb_events_i = 0;
            event_i     = struct('type','','latency',0,'urevent',0);
            EEG.event   = struct('type', {}, 'latency',{}, 'urevent',{});
        end
        % Add stim events (keep original event (there is one event always
        % present.. ?! ..) )
        for j=1:length(stim_events_i)
            event_i = setfield (event_i,'type','stim');
            % Convert time to index for the latency field
            event_i = setfield (event_i,'latency',floor(1+EEG.srate*(stim_events_i(j)*1E-3)));
            event_i = setfield (event_i,'urevent',nb_events_i+j);
            EEG.event(nb_events_i+j) = event_i;
        end
        EEG = eeg_checkset(EEG,'eventconsistency');
        
        [ALLEEG, EEG] = eeg_store (ALLEEG, EEG, i); % Store dataset
    end

    % Select only common channels
    commonChannelNames = getcommonchannelnames(ALLEEG);
    for i=1:length(ALLEEG)
        removedChannels = setdiff(commonChannelNames,{ALLEEG(i).chanlocs.labels});
        if ~isempty(removedChannels)
            warning(['Removing channel : ',[removedChannels{:}]]);
        end
        ALLEEG(i)       = pop_select(ALLEEG(i),'channel',commonChannelNames);
    end
    
    % Merge all datasets and store it
    if length(ALLEEG) > 1
        EEGenc          = pop_mergeset (ALLEEG, 1:nb_edf_enc_files, 0);
        EEGenc.setname  = 'Merged_ENC_Datasets';
    end

    disp ('Detecting bad channels...');
    bad_channel_ind = searchbadchannels (EEGenc);
    if ~isempty(bad_channel_ind)
        message = [num2str(length(bad_channel_ind)),' bad channel(s) have been detected in ENC files : '];
        for i=1:length(bad_channel_ind)
             message = [message,' ',EEGenc.chanlocs(bad_channel_ind(i)).labels];
        end
        %- plot the bad channels
        figure(); set(gcf,'name','Bad Channels Detected');
        nbadchannels = length(bad_channel_ind);
        for j=1:nbadchannels
           subplot(nbadchannels,1,j);
           plot (EEGenc.times,EEGenc.data(bad_channel_ind(j),:));
        end
        %- Ask if user want to keep them
        message = [message,'. Do you want to remove these channel(s) ?'];
        
        choice = questdlg(message,'Bad Channel Detection,','yes','no','yes');
        if strcmp (choice,'yes')
            channels_to_keep = 1:EEGenc.nbchan;
            channels_to_keep (bad_channel_ind) = [];
            EEGenc = pop_select (EEGenc, 'channel', channels_to_keep);
            enc_channels_removed = bad_channel_ind;
        end
    end
    
    % Extract epochs based on trigger events and store the dataset
    EEGenc        	= pop_epoch (EEGenc, {'stim'}, [-EPOCH_TIME_BEFORE_STIM, EPOCH_TIME_AFTER_STIM]);
    % Remove baseline means
    EEGenc = pop_rmbase (EEGenc, [-100, 0]);
    EEGenc.setname 	= 'Merged_ENC_Datasets_Epochs';
        
end


%% RECognition files

hits = []; correctRejects = []; reactionTimes = [];
ALLEEG = [];

if nb_edf_rec_files == 0
    disp ('Could not detect any .edf REC files - Cannot Plot ERP Images');
elseif nb_eprime_rec_files == 0
    disp ('Could not detect any eprime REC files - Cannot Plot ERP Images');
else
    disp ('Recognition Files :');
    disp (['|',blanks(nb_edf_rec_files),'|']);
    for i=1:nb_edf_rec_files
        EEG = pop_biosig (cell2mat(edf_rec_path_array(i)));
        EEG.setname = ['SAB_REC_',num2str(i)];
        
        trigger_ind     = abs(EEG.data(EEG.nbchan,:)) > TRIGGER_THRESHOLD;
        trigger_ind     = [trigger_ind,0]==1 & [0,trigger_ind]==0;
        t_triggers_i    = EEG.times (trigger_ind);
        % Try to detect if triggers are doubled or not by looking at the
        % time difference between 2 triggers
        trig_time_diff  = sort(diff(t_triggers_i));
        if trig_time_diff (round(length(trig_time_diff)/3))<10 % Inferior to 10ms
            if TRIGGER_SIZE==1; 
                warning (['Triggers seem to be doubled in the file : ',cell2mat(edf_rec_path_array(i))]); 
                figure; plot(EEG.times/1000,EEG.data(EEG.nbchan,:));
            end
        else
            if TRIGGER_SIZE==2; 
                warning (['Triggers seem NOT to be doubled in the file : ',cell2mat(edf_rec_path_array(i))]);
                figure; plot(EEG.times/1000,EEG.data(EEG.nbchan,:));
            end
        end
        
        disp (['|',repmat('-',1,i),blanks(nb_edf_rec_files-i),'|']);
        
        % Read associated eprime text file - Should not be any character in the file; only numbers
        fid_eprime_rec_i    = fopen(cell2mat(eprime_rec_path_array(i)));
        % Get columns names (first line)
        field_names         = textscan(fid_eprime_rec_i,'%s%s%s%s%s%s',1,'delimiter','\t');
        field_names         = [field_names{:}];
        % Get the rest of the file
        eprime_rec_data_i   = textscan(fid_eprime_rec_i,'%s%s%s%s%s%s','delimiter','\t');
        eprime_rec_data_i   = [eprime_rec_data_i{:}];
        fclose(fid_eprime_rec_i);

        % Get order of columns
        onset_time_col  = strcmpi(field_names,EPRIME_ONSET_TIME_STR)*(1:length(field_names)).';
        rt_time_col 	= strcmpi(field_names,EPRIME_RT_TIME_STR)  *(1:length(field_names)).';
        rt_col          = strcmpi(field_names,EPRIME_RT_STR)       *(1:length(field_names)).';
        cresp_col       = strcmpi(field_names,EPRIME_CRESP_STR)    *(1:length(field_names)).';
        resp_col        = strcmpi(field_names,EPRIME_RESP_STR)     *(1:length(field_names)).';
        imagename_col   = strcmpi(field_names,EPRIME_IMAGE_STR)    *(1:length(field_names)).';

        if onset_time_col == 0
            error ('Did you forget the Onset Time field in the eprime file ?');
        elseif rt_time_col == 0
            error ('Did you forget the RTTime field in the eprime file ?');
        elseif rt_col  == 0
            error ('Did you forget the RT field in the eprime file ?');
        elseif cresp_col  == 0
            error ('Did you forget the CRESP field in the eprime file ?');
        elseif resp_col == 0
            error ('Did you forget the RESP field in the eprime file ?');
        elseif imagename_col == 0
            warning ('Did you forget the Image Name field in the eprime file ?');
        end        
        
        % Delete the lines whith no information (first 3 starting triggers and some
        % at the end (?))        
        empty_lines_ind     = strcmp(eprime_rec_data_i(:,onset_time_col),'');
        eprime_rec_data_i   = eprime_rec_data_i(~empty_lines_ind,:);
        
        hits_i              = (str2double(eprime_rec_data_i(:,cresp_col)) == 1) & (str2double(eprime_rec_data_i(:,resp_col)) == 1);         % logical AND
        correct_rejects_i   = isnan(str2double(eprime_rec_data_i(:,cresp_col))) & isnan(str2double(eprime_rec_data_i(:,resp_col)));     % logical AND
        false_alarms_i      = isnan(str2double(eprime_rec_data_i(:,cresp_col)))  & (str2double(eprime_rec_data_i(:,resp_col)) == 1);         % logical AND
        omissions_i         = (str2double(eprime_rec_data_i(:,cresp_col)) == 1) & isnan(str2double(eprime_rec_data_i(:,resp_col)));         % logical AND
        if ~isequal(1,unique(sum([hits_i,correct_rejects_i,false_alarms_i,omissions_i],2)))
            error(['Something went wrong - Error in eprime resp and cresp fields detected in file ',cell2mat(eprime_rec_path_array(i))]);
        end

        delay = -1; 
        j = 1;
        while delay <= 0
            % Calcul the delay in ms on the first image (which time appear on both the
            % eprime file and in the eeg file (trigger)
            delay = str2double(eprime_rec_data_i(j,onset_time_col)) - t_triggers_i(3*TRIGGER_SIZE+1);
            j = j+1;
        end
        delay = round(delay);
        % Synchronize .edf and eprime files and get and stim_events rt_events (finger raised)
        % Not used but useful for visualization (rt_events)
        rt_events_i     = str2double(eprime_rec_data_i (:,rt_time_col));
        rt_events_i     = rt_events_i (rt_events_i > 0);  % Delete row whitout rt event
        rt_events_i     = rt_events_i - delay;            % substract delay
        stim_events_i   = str2double(eprime_rec_data_i (:,onset_time_col));
        stim_events_i   = stim_events_i (stim_events_i > 0);
        stim_events_i   = stim_events_i - delay;
        rt_i            = str2double(eprime_rec_data_i (:,rt_col));
        if imagename_col~=0
            imagenames_i    = eprime_rec_data_i(:,imagename_col);
        end
        
        % Supposed to have 60 stim_events for each rec file - check that
        if length (stim_events_i) ~= NB_STIM_EVENT_PER_REC_FILE
            warning (['An anormal number of stimulus was detected for recognition: ', num2str(length(stim_events_i)), ...
                ' stimulus detected - ', num2str(NB_STIM_EVENT_PER_REC_FILE) ,' were expected. In file : ',  ...
                cell2mat(eprime_rec_path_array(i)) ]);
        end

        
        if ~isempty(EEG.event)
            nb_events_i = length(EEG.event);
            event_i     = EEG.event(1);
        else
            nb_events_i = 0;
            event_i     = struct('type','','latency',0,'urevent',0);
            EEG.event   = struct('type', {}, 'latency',{}, 'urevent',{});
        end
        % Add stim events (keep original event (there is one event always
        % present.. ?! ..) )
        for j=1:length(stim_events_i)
            event_i = setfield (event_i,'type','stim');
            % Convert time to index for the latency field
            event_i = setfield (event_i,'latency',floor(1+EEG.srate*(stim_events_i(j)*1E-3)));
            event_i = setfield (event_i,'urevent',nb_events_i+j);
            EEG.event(nb_events_i+j) = event_i;
        end
        nb_events_i     = length(EEG.event);
        for j=1:length(rt_events_i)
            event_i = setfield (event_i,'type','rt');
            event_i = setfield (event_i,'latency',floor(1+rt_events_i(j)*1E-3*EEG.srate));
            event_i = setfield (event_i,'urevent',nb_events_i+j);
            EEG.event(nb_events_i+j) = event_i;
        end        
        EEG = eeg_checkset(EEG);
        
        % Concatenate global hits and correct reject vectors
        hits_i              = hits_i(:); 
        correct_rejects_i   = correct_rejects_i (:);
        rt_i                = rt_i (:);
        hits                = [hits;hits_i];
        correctRejects      = [correctRejects;correct_rejects_i];
        falseAlarms         = [falseAlarms;false_alarms_i];
        omissions           = [omissions;omissions_i];
        reactionTimes    	= [reactionTimes;rt_i];
        if imagename_col~=0
            imageNames          = [imageNames;imagenames_i];
        end
        
        % Store dataset
        [ALLEEG, EEG] = eeg_store (ALLEEG, EEG, i);
        
    end

    % Select only common channels
    commonChannelNames = getcommonchannelnames(ALLEEG);
    for i=2:length(ALLEEG)
        removedChannels = setdiff(commonChannelNames,{ALLEEG(i).chanlocs.labels});
        if ~isempty(removedChannels)
            warning(['Removing channel : ',[removedChannels{:}]]);
        end
        ALLEEG(i)       = pop_select(ALLEEG(i),'channel',commonChannelNames);
    end
    
    % Merge all datasets and store it (start from 2 because of the
    % merged_enc_datasets_epochs in position 1)
    if length(ALLEEG) > 1
        EEGrec          = pop_mergeset (ALLEEG, 1:nb_edf_rec_files, 0);
        EEGrec.setname  = 'Merged_REC_Datasets';
    end

    % Remove the bad channel detected and removed from ENC files
    if rec_only
        disp ('Detecting bad channels...');
        bad_channel_ind = searchbadchannels (EEGrec);
        if ~isempty(bad_channel_ind)
            message = [num2str(length(bad_channel_ind)),' bad channel(s) have been detected in ENC files : '];
            for i=1:length(bad_channel_ind)
                 message = [message,' ',EEGrec.chanlocs(bad_channel_ind(i)).labels];
            end
            message = [message,'. Do you want to remove these channel(s) ?'];
            %- plot the bad channels
            figure(); set(gcf,'name','Bad Channels Detected');
            nbadchannels = length(bad_channel_ind);
            for j=1:nbadchannels
                subplot(nbadchannels,1,j); hold on;
                plot (EEGrec.times/1000,EEGrec.data(bad_channel_ind(j),:)); hold on;
                stimTimes = EEGrec.times([EEGrec.event(strcmp({EEGrec.event.type},'stim')).latency]);
                plot([stimTimes;stimTimes]/1000,repmat(ylim',1,length(stimTimes)),'r');
                title (EEGrec.chanlocs(bad_channel_ind(j)).labels);
                xlabel ('time (s)'); ylabel('Amplitude (uV)'); axis tight;
            end
            %- Ask if user want to keep them
            choice = questdlg(message,'Bad Channel Detection,','yes','no','yes');
            if strcmp (choice,'yes')
                channels_to_keep = 1:EEGrec.nbchan;
                channels_to_keep (bad_channel_ind) = [];
                EEGrec = pop_select (EEGrec, 'channel', channels_to_keep);
                enc_channels_removed = bad_channel_ind;
            end
        end
    else
        if ~isempty(bad_channel_ind)
            channels_to_keep = 1:EEGrec.nbchan;
            channels_to_keep (enc_channels_removed) = [];
            EEGrec = pop_select (EEGrec, 'channel', channels_to_keep);
        end
    end
    
    % Extract epochs based on trigger events and store the dataset
    EEGrec             = pop_epoch (EEGrec, {'stim'}, [-EPOCH_TIME_BEFORE_STIM, EPOCH_TIME_AFTER_STIM]);
    % Remove baseline means
    EEGrec = pop_rmbase (EEGrec, [-100, 0]);
    EEGrec.setname     = 'Merged_REC_Datasets_Epochs';

end

hits           = logical (hits);
correctRejects = logical (correctRejects);
falseAlarms    = logical (falseAlarms);
omissions      = logical (omissions);

if save_mat
    if EPOCH_TIME_AFTER_STIM<1
        epochMaxStr = [num2str(1000*EPOCH_TIME_AFTER_STIM),'ms'];
    else
        epochMaxStr = [num2str(EPOCH_TIME_AFTER_STIM),'s'];
    end
    save_dir = fullfile(eeg_dir_path,'..'); % Save datasets in the patient folder
    if macroData
        save_dir = createuniquedir(fullfile(save_dir,['datasets_',epochMaxStr]));
    else
        save_dir = createuniquedir(fullfile(save_dir,['datasets_',epochMaxStr','_micro']));
    end
    if isempty(save_dir); warning ('Empty directory, cannot save the datasets'); return; end;
    if ~isdir(save_dir);
        mkdir (save_dir);
    end
    if ~rec_only; save (fullfile(save_dir,'EEGenc.mat'),'EEGenc'); end;
    save (fullfile(save_dir,'EEGrec.mat'),'EEGrec');
    save (fullfile(save_dir,'hits.mat'),'hits');
    save (fullfile(save_dir,'correctRejects.mat'),'correctRejects');
    save (fullfile(save_dir,'falseAlarms.mat'),'falseAlarms');
    save (fullfile(save_dir,'omissions.mat'),'omissions');
    save (fullfile(save_dir,'reactionTimes.mat'),'reactionTimes');
    if imagename_col~=0
        save (fullfile(save_dir,'imageNames.mat'),'imageNames');
    end
    %- Create log file
%     logFile = fopen(fullfile(save_dir,'logFile.txt'),'a+');
end


