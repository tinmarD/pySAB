function [enc_path_array, rec_path_array] = sab_getpatharrays (file_struct)
%[enc_path_array, rec_path_array] = SAB_GETPATHARRAY (file_struct)
%   [ENC_PATH_ARRAY, REC_FILE_ARRAY] = SAB_GETPATHARRAYS (FILE_STRUCT) 
%   Looks into file_struct and discriminate encodage files (ENC) and 
%   recognition files (REC) and sort the array given the trial number.
%
%   Expect files like 'SAB REC 2.edf', SAB-ENC-2-550ms.edf' or 'SAB_ENC_3.edf'
%   i.e.    Delimiters : ' ' or '-' or '_'
%           Trial number in 3rd position
%
%   file_struct is the structure returned by the rdir function
%
%   See also : rdir, sab_getdatasets
%
%   Author(s): Martin Deudon (2016)
%


nb_files            = length(file_struct);
enc_path_array      = {};
rec_path_array      = {};
enc_trial_nb        = [];
rec_trial_nb        = [];

delimiters = {' ','_','-'};

for i=1:nb_files
    path_i  = file_struct(i).name;
    tmp_ind = strfind (path_i,filesep);
    name_i  = path_i(tmp_ind(end)+1:end);
    n_trial = [];
    
    % Find the trial number - must be in 3rd position (eg:
    % 'SAB_ENC_3.edf' or 'SAB ENC 4 550ms.edf')
    % For each delimiter :
    for d = delimiters
        % Do not use strsplit because of MAC OS compatibility
%         name_split  = strsplit (name_i,d);
        name_split = regexp (name_i,cell2mat(['[^',d,']*']),'match');
        if length(name_split) >= 3
            n_str_trial = cell2mat(name_split(3));
            n_trial     = str2double(n_str_trial);
            % n_str_trial might contain something like '3.edf'
            if isnan(n_trial) % case like 'SAB_ENC_3.edf'
                tmp_split   = regexp (n_str_trial,'[^.]*','match');
%                 tmp_split   = strsplit (n_str_trial,'.');
                n_str_trial = cell2mat(tmp_split(1));
                n_trial     = str2double(n_str_trial);
            end
        end
        % if found n_trial go on with next signal
        if ~isempty (n_trial); break; end;
    end

    
    % is the file ENC or REC ?
    % ENC FILES
    if ~isempty(strfind(name_i,'ENC')) || ~isempty(strfind(name_i,'enc'))
        % if n_trial was found add the path and n_trial to the arrays
        if ~isempty (n_trial) && ~isnan (n_trial)
            enc_path_array(end+1)   = cellstr(path_i);
            enc_trial_nb(end+1)     = n_trial;
        else
            warning (['The trial number of file ',path_i,' could not be detected. This file will be ignored']);
        end  
    % REC FILES         
    elseif ~isempty(strfind(name_i,'REC')) || ~isempty(strfind(name_i,'rec'))
        % if n_trial was found add the path and n_trial to the arrays
        if ~isempty (n_trial)
            rec_path_array(end+1)   = cellstr(path_i);
            rec_trial_nb(end+1)     = n_trial;
        else
            warning (['The trial number of file ',path_i,' could not be detected. This file will be ignored']);
        end

    else
        warning (['file ',path_i,' could not be recognized as ENC or REC type file. This file will be ignored.']);
    end
    
end

% Sort the path list given the trial number
%#ok<*NCOMMA,*ASGLU>
[tmp ind] = sort (enc_trial_nb);
enc_path_array = enc_path_array (ind);
[tmp ind] = sort (rec_trial_nb); 
rec_path_array = rec_path_array (ind);

end

