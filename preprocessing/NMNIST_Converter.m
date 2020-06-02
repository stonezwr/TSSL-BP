%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A function to generate the spike sample file for the N-MNIST dataset
% This function read the binary file provided by the authors and generate
% a spike sample file for each class in the following form:
%
% 
% [#channel_id] [spike_time_0] [spike_time_1] ...
% ...
% #
%
% Each line represent a channel, starting with a channel id and followed by
% the spike times. 
% the '#' is a decimal for separating different samples in the same class
%
% *** Notice that the channel id is 1-based.
% *** Notice that the spike times starts from zero
%
% Each of the sample file has the name '[Train/Test]_[#cls_id].dat'
% where #cls_id is the label.
%
% Since the original dataset is too fine-grained pretty large (0.01 ms), we 
% further bin the events into buckets with certain size, and each bucket 
% represents a new time point to reduce the granularity. 
% There might be multiple spikes in the same bucket, which is a very
% rare case when the window size is not too big. 
% Therefore, we ignore those cases for simplicity.
% 
% 
% Input: train_or_test : reading the training/testing,can be 'Train' or 'Test'
%        use_two_channels: 1 means read both on and off events, -1 means
%                          read only on events.
%        time_window : the time window size to bin the long-scale spikes
%                      into the buckets.
%   
% Output: the output directory of dataset with name: directory_des
%       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function NMNIST_Converter(train_or_test, use_two_channels, time_window)
    if strcmp(train_or_test, 'Train') == 1
        total_sample = 60000;
        max_times = zeros(60000, 1);
    elseif strcmp(train_or_test, 'Test') == 1
        total_sample = 10000;
        max_times = zeros(10000, 1);
    else
        fprintf('The dataset argument can only be `Train` or `Test`! Unrecognitized arg: %s\n', train_or_test)
        exit(0);
    end
    x_dim = 34;
    y_dim = 34;
    lost_spikes = 0; % total number of spikes ignored by the compression
    sample_id = 1;
    h = waitbar(0,'Initializing waitbar...');
    for class=0:9
        directory_src = strcat(train_or_test,'/',num2str(class),'/');
        if use_two_channels == 1
            total_channels = x_dim * y_dim * 2;
        else
            total_channels = x_dim * y_dim;
        end
        % handle the output file:
        directory_des = sprintf('%d_%d_stable/', total_channels, time_window);
        Readfiles = dir(fullfile(directory_src,'*.bin'));
        file_num = length(Readfiles);
        if ~exist(directory_des, 'dir')
            mkdir(directory_des);
        end
        filename = strcat(directory_des, train_or_test,'_',num2str(class),'.dat');
        fid = fopen(filename,'w');
        if fid == -1
            fprintf('Cannot open the file: %s for writing!', filename);
        end
        
        % for samples in the same class
        for ii=1:file_num
            TD = Read_Ndataset(strcat(directory_src, Readfiles(ii).name));         
            TD = stabilize(TD);
            % remove the possible points that are out of bound
            nulls = (TD.x>34) | (TD.y>34);
            TD = RemoveNulls(TD, nulls);
            x = TD.x;
            y = TD.y;
            p = TD.p;
            ts = TD.ts;
            
            % the x and y is 1-based index, the channel index starts from 1
            channel = (x - 1)*y_dim + y;
            % if single channel is used, collapse the on and off events
            % otherwise differentiate the on and off events
            if use_two_channels == 1
                off_inds = find(p == 2);
                channel(off_inds) = channel(off_inds) + x_dim * y_dim;
            end
            % compress the spike train:
            ts_compressed = ceil(ts/time_window);
            
            % the binary spike matrix:
            mat = zeros(total_channels, max(ts_compressed + 1));
            mat(sub2ind(size(mat), channel, ts_compressed + 1)) = 1;
            lost_spikes = lost_spikes + length(ts_compressed) - sum(sum(mat));
            max_times(sample_id) = max(ts_compressed + 1);
            sample_id = sample_id + 1;
            
            % dump each channels to the file
            for i = 1:total_channels
                inds = find(channel == i);
                if ~isempty(inds)
                    fprintf(fid,'%d\t', [i, ts_compressed(inds)']);
                    fprintf(fid,'\n');
                end
            end
            % separator:
            fprintf(fid,'#\n');
            % the following code is for visualization
            % figure
            % plotSpikeRaster(mat == 1);
            perc = 100*(sample_id/total_sample);
            waitbar(perc/100, h, sprintf('Processing %.2f%% ...',perc));
        end
        fclose(fid);
    end
    close(h);
    fprintf('The average ignored spike per sample is %f\n', lost_spikes / sample_id);
    fprintf('The max spike train duration is %d\n', max(max_times));
    fprintf('The median spike train duration is %d\n', median(max_times));
end


