function result =  RemoveNulls(result, indices)
% TDout =  remove_nulls(TD, Null_indices)
% removes all events at the logical indices 'Null_indices'
% 
% TAKES IN:
% 'TD'
%   A struct of events where each field has the same number of elements.
%   Typical format is:
%       TD.x =  pixel X locations
%       TD.y =  pixel Y locations
%   	TD.p =  event polarity
%       TD.ts = event timestamps in microseconds
% 
% 'Null_indices'
%   A logical array of the same size as the fields of TD 
%   i.e. length(TD.x) = length(Null_indices)
%   Null_indices has value '0' wherever elements are to be preserved and
%   value '1' where elements are to be removed
% 
% 
% RETURNS:
% 'TDout'
%   A struct with the same field as 'TD', but with elements removed
%   at the locations where 'Null_indices' is '1'.
% 
% 
% EXAMPLE USE:
% % to remove all events at pixel location (3,5):
% Null_indices = (TD.x == 3) && (TD.y == 5);
% newTD =  RemoveNulls(TD, Null_indices);
% 
% % to remove the first half of the events:
% Null_indices = zeros(size(TD.x));
% Null_indices(1:round(length(Null_indices)/2)) = 1;
% TD_secondhalf =  RemoveNulls(TD, Null_indices);
% 
% 
% written by Garrick Orchard - June 2014
% garrickorchard@gmail.com


indices = logical(indices);
fieldnames = fields(result);
for i = 1:length(fieldnames)
    if ~strcmp(fieldnames{i}, 'meta')
        result.(fieldnames{i})(indices)  = [];
    end
end