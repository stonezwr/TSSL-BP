function TD_stabilized = stabilize(TD)
% TD_stabilized = stabilize(TD)
% Compensates for the image motion induced by egorotation of the ATIS
% sensor during recording of the N-MNIST and N-Caltech101 datasets
% 
% written by Garrick Orchard - August 2015
% garrickorchard@gmail.com
% relies on some common AER Matlab %functions which can be found at http://www.garrickorchard.com/code/matlab-AER-functions

saccade_1_index = TD.ts<=105e3;
TD.x(saccade_1_index) = TD.x(saccade_1_index) - 3.5*TD.ts(saccade_1_index)/105e3;
TD.y(saccade_1_index) = TD.y(saccade_1_index) - 7*TD.ts(saccade_1_index)/105e3;

saccade_2_index = (TD.ts>105e3) & (TD.ts<=210e3);
TD.x(saccade_2_index) = TD.x(saccade_2_index) - 3.5 - 3.5*(TD.ts(saccade_2_index)-105e3)/105e3;
TD.y(saccade_2_index) = TD.y(saccade_2_index) - 7 + 7*(TD.ts(saccade_2_index)-105e3)/105e3;

saccade_3_index = (TD.ts>210e3); 
TD.x(saccade_3_index) = TD.x(saccade_3_index) - 7 + 7*(TD.ts(saccade_3_index)-210e3)/105e3;

% TD.y remains unchaged because it is a horizontal saccade
TD.x = round(TD.x);
TD.y = round(TD.y);
nulls = (TD.x<1) | (TD.y<1);
TD_stabilized = RemoveNulls(TD, nulls);
