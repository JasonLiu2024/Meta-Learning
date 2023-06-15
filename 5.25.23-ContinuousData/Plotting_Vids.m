%% Analyze
close all

%This line takes the name of your file and gives it a standard variable
%name "DataMatrix" the name of you file, that is loaded in the workspace,
%goes where "LEDTry6" is.
DataMatrix = RegressionTemp1g1;


% This is for data visualization, to give you a better idea of how your WGM
% spectra evolved overtime.A
% i - goes from 1 through the length of your data set(i.e. if you have 12
% spectra you would write "i = 1:12")

% There is also a "pause" command in here that is commented out, you can
% un-comment the line by removing the "%" sign. This sets a pause (in ms)
% between frames.

for i = 1:10000
    plot(DataMatrix(i,:));
    axis([0 10000 0.2 1.65])
    %pause(1)
    title(i)
    vid(i) = getframe();
end