function []=ps_load_init()
%PS_LOAD_INIT setparm

logit('UPDATING param.m')

rscname = ['./rsc.txt'];
fid=fopen(rscname);
if fid<0
    error([rscname,' does not exist'])
end
rslcpar=textscan(fid,'%s');
rslcpar=rslcpar{1}{1};
fclose(fid);

heading=readparm(rslcpar,'heading:');
setparm('heading',heading,1);

freq=readparm(rslcpar,'radar_frequency:');
lambda=299792458/freq;
setparm('lambda',lambda,1);

sensor=readparm(rslcpar,'sensor:');
if ~isempty(strfind('sensor','ASAR'))
    platform='ENVISAT';
else
    platform=sensor; % S1A for Sentinel-1A
end
setparm('platform',platform,1);

logit('UPDATED param.m')