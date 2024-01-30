function stamps_patch(patch_dir)
%STAMPS Stanford Method for Persistent Scatterers
%   STAMPS(START_STEP,END_STEP,PATCH_PATH,PATCHES_FLAG,EST_GAMMA_FLAG) Default is to run all steps.
%   A subset of steps may be selected with START_STEP and/or END_STEP
%   STEP 1 = Initial load of data
%   STEP 2 = Estimate gamma 
%   STEP 3 = Select PS pixels
%   STEP 4 = Weed out adjacent pixels
%   STEP 5 = Correct wrapped phase for spatially-uncorrelated look angle error and merge patches
%   STEP 6 = Unwrap phase
%   STEP 7 = Calculate spatially correlated look angle (DEM) error 
%   STEP 8 = Filter spatially correlated noise 
%   STEP 0 = Continue from the last known stage till the end-stage selected
%   
%   PATCHES_FLAG Default 'y'. Set to 'n' to process all data as one patch
%
%   EST_GAMMA_PARM is an optional parameter passed to PS_EST_GAMMA_QUICK
%
%   PATCH_LIST_FILE is an optional argument specifying the file list of
%   patches to be processed. Note that from step 5 and above one should use
%   all patches to merge results.
%
%   If current directory is a single patch, stamps only operates in the
%   current directory, but if current directory contains many patches,
%   stamps operates on them all.
%
%   Andy Hooper, June 2006

%   =================================================================
%   07/2006 AH: END_STEP added
%   09/2006 AH: ps_load removed (obsolete)
%   09/2006 AH: small baselines added 
%   11/2006 AH: patches added
%   01/2007 AH: calculate spatially correlated look angle error added
%   03/2009 AH: simultaneously estimate velocity when SCLA estimated
%   03/2009 AH: smooth SCLA for unwrapping iteration
%   03/2010 AH: move ps_cal_ifg_std to after merge step
%   12/2012 AH: add gamma option
%   12/2012 DB: add patch_list_file argument as option
%   09/2013 DB: update the stamps version number 
%   09/2015 DB: Check if patches do have PS before proceeding with
%               processing.
%   09/2015 DB: Fix when running stamps in a patch folder mode when no PS are left
%   09/2015 AH: allow for non-differentiation of caps by dir
%   01/2016 DB: include stamps_save in step 1-4.
%   08/2016 AH: Fix bug of scn_kriging_flag not being set
%   06/2017 DB: Catching when no PS are left from step 1, allow for re-run
%               when parameters have changed.
%   06/2017 DB: Option to continue from last know processing step
%   08/2107 AH: Removed catch as proceeds also when error in Step 1
%   =================================================================

nfill=40;
fillstr=[repmat('#',1,nfill),'\n'];
skipstr='\n';
msgstr=fillstr;

fprintf(skipstr);
logit(fillstr);
msgstr(round(nfill)/2-12:round(nfill/2)+13)=' StaMPS/MTI Version 4.0b6 ';
logit(msgstr);
msgstr(round(nfill)/2-12:round(nfill/2)+13)='  Beta version, Jun 2018  ';
logit(msgstr);
logit(fillstr);
fprintf(skipstr);

quick_est_gamma_flag=getparm('quick_est_gamma_flag');

patches_flag='y';
est_gamma_parm=0;
stamps_PART1_flag='y';
reest_gamma_flag='n';

nfill=40;
fillstr=[repmat('#',1,nfill),'\n'];
msgstr=fillstr;

cd(patch_dir);

if isempty(patch_dir)
    logit(sprintf('THIS PATCH is EMPTY [%s] Exitting',patch_dir))
    exit;
end

logit(sprintf('Processing PATCH [%s]',patch_dir))

% Creating Initial no_ps_info.mat
stamps_step_no_ps = zeros([5 1 ]);       % keep for the first 5 steps only
save('no_ps_info.mat','stamps_step_no_ps')

% STEP 1
msgstr(round(nfill)/2-3:round(nfill/2)+4)=' Step 1 ';
fprintf(skipstr);
logit(fillstr);
logit(msgstr);
logit(fillstr);
fprintf(skipstr);
ps_load_initial_gamma;
load('no_ps_info.mat');
% reset as we are currently re-processing
stamps_step_no_ps(1:end)=0;
save('no_ps_info.mat','stamps_step_no_ps')


% STEP 2
msgstr(round(nfill)/2-3:round(nfill/2)+4)=' Step 2 ';
fprintf(skipstr);
logit(fillstr);
logit(msgstr);
logit(fillstr)
fprintf(skipstr);

% check if step 1 had more than 0 PS points
load('no_ps_info.mat');
% reset as we are currently re-processing
stamps_step_no_ps(2:end)=0;

if stamps_step_no_ps(1)==0
    if strcmpi(quick_est_gamma_flag,'y')
        ps_est_gamma_quick(est_gamma_parm);
    else
        ps_est_gamma(est_gamma_parm);
    end
else
    stamps_step_no_ps(2)=1;
    fprintf('No PS left in step 1, so will skip step 2 \n')
    exit;
end  
save('no_ps_info.mat','stamps_step_no_ps')


% STEP 3
msgstr(round(nfill)/2-3:round(nfill/2)+4)=' Step 3 ';
fprintf(skipstr);
logit(fillstr);
logit(msgstr);
logit(fillstr)
fprintf(skipstr);

% check if step 2 had more than 0 PS points
load('no_ps_info.mat');
% reset as we are currently re-processing
stamps_step_no_ps(3:end)=0;

% run step 3 when there are PS left in step 2
if stamps_step_no_ps(2)==0
    if strcmpi(quick_est_gamma_flag,'y') & strcmpi(reest_gamma_flag,'y')
        ps_select;
    else
        ps_select(1);
    end
else
    fprintf('No PS left in step 2, so will skip step 3 \n')
    stamps_step_no_ps(3)=1;
    exit
end              
save('no_ps_info.mat','stamps_step_no_ps')

% STEP 4
msgstr(round(nfill)/2-3:round(nfill/2)+4)=' Step 4 ';
fprintf(skipstr);
logit(fillstr);
logit(msgstr);
logit(fillstr)
fprintf(skipstr);


% check if step 3 had more than 0 PS points
load('no_ps_info.mat');
% reset as we are currently re-processing
stamps_step_no_ps(4:end) =0;       % keep for the first 5 steps only


% run step 4 when there are PS left in step 3
if stamps_step_no_ps(3)==0
    ps_weed;
else
    fprintf('No PS left in step 3, so will skip step 4 \n')
    stamps_step_no_ps(4)=1;
    exit
end
save('no_ps_info.mat','stamps_step_no_ps')
