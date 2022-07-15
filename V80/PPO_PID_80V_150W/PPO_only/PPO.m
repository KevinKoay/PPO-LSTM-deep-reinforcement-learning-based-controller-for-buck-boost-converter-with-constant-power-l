clear;
clc;
%% Open the file
mdl = 'buck_boost_model';
open_system(mdl)
%% Set random seed to 0 for reproducibility
rng(0)
%% Turn off data logging simulink to save memory
Simulink.sdi.setArchiveRunLimit(0);
Simulink.sdi.setAutoArchiveMode(false);
Simulink.sdi.clear
sdi.Repository.clearRepositoryFile
%% Reward Generator
% blk = 'buck_boost_model/BuckBoostStepResponse';
% generateRewardFunction(blk)
%%
% GLOBAL PARAMETERS 
% Parameter values
num_episodes = 30000;
numValidationExperiments = 20;
%%
init_action = 0.25; 
%% 
Ts = 0.00001;
Tf = 0.007; 
V_ref =80; 
max_steps = ceil(Tf/Ts);

%% Setting Action and Observation Parameter
agentblk = [mdl '/RL Agent'];

numObs = 2; % [v0, e, de/dt]
observationInfo = rlNumericSpec([numObs,1],...
    'LowerLimit',[0 -100]',...
    'UpperLimit',[100 100]');
observationInfo.Name = 'observations';
observationInfo.Description = 'voltage output';

PWMduty = 0.25 : 0.002 : 0.65;
length(PWMduty)
actionInfo = rlFiniteSetSpec(num2cell(PWMduty));
actionInfo.Name = 'actions';

%% Initialize enviroment
env = rlSimulinkEnv(mdl,agentblk,observationInfo,actionInfo);
env.ResetFcn = @(in)localResetFcn(in);
%%

modeAgentOpts = rlPPOAgentOptions(...
    "SampleTime",Ts,...
    "DiscountFactor",0.995,... 
    "ExperienceHorizon",floor(Tf/Ts), ...
    "MiniBatchSize",350, ...  
    "EntropyLossWeight",0.01,...
    "ClipFactor",0.3); 
modeAgentOpts.ActorOptimizerOptions.LearnRate = 1e-4;
modeAgentOpts.ActorOptimizerOptions.GradientThreshold = 1;
modeAgentOpts.CriticOptimizerOptions.LearnRate = 4e-4;   
initOptions = rlAgentInitializationOptions("NumHiddenUnit",256,"UseRNN",true);
agent = rlPPOAgent(observationInfo,actionInfo,initOptions,modeAgentOpts);

trainOpts = rlTrainingOptions(...
    'MaxEpisodes',num_episodes ,...
    'MaxStepsPerEpisode',max_steps,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','EpisodeCount',...
    'StopTrainingValue',num_episodes,...
    'ScoreAveragingWindowLength',100, ...
'SaveAgentCriteria', "EpisodeReward",...
"SaveAgentValue", 1200 ...
   );
%%
doTraining = false;
if doTraining
    trainingStats = train(agent,env,trainOpts);
else
    rng(0)
   load('Agent6457.mat','saved_agent')
    agent = saved_agent
    sim(mdl);
end

%% Get Data
meanVout = mean(V_simout.Data)
stdV_out= std(V_simout.Data)
meanSquareError = (sum(V_error.Data)^2)/V_error.Length
meanAbsoluteError = (sum(V_error.Data))/V_error.Length
