%% PamSim
%%

%%
% actinic light input values
%%

act=[ 1000  ]';
%act=500
%%
% Load Parameters
%%
params=getparamsfromfilename('params.txt');
simnow=1;
qtypes=[ 1 ]; % quenching type
ll=1;
if simnow==1
    for k=1:length(qtypes)
        for kk=1:length(act)
            simtype='PSIITrapLake';
            tic
            %%
            % Load light  inputs for PAM experiment using
            % <setupPAMIntensities.html |setupPAMIntensities|>
            %%
            [LightIntensities durat flashidx]=setupPAMIntensities(act(kk));
            
            %%
            % Run Simulation using
            %<chloroplastSim.html |chloroplastSim|>
            %%
            samplepam{kk,k}=chloroplastSim(LightIntensities, durat, params, qtypes(k), simtype);
            samplepam{kk,k}.flashidx=flashidx;
            toc
            %%
            % Plot results of simulation
            %%
            npq{k,kk}=plotNPQ(samplepam{kk,k}, ll, act(kk), qtypes(k), 'k');
            %figure(ll)
            %title(act(kk))
            ll=ll+2;
        end
        
    end
end
