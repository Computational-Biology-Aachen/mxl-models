function npq = plotNPQ(sim, fnum, plotstyle)

npq = calcNPQfromsim(sim);

if nargin < 2 || isempty(fnum)
    fnum = figure;
elseif ~ishandle(fnum)
    fnum = figure(fnum);
end

if nargin < 3 || isempty(plotstyle)
    plotstyle = 'b';
end

[s, q] = getStaticVals(sim);

% Figure 1
fig1 = figure(fnum);
ax1 = axes(fig1);
plot(ax1, double(npq.pulsetime), double(npq.qEpulse), plotstyle, 'LineWidth', 1.5);
grid(ax1, 'off')
title(ax1, 'Chl fluorescence yield')
xlabel(ax1, 'seconds')
ylabel(ax1, '\Phi_F')
%ylim(ax1, [0 0.15])
ax1.FontSize = 22;

% Figure 2
fig2 = figure(2);
ax2 = axes(fig2);
plot(ax2, sim.timevalues, q.Zea, ...
         sim.timevalues, q.QuenchersXanthophyll, ...
         sim.timevalues, q.Anth, ...
         sim.timevalues, q.QuenchersLutein, ...
         sim.timevalues, q.QuenchersXanthophyll + q.QuenchersLutein);

legend(ax2, 'Zea', 'Zea Quenchers', 'Anth', 'Lutein Quenchers', 'Total Quenchers')
title(ax2, 'Quenching-Active Species')
grid(ax2, 'off')
ax2.FontSize = 22;

end