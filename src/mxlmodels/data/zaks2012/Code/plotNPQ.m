function npq = plotNPQ(sim, fnum, lightint, qtype, plotstyle)

npq = calcNPQfromsim(sim);

[s, q] = getStaticVals(sim);

nameform = "qE %d uE qtype %d";
name1 = sprintf(nameform, lightint, qtype);
% Figure 1
fig1 = figure(fnum);
ax1 = axes(fig1);
plot(ax1, double(npq.pulsetime), double(npq.qEpulse), plotstyle, 'LineWidth', 1.5);
grid(ax1, 'off')
title(ax1, name1)
xlabel(ax1, 'seconds')
ylabel(ax1, 'qE') %\Phi_F
%ylim(ax1, [0 0.15])
ax1.FontSize = 22;

saveas(fig1,name1, "png")

% Figure 2

nameform2 = "Quencher %d uE qtype %d";
name2 = sprintf(nameform2, lightint, qtype);

fig2 = figure(fnum+1);
ax2 = axes(fig2);
plot(ax2, sim.timevalues, q.Zea, ...
         sim.timevalues, q.QuenchersXanthophyll, ...
         sim.timevalues, q.Anth, ...
         sim.timevalues, q.QuenchersLutein, ...
         sim.timevalues, q.QuenchersXanthophyll + q.QuenchersLutein);

legend(ax2, 'Zea', 'Zea Quenchers', 'Anth', 'Lutein Quenchers', 'Total Quenchers')
title(ax2, name2)
grid(ax2, 'off')
ax2.FontSize = 22;

saveas(fig2,name2, "png")

end