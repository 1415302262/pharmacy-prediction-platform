metrics = readtable('results/scaffold_repeat_summary.csv');
disp(metrics)

figure('Position', [100, 100, 1100, 420]);
subplot(1,2,1)
bar(categorical(metrics.model), metrics.rmse_mean)
title('Scaffold Repeat RMSE Mean')
ylabel('RMSE')

subplot(1,2,2)
bar(categorical(metrics.model), metrics.r2_mean)
title('Scaffold Repeat R2 Mean')
ylabel('R2')

sgtitle('Rigorous Scaffold Validation Summary')

