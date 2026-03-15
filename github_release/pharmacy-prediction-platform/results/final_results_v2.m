metrics = readtable('results/metrics_summary.csv');
test_metrics = metrics(strcmp(metrics.split, 'test'), :);
[~, idx] = sort(test_metrics.rmse);
test_metrics = test_metrics(idx, :);
disp(test_metrics)

figure('Position', [100, 100, 1000, 420]);
subplot(1,2,1)
bar(categorical(test_metrics.model), test_metrics.rmse)
title('Enhanced QSAR Test RMSE')
ylabel('RMSE')

subplot(1,2,2)
bar(categorical(test_metrics.model), test_metrics.r2)
title('Enhanced QSAR Test R2')
ylabel('R2')

sgtitle('Enhanced Lipophilicity Modeling Results')
