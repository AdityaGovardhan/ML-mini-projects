format short
clear

data = xlsread("p4data.xlsx");

%% principal component analysis (PCA)

[~,dataPca,~,~,explained,~] = pca(data);

fprintf("Percent of total variance in first two principal components is: %f\n\n", sum(explained(1:2)))

data2D = dataPca(:, 1:2);

figure()
plot(data2D(:, 1), data2D(:, 2), '.')
title("Data after PCA")
xlabel("1st Principal Component")
ylabel("2nd Principal Component")

%% hierarchical agglomerative clustering (HAC)

disp("#############")
disp("#### HAC ####")
disp("#############")


linkEval = evalclusters(data, "linkage", "gap", "KList", 1:6, "Distance", "Euclidean")

figure()
plot(linkEval)
title("Gap Statistics for HAC")

linkStruct = linkage(data2D, 'ward', 'euclidean');

cutoff = median([linkStruct(end-4,3) linkStruct(end-3,3)]);

figure()
dendrogram(linkStruct, "ColorThreshold", cutoff)
hold on
yline(cutoff, "--")
title("Dendrogram for HAC")
hold off

hacInd = cluster(linkStruct, "maxclust", 5);

figure()
gscatter(data2D(:, 1), data2D(:, 2), hacInd, "rgbmc", ".", [], "off")
title("HAC clustering")

%% K-Means clustering

disp("###############")
disp("### K-Means ###")
disp("###############")

kmsEval = evalclusters(data, "kmeans", "gap", "KList", 1:6, "Distance", "sqEuclidean")

figure()
plot(kmsEval)
title("Gap Statistics for K-Means")

[kmsInd, centroids] = kmeans(data2D, 5);

figure()
gscatter(data2D(:, 1), data2D(:, 2), kmsInd, "rgbmc", ".", [], "off")
title("K-Means clustering")
hold on
plot(centroids(:, 1), centroids(:, 2), "+", "MarkerFaceColor", "k", "MarkerEdgeColor","k")

centroids