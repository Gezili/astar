clear 
clc
close all

state_hist = load('state_hist.mat')
position_hist = state_hist.position_hist'
Z = position_hist(1:9, :)
rotate3d on
bar3(Z, 0.3)

yticks([1 2 3 4 5 6 7 8 9])
yticklabels({'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'})
xticks([1 2 3 4 5 6 7 8 9])
xticklabels({0 1 2 3 4 5 6 7 8 9})
xlabel('Iteration')
ylabel('Estimated Position')
zlabel('P(x)')