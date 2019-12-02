function [f, g1, g2] = evaluate(X)
n = size(X, 1);

f = zeros(n,1);
g1 = zeros(n,1);
g2 = zeros(n,1);

sections = X(:, 1:260);
coords = X(:, 261:end);

for i=1:n
    [weight, constr1, constr2] = ISCSO_2019(sections(i,:), coords(i,:), 0);
    f(i) = weight;
    g1(i) = constr1;
    g2(i) = constr2;
end

    
end



